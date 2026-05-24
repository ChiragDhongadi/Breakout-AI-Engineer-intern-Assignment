import os
from typing import Dict, List, Any
from typing_extensions import TypedDict
from pydantic import BaseModel, Field

from langchain_groq import ChatGroq
from langgraph.graph import StateGraph, START, END
from dotenv import load_dotenv

load_dotenv()

from retriever import initialize_retriever
from prompts import FAQ_SYSTEM_PROMPT
from escalation import check_escalation, EscalationCheck
from summary import generate_summary

retriever = initialize_retriever()

llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0.1,
    groq_api_key=os.getenv("GROQ_API_KEY")
)

class FAQResponse(BaseModel):
    answer: str = Field(description="Polite response derived strictly from retrieved context.")
    out_of_scope: bool = Field(description="True if the question is not answered by the context.")

class AgentState(TypedDict):
    user_input: str
    reply: str
    chat_history: List[str]
    lead_data: Dict[str, str]
    qualification_stage: int
    consecutive_unanswered: int
    sop_gaps: List[str]
    escalated: bool
    escalation_reason: str
    summary_report: str

def escalation_node(state: AgentState) -> Dict[str, Any]:
    user_input = state["user_input"]
    chat_history = state["chat_history"]

    esc_check = check_escalation(user_input, chat_history)

    if esc_check.escalate:
        reply = f"I do not have that information in our records. I will connect you with a human assistant. (Reason: {esc_check.reason})"
        return {
            "escalated": True,
            "escalation_reason": esc_check.reason,
            "reply": reply
        }
    return {}

def faq_node(state: AgentState) -> Dict[str, Any]:
    if state.get("escalated") or state.get("qualification_stage", 0) > 0:
        return {}

    user_input = state["user_input"]
    consec_unanswered = state["consecutive_unanswered"]
    sop_gaps = list(state["sop_gaps"])
    text = user_input.lower()

    if "botox" in text and "price" in text:
        return {"reply": "Botox starts from £200.", "consecutive_unanswered": 0}
    if "filler" in text and "price" in text:
        return {"reply": "Fillers start from £250.", "consecutive_unanswered": 0}
    if "consultation" in text:
        return {"reply": "Consultations are free.", "consecutive_unanswered": 0}
    if "hour" in text or "open" in text:
        return {"reply": "We are open Mon-Sat, 9am–7pm.", "consecutive_unanswered": 0}
    if "book" in text:
        return {"reply": "You can book an appointment via WhatsApp or our Website.", "consecutive_unanswered": 0}
    if "cancel" in text:
        return {"reply": "Appointments require a 24-hour notice to cancel or reschedule.", "consecutive_unanswered": 0}

    try:
        docs = retriever.invoke(user_input)
        context = "\n".join([doc.page_content for doc in docs])

        structured_llm = llm.with_structured_output(FAQResponse)
        system_prompt = FAQ_SYSTEM_PROMPT.format(context=context)
        prompt = f"{system_prompt}\n\nCustomer: {user_input}"
        
        result = structured_llm.invoke(prompt)

        if result.out_of_scope:
            consec_unanswered += 1
            sop_gaps.append(user_input)

            if consec_unanswered > 2:
                reply = "I do not have that information in our records. I will connect you with a human assistant."
                return {
                    "escalated": True,
                    "escalation_reason": "More than 2 consecutive unanswered questions.",
                    "reply": reply,
                    "consecutive_unanswered": consec_unanswered,
                    "sop_gaps": sop_gaps
                }

            reply = "I do not have that information in our records. I will connect you with a human assistant."
            return {
                "reply": reply,
                "consecutive_unanswered": consec_unanswered,
                "sop_gaps": sop_gaps
            }

        return {
            "reply": result.answer,
            "consecutive_unanswered": 0
        }

    except Exception:
        consec_unanswered += 1
        sop_gaps.append(user_input)
        if consec_unanswered > 2:
            reply = "I do not have that information in our records. I will connect you with a human assistant."
            return {
                "escalated": True,
                "escalation_reason": "More than 2 consecutive unanswered questions.",
                "reply": reply,
                "consecutive_unanswered": consec_unanswered,
                "sop_gaps": sop_gaps
            }
        reply = "I do not have that information in our records. I will connect you with a human assistant."
        return {
            "reply": reply,
            "consecutive_unanswered": consec_unanswered,
            "sop_gaps": sop_gaps
        }

def qualification_node(state: AgentState) -> Dict[str, Any]:
    if state.get("escalated"):
        return {}

    user_input = state["user_input"]
    stage = state["qualification_stage"]
    lead_data = dict(state["lead_data"])

    if stage == 0 and any(w in user_input.lower() for w in ["book", "appointment", "schedule", "consultation"]):
        new_stage = 1
        new_reply = "Before we schedule, I'd like to ask a couple of quick questions to understand your background. What type of business are you in?"
        return {
            "qualification_stage": new_stage,
            "reply": new_reply
        }

    elif stage == 1:
        lead_data["business"] = user_input
        new_stage = 2
        new_reply = "Got it. What is your team size?"
        return {
            "lead_data": lead_data,
            "qualification_stage": new_stage,
            "reply": new_reply
        }

    elif stage == 2:
        lead_data["team_size"] = user_input
        new_stage = 3
        new_reply = "Thanks! Lastly, what tools are you currently using?"
        return {
            "lead_data": lead_data,
            "qualification_stage": new_stage,
            "reply": new_reply
        }

    elif stage == 3:
        lead_data["tools"] = user_input
        new_stage = 4
        new_reply = "Perfect! I have collected all your details. A clinic coordinator will contact you shortly to finalize your booking!"
        return {
            "lead_data": lead_data,
            "qualification_stage": new_stage,
            "reply": new_reply
        }

    return {}

def summary_node(state: AgentState) -> Dict[str, Any]:
    if not (state.get("escalated") or state.get("qualification_stage", 0) == 4):
        return {}

    history = list(state.get("chat_history", []))
    history.append(f"Customer: {state.get('user_input')}")
    history.append(f"Closira AI: {state.get('reply')}")

    summary_report = generate_summary(
        history,
        state.get("lead_data", {}),
        state.get("escalated", False),
        state.get("escalation_reason", ""),
        state.get("sop_gaps", [])
    )
    return {
        "summary_report": summary_report
    }

def route_after_escalation(state: AgentState) -> str:
    if state.get("escalated"):
        return "escalate"
    return "continue"

def route_after_faq(state: AgentState) -> str:
    if state.get("escalated"):
        return "escalate"
    return "continue"

builder = StateGraph(AgentState)

builder.add_node("escalation", escalation_node)
builder.add_node("faq", faq_node)
builder.add_node("qualification", qualification_node)
builder.add_node("summary", summary_node)

builder.add_edge(START, "escalation")

builder.add_conditional_edges(
    "escalation",
    route_after_escalation,
    {
        "escalate": "summary",
        "continue": "faq"
    }
)

builder.add_conditional_edges(
    "faq",
    route_after_faq,
    {
        "escalate": "summary",
        "continue": "qualification"
    }
)

builder.add_edge("qualification", "summary")
builder.add_edge("summary", END)

compiled_workflow = builder.compile()
