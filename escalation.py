import os
from typing import Optional
from pydantic import BaseModel, Field
from langchain_groq import ChatGroq
from dotenv import load_dotenv

load_dotenv()

class EscalationCheck(BaseModel):
    escalate: bool = Field(description="True if escalation criteria are met.")
    reason: Optional[str] = Field(None, description="The reason for human handoff.")

llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0.1,
    groq_api_key=os.getenv("GROQ_API_KEY")
)

def check_escalation(user_input, chat_history):
    text = user_input.lower()

    angry_keywords = ["frustrated", "terrible", "bad service", "unhappy", "angry", "disappointed", "worst", "hate"]
    for word in angry_keywords:
        if word in text:
            return EscalationCheck(escalate=True, reason="Angry sentiment detected")

    complaint_keywords = ["complain", "reaction", "pain", "swollen", "bruised", "infection", "bad job", "ruined"]
    for word in complaint_keywords:
        if word in text:
            return EscalationCheck(escalate=True, reason="Customer complaint detected")

    medical_keywords = ["pregnant", "safe", "allergy", "allergic", "migraine", "side effect", "medical", "disease", "treatment safe"]
    for word in medical_keywords:
        if word in text:
            return EscalationCheck(escalate=True, reason="Medical question detected")

    pricing_keywords = ["discount", "negotiate", "bargain", "cheaper", "reduce price", "promo code", "package deal"]
    for word in pricing_keywords:
        if word in text:
            return EscalationCheck(escalate=True, reason="Pricing negotiation detected")

    human_keywords = ["human", "speak to someone", "representative", "real person", "agent", "connect me", "operator", "practitioner"]
    for word in human_keywords:
        if word in text:
            return EscalationCheck(escalate=True, reason="Explicit human request")

    try:
        history_context = "\n".join(chat_history[-3:])
        structured_llm = llm.with_structured_output(EscalationCheck)
        
        prompt = (
            "You are a safety supervisor checking if a conversation must be escalated to a human clinic staff.\n"
            "Trigger escalation if customer expresses complaints, medical issues, bargaining, or angry sentiment.\n\n"
            f"History:\n{history_context}\n"
            f"Customer Message: {user_input}"
        )
        return structured_llm.invoke(prompt)
    except Exception:
        return EscalationCheck(escalate=False)
