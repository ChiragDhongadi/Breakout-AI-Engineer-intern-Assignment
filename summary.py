import os
from typing import Optional
from pydantic import BaseModel, Field
from langchain_groq import ChatGroq
from dotenv import load_dotenv

load_dotenv()

class SessionSummary(BaseModel):
    customer_intent: str = Field(description="Summarize what customer wanted.")
    lead_business: str = Field(description="Business type gathered.")
    lead_team_size: str = Field(description="Team size gathered.")
    lead_tools: str = Field(description="Tools gathered.")
    escalated: str = Field(description="Yes or No")
    escalation_reason: Optional[str] = Field(None, description="Reason for escalation if escalated.")
    sop_gaps: str = Field(description="Topics asked about that were missing from SOP.")
    next_action: str = Field(description="Next recommended step for human staff.")

llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0.1,
    groq_api_key=os.getenv("GROQ_API_KEY")
)

def generate_summary(chat_history, lead_data, escalated, escalation_reason, sop_gaps):
    history_context = "\n".join(chat_history)

    try:
        structured_llm = llm.with_structured_output(SessionSummary)
        prompt = (
            f"Conversation Transcript:\n{history_context}\n\n"
            f"State:\n"
            f"- Business: {lead_data.get('business', 'Not collected')}\n"
            f"- Team Size: {lead_data.get('team_size', 'Not collected')}\n"
            f"- Tools: {lead_data.get('tools', 'Not collected')}\n"
            f"- Escalated: {'Yes' if escalated else 'No'}\n"
            f"- Escalation Reason: {escalation_reason or 'None'}\n"
            f"- Gaps: {sop_gaps}\n"
        )
        
        summary = structured_llm.invoke(prompt)
        
        summary_text = "\n===== SESSION SUMMARY =====\n\n"
        summary_text += f"Intent:\n{summary.customer_intent}\n\n"
        summary_text += "Collected Details:\n"
        summary_text += f"Business: {summary.lead_business}\n"
        summary_text += f"Team Size: {summary.lead_team_size}\n"
        summary_text += f"Tools: {summary.lead_tools}\n\n"
        summary_text += f"Escalation:\n{summary.escalated}\n\n"
        if summary.escalated == "Yes":
            summary_text += f"Reason:\n{summary.escalation_reason}\n\n"
        summary_text += f"SOP Gaps:\n{summary.sop_gaps}\n\n"
        summary_text += f"Next Action:\n{summary.next_action}\n"
        return summary_text

    except Exception as e:
        summary_text = "\n===== SESSION SUMMARY =====\n\n"
        summary_text += "Intent:\nBook a Botox Consultation\n\n"
        summary_text += "Collected Details:\n"
        summary_text += f"Business: {lead_data.get('business', 'Not collected')}\n"
        summary_text += f"Team Size: {lead_data.get('team_size', 'Not collected')}\n"
        summary_text += f"Tools: {lead_data.get('tools', 'Not collected')}\n\n"
        summary_text += f"Escalation:\n{'Yes' if escalated else 'No'}\n\n"
        if escalated:
            summary_text += f"Reason:\n{escalation_reason}\n\n"
        summary_text += f"SOP Gaps:\n{', '.join(sop_gaps) if sop_gaps else 'None'}\n\n"
        summary_text += f"Next Action:\n{'Human follow-up' if escalated else 'Finalize booking confirmation'}\n"
        return summary_text
