import os
import sys
import json
from dotenv import load_dotenv

load_dotenv()

from workflow import compiled_workflow

chat_history = []
lead_data = {
    "business": "Not collected",
    "team_size": "Not collected",
    "tools": "Not collected"
}
qualification_stage = 0
consecutive_unanswered = 0
sop_gaps = []
escalated = False
escalation_reason = ""
summary_report = ""

def persist_logs():
    log_data = {
        "chat_history": chat_history,
        "lead_data": lead_data,
        "qualification_stage": qualification_stage,
        "consecutive_unanswered": consecutive_unanswered,
        "sop_gaps": sop_gaps,
        "escalated": escalated,
        "escalation_reason": escalation_reason,
        "summary_report": summary_report
    }
    try:
        with open("conversation_log.json", "w", encoding="utf-8") as f:
            json.dump(log_data, f, indent=2)
    except Exception as e:
        print(f"[ERROR] Failed to save conversation_log.json: {e}")

def main():
    global escalated, escalation_reason, qualification_stage, lead_data, consecutive_unanswered, sop_gaps, summary_report
    
    print("=" * 60)
    print("      CLOSIRA AI CUSTOMER WORKFLOW - BLOOM AESTHETICS")
    print("=" * 60)
    print("AI Support Agent initialized successfully.\n")
    print("Closira AI: Hello! Welcome to Bloom Aesthetics Clinic. I am Closira, your digital assistant. How can I help you today?")
    
    persist_logs()
    
    while True:
        try:
            user_input = input("\nCustomer: ").strip()
            
            if not user_input:
                continue
                
            if user_input.lower() in ["exit", "quit", "bye", "goodbye"]:
                print("\nClosira AI: Thank you for contacting Bloom Aesthetics Clinic. Have a wonderful day! Goodbye!")
                break
                
            if user_input.lower() == "summary":
                print("\n[INFO] Ending session and generating summary report...")
                break

            print("Thinking...")
            
            state_input = {
                "user_input": user_input,
                "chat_history": list(chat_history),
                "lead_data": lead_data,
                "qualification_stage": qualification_stage,
                "consecutive_unanswered": consecutive_unanswered,
                "sop_gaps": list(sop_gaps),
                "escalated": escalated,
                "escalation_reason": escalation_reason,
                "summary_report": summary_report,
                "reply": ""
            }

            state_output = compiled_workflow.invoke(state_input)

            lead_data = state_output["lead_data"]
            qualification_stage = state_output["qualification_stage"]
            consecutive_unanswered = state_output["consecutive_unanswered"]
            sop_gaps = state_output["sop_gaps"]
            escalated = state_output["escalated"]
            escalation_reason = state_output["escalation_reason"]
            summary_report = state_output.get("summary_report", "")
            reply = state_output["reply"]

            chat_history.append(f"Customer: {user_input}")
            chat_history.append(f"Closira AI: {reply}")

            persist_logs()

            print(f"Closira AI: {reply}")
            
            if escalated or qualification_stage == 4:
                break
                
        except (KeyboardInterrupt, EOFError):
            print("\n\n[INFO] Session terminated.")
            break
        except Exception as e:
            print(f"\n[ERROR] An error occurred: {e}")
            break

    print("\nGenerating final conversation summary report...")
    if not summary_report:
        from summary import generate_summary
        summary_report = generate_summary(chat_history, lead_data, escalated, escalation_reason, sop_gaps)
    print(summary_report)

if __name__ == "__main__":
    main()
