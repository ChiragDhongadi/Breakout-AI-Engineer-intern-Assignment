# Closira AI Customer Support Workflow (LangGraph, LangChain, FAISS)

Welcome to the **Closira AI Customer Support Agent** prototype! This Python-based agentic workflow is built to handle inbound customer enquiries for a simulated premium small-and-medium business (SMB)—**Bloom Aesthetics Clinic**—using a state-of-the-art but highly clear and student-friendly stack: **LangGraph**, **LangChain**, and **FAISS**.

The system implements a structured **four-stage pipeline** (FAQ answering, Lead Qualification, Escalation detection, and Session Summarization) utilizing standard JSON states, keyword scans, vector search retrievals, and Pydantic validation schemas.

## Demo Video link
https://drive.google.com/file/d/1VCg7Xs9UUmm6fL-pMkocqKP5QVAxgRqM/view?usp=sharing

---

## Project Structure

This project has been built in a clean, simple, and modular folder structure:

```
project/
├── app.py                  # Main CLI interactive loop and conversation_log.json state writer
├── workflow.py             # LangGraph StateGraph builder, AgentState, and graph nodes
├── prompts.py              # Centralized FAQ system prompt template
├── retriever.py            # FAISS CPU initialization and sentence-transformers embeddings
├── escalation.py           # Hybrid keyword and LangChain Pydantic escalation checks
├── summary.py              # Pydantic-based session summary generator
├── sop.json                # Standard Operating Procedure (SOP) clinic rules
├── conversation_log.json   # Session logs persisted dynamically on each turn
├── requirements.txt        # Modular dependency tracker
├── prompt_design.md        # RAG pipelines, safety checks, and design justifications
├── README.md               # Setup instructions, features, and walkthroughs
└── .env                    # Active environment configurations
```

---

## 🛠️ Setup & Installation

### 1. Install Dependencies
Ensure you have Python 3.10+ installed. In your terminal, run:
```bash
pip install -r requirements.txt
```

### 2. Configure Environment
Create a `.env` file in the project folder and paste your Groq API Key:
```env
GROQ_API_KEY=your_groq_api_key_here
```

### 3. Run the App
Start the interactive session:
```bash
python app.py
```

---

## Graph Nodes Architecture

We use a compiled **LangGraph** `StateGraph` which invokes three sequential nodes on each customer message turn:

1.  **Escalation Node (`escalation`)**:
    Pre-emptively intercepts user inputs to check for anger sentiment, pain complaints, clinical queries, or bargaining. Uses simple keyword triggers backed up by a Pydantic `EscalationCheck` LangChain model.
2.  **FAQ Node (`faq`)**:
    If not escalated, retrieves relevant documents from **FAISS** (embedded via `sentence-transformers/all-MiniLM-L6-v2`) and passes them as grounded context to the **Groq LLM** to check if the question is answerable.
3.  **Qualification Node (`qualification`)**:
    If the customer shows booking intent, sequentially prompts the customer for **three lead questions**: *What type of business are you in?*, *What is your team size?*, and *What tools are you currently using?*. All values are written to `lead_data`.

After the LangGraph compiled state runs, the session is persisted directly to [conversation_log.json](file:///d:/Breakout-assignment/conversation_log.json). When the call ends, a complete [summary.py](file:///d:/Breakout-assignment/summary.py) report is printed in the console!

---

## Prototype Limitations & Trade-offs

- **Keyword Matching Intercepts**: Scans keywords first before checking with the LLM to save token cost and guarantee absolute safety on common triggers.
- **Local FAISS Indexing**: FAISS index is generated on boot in-memory using CPU embeddings. In production, this can be written to disk or replaced with a hosted vector database.
- **Single Session Persistence**: Logs are persisted locally to `conversation_log.json`. For multi-channel messaging (e.g. WhatsApp, Webchat), this state should be written to a database (like PostgreSQL or MongoDB) mapped to `session_id`.
