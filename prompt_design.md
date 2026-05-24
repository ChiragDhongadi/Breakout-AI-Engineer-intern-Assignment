# Prompt Engineering Design Document

Hey there! As an AI engineering intern at Closira, I designed a state-of-the-art but super clear prompting and validation system. 

Instead of hardcoding rules, we use a hybrid architecture of **LangChain**, **LangGraph**, and **FAISS** vector searches to answer grounded FAQ questions, validate guardrails, and summarize conversations!

---

## 1. Full FAQ System Prompt

My system prompt is defined in `prompts.py` and is injected dynamically with the retrieved context from our FAISS vector database:

```markdown
You are a customer support AI assistant for Bloom Aesthetics Clinic.

Rules:
1. Only answer from the provided retrieved context.
2. Never make up facts or generate information not present.
3. If information is unavailable, say you don't know and let the user know we will escalate.
4. Maintain a friendly and professional tone.
5. Keep responses concise.

Retrieved SOP Context:
{context}
```

### Rationale:
- **Retrieval Context Boundary**: The model is strictly instructed to *"Only answer from the provided retrieved context."* This keeps the model's active attention window completely focused on the local guidelines, preventing standard internet hallucinations.
- **Negative Constraints**: Instructs the model to strictly state *"I don't know"* if the answer is missing, which triggers our Python escalation logic!

---

## 2. Hallucination Prevention Strategy (RAG Pipeline)

To absolutely prevent hallucinations (which is super critical for a medical boutique), I designed a **Retrieval-Augmented Generation (RAG)** pipeline:

```
SOP JSON (sop.json)
      ↓
Document Chunking
      ↓
HuggingFace Embeddings (sentence-transformers/all-MiniLM-L6-v2)
      ↓
FAISS CPU Vector Index
      ↓
LangChain Retriever
      ↓
Grounded Groq LLM (llama-3.3-70b-versatile)
```

### Why it works:
When a customer asks a question:
1. We embed their question and retrieve the **top 2 most relevant chunks** from FAISS.
2. We inject this context directly into the prompt using LangChain.
3. If the user asks something out of scope, the context doesn't have it, the LLM says "I don't know", and we deflection-route or escalate immediately. The model never makes up prices or fake guidelines!

---

## 3. Structured Pydantic Guardrails & Escalations

I used Pydantic schemas bound directly to LangChain models to enforce strict, safe output shapes:

- `FAQResponse`: Forces the model to evaluate if a question is `out_of_scope: bool`.
- `EscalationCheck`: Analyzes conversation states and flags `escalate: bool` with a human-readable `reason`.
- `SessionSummary`: Restructures final logs into a standard text summary block.

### Confidence-Based Routing:
In `workflow.py`, if the `faq_node` receives an out-of-scope question, it increments the gaps counter. If it reaches **more than 2 consecutive unanswered questions**, the LangGraph state machine automatically triggers a human coordinator transition, complying with the clinic's escalation triggers.

---

## 4. Tone and Persona Design

Our bot's communication style is engineered for a **warm, highly professional aesthetics boutique**:
- **Empathetic and Warm**: Acknowledges patient requests supportively ("I would be absolutely delighted to help you book...").
- **Luxury Concierge Rationale**: Speaks politely and concisely. 
- **Graceful Transitions**: Instantly handoffs without arguing if a user expresses pain, dissatisfaction, or pricing disputes, keeping patient relations safe.
