import os
import json
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document

def initialize_retriever():
    try:
        with open("sop.json", "r", encoding="utf-8") as f:
            sop = json.load(f)
    except FileNotFoundError:
        print("[ERROR] sop.json file not found in directory.")
        return None

    documents = []

    hours_text = f"Bloom Aesthetics Clinic hours: open {sop.get('hours')}. Closed Sundays."
    documents.append(Document(page_content=hours_text, metadata={"section": "hours"}))

    booking_text = f"Booking instructions: schedule appointments through {sop.get('booking')}."
    documents.append(Document(page_content=booking_text, metadata={"section": "booking"}))

    cancel_text = f"Cancellation policy: a {sop.get('cancellation')} notice is strictly required."
    documents.append(Document(page_content=cancel_text, metadata={"section": "cancellation"}))

    for service, price in sop.get("services", {}).items():
        service_text = f"{service} treatment price at Bloom Aesthetics Clinic starts {price}."
        documents.append(Document(page_content=service_text, metadata={"section": "services"}))

    for rule in sop.get("escalate_if", []):
        rule_text = f"Escalation trigger condition: hand off to a human manager if there are {rule}."
        documents.append(Document(page_content=rule_text, metadata={"section": "escalation"}))

    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    db = FAISS.from_documents(documents, embeddings)
    return db.as_retriever(search_kwargs={"k": 2})
