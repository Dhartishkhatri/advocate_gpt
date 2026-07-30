# app/rag/api.py

from fastapi import FastAPI
from app.rag.retriever import retrieve
from app.rag.generator import generate_answer

app = FastAPI()

@app.get("/query")
def query(q):
    docs = retrieve(q)
    answer = generate_answer(q, docs)

    return {
        "answer": answer,
        "sources": docs
    }