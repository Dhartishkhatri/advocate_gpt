# app/rag/generator.py
from typing import Any, Dict, List
from dotenv import load_dotenv
from app.config import LLM_PROVIDER
from app.rag.utils.generate_answer import generate_answer_ollama
from app.logger import Logger
load_dotenv()

logger = Logger.get_logger(__name__)

def generate_answer(query,docs):
    """Generate a simple answer from retrieved documents."""
    try:
        answer = generate_answer_ollama(query, docs)

        return {
            "answer": answer,
            "provider": LLM_PROVIDER,
        }
    except Exception as exc:
        logger.error(f"Error generating answer: {exc}")
        return None
