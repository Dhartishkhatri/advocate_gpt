from typing import Any, Dict, List

import requests
from dotenv import load_dotenv
from app.config import OLLAMA_BASE_URL, OLLAMA_MODEL, TEMPERATURE, TIMEOUT
from app.logger import Logger
load_dotenv()

logger = Logger.get_logger(__name__)

def generate_answer_ollama(query, docs):
    """Generate a response from Ollama using the retrieved context."""
    context_parts = [d.get("text", "") for d in docs if isinstance(d, dict) and d.get("text")]
    context = "\n\n".join(context_parts) if context_parts else "No relevant context available."

    prompt = f"""You are a legal assistant.

Answer only from the provided context.
Keep the answer concise and relevant.

Context:
{context[:2000]}

Question:
{query}

Answer:"""
    response = requests.post(
        f"{OLLAMA_BASE_URL}/api/generate",
        json={
            "model": OLLAMA_MODEL,
            "prompt": prompt,
            "stream": False,
            "temperature": TEMPERATURE,
        },
        timeout=TIMEOUT,
    )
    if response.status_code == 200:
        answer = response.json().get("response", "No response generated")
        logger.info("Generated answer with Ollama")
        return answer

    logger.error(f"Ollama error: {response.status_code} - {response.text}")
    return None