# app/ingestion/embedder.py

from typing import List
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from app.config import EMBEDDING_MODEL


# Initialize model
model = SentenceTransformer(EMBEDDING_MODEL)

def embed_batch(texts):
    """Generate embeddings using SentenceTransformer
    parameters:
    -----------
    texts : list
        List of text strings to embed.
        
    returns:
    --------
    embeddings : np.ndarray
        Array of embeddings for the input texts.
    """
    embeddings = model.encode(texts, convert_to_numpy=True)
    embeddings = np.asarray(embeddings, dtype=np.float32)
    faiss.normalize_L2(embeddings)
    return embeddings
