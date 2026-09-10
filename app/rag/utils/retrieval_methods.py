import json
import math
import re
from collections import Counter
from typing import Any, Dict, List, Optional
from app.ingestion.pipeline import tokenize_bm25
import numpy as np
import requests
from app import config
from app.ingestion.embedder import embed_batch

def hyde_retrieval(query: str, store, k: int = 5, chunk_size: Optional[int] = None, temperature: float = 0):
    """Generate a hypothetical answer document, then retrieve the chunks closest to it.

    The hypothetical document is written by the configured Ollama model and used as
    the search text instead of the raw question, so the query and the indexed chunks
    sit in the same "answer shaped" region of the embedding space.
    """


    chunk_size = chunk_size or config.CHUNK_SIZE

    prompt = (
        f"Given the question '{query}', generate a hypothetical legal document that "
        "directly answers this question. The document should be detailed and in-depth. "
        f"The document size has to be exactly {chunk_size} characters."
    )

    response = requests.post(
        f"{config.OLLAMA_BASE_URL}/api/generate",
        json={
            "model": config.OLLAMA_MODEL,
            "prompt": prompt,
            "stream": False,
            "temperature": temperature,
        },
        timeout=config.TIMEOUT,
    )
    response.raise_for_status()

    hypothetical_document = response.json().get("response", "").strip()
    # Fall back to the raw question if the model returned nothing usable.
    search_text = hypothetical_document or query

    query_vector = embed_batch([search_text])
    search_k = min(k, store.index.ntotal)
    if search_k <= 0:
        return []

    distances, indices = store.index.search(query_vector, search_k)

    return [
        {
            **store.metadata[index],
            "relevance_score": float(1 / (1 + distance)),
            "hypothetical_document": hypothetical_document,
        }
        for distance, index in zip(distances[0], indices[0])
        if index >= 0
    ]


def hierarchical_retrieval(summary_store, detail_store, query: str,k: int = 5,summary_k: int = 3):
    """Retrieve parent summaries first, then their most relevant child chunks."""

    from app.ingestion.embedder import embed_batch

    query_embedding = embed_batch([query])[0]
    query_vector = np.asarray([query_embedding], dtype=np.float32)

    summary_count = min(summary_k, summary_store.index.ntotal)
    _, summary_indices = summary_store.index.search(query_vector, summary_count)
    parent_ids = {summary_store.metadata[index]["metadata"]["parent_id"] for index in summary_indices[0] if index >= 0}

    if not parent_ids:
        return []

    distances, detail_indices = detail_store.index.search(query_vector, detail_store.index.ntotal)
    results = []
    for distance, index in zip(distances[0], detail_indices[0]):
        if index < 0:
            continue

        document = detail_store.metadata[index]
        if document.get("metadata", {}).get("parent_id") not in parent_ids:
            continue

        results.append({**document,"relevance_score": float(1 / (1 + distance))})
        if len(results) == k:
            break

    return results


def bm25_retrieval(query, bm25_index, k, k1=1.5, b=0.75):
    """Return the top k chunk IDs with their BM25 scores."""

    query_terms = set(tokenize_bm25(query))

    chunk_count = bm25_index["chunk_count"]
    average_chunk_length = bm25_index["average_chunk_length"]
    postings = bm25_index["postings"]
    chunk_frequencies = bm25_index["chunk_frequencies"]
    chunk_lengths = bm25_index["chunk_lengths"]

    scores = {}

    for term in query_terms:
        term_postings = postings.get(term, [])
        if not term_postings:
            continue
        document_frequency = chunk_frequencies[term]

        inverse_document_frequency = math.log(1 + ((chunk_count - document_frequency + 0.5)/ (document_frequency + 0.5)))

        # Each item is: [chunk_id, term_frequency]
        for chunk_id, term_frequency in term_postings:
            chunk_id = str(chunk_id)
            chunk_length = chunk_lengths[chunk_id]

            length_normalization = (1 - b + b * (chunk_length / average_chunk_length) if average_chunk_length else 1)

            term_score = inverse_document_frequency * (term_frequency * (k1 + 1) / (term_frequency + k1 * length_normalization))

            scores[chunk_id] = scores.get(chunk_id, 0.0) + term_score

    ranked_results = sorted(
        scores.items(),
        key=lambda item: item[1],
        reverse=True,
    )[:k]

    return [
        {
            "chunk_id": chunk_id,
            "bm25_score": round(score, 4),
        }
        for chunk_id, score in ranked_results
    ]


def get_best_segments(relevance_values: list, max_length: int, overall_max_length: int,minimum_value: float):
    """Find the highest-scoring non-overlapping segments of chunks."""
    best_segments = []
    scores = []
    total_length = 0

    while total_length < overall_max_length:
        best_segment = None
        best_value = -1000

        for start in range(len(relevance_values)):
            if relevance_values[start] < 0:
                continue

            for end in range(start + 1,min(start + max_length + 1, len(relevance_values) + 1)):
                if relevance_values[end - 1] < 0:
                    continue
                if any(start < segment_end and end > segment_start for segment_start, segment_end in best_segments):
                    continue
                if total_length + end - start > overall_max_length:
                    continue

                segment_value = sum(relevance_values[start:end])
                if segment_value > best_value:
                    best_value = segment_value
                    best_segment = (start, end)

        if best_segment is None or best_value < minimum_value:
            break

        best_segments.append(best_segment)
        scores.append(best_value)
        total_length += best_segment[1] - best_segment[0]

    return best_segments, scores
