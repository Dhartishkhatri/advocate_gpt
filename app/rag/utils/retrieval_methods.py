import json
import math
import re
from collections import Counter
from typing import Any, Dict, List, Optional

import numpy as np


def hierarchical_retrieval(
    summary_store,
    detail_store,
    query: str,
    k: int = 5,
    summary_k: int = 3,
) -> List[Dict[str, Any]]:
    """Retrieve parent summaries first, then their most relevant child chunks."""
    if not query or summary_store is None or detail_store is None or k <= 0:
        return []

    from app.ingestion.embedder import embed_batch

    query_embedding = embed_batch([query])[0]
    query_vector = np.asarray([query_embedding], dtype=np.float32)

    summary_count = min(summary_k, summary_store.index.ntotal)
    _, summary_indices = summary_store.index.search(
        query_vector,
        summary_count,
    )
    parent_ids = {
        summary_store.metadata[index]["metadata"]["parent_id"]
        for index in summary_indices[0]
        if index >= 0
    }
    if not parent_ids:
        return []

    distances, detail_indices = detail_store.index.search(
        query_vector,
        detail_store.index.ntotal,
    )

    results = []
    for distance, index in zip(distances[0], detail_indices[0]):
        if index < 0:
            continue

        document = detail_store.metadata[index]
        if document.get("metadata", {}).get("parent_id") not in parent_ids:
            continue

        results.append(
            {
                **document,
                "relevance_score": float(1 / (1 + distance)),
            }
        )
        if len(results) == k:
            break

    return results


def bm25_retrieval(
    query: str,
    documents: List[Dict[str, Any]],
    k: int = 5,
    k1: float = 1.5,
    b: float = 0.75,
    bm25_index: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, Any]]:
    """Return the top ``k`` documents ranked by their BM25 score."""
    if not query or not documents or k <= 0:
        return []

    query_terms = re.findall(r"\w+", query.lower())

    if not query_terms:
        return []

    index = bm25_index
    document_count = index["document_count"]
    if document_count != len(documents):
        raise ValueError("BM25 index does not match document metadata")

    average_length = index["average_document_length"]
    document_frequencies = index["document_frequencies"]

    scores = []
    for document_index, term_frequencies in enumerate(index["term_frequencies"]):
        document_length = index["document_lengths"][document_index]
        length_normalization = (
            1 - b + b * document_length / average_length
            if average_length
            else 1
        )
        score = 0.0

        for term in query_terms:
            frequency = term_frequencies.get(term, 0)
            if not frequency:
                continue

            inverse_document_frequency = math.log(
                1
                + (document_count - document_frequencies[term] + 0.5)
                / (document_frequencies[term] + 0.5)
            )
            score += inverse_document_frequency * (
                frequency * (k1 + 1)
                / (frequency + k1 * length_normalization)
            )

        scores.append((score, document_index))

    scores.sort(key=lambda item: item[0], reverse=True)
    return [
        {**documents[index], "relevance_score": score}
        for score, index in scores[:k]
        if score > 0
    ]


def fusion_retrieval(vectorstore, bm25: Dict[str, Any], query: str, k: int = 5, alpha: float = 0.5,) -> List[Dict[str, Any]]:
    """Combine BM25 and vector-search scores and return the top documents."""
    documents = vectorstore.metadata

    epsilon = 1e-8

    # Get BM25 scores in the same order as the vector-store metadata.
    bm25_results = bm25_retrieval(
        query,
        documents,
        len(documents),
        bm25_index=bm25,
    )
    bm25_by_id = {
        document["id"]: document["relevance_score"]
        for document in bm25_results
    }
    bm25_scores = np.array(
        [bm25_by_id.get(document.get("id"), 0.0) for document in documents],
        dtype=np.float32,
    )

    # FAISS returns L2 distances, so lower values are more relevant.
    from app.ingestion.embedder import embed_batch

    query_embedding = embed_batch([query])[0]
    distances, indices = vectorstore.index.search(
        np.asarray([query_embedding], dtype=np.float32),
        len(documents),
    )
    vector_scores = np.zeros(len(documents), dtype=np.float32)
    for distance, index in zip(distances[0], indices[0]):
        if index >= 0:
            vector_scores[index] = distance

    vector_scores = 1 - (
        (vector_scores - np.min(vector_scores))
        / (np.max(vector_scores) - np.min(vector_scores) + epsilon)
    )
    bm25_scores = (
        (bm25_scores - np.min(bm25_scores))
        / (np.max(bm25_scores) - np.min(bm25_scores) + epsilon)
    )

    combined_scores = alpha * vector_scores + (1 - alpha) * bm25_scores
    sorted_indices = np.argsort(combined_scores)[::-1]

    return [
        {
            **documents[index],
            "relevance_score": float(combined_scores[index]),
        }
        for index in sorted_indices[:k]
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

            for end in range(
                start + 1,
                min(start + max_length + 1, len(relevance_values) + 1),
            ):
                if relevance_values[end - 1] < 0:
                    continue
                if any(
                    start < segment_end and end > segment_start
                    for segment_start, segment_end in best_segments
                ):
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
