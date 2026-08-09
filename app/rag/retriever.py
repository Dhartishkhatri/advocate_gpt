import json

import numpy as np

from app import config
from app.ingestion.utils.faiss import FAISSStore
from app.logger import Logger
from app.rag.utils.reranking_methods import (
    greedy_dartboard_search,
    weighted_reciprocal_rank_fusion,
)
from app.rag.utils.retrieval_methods import (
    bm25_retrieval,
    fusion_retrieval,
    get_best_segments,
    hierarchical_retrieval,
)

logger = Logger.get_logger(__name__)


retrieval_methods = {
    "bm25_retrieval": bm25_retrieval,
    "fusion_retrieval": fusion_retrieval,
    "get_best_segments": get_best_segments,
    "hierarchical_retrieval": hierarchical_retrieval,
}

reranking_methods = {
    "greedy_dartboard_search": greedy_dartboard_search,
    "weighted_reciprocal_rank_fusion": weighted_reciprocal_rank_fusion,
}


def _embed_query(query):
    """Embed one query while keeping model initialization lazy."""
    from app.ingestion.embedder import embed_batch

    return np.asarray(embed_batch([query])[0], dtype=np.float32)


def _vector_ranking(vectorstore, query_vector):
    """Return all vector-store documents in nearest-neighbour order."""
    if vectorstore.index.ntotal <= 0:
        return []

    _, indices = vectorstore.index.search(
        np.asarray([query_vector], dtype=np.float32),
        vectorstore.index.ntotal,
    )
    return [
        vectorstore.metadata[index]
        for index in indices[0]
        if index >= 0
    ]


def _dartboard_rerank(vectorstore, query_vector, documents, k):
    """Rerank retrieved candidates for relevance and vector-space diversity."""
    if not documents or k <= 0:
        return []

    metadata_indices = {
        document.get("id"): index
        for index, document in enumerate(vectorstore.metadata)
        if document.get("id") is not None
    }
    candidates = [
        document for document in documents if document.get("id") in metadata_indices
    ]
    if not candidates:
        return documents[:k]

    candidate_vectors = np.asarray(
        [
            vectorstore.index.reconstruct(metadata_indices[document["id"]])
            for document in candidates
        ],
        dtype=np.float32,
    )
    vector_norms = np.linalg.norm(candidate_vectors, axis=1, keepdims=True)
    candidate_vectors = candidate_vectors / np.maximum(vector_norms, 1e-12)

    query_norm = max(float(np.linalg.norm(query_vector)), 1e-12)
    query_vector = query_vector / query_norm

    query_distances = np.maximum(
        0.0,
        1.0 - candidate_vectors @ query_vector,
    )
    document_distances = np.maximum(
        0.0,
        1.0 - candidate_vectors @ candidate_vectors.T,
    )
    selected_documents, selection_scores = greedy_dartboard_search(
        query_distances,
        document_distances,
        candidates,
        k,
    )
    return [
        {**document, "relevance_score": score}
        for document, score in zip(selected_documents, selection_scores)
    ]


def retrieve(question, store=None, k=5):
    """Retrieve documents using the method selected in app.config."""

    try:
        retrieval_method = retrieval_methods.get(config.RETRIEVAL_METHOD)

        if config.RETRIEVAL_METHOD == "hierarchical_retrieval":
            summary_store = FAISSStore.load(
                config.SUMMARY_VECTOR_STORE_PATH,
                config.SUMMARY_METADATA_PATH,
            )
            detail_store = store or FAISSStore.load()
            results = retrieval_method(
                summary_store,
                detail_store,
                question,
                k=k,
                summary_k=config.HIERARCHICAL_SUMMARY_K,
            )
        else:
            store = store or FAISSStore.load()
            if store is None:
                raise ValueError("FAISS index not found. Ingest PDFs first.")

            with open(config.BM25_INDEX_PATH,"r",encoding="utf-8") as index_file:
                bm25_index = json.load(index_file)

            if config.RETRIEVAL_METHOD == "fusion_retrieval":
                query_vector = _embed_query(question)
                vector_results = _vector_ranking(store, query_vector)
                bm25_results = bm25_retrieval(
                    question,
                    store.metadata,
                    k=len(store.metadata),
                    bm25_index=bm25_index,
                )
                candidate_count = min(len(store.metadata), max(k * 3, k))

            if config.RERANKING_METHOD == "greedy_dartboard_search":
                results = _dartboard_rerank(store, query_vector, candidates, k)
            elif config.RERANKING_METHOD == "weighted_reciprocal_rank_fusion":  
                candidates = weighted_reciprocal_rank_fusion(
                    [vector_results, bm25_results],
                    weights=[config.FUSION_ALPHA, 1 - config.FUSION_ALPHA],
                    limit=candidate_count,
                )
            else:
                if retrieval_method is None:
                    raise ValueError(
                        f"Unknown retrieval method: {config.RETRIEVAL_METHOD}"
                    )
                results = retrieval_method(
                    question,
                    store.metadata,
                    k=k,
                    bm25_index=bm25_index,
                )
    except Exception as exc:
        logger.error(f"Error retrieving documents: {exc}")
        return []

    return results
