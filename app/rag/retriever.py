import json

import numpy as np
import faiss

from app import config
from app.ingestion.utils.faiss import FAISSStore
from app.logger import Logger
from app.rag.utils.reranking_methods import (
    greedy_dartboard_search,
    weighted_reciprocal_rank_fusion,
)
from app.rag.utils.retrieval_methods import (
    bm25_retrieval,
    get_best_segments,
    hierarchical_retrieval
)
retrieval_methods = {"bm25_retrieval": bm25_retrieval,"get_best_segments": get_best_segments,
"hierarchical_retrieval": hierarchical_retrieval}

reranking_methods = {"greedy_dartboard_search": greedy_dartboard_search,
"weighted_reciprocal_rank_fusion": weighted_reciprocal_rank_fusion}



logger = Logger.get_logger(__name__)

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
        from app.ingestion.embedder import embed_batch
        retrieval_method = retrieval_methods.get(config.RETRIEVAL_METHOD)

        if config.RETRIEVAL_METHOD == "hierarchical_retrieval":
            summary_store = FAISSStore.load(config.SUMMARY_VECTOR_STORE_PATH, config.SUMMARY_METADATA_PATH)    
            results = retrieval_method(summary_store, store, question, k=k, summary_k=config.HIERARCHICAL_SUMMARY_K)
        
        elif config.RETRIEVAL_METHOD == "fusion_retrieval":
            with open(config.BM25_INDEX_PATH, "r", encoding="utf-8") as index_file:
                bm25_index = json.load(index_file)

            bm25_results = bm25_retrieval(question, bm25_index, k)
            logger.info(f"BM25 results: {bm25_results}")


            index = faiss.read_index(config.VECTOR_STORE_PATH)
            with open(config.METADATA_PATH, "r", encoding="utf-8") as f:
                metadata = json.load(f)
            metadata_by_id = {chunk["id"]: chunk for chunk in metadata}

            query_vector = embed_batch([question])
            search_k = min(k, index.ntotal)
            vector_scores, indices = index.search(query_vector, search_k)
            vector_results = [
                {
                    "chunk_id": metadata[chunk_index]["id"],
                    "vector_score": float(vector_score),
                }
                for vector_score, chunk_index in zip(vector_scores[0], indices[0])
                if chunk_index >= 0
            ]
            logger.info(f"Vector results: {vector_results}")


        if config.RERANKING_METHOD == "greedy_dartboard_search":
            results = _dartboard_rerank(store, query_vector, vector_results, k)
        elif config.RERANKING_METHOD == "weighted_reciprocal_rank_fusion":
            weights = {"bm25": config.FUSION_ALPHA, "vector": 1 - config.FUSION_ALPHA}
            chunks_to_return = weighted_reciprocal_rank_fusion(
                bm25_results, vector_results, weights, id_key="chunk_id"
            )
            results = [
                {
                    **metadata_by_id[chunk_info["id"]],     #unpacking dictionary to include all metadata_by_id fields
                    "rrf_score": chunk_info["rrf_score"],
                    "final_rank": chunk_info["final_rank"],
                }
                for chunk_info in chunks_to_return if chunk_info["id"] in metadata_by_id
            ]
            logger.info(f"RRF results: {chunks_to_return}")
            logger.info(f"Final results: {results}")

    except Exception as exc:
        logger.error(f"Error retrieving documents: {exc}")
        return []

    return results
