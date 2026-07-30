import json

from app import config
from app.ingestion.utils.faiss import FAISSStore
from app.logger import Logger
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

            with open(
                config.BM25_INDEX_PATH,
                "r",
                encoding="utf-8",
            ) as index_file:
                bm25_index = json.load(index_file)

            if config.RETRIEVAL_METHOD == "fusion_retrieval":
                results = retrieval_method(
                    store,
                    bm25_index,
                    question,
                    k=k,
                    alpha=config.FUSION_ALPHA,
                )
            else:
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
