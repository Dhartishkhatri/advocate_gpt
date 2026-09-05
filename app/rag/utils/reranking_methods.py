"""Helpers for combining the output of multiple retrieval methods."""

from collections.abc import Mapping, Sequence
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


DIVERSITY_WEIGHT = 1.0
RELEVANCE_WEIGHT = 1.0
SIGMA = 0.1


def _log_normal_kernel(distances: np.ndarray, sigma: float) -> np.ndarray:
    """Convert distances to Gaussian log-kernel values."""
    return (
        -np.log(sigma)
        - 0.5 * np.log(2 * np.pi)
        - np.square(distances) / (2 * sigma**2)
    )


def _logsumexp(values: np.ndarray, axis: int) -> np.ndarray:
    """Compute log-sum-exp without requiring SciPy."""
    maximum = np.max(values, axis=axis, keepdims=True)
    result = maximum + np.log(
        np.sum(np.exp(values - maximum), axis=axis, keepdims=True)
    )
    return np.squeeze(result, axis=axis)


def greedy_dartboard_search(query_distances,document_distances,documents,num_results, diversity_weight = DIVERSITY_WEIGHT,
    relevance_weight = RELEVANCE_WEIGHT,sigma = SIGMA):
    """
    Perform greedy dartboard search to select top k documents balancing relevance and diversity.
    
    Args:
        query_distances: Distance between query and each document
        document_distances: Pairwise distances between documents
        documents: List of document texts
        num_results: Number of documents to return
    
    Returns:
        Tuple containing:
        - List of selected document texts
        - List of selection scores for each document
    """
     # Avoid division by zero in probability calculations
    sigma = max(SIGMA, 1e-5)
    
    # Convert distances to probability distributions
    query_probabilities = _log_normal_kernel(query_distances, sigma)
    document_probabilities = _log_normal_kernel(document_distances, sigma)
    
    # Initialize with most relevant document
    
    most_relevant_idx = np.argmax(query_probabilities)
    selected_indices = np.array([most_relevant_idx])
    selection_scores = [1.0] # dummy score for the first document
    # Get initial distances from the first selected document
    max_distances = document_probabilities[most_relevant_idx]
    
    # Select remaining documents
    while len(selected_indices) < num_results:
        # Update maximum distances considering new document
        updated_distances = np.maximum(max_distances, document_probabilities)
        
        # Calculate combined diversity and relevance scores
        combined_scores = (
            updated_distances * diversity_weight +
            query_probabilities * relevance_weight
        )
        
        # Normalize scores and mask already selected documents
        normalized_scores = _logsumexp(combined_scores, axis=1)
        normalized_scores[selected_indices] = -np.inf
        
        # Select best remaining document
        best_idx = np.argmax(normalized_scores)
        best_score = np.max(normalized_scores)
        
        # Update tracking variables
        max_distances = updated_distances[best_idx]
        selected_indices = np.append(selected_indices, best_idx)
        selection_scores.append(best_score)
    
    # Return selected documents and their scores
    selected_documents = [documents[i] for i in selected_indices]
    return selected_documents, selection_scores


def weighted_reciprocal_rank_fusion(bm_25_results, vector_results, weights, rank_constant = 60, limit = None, id_key: str = "id"):
    
    """Combine ranked document lists using weighted reciprocal rank fusion.
    Parameters:
    -----------
    bm_25_results : List[Dict[str, Any]]
        Ranked list of documents from BM25 retrieval.
        example: [{"id": "doc1", "score": 0.9}, {"id": "doc2", "score": 0.8}, ...]
    vector_results : List[Dict[str, Any]]
        Ranked list of documents from vector-based retrieval.
        example: [{"id": "doc3", "score": 0.95}, {"id": "doc4", "score": 0.85}, ...]
    weights : dict (optional)
        Weights for each ranked list. If None, equal weights are used.
        example: {"bm25": 0.6, "vector": 0.4} for BM25 and vector results respectively.
    rank_constant : int (optional)
        Constant to adjust the influence of rank in the scoring formula.
    limit : int (optional)
        Maximum number of documents to return. If None, all documents are returned.
    id_key : str (optional)
        Key in the document mapping to use as the unique identifier. Default is 'id'.
    
    Returns:
    --------
    ranked_ids : List[Dict[str, Any]]
        A list of document IDs sorted by their combined scores, limited to the specified number.
    """
    fused_scores = {}

    for rank, result in enumerate(bm_25_results, start=1):
        chunk_id = result[id_key]
        fused_scores[chunk_id] = fused_scores.get(chunk_id, 0) +  weights.get("bm25", 1) / (rank_constant + rank)

    for rank, result in enumerate(vector_results, start=1):
        chunk_id = result[id_key]
        fused_scores[chunk_id] = fused_scores.get(chunk_id, 0) + weights.get("vector", 1) / (rank_constant + rank)

    ranked_results = sorted(fused_scores.items(), key=lambda item: item[1], reverse=True)

    return [
        {
            "id": chunk_id,
            "rrf_score": round(score, 6),
            "final_rank": rank,
        }
        for rank, (chunk_id, score) in enumerate(ranked_results, start=1)
    ]
