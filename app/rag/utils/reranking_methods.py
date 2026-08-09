"""Helpers for combining the output of multiple retrieval methods."""

from collections.abc import Mapping, Sequence
from numbers import Real
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


def greedy_dartboard_search(
    query_distances: np.ndarray,
    document_distances: np.ndarray,
    documents: Sequence[Any],
    num_results: int,
    diversity_weight: float = DIVERSITY_WEIGHT,
    relevance_weight: float = RELEVANCE_WEIGHT,
    sigma: float = SIGMA,
) -> Tuple[List[Any], List[float]]:
    """Select a relevant and diverse subset using Greedy Dartboard Search.

    ``query_distances`` contains one distance per candidate and
    ``document_distances`` is the square pairwise-distance matrix for those
    candidates. The nearest candidate is selected first. Each later selection
    greedily balances relevance to the query with the additional coverage it
    provides relative to the candidates already selected.

    Returns the selected documents in selection order and their selection
    scores. The first result has the conventional seed score of ``1.0``.
    """
    if not isinstance(num_results, int) or isinstance(num_results, bool):
        raise TypeError("num_results must be an integer")
    if num_results < 0:
        raise ValueError("num_results must be non-negative")

    for name, value in (
        ("diversity_weight", diversity_weight),
        ("relevance_weight", relevance_weight),
        ("sigma", sigma),
    ):
        if not isinstance(value, Real) or isinstance(value, bool):
            raise TypeError(f"{name} must be numeric")
        if not np.isfinite(value):
            raise ValueError(f"{name} must be finite")

    if diversity_weight < 0 or relevance_weight < 0:
        raise ValueError("diversity_weight and relevance_weight must be non-negative")
    if diversity_weight == 0 and relevance_weight == 0:
        raise ValueError("at least one selection weight must be greater than zero")
    if sigma <= 0:
        raise ValueError("sigma must be greater than zero")

    candidate_documents = list(documents)
    candidate_count = len(candidate_documents)
    if candidate_count == 0 or num_results == 0:
        return [], []

    query_distance_array = np.asarray(query_distances, dtype=np.float64)
    if query_distance_array.ndim == 2 and 1 in query_distance_array.shape:
        query_distance_array = query_distance_array.reshape(-1)
    if query_distance_array.shape != (candidate_count,):
        raise ValueError("query_distances must contain one value per document")

    document_distance_array = np.asarray(
        document_distances,
        dtype=np.float64,
    )
    if document_distance_array.shape != (candidate_count, candidate_count):
        raise ValueError(
            "document_distances must be a square matrix matching documents"
        )
    if not np.all(np.isfinite(query_distance_array)):
        raise ValueError("query_distances must contain only finite values")
    if not np.all(np.isfinite(document_distance_array)):
        raise ValueError("document_distances must contain only finite values")
    if np.any(query_distance_array < 0) or np.any(document_distance_array < 0):
        raise ValueError("distances must be non-negative")

    result_count = min(num_results, candidate_count)
    query_probabilities = _log_normal_kernel(query_distance_array, float(sigma))
    document_probabilities = _log_normal_kernel(
        document_distance_array,
        float(sigma),
    )

    first_index = int(np.argmax(query_probabilities))
    selected_indices = [first_index]
    selection_scores = [1.0]
    covered_probabilities = document_probabilities[first_index].copy()

    while len(selected_indices) < result_count:
        updated_coverage = np.maximum(
            covered_probabilities,
            document_probabilities,
        )
        combined_scores = (
            float(diversity_weight) * updated_coverage
            + float(relevance_weight) * query_probabilities[np.newaxis, :]
        )
        candidate_scores = _logsumexp(combined_scores, axis=1)
        candidate_scores[selected_indices] = -np.inf

        best_index = int(np.argmax(candidate_scores))
        selected_indices.append(best_index)
        selection_scores.append(float(candidate_scores[best_index]))
        covered_probabilities = updated_coverage[best_index]

    return (
        [candidate_documents[index] for index in selected_indices],
        selection_scores,
    )


def weighted_reciprocal_rank_fusion(
    ranked_results: Sequence[Sequence[Mapping[str, Any]]],
    weights: Optional[Sequence[float]] = None,
    rank_constant: int = 60,
    limit: Optional[int] = None,
    id_key: str = "id",
) -> List[Dict[str, Any]]:
    """Combine ranked document lists using Weighted Reciprocal Rank Fusion.

    A document at one-based rank ``r`` in result list ``i`` contributes
    ``weights[i] / (rank_constant + r)`` to its final score. Documents may be
    absent from any of the lists. The returned documents contain the fused
    score in ``relevance_score`` and are ordered from most to least relevant.

    Duplicate documents in one input list contribute only at their first
    (best) rank. Documents are matched using ``id_key``.
    """
    if not isinstance(rank_constant, int) or isinstance(rank_constant, bool):
        raise TypeError("rank_constant must be an integer")
    if rank_constant < 0:
        raise ValueError("rank_constant must be non-negative")
    if limit is not None:
        if not isinstance(limit, int) or isinstance(limit, bool):
            raise TypeError("limit must be an integer or None")
        if limit < 0:
            raise ValueError("limit must be non-negative")
        if limit == 0:
            return []

    result_lists = list(ranked_results)
    if not result_lists:
        return []

    if weights is None:
        fusion_weights = [1.0] * len(result_lists)
    else:
        fusion_weights = list(weights)
        if len(fusion_weights) != len(result_lists):
            raise ValueError("weights must contain one value per ranked list")

    if any(
        not isinstance(weight, Real) or isinstance(weight, bool)
        for weight in fusion_weights
    ):
        raise TypeError("weights must be numeric")
    if any(weight < 0 for weight in fusion_weights):
        raise ValueError("weights must be non-negative")
    if not any(weight > 0 for weight in fusion_weights):
        raise ValueError("at least one weight must be greater than zero")

    scores = {}
    documents = {}
    first_seen = {}
    encounter_order = 0

    for results, weight in zip(result_lists, fusion_weights):
        seen_in_list = set()
        for rank, document in enumerate(results, start=1):
            if not isinstance(document, Mapping):
                raise TypeError("each ranked result must be a document mapping")
            if id_key not in document:
                raise ValueError(f"each document must contain the '{id_key}' key")

            document_id = document[id_key]
            try:
                already_seen = document_id in seen_in_list
            except TypeError as exc:
                raise TypeError(f"document '{id_key}' values must be hashable") from exc
            if already_seen:
                continue

            seen_in_list.add(document_id)
            if document_id not in documents:
                documents[document_id] = dict(document)
                first_seen[document_id] = encounter_order
                encounter_order += 1

            scores[document_id] = scores.get(document_id, 0.0) + (
                float(weight) / (rank_constant + rank)
            )

    ranked_ids = sorted(
        scores,
        key=lambda document_id: (-scores[document_id], first_seen[document_id]),
    )
    if limit is not None:
        ranked_ids = ranked_ids[:limit]

    return [
        {
            **documents[document_id],
            "relevance_score": scores[document_id],
        }
        for document_id in ranked_ids
    ]
