"""Reciprocal Rank Fusion (RRF) for combining search results.

RRF is a simple but effective method for fusing ranked lists from
multiple retrieval systems without requiring score normalization.
"""

from loguru import logger

from ..entities.search_result import SearchResult


def reciprocal_rank_fusion(
    result_lists: list[list[SearchResult]], k: int = 60, top_k: int = 5
) -> list[SearchResult]:
    """Combine multiple ranked result lists using RRF.

    RRF Formula: score(chunk) = sum(1 / (k + rank_i))
    where rank_i is the rank of the chunk in result list i.

    This method:
    - Doesn't require score normalization
    - Gives higher weight to chunks appearing in multiple lists
    - Downranks results that appear low in any list

    Args:
        result_lists: List of ranked result lists to combine
        k: RRF constant (default 60, as per original paper)
        top_k: Number of final results to return

    Returns:
        Combined and re-ranked results, sorted by RRF score descending

    Example:
        >>> vector_results = vector_store.search(query_vec, top_k=10)
        >>> text_results = text_store.search_fulltext(query_text, top_k=10)
        >>> fused = reciprocal_rank_fusion([vector_results, text_results])
    """
    if not result_lists:
        return []

    # Filter out empty lists
    result_lists = [lst for lst in result_lists if lst]
    if not result_lists:
        return []

    # Calculate RRF scores
    rrf_scores: dict[str, float] = {}  # chunk_id -> RRF score
    chunk_map: dict[str, SearchResult] = {}  # chunk_id -> SearchResult

    for result_list in result_lists:
        for rank, search_result in enumerate(result_list, start=1):
            chunk_id = search_result.chunk.id

            # Store chunk (use first occurrence)
            if chunk_id not in chunk_map:
                chunk_map[chunk_id] = search_result

            # Add RRF contribution
            rrf_contribution = 1.0 / (k + rank)
            rrf_scores[chunk_id] = rrf_scores.get(chunk_id, 0.0) + rrf_contribution

    # Create new SearchResults with RRF scores
    fused_results = [
        SearchResult(chunk=chunk_map[chunk_id].chunk, score=score)
        for chunk_id, score in rrf_scores.items()
    ]

    # Sort by RRF score descending
    fused_results.sort(key=lambda x: x.score, reverse=True)

    logger.debug(
        f"RRF fusion: combined {len(result_lists)} lists "
        f"({sum(len(lst) for lst in result_lists)} total results) "
        f"-> {len(fused_results)} unique chunks, returning top {top_k}"
    )

    return fused_results[:top_k]


def weighted_rrf(
    result_lists: list[list[SearchResult]], weights: list[float], k: int = 60, top_k: int = 5
) -> list[SearchResult]:
    """RRF with weighted contributions from each result list.

    Weighted RRF: score(chunk) = sum(w_i / (k + rank_i))
    where w_i is the weight for result list i.

    Use this when you want to favor one retrieval method over another.

    Args:
        result_lists: List of ranked result lists to combine
        weights: Weight for each result list (must match length)
        k: RRF constant (default 60)
        top_k: Number of final results to return

    Returns:
        Combined and re-ranked results with weighted RRF scores

    Raises:
        ValueError: If weights length doesn't match result_lists length

    Example:
        >>> # Favor vector search over text search
        >>> fused = weighted_rrf(
        ...     [vector_results, text_results],
        ...     weights=[0.7, 0.3]  # 70% vector, 30% text
        ... )
    """
    if not result_lists:
        return []

    # Validate weights
    if len(weights) != len(result_lists):
        raise ValueError(
            f"Weights length ({len(weights)}) must match result_lists length ({len(result_lists)})"
        )

    # Normalize weights to sum to 1.0
    weight_sum = sum(weights)
    if weight_sum == 0:
        raise ValueError("Sum of weights must be > 0")
    normalized_weights = [w / weight_sum for w in weights]

    # Filter out empty lists and corresponding weights
    filtered_lists = []
    filtered_weights = []
    for lst, weight in zip(result_lists, normalized_weights, strict=True):
        if lst:
            filtered_lists.append(lst)
            filtered_weights.append(weight)

    if not filtered_lists:
        return []

    # Calculate weighted RRF scores
    rrf_scores: dict[str, float] = {}
    chunk_map: dict[str, SearchResult] = {}

    for result_list, weight in zip(filtered_lists, filtered_weights, strict=True):
        for rank, search_result in enumerate(result_list, start=1):
            chunk_id = search_result.chunk.id

            if chunk_id not in chunk_map:
                chunk_map[chunk_id] = search_result

            # Add weighted RRF contribution
            weighted_contribution = weight / (k + rank)
            rrf_scores[chunk_id] = rrf_scores.get(chunk_id, 0.0) + weighted_contribution

    # Create new SearchResults with weighted RRF scores
    fused_results = [
        SearchResult(chunk=chunk_map[chunk_id].chunk, score=score)
        for chunk_id, score in rrf_scores.items()
    ]

    # Sort by weighted RRF score descending
    fused_results.sort(key=lambda x: x.score, reverse=True)

    logger.debug(
        f"Weighted RRF fusion: combined {len(filtered_lists)} lists "
        f"with weights {[f'{w:.2f}' for w in filtered_weights]} "
        f"-> {len(fused_results)} unique chunks, returning top {top_k}"
    )

    return fused_results[:top_k]
