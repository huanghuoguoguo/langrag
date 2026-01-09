"""Qwen Reranker - Reranking using DashScope API"""

from __future__ import annotations

from typing import TYPE_CHECKING

import httpx
from loguru import logger

from ..base import BaseReranker

if TYPE_CHECKING:
    from langrag.entities.query import Query
    from langrag.entities.search_result import SearchResult


class QwenReranker(BaseReranker):
    """Qwen Reranker.

    Reranks retrieval results using Alibaba Cloud DashScope API's Qwen model.

    Args:
        api_key: DashScope API key
        model: Model name, default 'qwen3-rerank'
        instruct: Reranking instruction, can be customized
        timeout: API timeout in seconds, default 30.0

    Usage example:
        >>> reranker = QwenReranker(
        ...     api_key="your-api-key",
        ...     model="qwen3-rerank"
        ... )
        >>> reranked = reranker.rerank(query, results, top_k=5)
    """

    def __init__(
        self,
        api_key: str,
        model: str = "qwen3-rerank",
        instruct: str = "Given a web search query, retrieve relevant passages that answer the query.",
        timeout: float = 30.0,
    ):
        """Initialize Qwen Reranker.

        Args:
            api_key: DashScope API key
            model: Model name
            instruct: Reranking instruction
            timeout: API timeout

        Raises:
            ValueError: If api_key is empty
        """
        if not api_key:
            raise ValueError("QwenReranker requires 'api_key' parameter")

        self.api_key = api_key
        self.model = model
        self.instruct = instruct
        self.timeout = timeout
        self.api_url = "https://dashscope.aliyuncs.com/compatible-api/v1/reranks"

        logger.info(f"Initialized QwenReranker with model='{model}', timeout={timeout}s")

    def rerank(
        self, query: Query, results: list[SearchResult], top_k: int | None = None
    ) -> list[SearchResult]:
        """Rerank retrieval results (sync method).

        Args:
            query: Query object
            results: List of results to rerank
            top_k: Number of results to return, None means all

        Returns:
            Reranked list of results
        """
        # Use sync call
        import asyncio

        try:
            # Run async code in sync context
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # If already in event loop, use run_until_complete
                # This may cause issues, but serves as a fallback
                logger.warning("Cannot run async rerank in running event loop, using fallback")
                return results[:top_k] if top_k else results
            else:
                return loop.run_until_complete(self.rerank_async(query, results, top_k))
        except Exception as e:
            logger.error(f"Failed to run async rerank: {e}")
            return results[:top_k] if top_k else results

    async def rerank_async(
        self, query: Query, results: list[SearchResult], top_k: int | None = None
    ) -> list[SearchResult]:
        """Rerank retrieval results (async method).

        Args:
            query: Query object
            results: List of results to rerank
            top_k: Number of results to return, None means all

        Returns:
            Reranked list of results
        """
        if not results:
            logger.debug("Empty results, nothing to rerank")
            return []

        # Determine number of results to return
        k = min(top_k, len(results)) if top_k else len(results)

        logger.info(
            f"Reranking {len(results)} results for query: '{query.text[:50]}...', returning top {k}"
        )

        try:
            # Extract document content
            documents = [result.chunk.content for result in results]

            # Prepare API request
            request_data = {
                "model": self.model,
                "query": query.text,
                "documents": documents,
                "top_n": k,
                "instruct": self.instruct,
            }

            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            }

            # Send API request
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.post(self.api_url, json=request_data, headers=headers)
                response.raise_for_status()
                api_response = response.json()

            # Parse response and rerank
            reranked_results = self._parse_and_rerank(api_response, results, k)

            logger.info(f"Qwen reranker: {len(results)} -> {len(reranked_results)} results")
            return reranked_results

        except httpx.HTTPStatusError as e:
            logger.error(f"Qwen API error (status {e.response.status_code}): {e.response.text}")
            # Fallback: return original results
            return results[:k]

        except httpx.TimeoutException:
            logger.error(f"Qwen API timeout after {self.timeout}s")
            return results[:k]

        except Exception as e:
            logger.error(f"Qwen reranker failed: {e}", exc_info=True)
            return results[:k]

    def _parse_and_rerank(
        self, api_response: dict, original_results: list[SearchResult], top_k: int
    ) -> list[SearchResult]:
        """Parse DashScope API response and rerank results.

        API response format:
        {
            "object": "list",
            "results": [
                {"index": 1, "relevance_score": 0.6171875690986707},
                {"index": 0, "relevance_score": 0.6073028761000254},
                ...
            ],
            "model": "qwen3-rerank",
            "usage": {"total_tokens": 2064}
        }

        Args:
            api_response: API response
            original_results: Original results list
            top_k: Number of results to return

        Returns:
            Reranked list of results
        """
        from langrag.entities.search_result import SearchResult

        results_data = api_response.get("results", [])

        if not results_data:
            logger.warning("API returned empty results")
            return original_results[:top_k]

        reranked_results = []
        used_indices = set()

        # Create reranked results in API-returned order
        for result_item in results_data:
            index = result_item.get("index")
            relevance_score = result_item.get("relevance_score", 0.0)

            if index is None or not (0 <= index < len(original_results)):
                logger.warning(f"Invalid index {index} from API")
                continue

            used_indices.add(index)
            original_result = original_results[index]

            # relevance_score: 0-1, higher is more relevant
            # Use directly as score (SearchResult's score is also higher-is-better)
            score = max(0.0, min(1.0, relevance_score))

            # Create new SearchResult (keep original chunk, update score)
            new_result = SearchResult(chunk=original_result.chunk, score=score)

            reranked_results.append(new_result)

            if len(reranked_results) >= top_k:
                break

        # If API returned fewer than top_k results, fill with original results
        if len(reranked_results) < top_k:
            logger.debug(
                f"API returned {len(reranked_results)} results, "
                f"filling to {top_k} with original results"
            )
            for i, result in enumerate(original_results):
                if i not in used_indices and len(reranked_results) < top_k:
                    reranked_results.append(result)

        logger.debug(
            f"Reranked results: "
            f"scores=[{', '.join(f'{r.score:.3f}' for r in reranked_results[:5])}...]"
        )

        return reranked_results[:top_k]
