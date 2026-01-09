"""Qwen-based context compressor using DashScope API."""

import asyncio
from typing import Any

from loguru import logger

from langrag.entities.search_result import SearchResult

from ..base import BaseCompressor

try:
    import httpx

    HTTPX_AVAILABLE = True
except ImportError:
    HTTPX_AVAILABLE = False
    logger.warning("httpx not installed. QwenCompressor will not be available.")


class QwenCompressor(BaseCompressor):
    """Context compressor using Qwen API.

    Compresses the content of each retrieval result by calling Qwen's
    Chat Completions API.

    Args:
        api_key: DashScope API Key
        model: Model name, default "qwen-plus"
        api_url: API endpoint, defaults to DashScope compatible mode endpoint
        temperature: Generation temperature, default 0.3 (more deterministic output)
        timeout: Request timeout in seconds, default 30
        max_concurrent: Maximum concurrent requests, default 5
    """

    def __init__(
        self,
        api_key: str,
        model: str = "qwen-plus",
        api_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions",
        temperature: float = 0.3,
        timeout: int = 30,
        max_concurrent: int = 5,
    ):
        """Initialize Qwen compressor.

        Args:
            api_key: DashScope API Key
            model: Model name
            api_url: API endpoint URL
            temperature: Generation temperature
            timeout: Request timeout
            max_concurrent: Maximum concurrent requests
        """
        if not HTTPX_AVAILABLE:
            raise ImportError(
                "httpx is required for QwenCompressor. Install it with: pip install httpx"
            )

        self.api_key = api_key
        self.model = model
        self.api_url = api_url
        self.temperature = temperature
        self.timeout = timeout
        self.max_concurrent = max_concurrent

        logger.info(f"Initialized QwenCompressor with model: {model}")

    def compress(
        self, query: str, results: list[SearchResult], target_ratio: float = 0.5
    ) -> list[SearchResult]:
        """Compress retrieval results (sync interface).

        Args:
            query: User query
            results: List of retrieval results
            target_ratio: Target compression ratio (0-1)

        Returns:
            List of compressed retrieval results
        """
        # Use async implementation, then run in sync context
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            # No running event loop, create a new one
            return asyncio.run(self.compress_async(query, results, target_ratio))
        else:
            # Event loop already exists, create task
            import nest_asyncio

            nest_asyncio.apply()
            return loop.run_until_complete(self.compress_async(query, results, target_ratio))

    async def compress_async(
        self, query: str, results: list[SearchResult], target_ratio: float = 0.5
    ) -> list[SearchResult]:
        """Compress retrieval results (async interface).

        Args:
            query: User query
            results: List of retrieval results
            target_ratio: Target compression ratio (0-1)

        Returns:
            List of compressed retrieval results
        """
        if not results:
            return results

        logger.info(
            f"Compressing {len(results)} results with target ratio {target_ratio:.1%}"
        )

        # Create semaphore to limit concurrency
        semaphore = asyncio.Semaphore(self.max_concurrent)

        # Compress all results concurrently
        tasks = [
            self._compress_single_result(query, result, target_ratio, semaphore)
            for result in results
        ]

        compressed_results = await asyncio.gather(*tasks, return_exceptions=True)

        # Handle exceptions, keep original results
        final_results = []
        for i, result in enumerate(compressed_results):
            if isinstance(result, Exception):
                logger.warning(f"Failed to compress result {i}: {result}, using original")
                final_results.append(results[i])
            else:
                final_results.append(result)

        # Calculate compression statistics
        original_length = sum(len(r.chunk.page_content) for r in results)
        compressed_length = sum(len(r.chunk.page_content) for r in final_results)
        actual_ratio = compressed_length / original_length if original_length > 0 else 1.0

        logger.info(
            f"Compression complete: {original_length} -> {compressed_length} chars "
            f"(actual ratio: {actual_ratio:.1%})"
        )

        return final_results

    async def _compress_single_result(
        self,
        query: str,
        result: SearchResult,
        target_ratio: float,
        semaphore: asyncio.Semaphore,
    ) -> SearchResult:
        """Compress a single retrieval result.

        Args:
            query: User query
            result: Retrieval result
            target_ratio: Target compression ratio
            semaphore: Concurrency control semaphore

        Returns:
            Compressed retrieval result
        """
        async with semaphore:
            try:
                compressed_content = await self._call_qwen_api(
                    query, result.chunk.page_content, target_ratio
                )

                # Create new Chunk and SearchResult (maintain immutability)
                from langrag.entities.document import Document

                compressed_chunk = Document(
                    id=result.chunk.id,
                    page_content=compressed_content,
                    vector=result.chunk.vector,
                    # source_doc_id=result.chunk.source_doc_id, # Document doesn't have source_doc_id field directly anymore, relies on metadata
                    metadata={
                        **result.chunk.metadata,
                        "source_doc_id": result.chunk.metadata.get("source_doc_id"),
                        "compressed": True,
                        "original_length": len(result.chunk.page_content),
                        "compressed_length": len(compressed_content),
                    },
                )

                return SearchResult(chunk=compressed_chunk, score=result.score)

            except Exception as e:
                logger.error(f"Failed to compress chunk {result.chunk.id}: {e}")
                raise

    async def _call_qwen_api(
        self, query: str, content: str, target_ratio: float
    ) -> str:
        """Call Qwen API for content compression.

        Args:
            query: User query
            content: Content to compress
            target_ratio: Target compression ratio

        Returns:
            Compressed content
        """
        # Build compression prompt
        target_length_desc = f"{int(target_ratio * 100)}% length"
        if target_ratio <= 0.3:
            target_length_desc = "minimal mode (keep core points only)"
        elif target_ratio <= 0.5:
            target_length_desc = "concise mode (keep key information)"
        elif target_ratio <= 0.7:
            target_length_desc = "moderate compression (keep main details)"

        system_prompt = (
            "You are a professional text compression assistant. Your task is to compress "
            "the given text to the specified length while preserving information most "
            "relevant to the user's query. "
            "Compression principles: 1) Prioritize content related to the query; "
            "2) Remove redundant and unimportant details; "
            "3) Maintain semantic coherence; 4) Do not add information not in the original."
        )

        user_prompt = f"""User query: {query}

Original text:
{content}

Please compress the above text to {target_length_desc}, focusing on content related to the query.
Output the compressed text directly without any explanation or prefix."""

        # Build request
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": self.temperature,
        }

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        # Send request
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            response = await client.post(
                self.api_url, headers=headers, json=payload
            )
            response.raise_for_status()

            result = response.json()

            # Parse response
            if "choices" in result and len(result["choices"]) > 0:
                compressed_text = result["choices"][0]["message"]["content"]

                # Log token usage
                if "usage" in result:
                    usage = result["usage"]
                    logger.debug(
                        f"Qwen API usage: "
                        f"prompt={usage.get('prompt_tokens', 0)}, "
                        f"completion={usage.get('completion_tokens', 0)}, "
                        f"total={usage.get('total_tokens', 0)}"
                    )

                return compressed_text.strip()
            else:
                raise ValueError(f"Unexpected API response format: {result}")

