"""LLM Template Reranker - Reranking using any configured LLM with custom templates"""

import httpx
from loguru import logger

from langrag.entities.search_result import SearchResult

from ..base import BaseReranker


class LLMTemplateReranker(BaseReranker):
    """
    LLM Template Reranker using any configured LLM with custom prompt templates.

    This reranker allows using any LLM (local or API-based) with custom templates
    for reranking documents based on relevance to a query.
    """

    DEFAULT_TEMPLATE = """
Given the following query and list of documents, please rerank the documents
based on their relevance to the query. Return only the indices of the documents
in order of relevance (most relevant first), separated by commas.

Query: {query}

Documents:
{documents}

Please provide the reranked indices (0-based) as a comma-separated list:
""".strip()

    def __init__(
        self,
        llm_model,  # LangRAG LLM model instance
        template: str | None = None,
        timeout: float = 30.0,
    ):
        """
        Initialize LLM Template Reranker.

        Args:
            llm_model: LangRAG LLM model instance (must have chat() method)
            template: Custom prompt template, uses default if None
            timeout: API timeout in seconds
        """
        self.llm_model = llm_model
        self.template = template or self.DEFAULT_TEMPLATE
        self.timeout = timeout

        logger.info(f"Initialized LLMTemplateReranker with model: {getattr(llm_model, 'model_name', 'unknown')}")

    def rerank(
        self, query: str, results: list[SearchResult], top_k: int | None = None
    ) -> list[SearchResult]:
        """
        Rerank results using LLM with template.

        Args:
            query: Search query
            results: Search results to rerank
            top_k: Number of results to return

        Returns:
            Reranked search results
        """
        if not results:
            return []

        q_text = query
        k = top_k if top_k is not None else len(results)

        try:
            # Format documents for the template
            documents_text = self._format_documents(results)

            # Create prompt using template
            prompt = self.template.format(
                query=q_text,
                documents=documents_text
            )

            # Call LLM
            response = self._call_llm(prompt)

            # Parse response to get indices
            indices = self._parse_indices(response, len(results))

            # Reorder results based on indices
            reranked_results = []
            for idx in indices[:k]:
                if 0 <= idx < len(results):
                    reranked_results.append(results[idx])

            # If we don't have enough results, fill with remaining
            used_indices = set(indices[:k])
            for i, result in enumerate(results):
                if i not in used_indices and len(reranked_results) < k:
                    reranked_results.append(result)

            logger.info(f"LLM reranker: {len(results)} -> {len(reranked_results)} results")
            return reranked_results

        except Exception as e:
            logger.error(f"LLM template reranker failed: {e}")
            # Fallback to original results
            return results[:k]

    def _format_documents(self, results: list[SearchResult]) -> str:
        """Format documents for the template."""
        formatted = []
        for i, result in enumerate(results):
            content = result.chunk.content[:500]  # Limit content length
            formatted.append(f"[{i}] {content}")

        return "\n".join(formatted)

    def _call_llm(self, prompt: str) -> str:
        """Call LLM with the formatted prompt."""
        try:
            # Format messages for LLM
            messages = [{"role": "user", "content": prompt}]

            # Call LLM's chat method (should return string directly)
            response = self.llm_model.chat(messages=messages, stream=False)

            # The response should be a string according to BaseLLM interface
            if isinstance(response, str):
                return response
            else:
                logger.warning(f"Unexpected LLM response type: {type(response)}, converting to string")
                return str(response)

        except Exception as e:
            logger.error(f"Failed to call LLM: {e}")
            return "0,1,2,3,4"  # Mock indices

    def _parse_indices(self, response: str, max_index: int) -> list[int]:
        """Parse comma-separated indices from LLM response."""
        try:
            # Extract indices from response
            # Look for patterns like "0,1,2,3,4" or "0, 1, 2, 3, 4"
            import re
            indices_str = re.search(r'[\d,\s]+', response.strip())
            if indices_str:
                indices = []
                for part in indices_str.group().split(','):
                    part = part.strip()
                    if part.isdigit():
                        idx = int(part)
                        if 0 <= idx < max_index:
                            indices.append(idx)
                return indices

            # Fallback: try to find any numbers
            numbers = re.findall(r'\d+', response)
            indices = []
            for num in numbers:
                idx = int(num)
                if 0 <= idx < max_index:
                    indices.append(idx)
            return indices[:max_index]  # Limit to available results

        except Exception as e:
            logger.error(f"Failed to parse indices from LLM response: {e}")
            return list(range(min(max_index, 5)))  # Fallback to first 5 indices