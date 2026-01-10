import httpx
from loguru import logger

from langrag.entities.query import Query
from langrag.entities.search_result import SearchResult
from langrag.retrieval.rerank.base import BaseReranker


class CohereReranker(BaseReranker):
    """
    Cohere Reranker implementation using Cohere API.
    """

    def __init__(self, api_key: str, model: str = "rerank-english-v3.0"):
        self.api_key = api_key
        self.model = model
        self.api_url = "https://api.cohere.com/v1/rerank"

    def rerank(
        self, query: str | Query, results: list[SearchResult], top_k: int | None = None
    ) -> list[SearchResult]:

        if not results:
            return []

        q_text = query.text if isinstance(query, Query) else query

        # Use sync client for simplicity and safety in sync context
        try:
            with httpx.Client(timeout=30.0) as client:
                documents = [r.chunk.page_content for r in results]

                k = top_k if top_k else len(documents)

                payload = {
                    "model": self.model,
                    "query": q_text,
                    "documents": documents,
                    "top_n": k,
                    "return_documents": False
                }

                headers = {
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                    "X-Client-Name": "LangRAG"
                }

                response = client.post(self.api_url, json=payload, headers=headers)
                response.raise_for_status()
                data = response.json()

                return self._parse_response(data, results)

        except httpx.HTTPStatusError as e:
            logger.error(f"Cohere API returned error status {e.response.status_code}: {e}")
            return results[:top_k] if top_k else results
        except httpx.RequestError as e:
            logger.error(f"Cohere API request failed (network error): {e}")
            return results[:top_k] if top_k else results
        except httpx.TimeoutException as e:
            logger.error(f"Cohere API request timed out: {e}")
            return results[:top_k] if top_k else results
        except (KeyError, ValueError) as e:
            logger.error(f"Failed to parse Cohere API response: {e}")
            return results[:top_k] if top_k else results

    def _parse_response(self, data: dict, original_results: list[SearchResult]) -> list[SearchResult]:
        reranked = []
        # Cohere 'results' list contains {index, relevance_score}
        for item in data.get("results", []):
            idx = item["index"]
            score = item["relevance_score"]

            if 0 <= idx < len(original_results):
                # Update score
                res = SearchResult(chunk=original_results[idx].chunk, score=score)
                reranked.append(res)

        return reranked
