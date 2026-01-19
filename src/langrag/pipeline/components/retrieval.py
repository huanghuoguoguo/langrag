"""
Retrieval Components - Components for search and retrieval pipelines.

These components support runtime model injection, allowing you to
dynamically change the reranker, LLM, or other models at query time.
"""

import logging
from typing import Any, Dict, List, Optional

from langrag.core.component.base import Component
from langrag.entities.document import Document
from langrag.entities.search_result import SearchResult
from langrag.datasource.vdb.base import BaseVector
from langrag.llm.embedder.base import BaseEmbedder
from langrag.retrieval.rerank.base import BaseReranker
from langrag.retrieval.rewriter.base import BaseRewriter
from langrag.llm.base import BaseLLM

logger = logging.getLogger(__name__)


class RetrievalComponent(Component):
    """
    Retrieves documents from vector stores.

    Supports runtime injection of embedder and vector stores.

    Input keys:
        - query: str
        - embedder (optional): BaseEmbedder - runtime override
        - vector_stores (optional): List[BaseVector] - runtime override
        - top_k (optional): int - default 10

    Output keys:
        - results: List[SearchResult]
        - query: str (pass through)
    """

    component_type = "retrieval"

    def __init__(
        self,
        embedder: Optional[BaseEmbedder] = None,
        vector_stores: Optional[List[BaseVector]] = None,
        top_k: int = 10,
        **kwargs
    ):
        super().__init__(**kwargs)
        self._default_embedder = embedder
        self._default_stores = vector_stores or []
        self._default_top_k = top_k

    async def _invoke(
        self,
        query: str,
        embedder: Optional[BaseEmbedder] = None,
        vector_stores: Optional[List[BaseVector]] = None,
        top_k: Optional[int] = None,
        **kwargs
    ) -> Dict[str, Any]:
        active_embedder = embedder or self._default_embedder
        active_stores = vector_stores or self._default_stores
        active_top_k = top_k or self._default_top_k

        if not active_stores:
            logger.warning("[Retrieval] No vector stores provided")
            return {"results": [], "query": query}

        # Get query embedding
        query_vector = None
        if active_embedder:
            import asyncio
            vectors = await asyncio.to_thread(active_embedder.embed, [query])
            query_vector = vectors[0] if vectors else None

        # Search all stores
        all_results: List[SearchResult] = []
        for store in active_stores:
            try:
                import asyncio
                docs = await asyncio.to_thread(
                    store.search,
                    query=query,
                    query_vector=query_vector,
                    top_k=active_top_k
                )
                for doc in docs:
                    all_results.append(SearchResult(
                        document=doc,
                        score=doc.metadata.get("score", 0.0),
                        source=store.collection_name,
                    ))
            except Exception as e:
                logger.warning(f"[Retrieval] Search failed for {store.collection_name}: {e}")

        # Sort by score
        all_results.sort(key=lambda x: x.score, reverse=True)

        logger.info(f"[Retrieval] Found {len(all_results)} results")
        return {"results": all_results[:active_top_k], "query": query}


class RerankComponent(Component):
    """
    Reranks search results using a reranker model.

    **Key Feature**: Supports runtime model injection.
    You can pass a different reranker at each invocation.

    Input keys:
        - query: str
        - results: List[SearchResult]
        - reranker (optional): BaseReranker - runtime override
        - top_k (optional): int

    Output keys:
        - results: List[SearchResult] (reranked)
        - reranked: bool
    """

    component_type = "reranker"

    def __init__(
        self,
        reranker: Optional[BaseReranker] = None,
        top_k: int = 10,
        **kwargs
    ):
        super().__init__(**kwargs)
        self._default_reranker = reranker
        self._default_top_k = top_k

    async def _invoke(
        self,
        query: str,
        results: List[SearchResult],
        reranker: Optional[BaseReranker] = None,
        top_k: Optional[int] = None,
        **kwargs
    ) -> Dict[str, Any]:
        active_reranker = reranker or self._default_reranker
        active_top_k = top_k or self._default_top_k

        if not results:
            return {"results": [], "reranked": False}

        if not active_reranker:
            logger.info("[Rerank] No reranker provided, skipping")
            return {"results": results[:active_top_k], "reranked": False}

        import asyncio
        reranked = await asyncio.to_thread(
            active_reranker.rerank,
            query=query,
            results=results,
            top_k=active_top_k
        )

        logger.info(f"[Rerank] Reranked {len(results)} -> {len(reranked)} results")
        return {"results": reranked, "reranked": True}


class RewriteComponent(Component):
    """
    Rewrites queries for better retrieval.

    Supports runtime LLM injection.

    Input keys:
        - query: str
        - rewriter (optional): BaseRewriter - runtime override
        - llm (optional): BaseLLM - runtime override (for LLM-based rewriting)

    Output keys:
        - query: str (rewritten)
        - original_query: str
        - rewritten: bool
    """

    component_type = "rewriter"

    def __init__(
        self,
        rewriter: Optional[BaseRewriter] = None,
        llm: Optional[BaseLLM] = None,
        **kwargs
    ):
        super().__init__(**kwargs)
        self._default_rewriter = rewriter
        self._default_llm = llm

    async def _invoke(
        self,
        query: str,
        rewriter: Optional[BaseRewriter] = None,
        llm: Optional[BaseLLM] = None,
        **kwargs
    ) -> Dict[str, Any]:
        active_rewriter = rewriter or self._default_rewriter

        if not active_rewriter:
            return {"query": query, "original_query": query, "rewritten": False}

        # If rewriter needs LLM, inject it
        active_llm = llm or self._default_llm
        if active_llm and hasattr(active_rewriter, 'llm'):
            active_rewriter.llm = active_llm

        import asyncio
        rewritten = await asyncio.to_thread(active_rewriter.rewrite, query)

        logger.info(f"[Rewrite] '{query}' -> '{rewritten}'")
        return {"query": rewritten, "original_query": query, "rewritten": True}


class GenerateComponent(Component):
    """
    Generates answers using LLM.

    Supports runtime LLM injection.

    Input keys:
        - query: str
        - results: List[SearchResult]
        - llm (optional): BaseLLM - runtime override

    Output keys:
        - answer: str
        - context: str (used context)
    """

    component_type = "generator"

    def __init__(self, llm: Optional[BaseLLM] = None, **kwargs):
        super().__init__(**kwargs)
        self._default_llm = llm

    async def _invoke(
        self,
        query: str,
        results: List[SearchResult],
        llm: Optional[BaseLLM] = None,
        **kwargs
    ) -> Dict[str, Any]:
        active_llm = llm or self._default_llm

        if not active_llm:
            raise ValueError("No LLM provided for generation")

        # Build context from results
        context_parts = []
        for i, r in enumerate(results[:5], 1):
            context_parts.append(f"[{i}] {r.document.page_content}")
        context = "\n\n".join(context_parts)

        prompt = f"""Based on the following context, answer the question.

Context:
{context}

Question: {query}

Answer:"""

        import asyncio
        answer = await asyncio.to_thread(
            active_llm.chat,
            messages=[{"role": "user", "content": prompt}]
        )

        return {"answer": answer, "context": context}
