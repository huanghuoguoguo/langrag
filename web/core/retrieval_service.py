"""
Retrieval service for the Web layer.

This module handles document retrieval including:
- Single and multi-knowledge base search
- Query rewriting (Agentic RAG)
- Hybrid search (vector + keyword)
- Result reranking
- Semantic caching for repeated queries
- Post-processing for special indexing techniques (QA, Parent-Child)

Design Decisions:
-----------------
1. **Agentic RAG Components**: The service optionally integrates with LangRAG's
   Router and Rewriter components for intelligent query processing. When these
   components are not configured, it falls back to direct search.

2. **Search Type Detection**: Automatically determines the best search strategy
   based on available capabilities (embedder, vector store type).

3. **Reranking Integration**: When a reranker is configured, retrieval expands
   the initial result set (5x top_k) then reranks to get the final results.

4. **Semantic Caching**: When a cache is configured, the service checks for
   semantically similar previous queries before executing a new search,
   reducing redundant embedding and search operations.

5. **Post-Processing Pipeline**: Handles special cases like QA indexing
   (swapping questions with answers) and Parent-Child indexing (fetching
   parent chunks from KV store).

Example Usage:
--------------
    from langrag.cache import SemanticCache

    cache = SemanticCache(similarity_threshold=0.95)
    service = RetrievalService(
        embedder=embedder,
        reranker=reranker,
        rewriter=rewriter,
        kv_store=kv_store,
        cache=cache
    )

    # Single KB search (with caching)
    results, search_type = service.search(
        store=vector_store,
        query="What is machine learning?",
        top_k=5
    )

    # Multi-KB search
    results, search_type = service.multi_search(
        stores={"kb1": store1, "kb2": store2},
        query="How does it work?",
        top_k=5
    )
"""

import logging

from langrag import BaseEmbedder, BaseVector
from langrag import Document as LangRAGDocument
from langrag.cache import SemanticCache
from langrag.datasource.kv.base import BaseKVStore
from langrag.entities.search_result import SearchResult
from langrag.retrieval.rerank.base import BaseReranker
from langrag.retrieval.rewriter.base import BaseRewriter

logger = logging.getLogger(__name__)


class RetrievalService:
    """
    Service for retrieving relevant documents from vector stores.

    This service encapsulates the complete retrieval pipeline including
    query processing, vector search, and result post-processing.

    The retrieval flow:
    1. Rewrite query (if rewriter is configured)
    2. Check semantic cache for similar queries
    3. Generate query embedding (if embedder is configured)
    4. Execute search on vector store(s)
    5. Post-process results (QA swap, parent fetch)
    6. Rerank results (if reranker is configured)
    7. Cache results for future queries
    8. Return final results with search type metadata

    Attributes:
        embedder: Embedding model for query vectorization
        reranker: Model for reranking initial results
        rewriter: LLM-based query rewriter for Agentic RAG
        kv_store: KV store for parent-child retrieval
        cache: Semantic cache for query deduplication
    """

    def __init__(
        self,
        embedder: BaseEmbedder | None = None,
        reranker: BaseReranker | None = None,
        rewriter: BaseRewriter | None = None,
        kv_store: BaseKVStore | None = None,
        cache: SemanticCache | None = None
    ):
        """
        Initialize the retrieval service.

        Args:
            embedder: Embedding model for generating query vectors.
                     If not provided, falls back to keyword search.
            reranker: Reranking model to improve result relevance.
                     When configured, initial retrieval fetches 5x top_k.
            rewriter: Query rewriter for Agentic RAG.
                     Rewrites user queries for better retrieval.
            kv_store: Key-value store for parent-child indexing.
                     Required to fetch parent chunks.
            cache: Semantic cache for query deduplication.
                     When configured, similar queries return cached results.
        """
        self.embedder = embedder
        self.reranker = reranker
        self.rewriter = rewriter
        self.kv_store = kv_store
        self.cache = cache

    def _embed_query(self, query: str) -> list[float] | None:
        """
        Generate embedding vector for the query.

        Args:
            query: The search query string

        Returns:
            Embedding vector or None if embedder is not configured
        """
        if not self.embedder:
            return None

        try:
            vectors = self.embedder.embed([query])
            return vectors[0] if vectors else None
        except Exception as e:
            logger.error(f"Query embedding failed: {e}")
            return None

    def _process_qa_results(self, results: list[LangRAGDocument]) -> None:
        """
        Post-process results from QA indexing.

        QA indexing stores questions in page_content and answers in metadata.
        This method swaps them back so the response contains the answer.

        The swap logic:
        - Original: content=question, metadata.answer=answer
        - After: content=answer, metadata.matched_question=question

        Args:
            results: List of documents to process (modified in place)
        """
        for doc in results:
            if doc.metadata.get("is_qa"):
                # Save the matched question
                question_text = doc.page_content
                doc.metadata["matched_question"] = question_text

                # Restore the original answer as content
                if "answer" in doc.metadata:
                    doc.page_content = doc.metadata["answer"]

                # Restore original document ID
                if "original_doc_id" in doc.metadata:
                    doc.id = doc.metadata["original_doc_id"]
                    doc.metadata["document_id"] = doc.metadata["original_doc_id"]

    def _process_parent_child_results(self, results: list[LangRAGDocument]) -> None:
        """
        Post-process results from parent-child indexing.

        Parent-child indexing stores child chunks in the vector store and
        parent chunks in a KV store. This method fetches the parent content
        to provide richer context in the response.

        Args:
            results: List of documents to process (modified in place)
        """
        if not self.kv_store:
            return

        # Collect parent IDs from results
        parent_ids = []
        for doc in results:
            if "parent_id" in doc.metadata:
                parent_ids.append(doc.metadata["parent_id"])

        if not parent_ids:
            return

        # Batch fetch parents from KV store
        parents_content = self.kv_store.mget(parent_ids)
        parent_map = dict(zip(parent_ids, parents_content))

        # Replace child content with parent content
        for doc in results:
            pid = doc.metadata.get("parent_id")
            if pid and pid in parent_map and parent_map[pid]:
                doc.page_content = parent_map[pid]
                doc.id = pid  # Update ID for deduplication
                doc.metadata["is_parent"] = True

    def _rerank_results(
        self,
        query: str,
        results: list[LangRAGDocument],
        top_k: int
    ) -> list[LangRAGDocument]:
        """
        Rerank results using the configured reranker.

        Reranking improves result quality by using a cross-encoder model
        to score query-document pairs more accurately than embedding similarity.

        Args:
            query: The search query
            results: Initial retrieval results
            top_k: Number of results to return after reranking

        Returns:
            Reranked and truncated result list
        """
        if not self.reranker or not results:
            return results[:top_k]

        # Convert to SearchResult objects for reranker
        search_results = [
            SearchResult(chunk=doc, score=doc.metadata.get('score', 0.0))
            for doc in results
        ]

        try:
            reranked = self.reranker.rerank(query, search_results, top_k=top_k)

            # Convert back to Document list and update scores
            final_results = []
            for res in reranked:
                doc = res.chunk
                doc.metadata['score'] = res.score
                final_results.append(doc)

            return final_results

        except Exception as e:
            logger.error(f"Rerank failed: {e}")
            return results[:top_k]

    def _determine_search_type(
        self,
        store: BaseVector,
        query_vector: list[float] | None
    ) -> str:
        """
        Determine the best search type based on capabilities.

        Search type selection logic:
        - SeekDB/DuckDB + vector → hybrid (best of both worlds)
        - Any store + vector → vector search
        - No vector → keyword search

        Args:
            store: The vector store to search
            query_vector: The query embedding (or None)

        Returns:
            Search type string: "hybrid", "vector", or "keyword"
        """
        store_class = store.__class__.__name__

        # Stores with hybrid search support (vector + FTS)
        hybrid_capable_stores = {'SeekDBVector', 'DuckDBVector'}

        if store_class in hybrid_capable_stores and query_vector:
            return "hybrid"
        elif query_vector:
            return "vector"
        else:
            return "keyword"

    def search(
        self,
        store: BaseVector,
        query: str,
        top_k: int = 5,
        rewrite: bool = True,
        use_cache: bool = True,
        search_mode: str | None = None,
        use_rerank: bool | None = None
    ) -> tuple[list[LangRAGDocument], str]:
        """
        Search a single vector store for relevant documents.

        This method implements the complete retrieval pipeline:
        1. Optionally rewrite the query using LLM
        2. Check semantic cache for similar queries
        3. Embed the query for vector search
        4. Execute search with appropriate strategy
        5. Post-process for special indexing techniques
        6. Rerank for improved relevance
        7. Cache results for future queries

        Args:
            store: The vector store to search
            query: The user's search query
            top_k: Number of results to return
            rewrite: Whether to apply query rewriting (default: True)
            use_cache: Whether to use semantic caching (default: True)
            search_mode: Force search mode ("hybrid", "vector", "keyword") or None for auto
            use_rerank: Force reranking (True/False) or None for default behavior

        Returns:
            Tuple of (results list, search type string)

        Example:
            >>> results, search_type = service.search(store, "What is RAG?", top_k=5)
            >>> print(f"Found {len(results)} results using {search_type}")
        """
        logger.info(f"Search: query='{query[:50]}...', top_k={top_k}, mode={search_mode}")

        # Step 1: Query rewriting (Agentic RAG)
        final_query = query
        if rewrite and self.rewriter:
            try:
                final_query = self.rewriter.rewrite(query)
                logger.info(f"[Agentic RAG] Query rewrite: '{query}' -> '{final_query}'")
            except Exception as e:
                logger.error(f"Query rewrite failed: {e}")

        # Step 2: Embed query
        query_vector = self._embed_query(final_query)

        # Step 3: Check semantic cache
        if use_cache and self.cache and query_vector:
            cache_hit = self.cache.get_by_similarity(query_vector)
            if cache_hit:
                search_type = cache_hit.metadata.get("search_type", "cached")
                logger.info(f"Semantic cache hit for query: '{final_query[:50]}...'")
                return cache_hit.results, f"{search_type}+cached"

        # Step 4: Determine search parameters
        # Use forced mode or auto-detect
        if search_mode and search_mode in ("hybrid", "vector", "keyword"):
            search_type = search_mode
        else:
            search_type = self._determine_search_type(store, query_vector)

        # Determine if reranking should be applied
        should_rerank = use_rerank if use_rerank is not None else (self.reranker is not None)

        # Expand retrieval if reranking
        k = top_k * 5 if should_rerank and self.reranker else top_k

        # Step 5: Execute search
        if search_type == "hybrid":
            results = store.search(
                final_query,
                query_vector=query_vector,
                top_k=k,
                search_type='hybrid'
            )
        elif search_type == "vector":
            results = store.search(
                final_query,
                query_vector=query_vector,
                top_k=k
            )
        else:
            results = store.search(final_query, top_k=k)

        # Step 6: Post-process results
        self._process_qa_results(results)
        self._process_parent_child_results(results)

        # Step 7: Rerank
        if should_rerank and self.reranker and results:
            results = self._rerank_results(final_query, results, top_k)
            search_type += "+rerank"
        else:
            results = results[:top_k]

        # Step 8: Cache results
        if use_cache and self.cache and query_vector and results:
            self.cache.set_with_embedding(
                query=final_query,
                embedding=query_vector,
                results=results,
                metadata={"search_type": search_type, "top_k": top_k}
            )

        return results, search_type

    def multi_search(
        self,
        stores: dict[str, BaseVector],
        query: str,
        top_k: int = 5,
        rewrite: bool = True,
        use_cache: bool = True
    ) -> tuple[list[LangRAGDocument], str]:
        """
        Search across multiple vector stores and merge results.

        This method searches all provided stores, combines results,
        and returns the top-scoring documents across all sources.

        The merging strategy:
        1. Check semantic cache for similar queries
        2. Search each store with the same query
        3. Tag results with their source KB ID
        4. Sort combined results by score (descending)
        5. Apply reranking if configured
        6. Cache results for future queries
        7. Return top_k results

        Note: Scores may not be directly comparable across different
        vector store types. Reranking is recommended for multi-KB search.

        Args:
            stores: Dictionary mapping KB IDs to vector stores
            query: The user's search query
            top_k: Number of results to return
            rewrite: Whether to apply query rewriting (default: True)
            use_cache: Whether to use semantic caching (default: True)

        Returns:
            Tuple of (results list, search type string)

        Example:
            >>> stores = {"kb1": store1, "kb2": store2}
            >>> results, search_type = service.multi_search(stores, "machine learning")
        """
        all_results = []
        primary_search_type = "keyword"

        # Step 1: Query rewriting (Agentic RAG)
        final_query = query
        if rewrite and self.rewriter:
            try:
                final_query = self.rewriter.rewrite(query)
                logger.info(f"[Agentic RAG] Query rewrite: '{query}' -> '{final_query}'")
            except Exception as e:
                logger.error(f"Query rewrite failed: {e}")

        # Step 2: Embed query once for all stores
        query_vector = self._embed_query(final_query)
        if query_vector:
            primary_search_type = "vector"

        # Step 3: Check semantic cache
        if use_cache and self.cache and query_vector:
            # Create context key from KB IDs for cache scoping
            context_key = ",".join(sorted(stores.keys()))
            cache_hit = self.cache.get_by_similarity(query_vector, context_key=context_key)
            if cache_hit:
                search_type = cache_hit.metadata.get("search_type", "cached")
                logger.info(f"Semantic cache hit for multi-search query: '{final_query[:50]}...'")
                return cache_hit.results, f"{search_type}+cached"

        # Expand retrieval if reranker is configured
        k = top_k * 5 if self.reranker else top_k

        # Step 4: Search each store
        for kb_id, store in stores.items():
            search_type = self._determine_search_type(store, query_vector)

            try:
                if search_type == "hybrid":
                    results = store.search(
                        final_query,
                        query_vector=query_vector,
                        top_k=k,
                        search_type='hybrid'
                    )
                    primary_search_type = "hybrid"
                elif search_type == "vector":
                    results = store.search(
                        final_query,
                        query_vector=query_vector,
                        top_k=k
                    )
                else:
                    results = store.search(final_query, top_k=k)

                # Tag results with source KB
                for doc in results:
                    doc.metadata['kb_id'] = kb_id

                all_results.extend(results)

            except Exception as e:
                logger.error(f"Search failed for KB {kb_id}: {e}")

        # Step 5: Sort by score
        all_results.sort(
            key=lambda x: x.metadata.get('score', 0),
            reverse=True
        )

        # Step 6: Post-process
        self._process_qa_results(all_results)
        self._process_parent_child_results(all_results)

        # Step 7: Rerank
        if self.reranker and all_results:
            all_results = self._rerank_results(final_query, all_results, top_k)
            primary_search_type += "+rerank"
        else:
            all_results = all_results[:top_k]

        # Step 8: Cache results
        if use_cache and self.cache and query_vector and all_results:
            context_key = ",".join(sorted(stores.keys()))
            self.cache.set_with_embedding(
                query=final_query,
                embedding=query_vector,
                results=all_results,
                metadata={
                    "search_type": primary_search_type,
                    "top_k": top_k,
                    "context_key": context_key
                }
            )

        return all_results, primary_search_type
