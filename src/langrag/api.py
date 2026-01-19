"""
LangRAG High-Level API.

This module provides simple, one-line functions for common RAG operations.
The goal is to minimize boilerplate code in the application layer.

Key Design Principles:
1. **One-line usage**: Most operations should be a single function call
2. **Runtime injection**: Models can be passed at call time, not just init time
3. **Sensible defaults**: Works out of the box, customizable when needed
4. **No boundary checks in app layer**: All validation happens here

Example Usage:
    import langrag

    # Index a document (one line)
    result = await langrag.index_document(
        file_path="doc.pdf",
        vector_store=store,
        embedder=embedder
    )

    # Search with runtime model override
    results = await langrag.search(
        query="What is RAG?",
        vector_stores=[store1, store2],
        embedder=embedder,
        reranker=my_custom_reranker,  # Override at runtime
        top_k=10
    )
"""

import asyncio
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

# Use direct imports to avoid circular dependency
from langrag.datasource.vdb.base import BaseVector
from langrag.llm.embedder.base import BaseEmbedder
from langrag.llm.base import BaseLLM
from langrag.retrieval.rerank.base import BaseReranker
from langrag.retrieval.rewriter.base import BaseRewriter
from langrag.entities.search_result import SearchResult
from langrag.entities.document import Document
from langrag.index_processor.splitter.providers.recursive_character import RecursiveCharacterChunker
from langrag.index_processor.extractor.factory import ParserFactory

logger = logging.getLogger(__name__)


# =============================================================================
# Result Types
# =============================================================================

@dataclass
class IndexResult:
    """Result of document indexing."""
    success: bool
    stored_count: int
    elapsed_time: float
    errors: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class RetrievalResult:
    """Result of search/retrieval."""
    results: List[SearchResult]
    query: str
    rewritten_query: Optional[str] = None
    elapsed_time: float = 0.0
    # Pipeline metadata for UI display
    pipeline: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RAGResult:
    """Result of full RAG (retrieve + generate)."""
    answer: str
    sources: List[SearchResult]
    query: str
    elapsed_time: float = 0.0


# =============================================================================
# Internal Helpers
# =============================================================================

def _get_parser(file_path: Path):
    """Auto-detect parser based on file extension."""
    ext = file_path.suffix.lower().lstrip(".")
    ext_map = {
        "md": "markdown", "markdown": "markdown",
        "htm": "html", "html": "html",
        "doc": "docx", "docx": "docx",
        "pdf": "pdf",
    }
    parser_type = ext_map.get(ext, "simple_text")
    try:
        return ParserFactory.create(parser_type)
    except ValueError:
        return ParserFactory.create("simple_text")


# =============================================================================
# High-Level API Functions
# =============================================================================

async def index_document(
    file_path: Union[str, Path],
    vector_store: BaseVector,
    embedder: Optional[BaseEmbedder] = None,
    chunk_size: int = 500,
    chunk_overlap: int = 50,
) -> IndexResult:
    """
    Index a document into a vector store.

    This is the simplest way to ingest a document. One line does everything:
    parse, chunk, embed, store.

    Args:
        file_path: Path to the document file.
        vector_store: Target vector store.
        embedder: Embedding model (optional for keyword-only search).
        chunk_size: Size of text chunks.
        chunk_overlap: Overlap between chunks.

    Returns:
        IndexResult with success status and stored count.

    Example:
        result = await langrag.index_document(
            "document.pdf",
            my_vector_store,
            my_embedder
        )
        print(f"Indexed {result.stored_count} chunks")
    """
    start = time.perf_counter()
    errors = []

    try:
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        # Step 1: Parse
        parser = _get_parser(path)
        documents = await asyncio.to_thread(parser.parse, path)
        logger.info(f"[index_document] Parsed {len(documents)} documents from {path.name}")

        # Step 2: Chunk
        chunker = RecursiveCharacterChunker(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
        chunks = await asyncio.to_thread(chunker.split, documents)
        logger.info(f"[index_document] Created {len(chunks)} chunks")

        # Step 3: Embed (if embedder provided)
        if embedder and chunks:
            texts = [c.page_content for c in chunks]
            vectors = await asyncio.to_thread(embedder.embed, texts)
            for chunk, vector in zip(chunks, vectors):
                chunk.vector = vector
            logger.info(f"[index_document] Embedded {len(chunks)} chunks")

        # Step 4: Store
        if chunks:
            await asyncio.to_thread(vector_store.add_texts, chunks)
            logger.info(f"[index_document] Stored {len(chunks)} chunks")

        elapsed = time.perf_counter() - start
        return IndexResult(
            success=True,
            stored_count=len(chunks),
            elapsed_time=elapsed,
            errors=[],
        )

    except Exception as e:
        logger.exception(f"[index_document] Failed: {e}")
        elapsed = time.perf_counter() - start
        return IndexResult(
            success=False,
            stored_count=0,
            elapsed_time=elapsed,
            errors=[{"error": str(e), "type": type(e).__name__}],
        )


async def search(
    query: str,
    vector_stores: List[BaseVector],
    embedder: Optional[BaseEmbedder] = None,
    reranker: Optional[BaseReranker] = None,
    rewriter: Optional[BaseRewriter] = None,
    router: Optional["BaseRouter"] = None,
    datasets: Optional[List["Dataset"]] = None,
    top_k: int = 10,
    rerank_top_k: Optional[int] = None,
) -> RetrievalResult:
    """
    Search across vector stores with optional reranking and routing.

    Supports runtime model injection - pass different models at each call.
    All boundary checks and error handling are done here, so callers can
    keep their code simple.

    Args:
        query: Search query.
        vector_stores: List of vector stores to search.
        embedder: Embedding model for query.
        reranker: Reranker model (optional, runtime injectable).
        rewriter: Query rewriter (optional, runtime injectable).
        router: Router for selecting which stores to search (optional).
        datasets: Dataset metadata for router (required if router is provided).
        top_k: Number of results to return.
        rerank_top_k: Number of results after reranking (defaults to top_k).

    Returns:
        RetrievalResult with search results and pipeline metadata.

    Example:
        # Basic search
        results = await langrag.search(
            "What is RAG?",
            [store1, store2],
            embedder
        )

        # With router to select KBs
        results = await langrag.search(
            "What is RAG?",
            [store1, store2],
            embedder,
            router=my_router,
            datasets=[dataset1, dataset2],  # Maps to vector_stores
        )

        # Check which pipeline steps were used
        print(results.pipeline)  # {"rewriter": {...}, "router": {...}, "reranker": {...}}
    """
    from langrag.retrieval.router.base import BaseRouter
    from langrag.entities.dataset import Dataset

    start = time.perf_counter()
    original_query = query
    rewritten_query = None

    # Pipeline metadata for UI
    pipeline_info: Dict[str, Any] = {
        "rewriter": {"enabled": False},
        "router": {"enabled": False},
        "reranker": {"enabled": False},
    }

    # Step 1: Rewrite query (if rewriter provided)
    if rewriter:
        pipeline_info["rewriter"]["enabled"] = True
        pipeline_info["rewriter"]["model"] = getattr(rewriter, "__class__", type(rewriter)).__name__
        try:
            rewritten_query = await asyncio.to_thread(rewriter.rewrite, query)
            if rewritten_query and rewritten_query != query:
                logger.info(f"[search] Rewrote query: '{query}' -> '{rewritten_query}'")
                query = rewritten_query
                pipeline_info["rewriter"]["output"] = rewritten_query
            else:
                rewritten_query = None
        except Exception as e:
            logger.warning(f"[search] Rewrite failed: {e}, using original query")
            pipeline_info["rewriter"]["error"] = str(e)

    # Step 2: Route to select stores (if router provided)
    selected_stores = vector_stores
    if router and datasets:
        pipeline_info["router"]["enabled"] = True
        pipeline_info["router"]["model"] = getattr(router, "__class__", type(router)).__name__
        pipeline_info["router"]["total_count"] = len(datasets)
        try:
            selected_datasets = await asyncio.to_thread(router.route, query, datasets)
            selected_ids = {d.id for d in selected_datasets}
            # Map selected datasets back to stores
            # Assumes datasets and vector_stores have same order or matching IDs
            selected_stores = []
            for i, store in enumerate(vector_stores):
                ds = datasets[i] if i < len(datasets) else None
                if ds and ds.id in selected_ids:
                    selected_stores.append(store)
            if not selected_stores:
                # Fallback to all stores if router selected none
                selected_stores = vector_stores
            pipeline_info["router"]["selected_count"] = len(selected_stores)
            pipeline_info["router"]["selected"] = [d.name for d in selected_datasets]
            logger.info(f"[search] Router selected {len(selected_stores)}/{len(vector_stores)} stores")
        except Exception as e:
            logger.warning(f"[search] Routing failed: {e}, using all stores")
            pipeline_info["router"]["error"] = str(e)

    # Step 3: Get query embedding
    query_vector = None
    if embedder:
        try:
            vectors = await asyncio.to_thread(embedder.embed, [query])
            query_vector = vectors[0] if vectors else None
        except Exception as e:
            logger.warning(f"[search] Embedding failed: {e}")

    # Step 4: Search selected stores
    retrieve_top_k = top_k * 2 if reranker else top_k
    all_results: List[SearchResult] = []

    for store in selected_stores:
        try:
            docs = await asyncio.to_thread(
                store.search,
                query=query,
                query_vector=query_vector,
                top_k=retrieve_top_k
            )
            for doc in docs:
                all_results.append(SearchResult(
                    document=doc,
                    score=doc.metadata.get("score", 0.0),
                    source=store.collection_name,
                ))
        except Exception as e:
            logger.warning(f"[search] Search failed for {store.collection_name}: {e}")

    # Sort by score
    all_results.sort(key=lambda x: x.score, reverse=True)
    all_results = all_results[:retrieve_top_k]

    # Step 5: Rerank (if reranker provided)
    if reranker and all_results:
        pipeline_info["reranker"]["enabled"] = True
        pipeline_info["reranker"]["model"] = getattr(reranker, "__class__", type(reranker)).__name__
        pipeline_info["reranker"]["input_count"] = len(all_results)
        try:
            final_top_k = rerank_top_k or top_k
            all_results = await asyncio.to_thread(
                reranker.rerank,
                query=query,
                results=all_results,
                top_k=final_top_k
            )
            pipeline_info["reranker"]["output_count"] = len(all_results)
            logger.info(f"[search] Reranked to {len(all_results)} results")
        except Exception as e:
            logger.warning(f"[search] Rerank failed: {e}")
            pipeline_info["reranker"]["error"] = str(e)

    elapsed = time.perf_counter() - start

    return RetrievalResult(
        results=all_results[:top_k],
        query=original_query,
        rewritten_query=rewritten_query,
        elapsed_time=elapsed,
        pipeline=pipeline_info,
    )


async def rag(
    query: str,
    vector_stores: List[BaseVector],
    llm: BaseLLM,
    embedder: Optional[BaseEmbedder] = None,
    reranker: Optional[BaseReranker] = None,
    rewriter: Optional[BaseRewriter] = None,
    top_k: int = 5,
) -> RAGResult:
    """
    Full RAG: retrieve relevant documents and generate an answer.

    All models can be injected at runtime.

    Args:
        query: User question.
        vector_stores: Vector stores to search.
        llm: LLM for answer generation (runtime injectable).
        embedder: Embedding model.
        reranker: Reranker model (optional).
        rewriter: Query rewriter (optional).
        top_k: Number of documents to use for context.

    Returns:
        RAGResult with answer and sources.

    Example:
        result = await langrag.rag(
            "How does RAG work?",
            [knowledge_base],
            llm=my_llm,
            embedder=my_embedder,
            reranker=my_reranker,  # Can be swapped per request
        )
        print(result.answer)
    """
    start = time.perf_counter()

    # Step 1: Search
    search_result = await search(
        query=query,
        vector_stores=vector_stores,
        embedder=embedder,
        reranker=reranker,
        rewriter=rewriter,
        top_k=top_k,
    )

    # Step 2: Build context
    context_parts = []
    for i, r in enumerate(search_result.results[:top_k], 1):
        context_parts.append(f"[{i}] {r.document.page_content}")
    context = "\n\n".join(context_parts)

    # Step 3: Generate
    prompt = f"""Based on the following context, answer the question.

Context:
{context}

Question: {query}

Answer:"""

    answer = await asyncio.to_thread(
        llm.chat,
        messages=[{"role": "user", "content": prompt}]
    )

    elapsed = time.perf_counter() - start

    return RAGResult(
        answer=answer,
        sources=search_result.results,
        query=query,
        elapsed_time=elapsed,
    )


# =============================================================================
# Sync Wrappers (for convenience)
# =============================================================================

def index_document_sync(
    file_path: Union[str, Path],
    vector_store: BaseVector,
    embedder: Optional[BaseEmbedder] = None,
    chunk_size: int = 500,
    chunk_overlap: int = 50,
) -> IndexResult:
    """Synchronous version of index_document."""
    return asyncio.run(index_document(
        file_path, vector_store, embedder, chunk_size, chunk_overlap
    ))


def search_sync(
    query: str,
    vector_stores: List[BaseVector],
    embedder: Optional[BaseEmbedder] = None,
    reranker: Optional[BaseReranker] = None,
    rewriter: Optional[BaseRewriter] = None,
    router: Optional["BaseRouter"] = None,
    datasets: Optional[List["Dataset"]] = None,
    top_k: int = 10,
    rerank_top_k: Optional[int] = None,
) -> RetrievalResult:
    """Synchronous version of search."""
    return asyncio.run(search(
        query, vector_stores, embedder, reranker, rewriter, router, datasets, top_k, rerank_top_k
    ))


def rag_sync(
    query: str,
    vector_stores: List[BaseVector],
    llm: BaseLLM,
    embedder: Optional[BaseEmbedder] = None,
    reranker: Optional[BaseReranker] = None,
    top_k: int = 5,
) -> RAGResult:
    """Synchronous version of rag."""
    return asyncio.run(rag(
        query, vector_stores, llm, embedder, reranker, top_k=top_k
    ))
