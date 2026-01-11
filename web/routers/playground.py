"""
Playground API - Feature visualization and comparison
"""

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlmodel import Session

from web.core.database import get_session
from web.core.rag_kernel import RAGKernel
from web.services.kb_service import KBService

router = APIRouter(prefix="/api/playground", tags=["playground"])


def get_rag_kernel():
    from web.core.context import rag_kernel
    return rag_kernel


# =============================================================================
# Request/Response Models
# =============================================================================

class SearchCompareRequest(BaseModel):
    kb_id: str
    query: str
    top_k: int = 5


class SearchModeResult(BaseModel):
    mode: str
    results: list[dict]
    count: int
    time_ms: float


class SearchCompareResponse(BaseModel):
    query: str
    modes: list[SearchModeResult]


class QueryRewriteRequest(BaseModel):
    query: str


class QueryRewriteResponse(BaseModel):
    original: str
    rewritten: str | None
    error: str | None


class RerankCompareRequest(BaseModel):
    kb_id: str
    query: str
    top_k: int = 5


class RerankResult(BaseModel):
    content: str
    original_score: float
    reranked_score: float | None
    rank_before: int
    rank_after: int | None
    rank_change: int | None


class RerankCompareResponse(BaseModel):
    query: str
    reranker_active: bool
    reranker_name: str | None
    results: list[RerankResult]


class CacheStatsResponse(BaseModel):
    enabled: bool
    hits: int
    misses: int
    hit_rate: float
    size: int
    max_size: int
    similarity_threshold: float
    ttl_seconds: int


class CacheTestRequest(BaseModel):
    kb_id: str
    query: str


class CacheTestResponse(BaseModel):
    query: str
    is_cache_hit: bool
    similarity_score: float | None
    matched_query: str | None
    search_type: str


# =============================================================================
# Endpoints
# =============================================================================

@router.post("/search-compare", response_model=SearchCompareResponse)
def compare_search_modes(
    req: SearchCompareRequest,
    session: Session = Depends(get_session),
    rag_kernel: RAGKernel = Depends(get_rag_kernel)
):
    """Compare different search modes (hybrid, vector, keyword) on same query"""
    import time

    kb = KBService.get_kb(session, req.kb_id)
    if not kb:
        raise HTTPException(status_code=404, detail="Knowledge base not found")

    store = rag_kernel.get_vector_store(req.kb_id)
    if not store:
        raise HTTPException(status_code=404, detail="Vector store not found")

    modes = ["hybrid", "vector", "keyword"]
    mode_results = []

    for mode in modes:
        try:
            start = time.time()
            results, search_type = rag_kernel.search(
                kb_id=req.kb_id,
                query=req.query,
                top_k=req.top_k,
                search_mode=mode,
                use_rerank=False,  # Disable rerank for fair comparison
                use_rewrite=False  # Disable rewrite for fair comparison
            )
            elapsed = (time.time() - start) * 1000

            mode_results.append(SearchModeResult(
                mode=mode,
                results=[{
                    "content": doc.page_content[:200] + "..." if len(doc.page_content) > 200 else doc.page_content,
                    "score": doc.metadata.get("score", 0),
                    "source": doc.metadata.get("source", "unknown")
                } for doc in results],
                count=len(results),
                time_ms=round(elapsed, 2)
            ))
        except Exception as e:
            mode_results.append(SearchModeResult(
                mode=mode,
                results=[],
                count=0,
                time_ms=0
            ))

    return SearchCompareResponse(
        query=req.query,
        modes=mode_results
    )


@router.post("/query-rewrite", response_model=QueryRewriteResponse)
def test_query_rewrite(
    req: QueryRewriteRequest,
    rag_kernel: RAGKernel = Depends(get_rag_kernel)
):
    """Test query rewriting"""
    if not rag_kernel.rewriter:
        return QueryRewriteResponse(
            original=req.query,
            rewritten=None,
            error="Query rewriter not configured. Please configure LLM first."
        )

    try:
        rewritten = rag_kernel.rewriter.rewrite(req.query)
        return QueryRewriteResponse(
            original=req.query,
            rewritten=rewritten,
            error=None
        )
    except Exception as e:
        return QueryRewriteResponse(
            original=req.query,
            rewritten=None,
            error=str(e)
        )


@router.post("/rerank-compare", response_model=RerankCompareResponse)
def compare_reranking(
    req: RerankCompareRequest,
    session: Session = Depends(get_session),
    rag_kernel: RAGKernel = Depends(get_rag_kernel)
):
    """Compare results with and without reranking"""
    kb = KBService.get_kb(session, req.kb_id)
    if not kb:
        raise HTTPException(status_code=404, detail="Knowledge base not found")

    # Get results without reranking
    results_no_rerank, _ = rag_kernel.search(
        kb_id=req.kb_id,
        query=req.query,
        top_k=req.top_k,
        use_rerank=False,
        use_rewrite=False
    )

    # Create baseline scores and rankings
    baseline = {
        doc.page_content: {
            "score": doc.metadata.get("score", 0),
            "rank": i + 1
        }
        for i, doc in enumerate(results_no_rerank)
    }

    # Get results with reranking (if reranker configured)
    reranker_active = rag_kernel.reranker is not None
    reranker_name = None
    results_reranked = []

    if reranker_active:
        reranker_name = rag_kernel.reranker.__class__.__name__
        results_reranked, _ = rag_kernel.search(
            kb_id=req.kb_id,
            query=req.query,
            top_k=req.top_k,
            use_rerank=True,
            use_rewrite=False
        )

    # Build comparison results
    comparison = []
    for i, doc in enumerate(results_no_rerank):
        content = doc.page_content
        original_score = baseline[content]["score"]
        rank_before = baseline[content]["rank"]

        # Find reranked position
        reranked_score = None
        rank_after = None
        rank_change = None

        if reranker_active and results_reranked:
            for j, rdoc in enumerate(results_reranked):
                if rdoc.page_content == content:
                    reranked_score = rdoc.metadata.get("score", 0)
                    rank_after = j + 1
                    rank_change = rank_before - rank_after
                    break

        comparison.append(RerankResult(
            content=content[:200] + "..." if len(content) > 200 else content,
            original_score=original_score,
            reranked_score=reranked_score,
            rank_before=rank_before,
            rank_after=rank_after,
            rank_change=rank_change
        ))

    return RerankCompareResponse(
        query=req.query,
        reranker_active=reranker_active,
        reranker_name=reranker_name,
        results=comparison
    )


@router.get("/cache-stats", response_model=CacheStatsResponse)
def get_cache_stats(rag_kernel: RAGKernel = Depends(get_rag_kernel)):
    """Get semantic cache statistics"""
    if not rag_kernel.cache:
        return CacheStatsResponse(
            enabled=False,
            hits=0,
            misses=0,
            hit_rate=0,
            size=0,
            max_size=0,
            similarity_threshold=0,
            ttl_seconds=0
        )

    stats = rag_kernel.cache_stats
    return CacheStatsResponse(
        enabled=True,
        hits=stats.get("hits", 0),
        misses=stats.get("misses", 0),
        hit_rate=stats.get("hit_rate", 0),
        size=stats.get("size", 0),
        max_size=stats.get("max_size", 1000),
        similarity_threshold=stats.get("similarity_threshold", 0.95),
        ttl_seconds=stats.get("ttl_seconds", 3600)
    )


@router.post("/cache-test", response_model=CacheTestResponse)
def test_cache(
    req: CacheTestRequest,
    session: Session = Depends(get_session),
    rag_kernel: RAGKernel = Depends(get_rag_kernel)
):
    """Test if a query would hit the cache"""
    kb = KBService.get_kb(session, req.kb_id)
    if not kb:
        raise HTTPException(status_code=404, detail="Knowledge base not found")

    # First search (may populate cache)
    results1, search_type1 = rag_kernel.search(
        kb_id=req.kb_id,
        query=req.query,
        top_k=5,
        use_rerank=False,
        use_rewrite=False
    )

    # Check if it was a cache hit
    is_cache_hit = "+cached" in search_type1

    return CacheTestResponse(
        query=req.query,
        is_cache_hit=is_cache_hit,
        similarity_score=1.0 if is_cache_hit else None,
        matched_query=req.query if is_cache_hit else None,
        search_type=search_type1
    )


@router.post("/cache-clear")
def clear_cache(rag_kernel: RAGKernel = Depends(get_rag_kernel)):
    """Clear the semantic cache"""
    rag_kernel.clear_cache()
    return {"message": "Cache cleared"}
