"""Search API"""


from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlmodel import Session

from web.core.database import get_session
from web.core.rag_kernel import RAGKernel
from web.services.kb_service import KBService

router = APIRouter(prefix="/api/search", tags=["search"])


class SearchRequest(BaseModel):
    kb_id: str
    query: str
    top_k: int = 5
    search_mode: str | None = None  # "hybrid", "vector", "keyword", or None for auto
    use_rerank: bool | None = None  # None = use default, True/False = force
    use_rewrite: bool = False  # Disabled by default for plain search


class SearchResultItem(BaseModel):
    content: str
    score: float
    source: str
    search_type: str


class SearchResponse(BaseModel):
    results: list[SearchResultItem]
    search_type: str
    original_query: str
    rewritten_query: str | None = None  # Only set if query was rewritten


def get_rag_kernel():
    """Dependency injection: Get RAG Kernel singleton"""
    from web.app import rag_kernel
    return rag_kernel


@router.post("", response_model=SearchResponse)
def search(
    req: SearchRequest,
    session: Session = Depends(get_session),
    rag_kernel: RAGKernel = Depends(get_rag_kernel)
):
    """Execute search with optional mode and rerank settings"""
    # Verify KB exists
    kb = KBService.get_kb(session, req.kb_id)
    if not kb:
        raise HTTPException(status_code=404, detail="Knowledge base not found")

    try:
        results, search_type, rewritten_query = rag_kernel.search(
            kb_id=req.kb_id,
            query=req.query,
            top_k=req.top_k,
            search_mode=req.search_mode,
            use_rerank=req.use_rerank,
            use_rewrite=req.use_rewrite
        )

        items = [
            SearchResultItem(
                content=doc.page_content,
                score=doc.metadata.get('score', 0),
                source=doc.metadata.get('source', 'unknown'),
                search_type=search_type
            )
            for doc in results
        ]

        return SearchResponse(
            results=items,
            search_type=search_type,
            original_query=req.query,
            rewritten_query=rewritten_query
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
