"""检索 API"""

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from typing import List
from sqlmodel import Session

from web.core.database import get_session
from web.core.rag_kernel import RAGKernel
from web.services.kb_service import KBService

router = APIRouter(prefix="/api/search", tags=["search"])


class SearchRequest(BaseModel):
    kb_id: str
    query: str
    top_k: int = 5


class SearchResultItem(BaseModel):
    content: str
    score: float
    source: str
    search_type: str


class SearchResponse(BaseModel):
    results: List[SearchResultItem]
    search_type: str


def get_rag_kernel():
    """依赖注入：获取 RAG Kernel 单例"""
    from web.app import rag_kernel
    return rag_kernel


@router.post("", response_model=SearchResponse)
def search(
    req: SearchRequest,
    session: Session = Depends(get_session),
    rag_kernel: RAGKernel = Depends(get_rag_kernel)
):
    """执行检索"""
    # Verify KB exists
    kb = KBService.get_kb(session, req.kb_id)
    if not kb:
        raise HTTPException(status_code=404, detail="Knowledge base not found")
    
    try:
        results, search_type = rag_kernel.search(
            kb_id=req.kb_id,
            query=req.query,
            top_k=req.top_k
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
            search_type=search_type
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
