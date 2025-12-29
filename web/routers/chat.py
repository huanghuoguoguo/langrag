"""对话 API"""

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from typing import List, Optional
from sqlmodel import Session

from web.core.database import get_session
from web.core.rag_kernel import RAGKernel
from web.services.kb_service import KBService

router = APIRouter(prefix="/api/chat", tags=["chat"])


class Message(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    kb_ids: List[str] = []
    query: str
    history: List[Message] = []


class SourceItem(BaseModel):
    content: str
    score: float
    source: str
    kb_id: Optional[str] = None


class ChatResponse(BaseModel):
    answer: str
    sources: List[SourceItem]


def get_rag_kernel():
    """依赖注入：获取 RAG Kernel 单例"""
    from web.app import rag_kernel
    return rag_kernel


@router.post("", response_model=ChatResponse)
async def chat(
    req: ChatRequest,
    session: Session = Depends(get_session),
    rag_kernel: RAGKernel = Depends(get_rag_kernel)
):
    """执行 RAG 对话"""
    # Verify KBs exist (optional, kernel will skip invalid ones)
    
    try:
        # Convert Pydantic models to dicts for internal use
        history_dicts = [{"role": m.role, "content": m.content} for m in req.history]
        
        target_kb_ids = req.kb_ids
        # Auto-select all KBs if none provided
        if not target_kb_ids:
            all_kbs = KBService.list_kbs(session)
            target_kb_ids = [kb.kb_id for kb in all_kbs]
            
        result = await rag_kernel.chat(
            kb_ids=target_kb_ids,
            query=req.query,
            history=history_dicts
        )
        
        return ChatResponse(
            answer=result["answer"],
            sources=[
                SourceItem(**item) for item in result["sources"]
            ]
        )
        
    except ValueError as e:
        # Typically "LLM not configured"
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
