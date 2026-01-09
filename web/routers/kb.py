"""Knowledge Base API"""


from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlmodel import Session

from web.core.database import get_session
from web.core.rag_kernel import RAGKernel
from web.services.kb_service import KBService

router = APIRouter(prefix="/api/kb", tags=["knowledge_base"])


class KBCreateRequest(BaseModel):
    name: str
    description: str | None = None
    vdb_type: str = "chroma"  # chroma, duckdb, seekdb
    embedder_name: str  # Required - must configure embedder first
    chunk_size: int = 1000
    chunk_overlap: int = 100


class KBResponse(BaseModel):
    id: str
    name: str
    description: str | None
    vdb_type: str
    embedder_name: str | None
    collection_name: str
    chunk_size: int
    chunk_overlap: int


def get_rag_kernel():
    """Dependency injection: Get RAG Kernel singleton"""
    from web.app import rag_kernel
    return rag_kernel


@router.post("", response_model=KBResponse)
def create_kb(
    req: KBCreateRequest,
    session: Session = Depends(get_session),
    rag_kernel: RAGKernel = Depends(get_rag_kernel)
):
    """Create knowledge base"""
    # Validate embedder_name is required
    if not req.embedder_name:
        raise HTTPException(
            status_code=400,
            detail="Embedder is required. Please configure an embedder in 'Model Configuration' page first."
        )

    kb = KBService.create_kb(
        session,
        rag_kernel,
        name=req.name,
        description=req.description,
        vdb_type=req.vdb_type,
        embedder_name=req.embedder_name,
        chunk_size=req.chunk_size,
        chunk_overlap=req.chunk_overlap
    )

    return KBResponse(
        id=kb.kb_id,
        name=kb.name,
        description=kb.description,
        vdb_type=kb.vdb_type,
        embedder_name=kb.embedder_name,
        collection_name=kb.collection_name,
        chunk_size=kb.chunk_size,
        chunk_overlap=kb.chunk_overlap
    )


@router.get("/{kb_id}", response_model=KBResponse)
def get_kb(
    kb_id: str,
    session: Session = Depends(get_session)
):
    """Get knowledge base details"""
    kb = KBService.get_kb(session, kb_id)
    if not kb:
        raise HTTPException(status_code=404, detail="Knowledge base not found")

    return KBResponse(
        id=kb.kb_id,
        name=kb.name,
        description=kb.description,
        vdb_type=kb.vdb_type,
        embedder_name=kb.embedder_name,
        collection_name=kb.collection_name,
        chunk_size=kb.chunk_size,
        chunk_overlap=kb.chunk_overlap
    )


@router.get("", response_model=list[KBResponse])
def list_kbs(session: Session = Depends(get_session)):
    """List all knowledge bases"""
    kbs = KBService.list_kbs(session)
    return [
        KBResponse(
            id=kb.kb_id,
            name=kb.name,
            description=kb.description,
            vdb_type=kb.vdb_type,
            embedder_name=kb.embedder_name,
            collection_name=kb.collection_name,
            chunk_size=kb.chunk_size,
            chunk_overlap=kb.chunk_overlap
        )
        for kb in kbs
    ]


@router.delete("/{kb_id}")
def delete_kb(
    kb_id: str,
    session: Session = Depends(get_session),
    rag_kernel: RAGKernel = Depends(get_rag_kernel)
):
    """Delete knowledge base"""
    success = KBService.delete_kb(session, rag_kernel, kb_id)
    if not success:
        raise HTTPException(status_code=404, detail="Knowledge base not found")

    return {"message": "Knowledge base deleted successfully"}
