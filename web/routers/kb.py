"""Knowledge Base API"""


from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlmodel import Session

from web.core.database import get_session
from web.core.rag_kernel import RAGKernel
from web.services.kb_service import KBService

router = APIRouter(prefix="/api/kb", tags=["knowledge_base"])


class RerankerConfigRequest(BaseModel):
    """Reranker 配置请求"""
    enabled: bool = False
    reranker_type: str | None = None  # "cohere", "qwen", "noop"
    model: str | None = None
    api_key: str | None = None
    top_k: int | None = None  # Rerank 后返回数量


class RewriterConfigRequest(BaseModel):
    """Rewriter 配置请求"""
    enabled: bool = False
    llm_name: str | None = None  # 使用的 LLM 配置名称


class KBCreateRequest(BaseModel):
    name: str
    description: str | None = None
    vdb_type: str = "chroma"  # chroma, duckdb, seekdb
    embedder_name: str  # Required - must configure embedder first
    chunk_size: int = 1000
    chunk_overlap: int = 100
    # 检索配置
    search_mode: str = "hybrid"  # "hybrid", "vector", "keyword"
    top_k: int = 5
    score_threshold: float = 0.0
    # 组件配置
    reranker: RerankerConfigRequest | None = None
    rewriter: RewriterConfigRequest | None = None


class KBUpdateRequest(BaseModel):
    """知识库更新请求"""
    name: str | None = None
    description: str | None = None
    # 检索配置
    search_mode: str | None = None
    top_k: int | None = None
    score_threshold: float | None = None
    # 组件配置
    reranker: RerankerConfigRequest | None = None
    rewriter: RewriterConfigRequest | None = None


class RerankerConfigResponse(BaseModel):
    """Reranker 配置响应"""
    enabled: bool
    reranker_type: str | None
    model: str | None
    top_k: int | None


class RewriterConfigResponse(BaseModel):
    """Rewriter 配置响应"""
    enabled: bool
    llm_name: str | None


class KBResponse(BaseModel):
    id: str
    name: str
    description: str | None
    vdb_type: str
    embedder_name: str | None
    collection_name: str
    chunk_size: int
    chunk_overlap: int
    # 检索配置
    search_mode: str
    top_k: int
    score_threshold: float
    # 组件配置
    reranker: RerankerConfigResponse
    rewriter: RewriterConfigResponse


def get_rag_kernel():
    """Dependency injection: Get RAG Kernel singleton"""
    from web.app import rag_kernel
    return rag_kernel


def _build_kb_response(kb) -> KBResponse:
    """构建 KB 响应对象"""
    return KBResponse(
        id=kb.kb_id,
        name=kb.name,
        description=kb.description,
        vdb_type=kb.vdb_type,
        embedder_name=kb.embedder_name,
        collection_name=kb.collection_name,
        chunk_size=kb.chunk_size,
        chunk_overlap=kb.chunk_overlap,
        search_mode=kb.search_mode or "hybrid",
        top_k=kb.top_k or 5,
        score_threshold=kb.score_threshold or 0.0,
        reranker=RerankerConfigResponse(
            enabled=kb.reranker_enabled or False,
            reranker_type=kb.reranker_type,
            model=kb.reranker_model,
            top_k=kb.reranker_top_k
        ),
        rewriter=RewriterConfigResponse(
            enabled=kb.rewriter_enabled or False,
            llm_name=kb.rewriter_llm_name
        )
    )


@router.post("", response_model=KBResponse)
def create_kb(
    req: KBCreateRequest,
    session: Session = Depends(get_session),
    rag_kernel: RAGKernel = Depends(get_rag_kernel)
):
    """Create knowledge base with retrieval configuration"""
    # Validate embedder_name is required
    if not req.embedder_name:
        raise HTTPException(
            status_code=400,
            detail="Embedder is required. Please configure an embedder in 'Model Configuration' page first."
        )

    # 解析 reranker 配置
    reranker_enabled = False
    reranker_type = None
    reranker_model = None
    reranker_api_key = None
    reranker_top_k = None
    if req.reranker:
        reranker_enabled = req.reranker.enabled
        reranker_type = req.reranker.reranker_type
        reranker_model = req.reranker.model
        reranker_api_key = req.reranker.api_key
        reranker_top_k = req.reranker.top_k
        
    # 解析 rewriter 配置
    rewriter_enabled = False
    rewriter_llm_name = None
    if req.rewriter:
        rewriter_enabled = req.rewriter.enabled
        rewriter_llm_name = req.rewriter.llm_name

    kb = KBService.create_kb(
        session,
        rag_kernel,
        name=req.name,
        description=req.description,
        vdb_type=req.vdb_type,
        embedder_name=req.embedder_name,
        chunk_size=req.chunk_size,
        chunk_overlap=req.chunk_overlap,
        # 检索配置
        search_mode=req.search_mode,
        top_k=req.top_k,
        score_threshold=req.score_threshold,
        # Reranker 配置
        reranker_enabled=reranker_enabled,
        reranker_type=reranker_type,
        reranker_model=reranker_model,
        reranker_api_key=reranker_api_key,
        reranker_top_k=reranker_top_k,
        # Rewriter 配置
        rewriter_enabled=rewriter_enabled,
        rewriter_llm_name=rewriter_llm_name
    )

    return _build_kb_response(kb)


@router.get("/{kb_id}", response_model=KBResponse)
def get_kb(
    kb_id: str,
    session: Session = Depends(get_session)
):
    """Get knowledge base details with retrieval configuration"""
    kb = KBService.get_kb(session, kb_id)
    if not kb:
        raise HTTPException(status_code=404, detail="Knowledge base not found")

    return _build_kb_response(kb)


@router.get("", response_model=list[KBResponse])
def list_kbs(session: Session = Depends(get_session)):
    """List all knowledge bases with retrieval configuration"""
    kbs = KBService.list_kbs(session)
    return [_build_kb_response(kb) for kb in kbs]


@router.put("/{kb_id}", response_model=KBResponse)
def update_kb(
    kb_id: str,
    req: KBUpdateRequest,
    session: Session = Depends(get_session),
    rag_kernel: RAGKernel = Depends(get_rag_kernel)
):
    """Update knowledge base with retrieval configuration"""
    # 解析 reranker 配置
    reranker_enabled = None
    reranker_type = None
    reranker_model = None
    reranker_api_key = None
    reranker_top_k = None
    if req.reranker:
        reranker_enabled = req.reranker.enabled
        reranker_type = req.reranker.reranker_type
        reranker_model = req.reranker.model
        reranker_api_key = req.reranker.api_key
        reranker_top_k = req.reranker.top_k
        
    # 解析 rewriter 配置
    rewriter_enabled = None
    rewriter_llm_name = None
    if req.rewriter:
        rewriter_enabled = req.rewriter.enabled
        rewriter_llm_name = req.rewriter.llm_name

    kb = KBService.update_kb(
        session,
        rag_kernel,
        kb_id=kb_id,
        name=req.name,
        description=req.description,
        # 检索配置
        search_mode=req.search_mode,
        top_k=req.top_k,
        score_threshold=req.score_threshold,
        # Reranker 配置
        reranker_enabled=reranker_enabled,
        reranker_type=reranker_type,
        reranker_model=reranker_model,
        reranker_api_key=reranker_api_key,
        reranker_top_k=reranker_top_k,
        # Rewriter 配置
        rewriter_enabled=rewriter_enabled,
        rewriter_llm_name=rewriter_llm_name
    )
    
    if not kb:
        raise HTTPException(status_code=404, detail="Knowledge base not found")

    return _build_kb_response(kb)


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
