"""Knowledge Base Business Logic Service"""

import os
from datetime import datetime

from sqlmodel import Session, select

from web.core.rag_kernel import RAGKernel
from web.core.kb_retrieval_config import KBRetrievalConfig, RerankerConfig, RewriterConfig
from web.models.database import KnowledgeBase


class KBService:
    """Knowledge Base Service"""

    @staticmethod
    def create_kb(
        session: Session,
        rag_kernel: RAGKernel,
        name: str,
        description: str | None = None,
        vdb_type: str = "chroma",
        embedder_name: str | None = None,
        chunk_size: int = 1000,
        chunk_overlap: int = 100,
        # 检索配置
        search_mode: str = "hybrid",
        top_k: int = 5,
        score_threshold: float = 0.0,
        # Reranker 配置
        reranker_enabled: bool = False,
        reranker_type: str | None = None,
        reranker_model: str | None = None,
        reranker_api_key: str | None = None,
        reranker_top_k: int | None = None,
        # Rewriter 配置
        rewriter_enabled: bool = False,
        rewriter_llm_name: str | None = None
    ) -> KnowledgeBase:
        """Create a knowledge base with retrieval configuration"""
        # Generate unique ID
        kb_id = f"kb_{os.urandom(4).hex()}"
        collection_name = f"col_{os.urandom(4).hex()}"

        # Create DB record
        kb = KnowledgeBase(
            kb_id=kb_id,
            name=name,
            description=description,
            vdb_type=vdb_type,
            embedder_name=embedder_name,
            collection_name=collection_name,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            # 检索配置
            search_mode=search_mode,
            top_k=top_k,
            score_threshold=score_threshold,
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
        session.add(kb)
        session.commit()
        session.refresh(kb)

        # Initialize vector store in RAG kernel
        rag_kernel.create_vector_store(kb_id, collection_name, vdb_type, name=name)
        
        # 设置 KB 级别的检索配置
        retrieval_config = KBRetrievalConfig.from_kb_model(kb)
        rag_kernel.set_kb_retrieval_config(retrieval_config)

        return kb

    @staticmethod
    def get_kb(session: Session, kb_id: str) -> KnowledgeBase | None:
        """Get a knowledge base"""
        statement = select(KnowledgeBase).where(KnowledgeBase.kb_id == kb_id)
        return session.exec(statement).first()

    @staticmethod
    def list_kbs(session: Session) -> list[KnowledgeBase]:
        """List all knowledge bases"""
        statement = select(KnowledgeBase)
        return list(session.exec(statement).all())

    @staticmethod
    def update_kb(
        session: Session,
        rag_kernel: RAGKernel,
        kb_id: str,
        name: str | None = None,
        description: str | None = None,
        # 检索配置
        search_mode: str | None = None,
        top_k: int | None = None,
        score_threshold: float | None = None,
        # Reranker 配置
        reranker_enabled: bool | None = None,
        reranker_type: str | None = None,
        reranker_model: str | None = None,
        reranker_api_key: str | None = None,
        reranker_top_k: int | None = None,
        # Rewriter 配置
        rewriter_enabled: bool | None = None,
        rewriter_llm_name: str | None = None
    ) -> KnowledgeBase | None:
        """Update a knowledge base with retrieval configuration"""
        kb = KBService.get_kb(session, kb_id)
        if not kb:
            return None

        # 基本信息
        if name:
            kb.name = name
        if description is not None:
            kb.description = description
            
        # 检索配置
        if search_mode is not None:
            kb.search_mode = search_mode
        if top_k is not None:
            kb.top_k = top_k
        if score_threshold is not None:
            kb.score_threshold = score_threshold
            
        # Reranker 配置
        if reranker_enabled is not None:
            kb.reranker_enabled = reranker_enabled
        if reranker_type is not None:
            kb.reranker_type = reranker_type
        if reranker_model is not None:
            kb.reranker_model = reranker_model
        if reranker_api_key is not None:
            kb.reranker_api_key = reranker_api_key
        if reranker_top_k is not None:
            kb.reranker_top_k = reranker_top_k
            
        # Rewriter 配置
        if rewriter_enabled is not None:
            kb.rewriter_enabled = rewriter_enabled
        if rewriter_llm_name is not None:
            kb.rewriter_llm_name = rewriter_llm_name

        kb.updated_at = datetime.utcnow()
        session.add(kb)
        session.commit()
        session.refresh(kb)
        
        # 更新 RAG Kernel 中的检索配置
        retrieval_config = KBRetrievalConfig.from_kb_model(kb)
        rag_kernel.set_kb_retrieval_config(retrieval_config)
        
        return kb

    @staticmethod
    def delete_kb(session: Session, rag_kernel: RAGKernel, kb_id: str) -> bool:
        """Delete a knowledge base"""
        kb = KBService.get_kb(session, kb_id)
        if not kb:
            return False

        # Delete from DB
        session.delete(kb)
        session.commit()

        # Clean up vector store
        if kb_id in rag_kernel.vector_stores:
            del rag_kernel.vector_stores[kb_id]

        return True
