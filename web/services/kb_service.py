"""Knowledge Base Business Logic Service"""

import os
from datetime import datetime

from sqlmodel import Session, select

from web.core.rag_kernel import RAGKernel
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
        chunk_overlap: int = 100
    ) -> KnowledgeBase:
        """Create a knowledge base"""
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
            chunk_overlap=chunk_overlap
        )
        session.add(kb)
        session.commit()
        session.refresh(kb)

        # Initialize vector store in RAG kernel
        rag_kernel.create_vector_store(kb_id, collection_name, vdb_type, name=name)

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
        kb_id: str,
        name: str | None = None,
        description: str | None = None
    ) -> KnowledgeBase | None:
        """Update a knowledge base"""
        kb = KBService.get_kb(session, kb_id)
        if not kb:
            return None

        if name:
            kb.name = name
        if description is not None:
            kb.description = description

        kb.updated_at = datetime.utcnow()
        session.add(kb)
        session.commit()
        session.refresh(kb)
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
