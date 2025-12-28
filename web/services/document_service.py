"""文档处理服务"""

from datetime import datetime
from pathlib import Path
from sqlmodel import Session, select
from typing import List, Optional

from web.models.database import Document, KnowledgeBase
from web.core.rag_kernel import RAGKernel


class DocumentService:
    """文档处理服务"""
    
    @staticmethod
    def create_document(
        session: Session,
        kb_id: str,
        filename: str,
        file_size: int
    ) -> Document:
        """创建文档记录"""
        doc = Document(
            kb_id=kb_id,
            filename=filename,
            file_size=file_size,
            status="pending"
        )
        session.add(doc)
        session.commit()
        session.refresh(doc)
        return doc
    
    @staticmethod
    def process_document(
        session: Session,
        rag_kernel: RAGKernel,
        doc_id: int,
        file_path: Path
    ) -> Document:
        """处理文档"""
        import logging
        logger = logging.getLogger(__name__)
        
        logger.info(f"[DocumentService] Starting to process document ID: {doc_id}")
        
        # Get document record
        doc = session.get(Document, doc_id)
        if not doc:
            logger.error(f"[DocumentService] Document {doc_id} not found in database")
            raise ValueError(f"Document {doc_id} not found")
        
        logger.info(f"[DocumentService] Document found: {doc.filename}, KB: {doc.kb_id}")
        
        # Get KB config
        statement = select(KnowledgeBase).where(KnowledgeBase.kb_id == doc.kb_id)
        kb = session.exec(statement).first()
        if not kb:
            logger.error(f"[DocumentService] Knowledge base {doc.kb_id} not found in database")
            raise ValueError(f"Knowledge base {doc.kb_id} not found")
        
        logger.info(f"[DocumentService] KB config loaded: chunk_size={kb.chunk_size}, chunk_overlap={kb.chunk_overlap}")
        
        # Update status
        doc.status = "processing"
        session.add(doc)
        session.commit()
        logger.info(f"[DocumentService] Document status updated to 'processing'")
        
        try:
            logger.info(f"[DocumentService] Calling RAG kernel to process document...")
            logger.info(f"[DocumentService] File path: {file_path}")
            logger.info(f"[DocumentService] KB ID: {doc.kb_id}")
            
            # Process with RAG kernel
            chunk_count = rag_kernel.process_document(
                file_path,
                kb_id=doc.kb_id,
                chunk_size=kb.chunk_size,
                chunk_overlap=kb.chunk_overlap
            )
            
            logger.info(f"[DocumentService] RAG kernel processing completed. Chunks created: {chunk_count}")
            
            # Update success
            doc.chunk_count = chunk_count
            doc.status = "completed"
            doc.processed_at = datetime.utcnow()
            logger.info(f"[DocumentService] Document status updated to 'completed'")
            
        except Exception as e:
            # Update failure
            logger.error(f"[DocumentService] Processing failed with error: {type(e).__name__}: {str(e)}")
            logger.exception(f"[DocumentService] Full traceback:")
            doc.status = "failed"
            raise e
        finally:
            session.add(doc)
            session.commit()
            session.refresh(doc)
        
        logger.info(f"[DocumentService] Document processing finished for ID: {doc_id}")
        return doc
    
    @staticmethod
    def list_documents(session: Session, kb_id: str) -> List[Document]:
        """列出知识库的所有文档"""
        statement = select(Document).where(Document.kb_id == kb_id)
        return list(session.exec(statement).all())
    
    @staticmethod
    def get_document(session: Session, doc_id: int) -> Optional[Document]:
        """获取文档"""
        return session.get(Document, doc_id)
