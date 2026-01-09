"""Document Upload and Processing API"""

import shutil
import tempfile
from pathlib import Path

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from pydantic import BaseModel
from sqlmodel import Session

from web.core.database import get_session
from web.core.rag_kernel import RAGKernel
from web.services.document_service import DocumentService
from web.services.kb_service import KBService

router = APIRouter(prefix="/api/upload", tags=["document"])


class UploadResponse(BaseModel):
    processed_files: int
    total_chunks: int
    failed_files: list[str]


def get_rag_kernel():
    """Dependency injection: Get RAG Kernel singleton"""
    from web.app import rag_kernel
    return rag_kernel


@router.post("", response_model=UploadResponse)
async def upload_documents(
    kb_id: str = Form(...),
    files: list[UploadFile] = File(...),
    session: Session = Depends(get_session),
    rag_kernel: RAGKernel = Depends(get_rag_kernel)
):
    """Upload and process documents"""
    # Verify KB exists
    kb = KBService.get_kb(session, kb_id)
    if not kb:
        raise HTTPException(status_code=404, detail="Knowledge base not found")

    processed_count = 0
    total_chunks = 0
    failed_files = []

    with tempfile.TemporaryDirectory() as temp_dir:
        for file in files:
            try:
                # Save file temporarily
                temp_path = Path(temp_dir) / file.filename
                with open(temp_path, "wb") as buffer:
                    shutil.copyfileobj(file.file, buffer)

                # Create document record
                doc = DocumentService.create_document(
                    session,
                    kb_id=kb_id,
                    filename=file.filename,
                    file_size=temp_path.stat().st_size
                )

                # Process document
                processed_doc = DocumentService.process_document(
                    session,
                    rag_kernel,
                    doc_id=doc.id,
                    file_path=temp_path
                )

                processed_count += 1
                total_chunks += processed_doc.chunk_count

            except Exception as e:
                failed_files.append(f"{file.filename}: {str(e)}")

    return UploadResponse(
        processed_files=processed_count,
        total_chunks=total_chunks,
        failed_files=failed_files
    )


@router.get("/documents/{kb_id}")
def list_documents(
    kb_id: str,
    session: Session = Depends(get_session)
):
    """List all documents in a knowledge base"""
    docs = DocumentService.list_documents(session, kb_id)
    return [
        {
            "id": doc.id,
            "filename": doc.filename,
            "file_size": doc.file_size,
            "chunk_count": doc.chunk_count,
            "status": doc.status,
            "created_at": doc.created_at,
            "processed_at": doc.processed_at
        }
        for doc in docs
    ]
