import pytest
from langrag.entities.document import Document, DocumentType

def test_document_initialization():
    doc = Document(page_content="Test content", metadata={"key": "value"})
    assert doc.page_content == "Test content"
    assert doc.metadata["key"] == "value"
    assert doc.type == DocumentType.ORIGINAL
    assert doc.id is not None

def test_document_chunk_type():
    doc = Document(
        page_content="Chunk content", 
        type=DocumentType.CHUNK,
        vector=[0.1, 0.2]
    )
    assert doc.type == DocumentType.CHUNK
    assert doc.vector == [0.1, 0.2]

def test_document_serialization():
    doc = Document(page_content="Hello", metadata={"a": 1})
    data = doc.model_dump()
    assert data["page_content"] == "Hello"
    assert data["metadata"] == {"a": 1}

def test_document_hash_id_generation():
    doc1 = Document(page_content="Same content")
    doc2 = Document(page_content="Same content")
    # IDs are typically UUIDs and random by default in pydantic models unless default factory logic is deterministic
    # The current Document model likely uses uuid4, so ids should be different
    assert doc1.id != doc2.id
