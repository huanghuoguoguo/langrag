import pytest
from langrag.entities.document import Document, DocumentType
from langrag.index_processor.splitter import RecursiveCharacterChunker

def test_recursive_splitter_initialization():
    splitter = RecursiveCharacterChunker(chunk_size=100, chunk_overlap=20)
    assert splitter.chunk_size == 100
    assert splitter.chunk_overlap == 20

def test_split_simple_document():
    text = "a" * 200
    doc = Document(page_content=text)
    splitter = RecursiveCharacterChunker(chunk_size=50, chunk_overlap=0)
    
    chunks = splitter.split([doc])
    
    assert len(chunks) >= 4
    for chunk in chunks:
        assert isinstance(chunk, Document)
        assert chunk.type == DocumentType.CHUNK
        assert len(chunk.page_content) <= 50
        assert chunk.metadata.get("source", "unknown") == doc.metadata.get("source", "unknown")

def test_split_with_metadata_preservation():
    doc = Document(
        page_content="Test content that is long enough to split.", 
        metadata={"source": "test.txt", "author": "me"}
    )
    splitter = RecursiveCharacterChunker(chunk_size=10, chunk_overlap=0)
    chunks = splitter.split([doc])
    
    assert len(chunks) > 1
    for chunk in chunks:
        assert chunk.metadata["source"] == "test.txt"
        assert chunk.metadata["author"] == "me"

