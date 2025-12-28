import pytest
from pathlib import Path
from langrag.entities.dataset import Dataset
from langrag.entities.document import Document
from langrag.index_processor.extractor import SimpleTextParser
from langrag.index_processor.splitter import RecursiveCharacterChunker
from tests.utils.in_memory_vector_store import InMemoryVectorStore

@pytest.mark.integration
def test_full_indexing_and_retrieval_flow(tmp_path):
    # 1. Setup Data
    f = tmp_path / "knowledge.txt"
    f.write_text("""
    LangRAG is a modular RAG framework.
    It separates indexing and retrieval concerns.
    It supports multiple vector stores.
    """, encoding="utf-8")

    # 2. Setup Components
    parser = SimpleTextParser()
    chunker = RecursiveCharacterChunker(chunk_size=50, chunk_overlap=0)
    dataset = Dataset(name="int_ds", collection_name="integration_test")
    vector_store = InMemoryVectorStore(dataset=dataset)

    # 3. Indexing Pipeline Execution
    # Parse
    original_docs = parser.parse(f)
    assert len(original_docs) == 1
    
    # Split
    chunks = chunker.split(original_docs)
    assert len(chunks) > 1
    
    # Embed (Mock)
    for chunk in chunks:
        # Simple mock embedding: check for "modular" keyword for a specific dim
        val = 0.5 if "modular" in chunk.page_content else 0.1
        chunk.vector = [val] * 10
        
    # Store
    vector_store.add_texts(chunks)
    
    # 4. Retrieval
    # Search for "modular"
    query_vector = [0.5] * 10
    results = vector_store.search("modular", query_vector=query_vector, top_k=2)
    
    assert len(results) > 0
    assert "modular" in results[0].page_content
    # Score injected by InMemoryVectorStore
    assert results[0].metadata['score'] > 0.9  # Should be high similarity

@pytest.mark.integration
def test_batch_processing_flow(sample_document_files):
    # Setup
    parser = SimpleTextParser()
    chunker = RecursiveCharacterChunker(chunk_size=100)
    dataset = Dataset(name="batch_ds", collection_name="batch_test")
    store = InMemoryVectorStore(dataset=dataset)
    
    # Process batch
    all_chunks = []
    for file_path in sample_document_files:
        docs = parser.parse(file_path)
        chunks = chunker.split(docs)
        # Mock embed
        for c in chunks:
            c.vector = [0.1] * 10
        all_chunks.extend(chunks)
    
    store.add_texts(all_chunks)
    
    # Assert
    # We rely on internal list length of InMemoryVectorStore
    assert len(store._docs) == len(all_chunks)
