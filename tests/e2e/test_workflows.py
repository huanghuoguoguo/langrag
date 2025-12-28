"""End-to-end tests for LangRAG framework using new modular architecture."""

import pytest
from pathlib import Path
from typing import List

from langrag.entities.dataset import Dataset
from langrag.entities.document import Document
from langrag.index_processor.extractor import SimpleTextParser
from langrag.index_processor.splitter import RecursiveCharacterChunker
from langrag.retrieval.workflow import RetrievalWorkflow
# from langrag.retrieval.retriever import Retriever # In case we need it, but mostly Workflow

# Use our utility store
from tests.utils.in_memory_vector_store import InMemoryVectorStore


class SimpleRAGEngine:
    """Helper facade to mimic RAGEngine for E2E tests using new modular components."""
    
    def __init__(self, collection_name="e2e_test"):
        self.dataset = Dataset(name="e2e_ds", collection_name=collection_name)
        self.parser = SimpleTextParser()
        self.chunker = RecursiveCharacterChunker(chunk_size=200, chunk_overlap=20)
        # For simplicity, we directly instantiate the store. 
        # In real workflow, RetrievalService instantiates it.
        # But RetrievalWorkflow allows passing vector_store_cls.
        # Here we need a way to populate the same store that RetrievalService will read.
        # This is tricky because InMemoryStore is transient.
        # Solution: Subclass InMemoryStore to be a Singleton or pass partial functional.
        # For E2E tests, let's keep it simple: we pass the instantiated store to our workflow if possible?
        # RetrievalWorkflow takes vector_store_cls.
        
        # Hack for testing: We will mock RetrievalService's VDB instantiation 
        # OR we just use Manual Pipeline for E2E validation as seen in main.py
        
        self.store = InMemoryVectorStore(self.dataset)

    def index(self, file_path: Path) -> int:
        docs = self.parser.parse(file_path)
        chunks = self.chunker.split(docs)
        
        # Simulate Embedding
        for c in chunks:
            c.vector = [0.1] * 10
            
        self.store.add_texts(chunks)
        return len(chunks)

    def index_batch(self, file_paths: List[Path]) -> int:
        total = 0
        for f in file_paths:
            total += self.index(f)
        return total

    def retrieve(self, query: str, top_k=5):
        # We manually call store search to simulate retrieval
        # since we can't easily inject the populated InMemoryStore into RetrievalWorkflow 
        # which expects to instantiate it.
        # (Unless we patch RetrievalService)
        
        # Simulate query vector
        query_vector = [0.1] * 10
        
        docs = self.store.search(query, query_vector=query_vector, top_k=top_k)
        return docs


@pytest.mark.e2e
class TestBasicRAGWorkflow:
    
    def test_complete_rag_pipeline(self, tmp_path):
        # 1. Prepare
        f = tmp_path / "rag_test.txt"
        f.write_text("RAG systems use vector databases for retrieval.", encoding="utf-8")
        
        # 2. Engine
        engine = SimpleRAGEngine()
        
        # 3. Index
        count = engine.index(f)
        assert count > 0
        
        # 4. Retrieve
        results = engine.retrieve("vector databases")
        assert len(results) > 0
        assert "vector databases" in results[0].page_content

    def test_batch_indexing_workflow(self, tmp_path):
        files = []
        for i in range(3):
            p = tmp_path / f"doc_{i}.txt"
            p.write_text(f"Content for document {i}", encoding="utf-8")
            files.append(p)
            
        engine = SimpleRAGEngine()
        total = engine.index_batch(files)
        assert total >= 3
        
        res = engine.retrieve("Content")
        assert len(res) > 0


@pytest.mark.e2e
class TestLargeScaleIndexing:
    
    def test_large_document_indexing(self, large_document_file):
        engine = SimpleRAGEngine()
        count = engine.index(large_document_file)
        assert count > 10 # Should split into multiple chunks
        
        results = engine.retrieve("artificial intelligence", top_k=5)
        # Since we use dummy embedding, we might get random results or all results
        # But we check for existence
        assert len(results) > 0

    def test_multilingual_documents(self, multilingual_document_files):
        engine = SimpleRAGEngine()
        engine.index_batch(multilingual_document_files)
        
        zh_res = engine.retrieve("人工智能")
        assert len(zh_res) > 0
