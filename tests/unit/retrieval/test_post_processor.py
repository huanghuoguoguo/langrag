import pytest
from langrag.entities.document import Document
from langrag.retrieval.post_processor import PostProcessor

class TestPostProcessor:
    
    @pytest.fixture
    def processor(self):
        return PostProcessor()
        
    def test_deduplication_by_id(self, processor):
        doc1 = Document(page_content="foo", metadata={"document_id": "1"}, id="1")
        doc2 = Document(page_content="bar", metadata={"document_id": "1"}, id="1") # Duplicate ID
        doc3 = Document(page_content="baz", metadata={"document_id": "2"}, id="2")
        
        docs = [doc1, doc2, doc3]
        result = processor.run(docs)
        
        assert len(result) == 2
        ids = [d.id for d in result]
        assert "1" in ids
        assert "2" in ids

    def test_deduplication_by_content_hash(self, processor):
        # Force empty ID to trigger hash fallback
        doc1 = Document(page_content="same payload", id="")
        doc2 = Document(page_content="same payload", id="")
        doc3 = Document(page_content="diff payload", id="")
        
        docs = [doc1, doc2, doc3]
        result = processor.run(docs)
        
        assert len(result) == 2
        contents = [d.page_content for d in result]
        assert "same payload" in contents
        assert "diff payload" in contents
        
    def test_score_filtering(self, processor):
        doc1 = Document(page_content="A", metadata={"score": 0.9})
        # Note: Document generates random ID by default, so they are unique
        doc2 = Document(page_content="B", metadata={"score": 0.5})
        doc3 = Document(page_content="C", metadata={"score": 0.1})
        
        docs = [doc1, doc2, doc3]
        
        # Test threshold 0.6
        result = processor.run(docs, score_threshold=0.6)
        assert len(result) == 1
        assert result[0].page_content == "A"
        
        # Test threshold 0.5 (inclusive)
        result = processor.run(docs, score_threshold=0.5)
        assert len(result) == 2
        
        # Test threshold 0.0
        result = processor.run(docs, score_threshold=0.0)
        assert len(result) == 3

    def test_deduplication_and_filtering(self, processor):
        # D1: High score
        doc1 = Document(page_content="A", id="1", metadata={"score": 0.9})
        # D2: Duplicate of D1, lower score (logic preserves first found)
        # If default order is D1, D2 -> Keeps D1
        doc2 = Document(page_content="A_dup", id="1", metadata={"score": 0.2})
        # D3: Unique, Low score
        doc3 = Document(page_content="B", id="2", metadata={"score": 0.3})
        
        docs = [doc1, doc2, doc3]
        
        # Filter 0.5
        # D1 is kept (score 0.9), D2 is removed (duplicate of D1), D3 is removed (score 0.3)
        result = processor.run(docs, score_threshold=0.5)
        assert len(result) == 1
        assert result[0].id == "1"
        assert result[0].metadata["score"] == 0.9
