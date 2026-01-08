import pytest
from unittest.mock import MagicMock, patch
from langrag.entities.document import Document
from langrag.entities.search_result import SearchResult
from web.core.rag_kernel import RAGKernel

class TestRerankFlow:
    def test_search_with_rerank(self):
        # 1. Setup Kernel
        kernel = RAGKernel()
        kernel.workflow = MagicMock() # To bypass "workflow not initialized" warning if any
        
        # Mock Embedder
        embedder = MagicMock()
        embedder.embed.return_value = [[0.1, 0.2]]
        kernel.embedder = embedder
        
        # Mock VectorStore
        store = MagicMock()
        # Assume initial retrieval returns 3 docs with low/random scores
        initial_docs = [
            Document(page_content="Doc1", metadata={"score": 0.5}, id="1"),
            Document(page_content="Doc2", metadata={"score": 0.4}, id="2"),
            Document(page_content="Doc3", metadata={"score": 0.3}, id="3")
        ]
        store.search.return_value = initial_docs
        kernel.vector_stores["test_kb"] = store
        
        # Mock Reranker
        reranker = MagicMock()
        # Reranker reorders: Doc3 (best), Doc1, Doc2
        reranker.rerank.return_value = [
            SearchResult(chunk=initial_docs[2], score=0.99), # Doc3
            SearchResult(chunk=initial_docs[0], score=0.88)  # Doc1
        ]
        kernel.reranker = reranker
        
        # 2. Execute Search (top_k=2)
        results, search_type = kernel.search("test_kb", "query", top_k=2)
        
        # 3. Asserts
        
        # Verify initial retrieval expansion
        # top_k=2 -> k=10 passed to store.search
        store.search.assert_called_with("query", query_vector=[0.1, 0.2], top_k=10)
        
        # Verify rerank call
        reranker.rerank.assert_called_once()
        call_args = reranker.rerank.call_args
        assert call_args[0][0] == "query" # query arg
        assert len(call_args[0][1]) == 3 # input results (all 3 initial docs)
        assert call_args[1]['top_k'] == 2
        
        # Verify final results
        assert len(results) == 2
        assert results[0].id == "3" # Doc3 is now first
        assert results[0].metadata['score'] == 0.99
        assert results[1].id == "1"
        assert results[1].metadata['score'] == 0.88
        
        assert "+rerank" in search_type

    def test_set_reranker_factory(self):
        kernel = RAGKernel()
        with patch("web.core.rag_kernel.RerankerFactory") as mock_factory:
            kernel.set_reranker("cohere", api_key="test")
            mock_factory.create.assert_called_with("cohere", api_key="test")
            assert kernel.reranker is not None
