import pytest
from unittest.mock import MagicMock, patch
from langrag.entities.document import Document
from langrag.entities.search_result import SearchResult
from langrag.retrieval.rerank.providers.cohere import CohereReranker
from langrag.retrieval.workflow import RetrievalWorkflow
from langrag.entities.dataset import Dataset

class TestCohereReranker:

    @patch("langrag.retrieval.rerank.providers.cohere.httpx.Client")
    def test_rerank_sync(self, mock_client_cls):
        # Mock Response
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "results": [
                {"index": 1, "relevance_score": 0.99},
                {"index": 0, "relevance_score": 0.50}
            ]
        }
        mock_client = mock_client_cls.return_value.__enter__.return_value
        mock_client.post.return_value = mock_response
        
        reranker = CohereReranker(api_key="key")
        
        d0 = Document(page_content="doc0", id="0")
        d1 = Document(page_content="doc1", id="1")
        results = [
            SearchResult(chunk=d0, score=0.1),
            SearchResult(chunk=d1, score=0.2)
        ]
        
        reranked = reranker.rerank("query", results)
        
        # Expect d1 (index 1) checks out first with score 0.99
        # Then d0 (index 0) with score 0.50
        assert len(reranked) == 2
        assert reranked[0].chunk.id == "1"
        assert reranked[0].score == 0.99
        assert reranked[1].chunk.id == "0"
        assert reranked[1].score == 0.50

class TestWorkflowRerank:
    
    @patch("langrag.retrieval.workflow.RetrievalService")
    def test_workflow_uses_reranker(self, mock_service):
        mock_reranker = MagicMock()
        workflow = RetrievalWorkflow(reranker=mock_reranker)
        
        # Setup documents
        doc1 = Document(page_content="d1", metadata={"score": 0.5})
        doc2 = Document(page_content="d2", metadata={"score": 0.6})
        mock_service.retrieve.return_value = [doc1, doc2]
        
        # Setup Reranker return
        # Return reversed order with higher scores
        res1 = SearchResult(chunk=doc2, score=0.9)
        res2 = SearchResult(chunk=doc1, score=0.8)
        mock_reranker.rerank.return_value = [res1, res2]
        
        dataset = Dataset(name="ds", collection_name="col")
        results = workflow.retrieve("query", [dataset])
        
        # Verify reranker called
        mock_reranker.rerank.assert_called_once()
        
        # Verify results order and scores
        assert len(results) == 2
        assert results[0].content == "d2"
        assert results[0].score == 0.9
        assert results[1].content == "d1"
        assert results[1].score == 0.8
