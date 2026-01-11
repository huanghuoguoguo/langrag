import pytest
from unittest.mock import MagicMock
from langrag.entities.dataset import Dataset
from langrag.retrieval.router.llm_router import LLMRouter
from langrag.retrieval.rewriter.llm_rewriter import LLMRewriter
from langrag.retrieval.workflow import RetrievalWorkflow

class TestAgenticComponents:
    
    def test_llm_router_routing(self):
        mock_llm = MagicMock()
        # Mock JSON response
        mock_llm.chat.return_value = '```json\n{"dataset_names": ["ds1"]}\n```'
        
        router = LLMRouter(mock_llm)
        
        d1 = Dataset(name="ds1", collection_name="c1")
        d2 = Dataset(name="ds2", collection_name="c2")
        
        selected = router.route("query", [d1, d2])
        
        assert len(selected) == 1
        assert selected[0].name == "ds1"
        assert "query" in mock_llm.chat.call_args[0][0][0]["content"]

    def test_llm_rewriter_rewriting(self):
        mock_llm = MagicMock()
        mock_llm.chat.return_value = "Better Query"
        
        rewriter = LLMRewriter(mock_llm)
        res = rewriter.rewrite("bad query")
        
        assert res == "Better Query"
        assert "bad query" in mock_llm.chat.call_args[0][0][0]["content"]

    def test_workflow_integration(self):
        mock_rewriter = MagicMock()
        mock_rewriter.rewrite.return_value = "Rewritten"
        
        workflow = RetrievalWorkflow(rewriter=mock_rewriter)
        
        # Mock Router and Service
        # We need to mock RetrievalService.retrieve to return empty list or something to avoid hitting DB
        from langrag.entities.document import Document
        with pytest.MonkeyPatch.context() as m:
             mock_retrieve = MagicMock(return_value=[])
             m.setattr("langrag.retrieval.executor.RetrievalService.retrieve", mock_retrieve)
             
             ds = Dataset(name="ds", collection_name="c")
             workflow.retrieve("original", [ds])
             
             # Verify Rewriter was called
             mock_rewriter.rewrite.assert_called_with("original")
             
             # Verify Retrieve was called with NEW query
             assert mock_retrieve.call_args[1]['query'] == "Rewritten"
