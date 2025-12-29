import pytest
from unittest.mock import MagicMock, patch
from langrag.retrieval.workflow import RetrievalWorkflow
from langrag.entities.dataset import Dataset, RetrievalContext
from langrag.entities.document import Document

class TestRetrievalWorkflow:

    @pytest.fixture
    def workflow(self):
        return RetrievalWorkflow()

    def test_retrieve_empty_datasets(self, workflow):
        results = workflow.retrieve("query", [])
        assert results == []

    @patch("langrag.retrieval.workflow.RetrievalService")
    def test_retrieve_single_dataset(self, mock_service, workflow):
        # Mock Service response
        doc = Document(page_content="foo", metadata={"score": 0.9, "document_id": "d1"})
        mock_service.retrieve.return_value = [doc]
        
        dataset = Dataset(name="ds", collection_name="col")
        results = workflow.retrieve("query", [dataset])
        
        assert len(results) == 1
        assert results[0].content == "foo"
        assert results[0].score == 0.9
        
    @patch("langrag.retrieval.workflow.RetrievalService")
    def test_retrieve_filtering(self, mock_service, workflow):
        # Doc with low score
        doc = Document(page_content="foo", metadata={"score": 0.1, "document_id": "d1"})
        mock_service.retrieve.return_value = [doc]
        
        dataset = Dataset(name="ds", collection_name="col")
        # Threshold 0.5
        results = workflow.retrieve("query", [dataset], score_threshold=0.5)
        
        assert len(results) == 0

    @patch("langrag.retrieval.workflow.RetrievalService")
    def test_retrieve_workflow_router(self, mock_service, workflow):
        # Setup router
        mock_router = MagicMock()
        workflow.router = mock_router
        
        ds1 = Dataset(name="ds1", collection_name="col1")
        ds2 = Dataset(name="ds2", collection_name="col2")
        
        # Router selects only ds1
        mock_router.route.return_value = [ds1]
        mock_service.retrieve.return_value = []
        
        workflow.retrieve("query", [ds1, ds2])
        
        mock_router.route.assert_called_once()
        # Ensure service called only for ds1?
        # Check call args of service
        # Service.retrieve is called once for ds1
        # But wait, it's called in a loop.
        
        # Check how many times retrieve was called
        assert mock_service.retrieve.call_count == 1
        args = mock_service.retrieve.call_args[1]
        assert args['dataset'] == ds1

    @patch("langrag.retrieval.workflow.RetrievalService")
    def test_retrieve_error_handling(self, mock_service, workflow):
        # Service raises exception
        mock_service.retrieve.side_effect = Exception("DB Error")
        
        dataset = Dataset(name="ds", collection_name="col")
        results = workflow.retrieve("query", [dataset])
        
        # Should catch error and return empty list (or partial results if multiple datasets)
        assert results == []
