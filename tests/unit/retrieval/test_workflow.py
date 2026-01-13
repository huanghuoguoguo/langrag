from unittest.mock import MagicMock, patch

import pytest

from langrag.entities.dataset import Dataset
from langrag.entities.document import Document
from langrag.retrieval import DEFAULT_MAX_WORKERS, RetrievalWorkflow, RetrievalExecutor


class TestRetrievalWorkflow:

    @pytest.fixture
    def workflow(self):
        return RetrievalWorkflow()

    def test_retrieve_empty_datasets(self, workflow):
        results = workflow.retrieve("query", [])
        assert results == []

    @patch("langrag.retrieval.executor.RetrievalService")
    def test_retrieve_single_dataset(self, mock_service, workflow):
        # Mock Service response
        doc = Document(page_content="foo", metadata={"score": 0.9, "document_id": "d1"})
        mock_service.retrieve.return_value = [doc]

        dataset = Dataset(name="ds", collection_name="col")
        results = workflow.retrieve("query", [dataset])

        assert len(results) == 1
        assert results[0].content == "foo"
        assert results[0].score == 0.9

    @patch("langrag.retrieval.executor.RetrievalService")
    def test_retrieve_filtering(self, mock_service, workflow):
        # Doc with low score
        doc = Document(page_content="foo", metadata={"score": 0.1, "document_id": "d1"})
        mock_service.retrieve.return_value = [doc]

        dataset = Dataset(name="ds", collection_name="col")
        # Threshold 0.5
        results = workflow.retrieve("query", [dataset], score_threshold=0.5)

        assert len(results) == 0

    @patch("langrag.retrieval.executor.RetrievalService")
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

    @patch("langrag.retrieval.executor.RetrievalService")
    def test_retrieve_error_handling(self, mock_service, workflow):
        # Service raises exception
        mock_service.retrieve.side_effect = Exception("DB Error")

        dataset = Dataset(name="ds", collection_name="col")
        results = workflow.retrieve("query", [dataset])

        # Should catch error and return empty list (or partial results if multiple datasets)
        assert results == []

    def test_max_workers_parameter(self):
        """Test that max_workers parameter is accepted and stored."""
        workflow = RetrievalWorkflow(max_workers=10)
        assert workflow.max_workers == 10

    def test_default_max_workers(self):
        """Test default max_workers value."""
        workflow = RetrievalWorkflow()
        assert workflow.max_workers == DEFAULT_MAX_WORKERS

    @patch("langrag.retrieval.executor.RetrievalService")
    def test_parallel_retrieval_multiple_datasets(self, mock_service):
        """Test that multiple datasets are retrieved in parallel."""
        from langrag.retrieval.config import WorkflowConfig
        # Disable retry for simpler testing
        config = WorkflowConfig(max_workers=3, enable_retrieval_retry=False)
        workflow = RetrievalWorkflow(config=config)

        # Create mock documents for each dataset
        def mock_retrieve(dataset, query, retrieval_method, top_k, vector_manager=None):
            return [
                Document(
                    page_content=f"content from {dataset.name}",
                    metadata={"score": 0.8, "document_id": f"doc_{dataset.name}"}
                )
            ]

        mock_service.retrieve.side_effect = mock_retrieve

        datasets = [
            Dataset(name="ds1", collection_name="col1"),
            Dataset(name="ds2", collection_name="col2"),
            Dataset(name="ds3", collection_name="col3"),
        ]

        results = workflow.retrieve("query", datasets)

        # All three datasets should have been queried
        assert mock_service.retrieve.call_count == 3

        # Should have 3 results (one from each dataset)
        assert len(results) == 3

    @patch("langrag.retrieval.executor.RetrievalService")
    def test_parallel_retrieval_partial_failure(self, mock_service):
        """Test that partial failures don't prevent successful results."""
        from langrag.retrieval.config import WorkflowConfig
        config = WorkflowConfig(max_workers=3, enable_retrieval_retry=False)
        workflow = RetrievalWorkflow(config=config)

        call_count = [0]

        def mock_retrieve(dataset, query, retrieval_method, top_k, vector_manager=None):
            call_count[0] += 1
            if dataset.name == "ds2":
                raise Exception("Dataset 2 failed")
            return [
                Document(
                    page_content=f"content from {dataset.name}",
                    metadata={"score": 0.8, "document_id": f"doc_{dataset.name}"}
                )
            ]

        mock_service.retrieve.side_effect = mock_retrieve

        datasets = [
            Dataset(name="ds1", collection_name="col1"),
            Dataset(name="ds2", collection_name="col2"),
            Dataset(name="ds3", collection_name="col3"),
        ]

        results = workflow.retrieve("query", datasets)

        # All three datasets should have been attempted
        assert call_count[0] == 3

        # Should have 2 results (ds1 and ds3 succeeded, ds2 failed)
        assert len(results) == 2

    @patch("langrag.retrieval.executor.RetrievalService")
    def test_single_dataset_no_parallel(self, mock_service):
        """Test that single dataset retrieval doesn't use thread pool overhead."""
        workflow = RetrievalWorkflow()

        doc = Document(page_content="foo", metadata={"score": 0.9, "document_id": "d1"})
        mock_service.retrieve.return_value = [doc]

        dataset = Dataset(name="ds", collection_name="col")
        results = workflow.retrieve("query", [dataset])

        # Should still work correctly
        assert len(results) == 1
        assert results[0].content == "foo"

        # Service should be called once
        assert mock_service.retrieve.call_count == 1

    @patch("langrag.retrieval.executor.RetrievalService")
    def test_parallel_respects_max_workers(self, mock_service):
        """Test that parallel retrieval respects max_workers limit."""
        from langrag.retrieval.config import WorkflowConfig
        # Use a very small max_workers and disable retry
        config = WorkflowConfig(max_workers=2, enable_retrieval_retry=False)
        workflow = RetrievalWorkflow(config=config)

        # Create unique documents for each dataset to avoid deduplication
        call_counter = [0]

        def mock_retrieve(dataset, query, retrieval_method, top_k, vector_manager=None):
            call_counter[0] += 1
            return [
                Document(
                    page_content=f"content from {dataset.name}",
                    metadata={"score": 0.8, "document_id": f"doc_{dataset.name}"}
                )
            ]

        mock_service.retrieve.side_effect = mock_retrieve

        # Create more datasets than max_workers
        datasets = [
            Dataset(name=f"ds{i}", collection_name=f"col{i}")
            for i in range(5)
        ]

        # Use top_k larger than number of datasets to get all results
        results = workflow.retrieve("query", datasets, top_k=10)

        # All datasets should still be queried
        assert call_counter[0] == 5

        # Should have results from all datasets
        assert len(results) == 5

    def test_get_retrieval_method_economy(self):
        """Test retrieval method selection for economy indexing."""
        workflow = RetrievalWorkflow()
        dataset = Dataset(name="ds", collection_name="col", indexing_technique="economy")
        method = workflow._executor._get_retrieval_method(dataset)
        assert method == "keyword_search"

    def test_get_retrieval_method_default(self):
        """Test retrieval method selection for default indexing."""
        workflow = RetrievalWorkflow()
        dataset = Dataset(name="ds", collection_name="col")
        method = workflow._executor._get_retrieval_method(dataset)
        assert method == "semantic_search"

