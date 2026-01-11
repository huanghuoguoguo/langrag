import pytest
from unittest.mock import MagicMock, patch
from uuid import uuid4
from langrag.core.callbacks.manager import CallbackManager
from langrag.core.callbacks.base import BaseCallbackHandler
from langrag.retrieval.workflow import RetrievalWorkflow
from langrag.entities.dataset import Dataset
from langrag.entities.document import Document

class TestCallbackSystem:
    def test_manager_delegates(self):
        handler = MagicMock(spec=BaseCallbackHandler)
        manager = CallbackManager([handler])
        
        run_id = manager.on_retrieve_start("query")
        handler.on_retrieve_start.assert_called_once()
        args = handler.on_retrieve_start.call_args
        assert args[0][1] == run_id # run_id check
        
        manager.on_retrieve_end([], run_id)
        handler.on_retrieve_end.assert_called_once()

class TestWorkflowCallbacks:
    def test_workflow_triggers_callbacks(self):
        handler = MagicMock(spec=BaseCallbackHandler)
        manager = CallbackManager([handler])
        
        workflow = RetrievalWorkflow()
        workflow.set_callback_manager(manager)
        
        # Mock RetrievalService to avoid hitting DB
        with (
            pytest.MonkeyPatch.context() as m,
            patch("langrag.retrieval.executor.RetrievalService") as mock_service
        ):
            mock_service.retrieve.return_value = [Document(page_content="foo", metadata={"score": 0.5})]
            
            dataset = Dataset(name="ds", collection_name="col")
            workflow.retrieve("query", [dataset])
            
            handler.on_retrieve_start.assert_called_once()
            handler.on_retrieve_end.assert_called_once()
            
            # Check calling args - query is first positional arg
            start_args = handler.on_retrieve_start.call_args[0]
            assert start_args[0] == "query"

    def test_workflow_triggers_error_callback(self):
        handler = MagicMock(spec=BaseCallbackHandler)
        manager = CallbackManager([handler])
        
        workflow = RetrievalWorkflow()
        workflow.set_callback_manager(manager)
        
        # Mock PostProcessor to raise exception (one of the few things not swallowed inside loop)
        with patch.object(workflow.post_processor, 'run', side_effect=Exception("Critical Failure")):
            dataset = Dataset(name="ds", collection_name="col")
            
            with pytest.raises(Exception):
                workflow.retrieve("query", [dataset])
            
            handler.on_retrieve_start.assert_called_once()
            handler.on_error.assert_called_once()
