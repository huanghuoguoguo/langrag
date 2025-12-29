import pytest
from unittest.mock import MagicMock
from langrag.entities.dataset import Dataset
from langrag.entities.document import Document
from langrag.index_processor.processor.qa import QAIndexProcessor

class TestQAIndexProcessor:
    def test_qa_generation_flow(self):
        # Mock Dependencies
        vdb = MagicMock()
        llm = MagicMock()
        splitter = MagicMock()
        
        # Setup mocks
        splitter.split_documents.return_value = [Document(page_content="Answer Content", id="1")]
        # Mock Chat response
        llm.chat.return_value = "Generated Question?"
        # Mock Embed
        llm.embed_documents.return_value = [[0.1]*10]
        
        processor = QAIndexProcessor(vdb, llm, splitter)
        dataset = Dataset(name="ds", collection_name="col")
        
        processor.process(dataset, [Document(page_content="doc")])
        
        # Assertions
        # 1. Chat called
        llm.chat.assert_called_once()
        args = llm.chat.call_args
        assert "Answer Content" in args[0][0][0]["content"]
        
        # 2. VDB Create called with Question doc
        vdb.create.assert_called_once()
        saved_docs = vdb.create.call_args[0][0]
        assert len(saved_docs) == 1
        assert saved_docs[0].page_content == "Generated Question?"
        assert saved_docs[0].metadata["answer"] == "Answer Content"
