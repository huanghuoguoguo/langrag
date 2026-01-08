import pytest
from unittest.mock import MagicMock
from langrag.entities.dataset import Dataset
from langrag.entities.document import Document
from langrag.index_processor.processor.parent_child import ParentChildIndexProcessor
from langrag.datasource.kv.in_memory import InMemoryKV

class TestParentChildIndexProcessorV2:
    def test_processing_flow(self):
        # Mock Dependencies
        vdb = MagicMock()
        embedder = MagicMock()
        # Mock embed to return list of lists based on input length
        def mock_embed(texts):
            return [[0.1]*10] * len(texts)
        embedder.embed.side_effect = mock_embed
        
        # Splitters
        # Mock Parent Splitter: 1 Doc -> 1 Parent
        parent_splitter = MagicMock()
        parent_doc = Document(page_content="Parent Content", id="p1")
        parent_splitter.split_documents.return_value = [parent_doc]
        
        # Mock Child Splitter: 1 Parent -> 2 Children
        child_splitter = MagicMock()
        child_splitter.split_documents.return_value = [
            Document(page_content="Child 1", id="c1"),
            Document(page_content="Child 2", id="c2")
        ]
        
        kv_store = InMemoryKV()
        
        processor = ParentChildIndexProcessor(
            vector_store=vdb,
            kv_store=kv_store,
            embedder=embedder,
            parent_splitter=parent_splitter,
            child_splitter=child_splitter
        )
        
        dataset = Dataset(name="ds", collection_name="col")
        input_doc = Document(page_content="Original Doc", metadata={"document_id": "orig1"})
        
        # Act
        processor.process(dataset, [input_doc])
        
        # Assertions
        
        # 1. KV Store has parent
        assert kv_store.get("p1") == "Parent Content"
        
        # 2. VDB Create called with Children
        vdb.create.assert_called_once()
        saved_children = vdb.create.call_args[0][0]
        assert len(saved_children) == 2
        assert saved_children[0].page_content == "Child 1"
        assert saved_children[0].metadata['parent_id'] == "p1"
        assert saved_children[0].metadata['document_id'] == "orig1"
        assert saved_children[0].vector == [0.1]*10

    def test_retrieval_swap_logic(self):
        # Simulation of RAGKernel._process_parent_child_results
        
        # Setup KV with parent
        kv_store = InMemoryKV()
        kv_store.set("p1", "Full Parent Content")
        
        # Search Result (Child Chunk)
        retrieved_child = Document(
            page_content="Partial Child", 
            metadata={"parent_id": "p1", "score": 0.9},
            id="c1"
        )
        results = [retrieved_child]
        
        # Logic to test
        parent_ids = [d.metadata["parent_id"] for d in results if "parent_id" in d.metadata]
        parents_content = kv_store.mget(parent_ids)
        parent_map = dict(zip(parent_ids, parents_content))
        
        for doc in results:
            pid = doc.metadata.get("parent_id")
            if pid and pid in parent_map:
                doc.page_content = parent_map[pid]
                doc.id = pid
                
        # Verify swap
        assert results[0].page_content == "Full Parent Content"
        assert results[0].id == "p1"
