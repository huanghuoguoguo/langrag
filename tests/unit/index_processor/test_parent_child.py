import pytest
from unittest.mock import MagicMock
from langrag.entities.dataset import Dataset
from langrag.entities.document import Document, DocumentType
from langrag.index_processor.processor.parent_child import ParentChildIndexProcessor
from langrag.datasource.kv.in_memory import InMemoryKV

class TestParentChildProcessor:
    
    @pytest.fixture
    def mock_components(self):
        vdb = MagicMock()
        embedder = MagicMock()
        
        # Mock splitters
        parent_splitter = MagicMock()
        child_splitter = MagicMock()
        
        return vdb, embedder, parent_splitter, child_splitter
        
    def test_process_flow(self, mock_components):
        vdb, embedder, parent_splitter, child_splitter = mock_components
        kv = InMemoryKV()
        processor = ParentChildIndexProcessor(
            vector_store=vdb,
            kv_store=kv,
            embedder=embedder,
            parent_splitter=parent_splitter,
            child_splitter=child_splitter
        )
        
        # Setup Documents
        doc = Document(page_content="Full Text", metadata={"document_id": "orig_1"})
        
        # Setup Mock Returns
        # Parent Splitter returns 2 parents
        p1 = Document(page_content="Parent 1", id="p1")
        p2 = Document(page_content="Parent 2", id="p2")
        parent_splitter.split_documents.return_value = [p1, p2]
        
        # Child Splitter returns 2 children for each parent
        c1 = Document(page_content="Child 1-1", id="c1")
        c2 = Document(page_content="Child 1-2", id="c2")
        child_splitter.split_documents.side_effect = [
            [c1, c2], # for p1
            [Document(page_content="Child 2-1", id="c3")] # for p2
        ]
        
        # Embedder
        embedder.embed_documents.return_value = [[0.1]*10, [0.2]*10, [0.3]*10]
        
        # Run
        dataset = Dataset(name="ds", collection_name="col")
        processor.process(dataset, [doc])
        
        # Assertions
        
        # 1. KV Store
        assert kv.get("p1") == "Parent 1"
        assert kv.get("p2") == "Parent 2"
        
        # 2. VDB Create
        vdb.create.assert_called_once()
        saved_chunks = vdb.create.call_args[0][0]
        assert len(saved_chunks) == 3
        
        # Check Metadata
        assert saved_chunks[0].metadata["parent_id"] == "p1"
        assert saved_chunks[0].metadata["document_id"] == "orig_1"
        assert saved_chunks[2].metadata["parent_id"] == "p2"
        
        # Check Vectors
        assert saved_chunks[0].vector == [0.1]*10
