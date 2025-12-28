import pytest
from pathlib import Path
from langrag.entities.document import Document
from langrag.entities.dataset import Dataset
from langrag.index_processor.extractor import SimpleTextParser
from langrag.index_processor.splitter import RecursiveCharacterChunker
from tests.utils.in_memory_vector_store import InMemoryVectorStore


@pytest.mark.smoke
class TestCoreSmoke:
    def test_components_instantiation(self):
        """Verify we can instantiate all core components."""
        parser = SimpleTextParser()
        chunker = RecursiveCharacterChunker(chunk_size=100, chunk_overlap=0)
        dataset = Dataset(name="test", collection_name="smoke_test")
        store = InMemoryVectorStore(dataset)
        
        assert parser is not None
        assert chunker is not None
        assert dataset is not None
        assert store is not None

    def test_basic_flow_smoke(self, temp_file):
        """Smoke test for the critical path."""
        parser = SimpleTextParser()
        chunker = RecursiveCharacterChunker(chunk_size=50, chunk_overlap=0)
        dataset = Dataset(name="flow", collection_name="smoke_flow")
        store = InMemoryVectorStore(dataset)
        
        # Action
        docs = parser.parse(temp_file)
        chunks = chunker.split(docs)
        store.add_texts(chunks)
        results = store.search("Sample", query_vector=None, top_k=1)
        
        assert len(results) > 0
