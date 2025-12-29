import pytest
from langrag.entities.dataset import Dataset, RetrievalContext

class TestDataset:
    def test_dataset_initialization(self):
        dataset = Dataset(name="Test Dataset", collection_name="test_collection")
        assert dataset.name == "Test Dataset"
        assert dataset.collection_name == "test_collection"
        assert dataset.id is not None
        assert dataset.indexing_technique == "high_quality" # Default

    def test_dataset_optional_fields(self):
        dataset = Dataset(
            name="Test", 
            collection_name="col",
            description="Desc",
            indexing_technique="economy",
            tenant_id="tenant1"
        )
        assert dataset.description == "Desc"
        assert dataset.indexing_technique == "economy"
        assert dataset.tenant_id == "tenant1"

class TestRetrievalContext:
    def test_retrieval_context_init(self):
        ctx = RetrievalContext(
            document_id="doc1",
            content="content",
            score=0.9,
            metadata={"source": "file"}
        )
        assert ctx.score == 0.9
        assert ctx.metadata["source"] == "file"
