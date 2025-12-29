import pytest
from unittest.mock import MagicMock, patch
from langrag.entities.dataset import Dataset
from langrag.entities.document import Document
from langrag.datasource.vdb.chroma import ChromaVector

class TestChromaVector:
    
    @pytest.fixture
    def mock_chroma_client(self):
        with patch("langrag.datasource.vdb.chroma.chromadb.PersistentClient") as mock:
            client = MagicMock()
            mock.return_value = client
            yield client

    @pytest.fixture
    def dataset(self):
        return Dataset(name="test", collection_name="test_col")

    def test_init(self, mock_chroma_client, dataset):
        chroma = ChromaVector(dataset, persist_directory="/tmp/test")
        
        mock_chroma_client.get_or_create_collection.assert_called_with(
            name="test_col",
            metadata={"hnsw:space": "cosine"}
        )
        assert chroma._client == mock_chroma_client

    def test_add_texts(self, mock_chroma_client, dataset):
        mock_collection = MagicMock()
        mock_chroma_client.get_or_create_collection.return_value = mock_collection
        
        chroma = ChromaVector(dataset)
        
        docs = [
            Document(page_content="hello", id="1", vector=[0.1, 0.2], metadata={"a": 1}),
            Document(page_content="world", id="2", vector=[0.3, 0.4], metadata={"b": "str"})
        ]
        
        chroma.add_texts(docs)
        
        mock_collection.add.assert_called_once()
        call_args = mock_collection.add.call_args[1]
        assert call_args["ids"] == ["1", "2"]
        assert call_args["documents"] == ["hello", "world"]
        assert call_args["embeddings"] == [[0.1, 0.2], [0.3, 0.4]]
        # Metadata conversion verification (int preserved? Chroma supports int/float/str/bool)
        # Our code converts non-primitives, but int is primitive for Chroma.
        # But wait, looking at my implementation:
        # if isinstance(v, (str, int, float, bool)): keep
        # else: str(v)
        # So int should be kept as int.
        assert call_args["metadatas"][0]["a"] == 1 

    def test_search_vector(self, mock_chroma_client, dataset):
        mock_collection = MagicMock()
        mock_chroma_client.get_or_create_collection.return_value = mock_collection
        
        # Mock result
        # parsed: ids=[[]], documents=[[]], metadatas=[[]], distances=[[]]
        mock_collection.query.return_value = {
            "ids": [["1"]],
            "documents": [["doc content"]],
            "metadatas": [[{"a": 1}]],
            "distances": [[0.1]]
        }
        
        chroma = ChromaVector(dataset)
        results = chroma.search("query", [0.1, 0.2], top_k=1)
        
        mock_collection.query.assert_called_once_with(
            query_embeddings=[[0.1, 0.2]],
            n_results=1,
            include=["documents", "metadatas", "distances"]
        )
        
        assert len(results) == 1
        assert results[0].id == "1"
        # Score = 1 - distance = 1 - 0.1 = 0.9
        assert results[0].metadata["score"] == 0.9 

    def test_delete_by_ids(self, mock_chroma_client, dataset):
        mock_collection = MagicMock()
        mock_chroma_client.get_or_create_collection.return_value = mock_collection
        chroma = ChromaVector(dataset)
        
        chroma.delete_by_ids(["1", "2"])
        mock_collection.delete.assert_called_once_with(ids=["1", "2"])

    def test_delete_collection(self, mock_chroma_client, dataset):
        chroma = ChromaVector(dataset)
        chroma.delete()
        mock_chroma_client.delete_collection.assert_called_once_with("test_col")
