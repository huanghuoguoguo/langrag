"""Pytest configuration and global fixtures for LangRAG tests."""

import tempfile
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from langrag.entities.document import Document, DocumentType
from langrag.entities.dataset import Dataset
from langrag.llm.embedder.base import BaseEmbedder
from langrag.index_processor.extractor import SimpleTextParser
from langrag.index_processor.splitter import RecursiveCharacterChunker
from tests.utils.in_memory_vector_store import InMemoryVectorStore

# Import shared fixtures
from tests.fixtures.common import (  # noqa: F401
    sample_document_files,
    sample_documents_content,
    sample_search_results,
    sample_chunks,
    minimal_rag_config,
    large_document_file,
    multilingual_document_files,
    special_chars_document_file,
    empty_document_file,
    temp_workspace
)

@pytest.fixture
def temp_dir():
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)

@pytest.fixture
def temp_file(temp_dir):
    file_path = temp_dir / "test_file.txt"
    file_path.write_text("Sample content for testing")
    return file_path

@pytest.fixture
def sample_text():
    return """
    Retrieval-Augmented Generation (RAG) is a technique that combines
    information retrieval with large language models.
    """

@pytest.fixture
def sample_documents() -> list[Document]:
    return [
        Document(
            page_content="RAG combines retrieval and generation.",
            metadata={"source": "doc1.txt", "author": "Alice"},
        ),
        Document(
            page_content="Vector databases enable semantic search.",
            metadata={"source": "doc2.txt", "author": "Bob"},
        ),
    ]

# ==================== Component Fixtures ====================

@pytest.fixture
def mock_embedder():
    """Provide a mock embedder that matches BaseEmbedder interface."""
    class MockEmbedder(BaseEmbedder):
        def __init__(self, dimension=384):
            super().__init__()
            self._dimension = dimension

        def embed(self, texts: list[str]) -> list[list[float]]:
            # Return same vector for any input for simplicity, or random
            return [[0.1] * self._dimension for _ in texts]

        @property
        def dimension(self) -> int:
            return self._dimension

    return MockEmbedder()

@pytest.fixture
def recursive_chunker():
    return RecursiveCharacterChunker(chunk_size=500, chunk_overlap=50)

@pytest.fixture
def simple_text_parser():
    return SimpleTextParser()

@pytest.fixture
def in_memory_vector_store():
    dataset = Dataset(name="test_ds", collection_name="test_collection")
    return InMemoryVectorStore(dataset=dataset)

@pytest.fixture
def mock_vector_store(mocker):
    """Mock the BaseVector interface."""
    from langrag.datasource.vdb.base import BaseVector
    mock = mocker.Mock(spec=BaseVector)
    # Mocking capabilities property to simulate a standard vector store
    # Note: capabilities might not be directly on BaseVector anymore in new arch, 
    # but let's assume standard behavior.
    return mock

# ==================== Pytest Configuration ====================

def pytest_configure(config):
    config.addinivalue_line("markers", "unit: Unit tests")
    config.addinivalue_line("markers", "integration: Integration tests")
    config.addinivalue_line("markers", "e2e: End-to-end tests")
    config.addinivalue_line("markers", "smoke: Smoke tests")

def pytest_collection_modifyitems(config, items):
    for item in items:
        rel_path = Path(item.fspath).relative_to(Path(__file__).parent)
        if "unit" in rel_path.parts:
            item.add_marker(pytest.mark.unit)
        elif "integration" in rel_path.parts:
            item.add_marker(pytest.mark.integration)
        elif "e2e" in rel_path.parts:
            item.add_marker(pytest.mark.e2e)
        elif "smoke" in rel_path.parts:
            item.add_marker(pytest.mark.smoke)
