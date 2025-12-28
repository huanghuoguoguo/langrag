"""Shared test fixtures for all test types."""

from pathlib import Path

import pytest

from langrag import SearchResult, Document, DocumentType
from langrag.config.models import VectorStoreConfig, ComponentConfig, RAGConfig


@pytest.fixture
def sample_documents_content() -> list[str]:
    """Provide sample document contents."""
    return [
        "Machine learning is a subset of artificial intelligence. "
        "It focuses on teaching computers to learn from data. "
        "Common algorithms include neural networks, decision trees, and support vector machines.",
        "Python is a high-level programming language. "
        "It is widely used in data science, web development, and automation. "
        "Python has a simple syntax that makes it easy to learn.",
        "Natural language processing (NLP) is a field of AI. "
        "It deals with the interaction between computers and human language. "
        "Applications include chatbots, translation, and sentiment analysis.",
    ]


@pytest.fixture
def sample_document_files(tmp_path, sample_documents_content) -> list[Path]:
    """Create temporary files with sample content."""
    files = []
    for i, content in enumerate(sample_documents_content):
        file_path = tmp_path / f"doc_{i}.txt"
        file_path.write_text(content, encoding="utf-8")
        files.append(file_path)
    return files


@pytest.fixture
def sample_chunks() -> list[Document]:
    """Provide sample 'chunks' (Documents of type CHUNK) for testing."""
    return [
        Document(
            id="chunk_1",
            page_content="Python is a programming language",
            vector=[1.0, 0.0, 0.0],
            metadata={"topic": "python", "page": 1, "source_doc_id": "doc_1"},
            type=DocumentType.CHUNK
        ),
        Document(
            id="chunk_2",
            page_content="Machine learning uses algorithms",
            vector=[0.0, 1.0, 0.0],
            metadata={"topic": "ml", "page": 1, "source_doc_id": "doc_2"},
            type=DocumentType.CHUNK
        ),
        Document(
            id="chunk_3",
            page_content="Natural language processing is AI",
            vector=[0.0, 0.0, 1.0],
            metadata={"topic": "nlp", "page": 1, "source_doc_id": "doc_3"},
            type=DocumentType.CHUNK
        ),
    ]


@pytest.fixture
def sample_search_results(sample_chunks) -> list[SearchResult]:
    """Provide sample SearchResults."""
    # Note: SearchResult expects 'chunk' field which is now a Document
    return [
        SearchResult(chunk=sample_chunks[0], score=0.95),
        SearchResult(chunk=sample_chunks[1], score=0.85),
        SearchResult(chunk=sample_chunks[2], score=0.75),
    ]


@pytest.fixture
def minimal_rag_config() -> RAGConfig:
    """Provide minimal RAG config."""
    return RAGConfig(
        parser=ComponentConfig(type="simple_text"),
        chunker=ComponentConfig(type="fixed_size", params={"chunk_size": 200, "overlap": 50}),
        embedder=ComponentConfig(type="mock", params={"dimension": 384, "seed": 42}),
        vector_store=VectorStoreConfig(type="in_memory"),
        reranker=ComponentConfig(type="noop"),
    )


@pytest.fixture
def large_document_file(tmp_path) -> Path:
    """Create a large document."""
    file_path = tmp_path / "large_doc.txt"
    paragraphs = [
        "Artificial intelligence has revolutionized many industries.",
        "Machine learning algorithms can learn patterns from data.",
        "Deep learning uses neural networks with multiple layers.",
        "Natural language processing enables computers to understand text.",
        "Computer vision allows machines to interpret visual information.",
    ]
    content = "\n\n".join(paragraphs * 100)
    file_path.write_text(content, encoding="utf-8")
    return file_path


@pytest.fixture
def multilingual_document_files(tmp_path) -> list[Path]:
    """Create multilingual documents."""
    files = []
    
    en_file = tmp_path / "english.txt"
    en_file.write_text("This is an English document about artificial intelligence.", encoding="utf-8")
    files.append(en_file)

    zh_file = tmp_path / "chinese.txt"
    zh_file.write_text("这是一个关于人工智能的中文文档。", encoding="utf-8")
    files.append(zh_file)

    mixed_file = tmp_path / "mixed.txt"
    mixed_file.write_text("Machine Learning (机器学习) is a subset of AI.", encoding="utf-8")
    files.append(mixed_file)

    return files


@pytest.fixture
def empty_document_file(tmp_path) -> Path:
    """Create empty document."""
    file_path = tmp_path / "empty.txt"
    file_path.write_text("", encoding="utf-8")
    return file_path


@pytest.fixture
def special_chars_document_file(tmp_path) -> Path:
    """Create document with special characters."""
    file_path = tmp_path / "special_chars.txt"
    content = """
    Special Characters Test:
    - Emojis: 🎉 🚀 💻
    - Unicode: café résumé
    """
    file_path.write_text(content, encoding="utf-8")
    return file_path

@pytest.fixture(scope="session")
def temp_workspace():
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)
