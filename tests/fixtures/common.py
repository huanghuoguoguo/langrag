"""Shared test fixtures for all test types.

这个模块提供可在所有测试中重复使用的fixtures。
"""

import pytest
import tempfile
from pathlib import Path
from typing import List

from langrag import (
    Document,
    Chunk,
    SearchResult,
    RAGConfig,
    ComponentConfig
)
from langrag.config.models import VectorStoreConfig


@pytest.fixture
def sample_documents_content() -> List[str]:
    """提供示例文档内容列表"""
    return [
        "Machine learning is a subset of artificial intelligence. "
        "It focuses on teaching computers to learn from data. "
        "Common algorithms include neural networks, decision trees, and support vector machines.",

        "Python is a high-level programming language. "
        "It is widely used in data science, web development, and automation. "
        "Python has a simple syntax that makes it easy to learn.",

        "Natural language processing (NLP) is a field of AI. "
        "It deals with the interaction between computers and human language. "
        "Applications include chatbots, translation, and sentiment analysis."
    ]


@pytest.fixture
def sample_document_files(tmp_path, sample_documents_content) -> List[Path]:
    """创建包含示例内容的临时文件"""
    files = []
    for i, content in enumerate(sample_documents_content):
        file_path = tmp_path / f"doc_{i}.txt"
        file_path.write_text(content, encoding="utf-8")
        files.append(file_path)
    return files


@pytest.fixture
def sample_chunks() -> List[Chunk]:
    """提供示例Chunk列表用于测试"""
    return [
        Chunk(
            id="chunk_1",
            content="Python is a programming language",
            embedding=[1.0, 0.0, 0.0],
            source_doc_id="doc_1",
            metadata={"topic": "python", "page": 1}
        ),
        Chunk(
            id="chunk_2",
            content="Machine learning uses algorithms",
            embedding=[0.0, 1.0, 0.0],
            source_doc_id="doc_2",
            metadata={"topic": "ml", "page": 1}
        ),
        Chunk(
            id="chunk_3",
            content="Natural language processing is AI",
            embedding=[0.0, 0.0, 1.0],
            source_doc_id="doc_3",
            metadata={"topic": "nlp", "page": 1}
        ),
    ]


@pytest.fixture
def sample_search_results(sample_chunks) -> List[SearchResult]:
    """提供示例SearchResult列表"""
    return [
        SearchResult(chunk=sample_chunks[0], score=0.95),
        SearchResult(chunk=sample_chunks[1], score=0.85),
        SearchResult(chunk=sample_chunks[2], score=0.75),
    ]


@pytest.fixture
def minimal_rag_config() -> RAGConfig:
    """提供最小RAG配置用于测试"""
    return RAGConfig(
        parser=ComponentConfig(type="simple_text"),
        chunker=ComponentConfig(type="fixed_size", params={"chunk_size": 200, "overlap": 50}),
        embedder=ComponentConfig(type="mock", params={"dimension": 384, "seed": 42}),
        vector_store=VectorStoreConfig(type="in_memory"),
        reranker=ComponentConfig(type="noop")
    )


@pytest.fixture
def duckdb_rag_config(tmp_path) -> RAGConfig:
    """提供使用DuckDB的RAG配置"""
    db_path = tmp_path / "test.duckdb"
    return RAGConfig(
        parser=ComponentConfig(type="simple_text"),
        chunker=ComponentConfig(type="fixed_size", params={"chunk_size": 200}),
        embedder=ComponentConfig(type="mock", params={"dimension": 128, "seed": 42}),
        vector_store=VectorStoreConfig(
            type="duckdb",
            params={"database_path": str(db_path), "vector_dimension": 128}
        ),
        reranker=ComponentConfig(type="noop")
    )


@pytest.fixture
def large_document_file(tmp_path) -> Path:
    """创建一个大文档用于测试分块"""
    file_path = tmp_path / "large_doc.txt"

    # 创建大约5000字的文档
    paragraphs = [
        "Artificial intelligence has revolutionized many industries.",
        "Machine learning algorithms can learn patterns from data.",
        "Deep learning uses neural networks with multiple layers.",
        "Natural language processing enables computers to understand text.",
        "Computer vision allows machines to interpret visual information.",
    ]

    # 重复这些段落以创建大文档
    content = "\n\n".join(paragraphs * 100)
    file_path.write_text(content, encoding="utf-8")

    return file_path


@pytest.fixture
def multilingual_document_files(tmp_path) -> List[Path]:
    """创建多语言文档用于测试"""
    files = []

    # 英文文档
    en_file = tmp_path / "english.txt"
    en_file.write_text(
        "This is an English document about artificial intelligence and machine learning.",
        encoding="utf-8"
    )
    files.append(en_file)

    # 中文文档
    zh_file = tmp_path / "chinese.txt"
    zh_file.write_text(
        "这是一个关于人工智能和机器学习的中文文档。深度学习是人工智能的重要分支。",
        encoding="utf-8"
    )
    files.append(zh_file)

    # 混合文档
    mixed_file = tmp_path / "mixed.txt"
    mixed_file.write_text(
        "Machine Learning (机器学习) is a subset of AI (人工智能).",
        encoding="utf-8"
    )
    files.append(mixed_file)

    return files


@pytest.fixture
def empty_document_file(tmp_path) -> Path:
    """创建空文档用于测试边界情况"""
    file_path = tmp_path / "empty.txt"
    file_path.write_text("", encoding="utf-8")
    return file_path


@pytest.fixture
def special_chars_document_file(tmp_path) -> Path:
    """创建包含特殊字符的文档"""
    file_path = tmp_path / "special_chars.txt"
    content = """
    Special Characters Test:
    - Email: test@example.com
    - URL: https://example.com/path?param=value
    - Code: def func(x): return x**2
    - Math: ∑ ∫ ∂ ∇ ∞
    - Emoji: 😀 🚀 💻
    - Unicode: café résumé naïve
    """
    file_path.write_text(content, encoding="utf-8")
    return file_path


@pytest.fixture(scope="session")
def temp_workspace():
    """提供会话级别的临时工作空间"""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)
