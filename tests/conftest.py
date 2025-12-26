"""Pytest configuration and global fixtures for LangRAG tests."""

import tempfile
from pathlib import Path
from typing import List
import pytest

from langrag.core.document import Document
from langrag.core.chunk import Chunk
from langrag import RAGConfig, ComponentConfig
from langrag.config.models import VectorStoreConfig


# Import shared fixtures from fixtures module
from tests.fixtures.common import (
    sample_documents_content,
    sample_document_files,
    sample_chunks,
    sample_search_results,
    minimal_rag_config,
    duckdb_rag_config,
    large_document_file,
    multilingual_document_files,
    empty_document_file,
    special_chars_document_file,
    temp_workspace
)


# ==================== 临时目录夹具 ====================

@pytest.fixture
def temp_dir():
    """提供临时目录，测试结束后自动清理

    Returns:
        Path: 临时目录路径
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def temp_file(temp_dir):
    """提供临时文件，测试结束后自动清理

    Returns:
        Path: 临时文件路径
    """
    file_path = temp_dir / "test_file.txt"
    file_path.write_text("Sample content for testing")
    return file_path


# ==================== 示例数据夹具 ====================

@pytest.fixture
def sample_text():
    """提供示例文本数据

    Returns:
        str: 示例文本
    """
    return """
    Retrieval-Augmented Generation (RAG) is a technique that combines
    information retrieval with large language models. RAG retrieves
    relevant documents from a knowledge base and uses them to generate
    more accurate and contextual responses.

    Vector databases are essential for RAG systems as they enable
    efficient semantic search over large document collections.
    """


@pytest.fixture
def sample_documents() -> List[Document]:
    """提供示例文档列表

    Returns:
        List[Document]: 文档列表
    """
    return [
        Document(
            content="RAG combines retrieval and generation for better AI responses.",
            metadata={"source": "doc1.txt", "author": "Alice"}
        ),
        Document(
            content="Vector databases enable semantic search in RAG systems.",
            metadata={"source": "doc2.txt", "author": "Bob"}
        ),
        Document(
            content="Embedding models convert text into numerical vectors.",
            metadata={"source": "doc3.txt", "author": "Charlie"}
        ),
    ]


@pytest.fixture
def sample_chunks() -> List[Chunk]:
    """提供示例 chunk 列表（带 embedding）

    Returns:
        List[Chunk]: Chunk 列表
    """
    return [
        Chunk(
            id="chunk-1",
            content="RAG combines retrieval and generation.",
            embedding=[0.1] * 384,
            source_doc_id="doc1.txt",
            metadata={"position": 0}
        ),
        Chunk(
            id="chunk-2",
            content="Vector databases enable semantic search.",
            embedding=[0.2] * 384,
            source_doc_id="doc2.txt",
            metadata={"position": 0}
        ),
        Chunk(
            id="chunk-3",
            content="Embedding models convert text to vectors.",
            embedding=[0.3] * 384,
            source_doc_id="doc3.txt",
            metadata={"position": 0}
        ),
    ]


# ==================== 组件夹具 ====================

@pytest.fixture
def simple_embedder():
    """提供简单的嵌入器（用于测试）

    Returns:
        BaseEmbedder: 简单嵌入器实例
    """
    from langrag.embedder import SimpleEmbedder
    return SimpleEmbedder(dimension=384)


@pytest.fixture
def recursive_chunker():
    """提供递归分块器

    Returns:
        RecursiveCharacterChunker: 分块器实例
    """
    from langrag.chunker.providers.recursive_character import RecursiveCharacterChunker
    return RecursiveCharacterChunker(chunk_size=500, chunk_overlap=50)


@pytest.fixture
def simple_text_parser():
    """提供简单文本解析器

    Returns:
        SimpleTextParser: 解析器实例
    """
    from langrag.parser import SimpleTextParser
    return SimpleTextParser()


@pytest.fixture
def in_memory_vector_store():
    """提供内存向量存储

    Returns:
        InMemoryVectorStore: 向量存储实例
    """
    from langrag.vector_store import InMemoryVectorStore
    return InMemoryVectorStore()


# ==================== Mock 夹具 ====================

@pytest.fixture
def mock_embedder(mocker):
    """提供 Mock 嵌入器

    Returns:
        Mock: Mock 嵌入器对象
    """
    from langrag.embedder import BaseEmbedder
    mock = mocker.Mock(spec=BaseEmbedder)
    mock.embed.return_value = [[0.1] * 384]
    mock.dimension = 384
    return mock


@pytest.fixture
def mock_vector_store(mocker):
    """提供 Mock 向量存储

    Returns:
        Mock: Mock 向量存储对象
    """
    from langrag.vector_store import BaseVectorStore
    from langrag.vector_store.capabilities import VectorStoreCapabilities

    mock = mocker.Mock(spec=BaseVectorStore)
    mock.capabilities = VectorStoreCapabilities(
        supports_vector=True,
        supports_fulltext=False,
        supports_hybrid=False
    )
    mock.count.return_value = 0
    return mock


# ==================== RAG 引擎夹具 ====================

@pytest.fixture
def minimal_rag_config():
    """提供最小化的 RAG 配置

    Returns:
        RAGConfig: RAG 配置对象
    """
    from langrag.config.models import RAGConfig, ComponentConfig

    return RAGConfig(
        parser=ComponentConfig(type="simple_text", params={}),
        chunker=ComponentConfig(type="recursive", params={"chunk_size": 500}),
        embedder=ComponentConfig(type="simple", params={"dimension": 384}),
        vector_store=ComponentConfig(type="in_memory", params={}),
    )


@pytest.fixture
def rag_engine(minimal_rag_config):
    """提供 RAG 引擎实例

    Returns:
        RAGEngine: RAG 引擎实例
    """
    from langrag.engine import RAGEngine
    return RAGEngine(minimal_rag_config)


# ==================== 测试标记处理 ====================

def pytest_configure(config):
    """配置 pytest，添加自定义标记"""
    config.addinivalue_line(
        "markers", "unit: Unit tests - 快速、隔离的单元测试"
    )
    config.addinivalue_line(
        "markers", "integration: Integration tests - 组件间协作测试"
    )
    config.addinivalue_line(
        "markers", "e2e: End-to-end tests - 完整业务流程测试"
    )
    config.addinivalue_line(
        "markers", "smoke: Smoke tests - 快速验证核心功能"
    )
    config.addinivalue_line(
        "markers", "slow: Slow tests - 执行时间 > 1s 的测试"
    )


def pytest_collection_modifyitems(config, items):
    """根据文件路径自动添加标记"""
    for item in items:
        # 获取测试文件的相对路径
        rel_path = Path(item.fspath).relative_to(Path(__file__).parent)

        # 根据目录自动添加标记
        if "unit" in rel_path.parts:
            item.add_marker(pytest.mark.unit)
        elif "integration" in rel_path.parts:
            item.add_marker(pytest.mark.integration)
        elif "e2e" in rel_path.parts:
            item.add_marker(pytest.mark.e2e)
        elif "smoke" in rel_path.parts:
            item.add_marker(pytest.mark.smoke)


# ==================== 测试会话钩子 ====================

def pytest_sessionstart(session):
    """测试会话开始时的钩子"""
    print("\n" + "="*70)
    print("🧪 Starting LangRAG Test Suite")
    print("="*70)


def pytest_sessionfinish(session, exitstatus):
    """测试会话结束时的钩子"""
    print("\n" + "="*70)
    if exitstatus == 0:
        print("✅ All tests passed!")
    else:
        print("❌ Some tests failed")
    print("="*70)
