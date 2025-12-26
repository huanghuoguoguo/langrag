"""Smoke tests for LangRAG framework.

冒烟测试特点：
- 快速执行（< 5分钟）
- 验证核心功能可用
- 测试关键路径
- 适合 CI/CD 快速失败
"""

import pytest
import tempfile
from pathlib import Path

from langrag import RAGEngine, RAGConfig, ComponentConfig
from langrag.config.models import VectorStoreConfig


@pytest.mark.smoke
class TestCoreSmoke:
    """核心功能冒烟测试"""

    def test_rag_engine_can_initialize(self):
        """RAG引擎能够成功初始化"""
        config = RAGConfig(
            parser=ComponentConfig(type="simple_text"),
            chunker=ComponentConfig(type="fixed_size", params={"chunk_size": 200}),
            embedder=ComponentConfig(type="mock", params={"dimension": 384}),
            vector_store=VectorStoreConfig(type="in_memory"),
            reranker=ComponentConfig(type="noop")
        )

        engine = RAGEngine(config)

        assert engine is not None
        assert engine.parser is not None
        assert engine.chunker is not None
        assert engine.embedder is not None
        assert len(engine.vector_stores) > 0

    def test_can_index_and_retrieve(self):
        """能够索引文档并检索结果"""
        # 创建测试文件
        with tempfile.TemporaryDirectory() as tmpdir:
            test_file = Path(tmpdir) / "test.txt"
            test_file.write_text("Python is a programming language. " * 10, encoding="utf-8")

            # 创建引擎
            config = RAGConfig(
                parser=ComponentConfig(type="simple_text"),
                chunker=ComponentConfig(type="fixed_size", params={"chunk_size": 100}),
                embedder=ComponentConfig(type="mock", params={"dimension": 128, "seed": 42}),
                vector_store=VectorStoreConfig(type="in_memory"),
                reranker=ComponentConfig(type="noop")
            )
            engine = RAGEngine(config)

            # 索引
            num_chunks = engine.index(test_file)
            assert num_chunks > 0

            # 检索
            results = engine.retrieve("Python programming")
            assert len(results) > 0
            assert results[0].chunk is not None
            assert 0 <= results[0].score <= 1.0

    def test_vector_stores_available(self):
        """核心向量存储可用"""
        from langrag import VectorStoreFactory

        # InMemoryVectorStore 应该始终可用
        types = VectorStoreFactory.list_types()
        assert "in_memory" in types

        # 能够创建 InMemoryVectorStore
        store = VectorStoreFactory.create("in_memory")
        assert store is not None


@pytest.mark.smoke
class TestComponentFactories:
    """组件工厂冒烟测试"""

    def test_parser_factory_works(self):
        """Parser工厂能够创建组件"""
        from langrag import ParserFactory

        parser = ParserFactory.create("simple_text")
        assert parser is not None

    def test_chunker_factory_works(self):
        """Chunker工厂能够创建组件"""
        from langrag import ChunkerFactory

        chunker = ChunkerFactory.create("fixed_size", chunk_size=100)
        assert chunker is not None

    def test_embedder_factory_works(self):
        """Embedder工厂能够创建组件"""
        from langrag import EmbedderFactory

        embedder = EmbedderFactory.create("mock", dimension=128)
        assert embedder is not None

    def test_vector_store_factory_works(self):
        """VectorStore工厂能够创建组件"""
        from langrag import VectorStoreFactory

        store = VectorStoreFactory.create("in_memory")
        assert store is not None


def _chroma_available():
    """检查ChromaDB是否可用"""
    try:
        import chromadb
        return True
    except ImportError:
        return False


@pytest.mark.smoke
class TestVectorStoreSmoke:
    """向量存储冒烟测试"""

    def test_in_memory_vector_store_basic_flow(self):
        """InMemoryVectorStore基本流程"""
        from langrag import InMemoryVectorStore, Chunk

        store = InMemoryVectorStore()

        # 添加chunk
        chunk = Chunk(
            content="Test content",
            embedding=[0.1, 0.2, 0.3],
            source_doc_id="doc1",
            metadata={}
        )
        store.add([chunk])

        # 搜索
        results = store.search([0.1, 0.2, 0.3], top_k=1)
        assert len(results) == 1
        assert results[0].chunk.content == "Test content"

    def test_duckdb_vector_store_available(self):
        """DuckDB向量存储可用"""
        from langrag import VectorStoreFactory

        types = VectorStoreFactory.list_types()
        assert "duckdb" in types

        # 能够创建DuckDB store
        store = VectorStoreFactory.create(
            "duckdb",
            database_path=":memory:",
            vector_dimension=128
        )
        assert store is not None

    @pytest.mark.skipif(
        not _chroma_available(),
        reason="ChromaDB not available"
    )
    def test_chroma_vector_store_available(self):
        """ChromaDB向量存储可用（如果已安装）"""
        from langrag import VectorStoreFactory

        types = VectorStoreFactory.list_types()
        assert "chroma" in types


@pytest.mark.smoke
class TestIndexingSmoke:
    """索引流程冒烟测试"""

    def test_indexing_pipeline_works(self):
        """索引流程正常工作"""
        from langrag.indexing.pipeline import IndexingPipeline
        from langrag import ParserFactory, ChunkerFactory, EmbedderFactory, VectorStoreFactory

        with tempfile.TemporaryDirectory() as tmpdir:
            test_file = Path(tmpdir) / "test.txt"
            test_file.write_text("Test document content", encoding="utf-8")

            # 创建组件
            parser = ParserFactory.create("simple_text")
            chunker = ChunkerFactory.create("fixed_size", chunk_size=100)
            embedder = EmbedderFactory.create("mock", dimension=128)
            vector_store = VectorStoreFactory.create("in_memory")

            # 创建索引流程
            from langrag.config.models import StorageRole
            pipeline = IndexingPipeline(
                parser=parser,
                chunker=chunker,
                embedder=embedder,
                vector_stores=[(vector_store, StorageRole.PRIMARY)]
            )

            # 索引文件
            num_chunks = pipeline.index_file(test_file)
            assert num_chunks > 0


@pytest.mark.smoke
class TestRetrievalSmoke:
    """检索流程冒烟测试"""

    def test_retrieval_works(self):
        """检索流程正常工作"""
        from langrag import InMemoryVectorStore, Chunk, EmbedderFactory
        from langrag.retrieval.retriever import Retriever
        from langrag.config.models import StorageRole

        # 准备数据
        store = InMemoryVectorStore()
        chunks = [
            Chunk(
                id="c1",
                content="Python programming",
                embedding=[1.0, 0.0, 0.0],
                source_doc_id="doc1",
                metadata={}
            ),
            Chunk(
                id="c2",
                content="Java development",
                embedding=[0.0, 1.0, 0.0],
                source_doc_id="doc1",
                metadata={}
            ),
        ]
        store.add(chunks)

        # 创建检索器
        embedder = EmbedderFactory.create("mock", dimension=3, seed=42)
        retriever = Retriever.from_single_store(
            embedder=embedder,
            vector_store=store,
            storage_role=StorageRole.PRIMARY
        )

        # 执行检索
        import asyncio
        results = asyncio.run(retriever.retrieve("Python", top_k=2))

        assert len(results) > 0
        assert all(r.chunk is not None for r in results)
