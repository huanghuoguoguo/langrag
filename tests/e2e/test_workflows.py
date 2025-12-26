"""End-to-end tests for LangRAG framework.

端到端测试特点：
- 测试完整的用户流程
- 涵盖多个组件协作
- 真实场景模拟
- 执行时间较长但全面
"""

import pytest
from pathlib import Path

from langrag import RAGEngine, RAGConfig, ComponentConfig
from langrag.config.models import VectorStoreConfig


@pytest.mark.e2e
class TestBasicRAGWorkflow:
    """测试基本RAG工作流程"""

    def test_complete_rag_pipeline(self, tmp_path, sample_documents_content):
        """测试完整的RAG流程：索引 -> 检索 -> 验证"""
        # 1. 准备文档
        doc_file = tmp_path / "document.txt"
        doc_file.write_text(sample_documents_content[0], encoding="utf-8")

        # 2. 创建RAG引擎
        config = RAGConfig(
            parser=ComponentConfig(type="simple_text"),
            chunker=ComponentConfig(type="fixed_size", params={"chunk_size": 100}),
            embedder=ComponentConfig(type="mock", params={"dimension": 384, "seed": 42}),
            vector_store=VectorStoreConfig(type="in_memory"),
            reranker=ComponentConfig(type="noop")
        )
        engine = RAGEngine(config)

        # 3. 索引文档
        num_chunks = engine.index(doc_file)
        assert num_chunks > 0, "应该成功索引文档"

        # 4. 执行检索
        results = engine.retrieve("machine learning")
        assert len(results) > 0, "应该返回检索结果"

        # 5. 验证结果质量
        assert results[0].score >= 0.0, "分数应该有效"
        assert results[0].chunk.content, "chunk应该有内容"
        # 由于分块，结果可能包含部分内容，所以只验证有结果即可

    def test_batch_indexing_workflow(self, sample_document_files):
        """测试批量索引工作流程"""
        # 创建引擎
        config = RAGConfig(
            parser=ComponentConfig(type="simple_text"),
            chunker=ComponentConfig(type="fixed_size", params={"chunk_size": 150}),
            embedder=ComponentConfig(type="mock", params={"dimension": 256, "seed": 42}),
            vector_store=VectorStoreConfig(type="in_memory"),
            reranker=ComponentConfig(type="noop")
        )
        engine = RAGEngine(config)

        # 批量索引
        total_chunks = engine.index_batch(sample_document_files)
        assert total_chunks >= len(sample_document_files), "应该索引所有文档"

        # 验证可以检索到所有主题的内容
        ml_results = engine.retrieve("machine learning")
        python_results = engine.retrieve("Python programming")
        nlp_results = engine.retrieve("natural language processing")

        assert len(ml_results) > 0, "应该能检索到ML相关内容"
        assert len(python_results) > 0, "应该能检索到Python相关内容"
        assert len(nlp_results) > 0, "应该能检索到NLP相关内容"


@pytest.mark.e2e
class TestMultiVectorStoreWorkflow:
    """测试多向量存储工作流程"""

    def test_duckdb_persistence_workflow(self, tmp_path):
        """测试DuckDB持久化工作流程"""
        db_file = tmp_path / "test.duckdb"

        # 第一阶段：索引文档
        config1 = RAGConfig(
            parser=ComponentConfig(type="simple_text"),
            chunker=ComponentConfig(type="fixed_size", params={"chunk_size": 100}),
            embedder=ComponentConfig(type="mock", params={"dimension": 128, "seed": 42}),
            vector_store=VectorStoreConfig(
                type="duckdb",
                params={"database_path": str(db_file), "vector_dimension": 128}
            ),
            reranker=ComponentConfig(type="noop")
        )
        engine1 = RAGEngine(config1)

        # 索引文档
        doc_file = tmp_path / "doc.txt"
        doc_file.write_text("Python is used for data science and machine learning.", encoding="utf-8")
        num_chunks = engine1.index(doc_file)
        assert num_chunks > 0

        # 第二阶段：重新加载并检索
        config2 = RAGConfig(
            parser=ComponentConfig(type="simple_text"),
            chunker=ComponentConfig(type="fixed_size", params={"chunk_size": 100}),
            embedder=ComponentConfig(type="mock", params={"dimension": 128, "seed": 42}),
            vector_store=VectorStoreConfig(
                type="duckdb",
                params={"database_path": str(db_file), "vector_dimension": 128}
            ),
            reranker=ComponentConfig(type="noop")
        )
        engine2 = RAGEngine(config2)

        # 验证能检索到之前索引的内容
        results = engine2.retrieve("Python data science")
        assert len(results) > 0, "应该能检索到持久化的数据"


@pytest.mark.e2e
class TestLargeScaleIndexing:
    """测试大规模索引场景"""

    def test_large_document_indexing(self, large_document_file):
        """测试大文档索引"""
        config = RAGConfig(
            parser=ComponentConfig(type="simple_text"),
            chunker=ComponentConfig(type="fixed_size", params={"chunk_size": 200, "overlap": 50}),
            embedder=ComponentConfig(type="mock", params={"dimension": 256, "seed": 42}),
            vector_store=VectorStoreConfig(type="in_memory"),
            reranker=ComponentConfig(type="noop")
        )
        engine = RAGEngine(config)

        # 索引大文档
        num_chunks = engine.index(large_document_file)

        # 验证
        assert num_chunks >= 10, "大文档应该被分成多个chunks"

        # 测试检索
        results = engine.retrieve("artificial intelligence", top_k=10)
        assert len(results) > 0, "应该能从大文档中检索"
        assert len(results) <= 10, "不应该超过top_k限制"

    def test_multilingual_documents(self, multilingual_document_files):
        """测试多语言文档索引和检索"""
        config = RAGConfig(
            parser=ComponentConfig(type="simple_text"),
            chunker=ComponentConfig(type="fixed_size", params={"chunk_size": 150}),
            embedder=ComponentConfig(type="mock", params={"dimension": 384, "seed": 42}),
            vector_store=VectorStoreConfig(type="in_memory"),
            reranker=ComponentConfig(type="noop")
        )
        engine = RAGEngine(config)

        # 索引所有多语言文档
        total_chunks = engine.index_batch(multilingual_document_files)
        assert total_chunks >= len(multilingual_document_files), "应该索引所有语言的文档"

        # 测试英文检索
        en_results = engine.retrieve("machine learning")
        assert len(en_results) > 0, "应该能检索英文内容"

        # 测试中文检索
        zh_results = engine.retrieve("人工智能")
        assert len(zh_results) > 0, "应该能检索中文内容"


@pytest.mark.e2e
class TestErrorHandling:
    """测试错误处理场景"""

    def test_empty_query_handling(self, minimal_rag_config, sample_document_files):
        """测试空查询处理"""
        engine = RAGEngine(minimal_rag_config)
        engine.index_batch(sample_document_files)

        # 空查询应该抛出错误
        with pytest.raises(ValueError, match="query cannot be empty"):
            engine.retrieve("")

    def test_nonexistent_file_handling(self, minimal_rag_config):
        """测试不存在的文件处理"""
        engine = RAGEngine(minimal_rag_config)

        # 不存在的文件应该抛出错误
        with pytest.raises(FileNotFoundError):
            engine.index("/nonexistent/path/file.txt")

    def test_special_characters_handling(self, minimal_rag_config, special_chars_document_file):
        """测试特殊字符处理"""
        engine = RAGEngine(minimal_rag_config)

        # 应该能够索引包含特殊字符的文档
        num_chunks = engine.index(special_chars_document_file)
        assert num_chunks > 0, "应该能索引特殊字符文档"

        # 应该能够检索
        results = engine.retrieve("Special Characters")
        assert len(results) > 0, "应该能检索到特殊字符内容"


@pytest.mark.e2e
class TestRetrievalQuality:
    """测试检索质量"""

    def test_relevance_ranking(self, tmp_path):
        """测试相关性排序"""
        # 创建有明确主题的文档
        docs = [
            ("python.txt", "Python is a programming language used for web development."),
            ("java.txt", "Java is a programming language used for enterprise applications."),
            ("ml.txt", "Machine learning is a field of artificial intelligence.")
        ]

        for filename, content in docs:
            (tmp_path / filename).write_text(content, encoding="utf-8")

        # 索引
        config = RAGConfig(
            parser=ComponentConfig(type="simple_text"),
            chunker=ComponentConfig(type="fixed_size", params={"chunk_size": 200}),
            embedder=ComponentConfig(type="mock", params={"dimension": 128, "seed": 42}),
            vector_store=VectorStoreConfig(type="in_memory"),
            reranker=ComponentConfig(type="noop")
        )
        engine = RAGEngine(config)

        for filename, _ in docs:
            engine.index(tmp_path / filename)

        # 检索Python相关内容
        results = engine.retrieve("Python programming language")

        # 验证分数降序排列
        scores = [r.score for r in results]
        assert scores == sorted(scores, reverse=True), "结果应该按分数降序排列"

    def test_top_k_limiting(self, sample_document_files, minimal_rag_config):
        """测试top_k限制"""
        engine = RAGEngine(minimal_rag_config)
        engine.index_batch(sample_document_files)

        # 测试不同的top_k值
        for k in [1, 3, 5]:
            results = engine.retrieve("machine learning", top_k=k)
            assert len(results) <= k, f"结果数量不应该超过top_k={k}"
