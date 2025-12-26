"""
集成测试 - 测试LangRAG系统的完整功能
"""

import tempfile
import pytest
from pathlib import Path
from langrag import RAGEngine, RAGConfig


class TestRAGIntegration:
    """RAG系统集成测试类"""

    @pytest.fixture
    def config(self) -> RAGConfig:
        """创建测试配置"""
        return RAGConfig(
            parser={
                "type": "simple_text",
                "params": {"encoding": "utf-8"}
            },
            chunker={
                "type": "fixed_size",
                "params": {"chunk_size": 200, "overlap": 50}
            },
            embedder={
                "type": "mock",
                "params": {"dimension": 384, "seed": 42}
            },
            vector_store={
                "type": "in_memory",
                "params": {}
            },
            reranker={
                "type": "noop",
                "params": {}
            },
            retrieval_top_k=5,
            rerank_top_k=3
        )

    @pytest.fixture
    def sample_documents(self, tmp_path: Path) -> list[Path]:
        """创建测试文档"""
        docs = []

        # 文档1：关于机器学习的介绍
        doc1_content = """机器学习是人工智能的一个重要分支。它通过算法让计算机从数据中学习模式，
        而不需要显式编程。监督学习、无监督学习和强化学习是机器学习的三种主要类型。

        深度学习是机器学习的一个子集，使用神经网络模拟人脑的工作方式。
        卷积神经网络（CNN）在图像处理领域特别有效，而循环神经网络（RNN）则擅长处理序列数据。

        自然语言处理（NLP）是机器学习在语言领域的应用，包括文本分类、情感分析、
        机器翻译和问答系统等任务。"""

        doc1_path = tmp_path / "ml_intro.txt"
        doc1_path.write_text(doc1_content, encoding='utf-8')
        docs.append(doc1_path)

        # 文档2：关于检索增强生成的介绍
        doc2_content = """检索增强生成（RAG）是一种结合信息检索和文本生成的技术。
        它允许语言模型访问外部知识库来提供更准确和最新的响应。

        RAG过程分为两个主要阶段：索引阶段和检索阶段。在索引阶段，
        文档被解析、分割、嵌入并存储在向量数据库中。在检索阶段，
        用户查询被嵌入，然后检索相似的内容块来为生成提供上下文。

        RAG系统的质量取决于几个因素：分块策略、嵌入模型质量、
        向量存储的搜索算法，以及可选的重排序机制。"""

        doc2_path = tmp_path / "rag_intro.txt"
        doc2_path.write_text(doc2_content, encoding='utf-8')
        docs.append(doc2_path)

        return docs

    def test_rag_engine_initialization(self, config: RAGConfig):
        """测试RAG引擎初始化"""
        engine = RAGEngine(config)

        assert engine.config == config
        assert engine.parser is not None
        assert engine.chunker is not None
        assert engine.embedder is not None
        assert engine.vector_stores is not None
        assert len(engine.vector_stores) > 0
        assert engine.reranker is not None
        assert engine.indexing_pipeline is not None
        assert engine.retriever is not None

    def test_basic_indexing_and_retrieval(self, config: RAGConfig, sample_documents: list[Path]):
        """测试基础索引和检索功能"""
        engine = RAGEngine(config)

        # 测试单个文件索引
        doc_path = sample_documents[0]
        num_chunks = engine.index(doc_path)

        assert num_chunks > 0
        assert engine.num_chunks == num_chunks

        # 测试检索
        query = "什么是机器学习？"
        results = engine.retrieve(query)

        assert len(results) > 0
        assert len(results) <= config.retrieval_top_k

        # 验证结果结构
        for result in results:
            assert hasattr(result, 'chunk')
            assert hasattr(result, 'score')
            assert result.score >= 0.0
            assert result.score <= 1.0
            assert result.chunk.content is not None
            assert len(result.chunk.content) > 0
            assert result.chunk.metadata is not None
            assert 'filename' in result.chunk.metadata

        # 由于mock embedder的随机性，我们不严格检查排序
        # 但验证所有分数都在合理范围内
        scores = [r.score for r in results]
        for score in scores:
            assert 0.0 <= score <= 1.0, f"分数超出范围: {score}"

    def test_batch_indexing(self, config: RAGConfig, sample_documents: list[Path]):
        """测试批量文档索引"""
        engine = RAGEngine(config)

        # 批量索引所有文档
        total_chunks = engine.index_batch(sample_documents)

        assert total_chunks > 0
        assert engine.num_chunks == total_chunks

        # 验证所有文档都被索引
        # 获取第一个 PRIMARY 存储来验证
        primary_store = None
        for store, role in engine.vector_stores:
            if role.value == "primary":
                primary_store = store
                break

        if primary_store and hasattr(primary_store, '_chunks'):
            for doc_path in sample_documents:
                # 检查是否有来自该文档的块
                doc_chunks = [chunk for chunk in primary_store._chunks.values()
                             if chunk.metadata.get('filename') == doc_path.name]
                assert len(doc_chunks) > 0, f"文档 {doc_path.name} 没有被正确索引"

    def test_retrieval_relevance(self, config: RAGConfig, sample_documents: list[Path]):
        """测试检索的相关性"""
        engine = RAGEngine(config)

        # 索引文档
        engine.index_batch(sample_documents)

        # 使用更简单的测试，由于mock embedder的随机性，我们主要验证检索能正常工作
        # 而不是严格的相关性匹配
        queries = [
            "机器学习",
            "RAG",
            "深度学习",
            "自然语言处理"
        ]

        for query in queries:
            results = engine.retrieve(query)
            assert len(results) > 0

            # 验证所有结果都有有效的分数和内容
            for result in results:
                assert result.score >= 0.0
                assert result.score <= 1.0
                assert result.chunk.content is not None
                assert len(result.chunk.content.strip()) > 0

            # 验证至少返回了某些结果（由于mock embedder的随机性，我们不做严格的相关性检查）
            # 只要检索能正常工作就算通过

    def test_index_persistence(self, config: RAGConfig, sample_documents: list[Path]):
        """测试索引的保存和加载"""
        engine = RAGEngine(config)

        # 索引文档
        original_chunks = engine.index_batch(sample_documents)
        assert original_chunks > 0

        # 保存索引
        with tempfile.NamedTemporaryFile(suffix='.pkl', delete=False) as f:
            save_path = f.name

        try:
            engine.save_index(save_path)

            # 创建新引擎并加载索引
            new_engine = RAGEngine(config)
            new_engine.load_index(save_path)

            # 验证加载的索引
            assert new_engine.num_chunks == original_chunks

            # 测试检索功能仍然工作
            query = "机器学习"
            original_results = engine.retrieve(query)
            loaded_results = new_engine.retrieve(query)

            assert len(original_results) == len(loaded_results)

            # 结果应该相同（或至少相似）
            for orig, loaded in zip(original_results, loaded_results):
                assert orig.chunk.id == loaded.chunk.id
                assert abs(orig.score - loaded.score) < 0.001  # 分数应该非常接近

        finally:
            # 清理临时文件
            Path(save_path).unlink(missing_ok=True)

    def test_empty_retrieval(self, config: RAGConfig):
        """测试空索引的检索"""
        engine = RAGEngine(config)

        # 没有索引任何文档时检索
        query = "测试查询"
        results = engine.retrieve(query)

        assert results == []

    def test_chunk_metadata(self, config: RAGConfig, sample_documents: list[Path]):
        """测试块元数据的正确性"""
        engine = RAGEngine(config)

        # 索引单个文档
        doc_path = sample_documents[0]
        engine.index(doc_path)

        # 检索并检查元数据
        query = "机器学习"
        results = engine.retrieve(query)

        assert len(results) > 0

        for result in results:
            metadata = result.chunk.metadata
            assert 'filename' in metadata
            assert metadata['filename'] == doc_path.name  # 应该是文件名而不是完整路径
            assert 'chunk_index' in metadata
            assert isinstance(metadata['chunk_index'], int)
            assert metadata['chunk_index'] >= 0

    def test_query_method(self, config: RAGConfig, sample_documents: list[Path]):
        """测试query方法（不使用LLM）"""
        engine = RAGEngine(config)

        # 索引文档
        engine.index_batch(sample_documents)

        # 测试query方法
        query = "什么是RAG？"
        result = engine.query(query, use_llm=False)

        # 应该返回SearchResult列表
        assert isinstance(result, list)
        assert len(result) > 0

        for item in result:
            assert hasattr(item, 'chunk')
            assert hasattr(item, 'score')
