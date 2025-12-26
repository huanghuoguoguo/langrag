"""P0 问题修复的回归测试

测试内容：
1. save_index/load_index 支持单存储和多存储
2. 异步事件循环正确处理
"""

import asyncio
import tempfile
from pathlib import Path
import pytest

from langrag import RAGEngine
from langrag.config.models import RAGConfig, ComponentConfig, VectorStoreConfig, StorageRole


class TestSaveLoadIndex:
    """测试 save_index/load_index 功能"""

    def test_save_load_single_store(self):
        """测试单存储的保存和加载"""
        # 创建配置 - 单存储
        config = RAGConfig(
            parser=ComponentConfig(type="simple_text", params={}),
            chunker=ComponentConfig(type="fixed_size", params={"chunk_size": 100}),
            embedder=ComponentConfig(type="mock", params={"dimension": 384}),
            vector_store=VectorStoreConfig(type="in_memory", params={}),
        )

        # 初始化引擎
        engine = RAGEngine(config)

        # 创建临时文件进行测试
        with tempfile.TemporaryDirectory() as tmpdir:
            test_file = Path(tmpdir) / "test.txt"
            test_file.write_text("This is a test document for indexing.")

            # 索引文档
            num_chunks = engine.index(test_file)
            assert num_chunks > 0, "Should index at least one chunk"

            # 保存索引
            index_path = Path(tmpdir) / "index.pkl"
            engine.save_index(index_path)
            assert index_path.exists(), "Index file should be created"

            # 创建新引擎并加载索引
            engine2 = RAGEngine(config)
            engine2.load_index(index_path)

            # 验证加载成功 - 通过检索验证
            results = engine2.retrieve("test document")
            assert len(results) > 0, "Should retrieve results after loading"

    def test_save_load_multi_store(self):
        """测试多存储的保存和加载"""
        # 创建配置 - 多存储
        config = RAGConfig(
            parser=ComponentConfig(type="simple_text", params={}),
            chunker=ComponentConfig(type="fixed_size", params={"chunk_size": 100}),
            embedder=ComponentConfig(type="mock", params={"dimension": 384}),
            vector_stores=[
                VectorStoreConfig(
                    type="in_memory",
                    params={},
                    role=StorageRole.PRIMARY
                ),
                VectorStoreConfig(
                    type="in_memory",
                    params={},
                    role=StorageRole.BACKUP
                ),
            ],
        )

        # 初始化引擎
        engine = RAGEngine(config)

        # 创建临时文件进行测试
        with tempfile.TemporaryDirectory() as tmpdir:
            test_file = Path(tmpdir) / "test.txt"
            test_file.write_text("Multi-store test document.")

            # 索引文档
            num_chunks = engine.index(test_file)
            assert num_chunks > 0, "Should index at least one chunk"

            # 保存索引到目录
            index_dir = Path(tmpdir) / "index_dir"
            engine.save_index(index_dir)
            assert index_dir.exists(), "Index directory should be created"
            assert index_dir.is_dir(), "Index path should be a directory"

            # 检查子目录是否创建
            subdirs = list(index_dir.iterdir())
            assert len(subdirs) >= 2, "Should create subdirectories for each store"

            # 创建新引擎并加载索引
            engine2 = RAGEngine(config)
            engine2.load_index(index_dir)

            # 验证加载成功
            results = engine2.retrieve("multi-store test")
            assert len(results) > 0, "Should retrieve results after loading"


class TestAsyncEventLoop:
    """测试异步事件循环处理"""

    def test_retrieve_without_event_loop(self):
        """测试在没有事件循环的环境中同步检索"""
        # 创建简单配置
        config = RAGConfig(
            parser=ComponentConfig(type="simple_text", params={}),
            chunker=ComponentConfig(type="fixed_size", params={"chunk_size": 100}),
            embedder=ComponentConfig(type="mock", params={"dimension": 384}),
            vector_store=VectorStoreConfig(type="in_memory", params={}),
        )

        engine = RAGEngine(config)

        # 索引测试数据
        with tempfile.TemporaryDirectory() as tmpdir:
            test_file = Path(tmpdir) / "test.txt"
            test_file.write_text("Testing event loop handling.")
            engine.index(test_file)

            # 同步检索（应该在没有事件循环时正常工作）
            results = engine.retrieve("event loop")
            assert isinstance(results, list), "Should return list of results"

    @pytest.mark.asyncio
    async def test_retrieve_async(self):
        """测试异步检索 API"""
        config = RAGConfig(
            parser=ComponentConfig(type="simple_text", params={}),
            chunker=ComponentConfig(type="fixed_size", params={"chunk_size": 100}),
            embedder=ComponentConfig(type="mock", params={"dimension": 384}),
            vector_store=VectorStoreConfig(type="in_memory", params={}),
        )

        engine = RAGEngine(config)

        # 索引测试数据
        with tempfile.TemporaryDirectory() as tmpdir:
            test_file = Path(tmpdir) / "test.txt"
            test_file.write_text("Testing async retrieval.")
            engine.index(test_file)

            # 异步检索
            results = await engine.retrieve_async("async")
            assert isinstance(results, list), "Should return list of results"

    def test_retrieve_with_running_loop(self):
        """测试在已运行事件循环中同步检索（使用线程池回退）"""
        config = RAGConfig(
            parser=ComponentConfig(type="simple_text", params={}),
            chunker=ComponentConfig(type="fixed_size", params={"chunk_size": 100}),
            embedder=ComponentConfig(type="mock", params={"dimension": 384}),
            vector_store=VectorStoreConfig(type="in_memory", params={}),
        )

        async def run_in_loop():
            """在事件循环中运行同步检索"""
            engine = RAGEngine(config)

            # 索引测试数据
            with tempfile.TemporaryDirectory() as tmpdir:
                test_file = Path(tmpdir) / "test.txt"
                test_file.write_text("Testing nested event loop.")
                engine.index(test_file)

                # 在事件循环中调用同步检索（应该使用线程池回退）
                results = engine.retrieve("nested loop")
                assert isinstance(results, list), "Should handle nested loop gracefully"
                return results

        # 在事件循环中运行
        results = asyncio.run(run_in_loop())
        assert len(results) >= 0, "Should complete without error"


class TestEnvironmentVariables:
    """测试环境变量配置"""

    def test_env_file_example_exists(self):
        """验证 .env.example 文件存在"""
        env_example = Path(__file__).parent.parent / ".env.example"
        assert env_example.exists(), ".env.example file should exist"

        # 验证包含必要的示例
        content = env_example.read_text()
        assert "QWEN_API_KEY" in content, "Should include QWEN_API_KEY example"

    def test_gitignore_protects_env(self):
        """验证 .gitignore 保护 .env 文件"""
        gitignore = Path(__file__).parent.parent / ".gitignore"
        assert gitignore.exists(), ".gitignore should exist"

        content = gitignore.read_text()
        assert ".env" in content, ".env should be in .gitignore"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
