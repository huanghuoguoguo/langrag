import pytest
import logging
import sys
import os
import tempfile
from pathlib import Path

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 确保可以从 src 导入
sys.path.append(os.path.join(os.getcwd(), "src"))

from langrag.datasource.vdb.duckdb import DuckDBVector
from langrag.entities.document import Document
from langrag.entities.dataset import Dataset


def _run_duckdb_test(db_path: str):
    """DuckDB 向量数据库验证的核心逻辑"""
    logger.info(f"开始 DuckDB 向量数据库验证，路径: {db_path}")

    # 创建虚拟数据集实体
    dataset = Dataset(
        id="test_ds",
        name="test_dataset",
        collection_name="test_collection"
    )

    logger.info("初始化 DuckDBVector...")
    try:
        # 初始化向量存储
        vector_store = DuckDBVector(
            dataset=dataset,
            database_path=db_path
        )
        logger.info("DuckDBVector 初始化成功。")
    except ImportError:
        logger.error("duckdb 未安装或未找到。")
        pytest.skip("duckdb not installed")
        return
    except Exception as e:
        logger.error(f"初始化失败: {e}")
        raise

    # 创建虚拟文档
    docs = [
        Document(id="1", page_content="Hello world", metadata={"source": "test"}, vector=[0.1]*768),
        Document(id="2", page_content="DuckDB is great for vector search", metadata={"source": "test"}, vector=[0.2]*768),
        Document(id="3", page_content="Vector databases are useful tools", metadata={"source": "test"}, vector=[0.3]*768),
    ]

    logger.info(f"添加 {len(docs)} 个文档...")
    try:
        vector_store.create(docs)
        logger.info("文档添加成功。")
    except Exception as e:
        logger.error(f"添加文档失败: {e}")
        raise e

    # 测试向量搜索
    query_vector = [0.1]*768
    logger.info("测试向量搜索...")
    try:
        results = vector_store.search(
            query="world",
            query_vector=query_vector,
            top_k=2,
            search_type="similarity"
        )
        logger.info(f"向量搜索返回 {len(results)} 个结果。")
        for doc in results:
            logger.info(f" - {doc.id}: {doc.page_content} (相似度分数: {doc.metadata.get('score')})")

        # 验证基本功能
        assert len(results) > 0, "未找到任何结果"
        assert len(results) <= 2, "返回结果超过 top_k 限制"

        # 验证返回的文档 ID 在我们添加的文档中
        valid_ids = {"1", "2", "3"}
        for doc in results:
            assert doc.id in valid_ids, f"返回了未知的文档 ID: {doc.id}"
            assert doc.page_content, "文档内容为空"
            assert "score" in doc.metadata, "缺少相似度分数"

    except Exception as e:
        logger.error(f"向量搜索失败: {e}")
        raise e

    # 测试关键词搜索 (DuckDB 支持 FTS)
    logger.info("测试关键词搜索...")
    try:
        kw_results = vector_store.search(
            query="DuckDB",
            query_vector=None,
            top_k=2,
            search_type="keyword"
        )
        logger.info(f"关键词搜索返回 {len(kw_results)} 个结果。")
        for doc in kw_results:
            logger.info(f" - {doc.id}: {doc.page_content} (相似度分数: {doc.metadata.get('score')})")

        # 验证关键词搜索功能
        assert len(kw_results) > 0, "关键词搜索未找到任何结果"
        assert len(kw_results) <= 2, "返回结果超过 top_k 限制"

        # 验证返回的文档 ID 在我们添加的文档中
        for doc in kw_results:
            assert doc.id in valid_ids, f"返回了未知的文档 ID: {doc.id}"
            assert doc.page_content, "文档内容为空"

    except Exception as e:
        logger.error(f"关键词搜索失败: {e}")
        raise e

    # 测试混合搜索 (DuckDB 支持混合搜索)
    logger.info("测试混合搜索...")
    try:
        hybrid_results = vector_store.search(
            query="vector",
            query_vector=query_vector,
            top_k=3,
            search_type="hybrid"
        )
        logger.info(f"混合搜索返回 {len(hybrid_results)} 个结果。")
        for doc in hybrid_results:
            logger.info(f" - {doc.id}: {doc.page_content} (相似度分数: {doc.metadata.get('score')})")

        # 验证混合搜索功能
        assert len(hybrid_results) > 0, "混合搜索未找到任何结果"
        assert len(hybrid_results) <= 3, "返回结果超过 top_k 限制"

        # 验证返回的文档 ID 在我们添加的文档中
        for doc in hybrid_results:
            assert doc.id in valid_ids, f"返回了未知的文档 ID: {doc.id}"
            assert doc.page_content, "文档内容为空"

    except Exception as e:
        logger.error(f"混合搜索失败: {e}")
        raise e

    # 清理资源
    try:
        vector_store.delete()  # 删除表
        logger.info("DuckDB 验证成功完成。")
    except Exception as e:
        logger.warning(f"清理失败: {e}")


@pytest.fixture
def temp_db_path():
    """创建临时目录和数据库路径，测试后自动清理"""
    with tempfile.TemporaryDirectory() as temp_dir:
        yield os.path.join(temp_dir, "test_duckdb.db")


def test_duckdb_with_path(temp_db_path):
    """DuckDB 向量数据库验证，使用临时目录"""
    _run_duckdb_test(temp_db_path)


def test_duckdb(temp_db_path):
    """向后兼容的测试函数"""
    _run_duckdb_test(temp_db_path)


@pytest.mark.integration
def test_duckdb_integration(temp_db_path):
    """集成测试：验证 DuckDB 向量数据库的完整功能"""
    _run_duckdb_test(temp_db_path)
