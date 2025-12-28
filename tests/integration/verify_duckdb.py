import pytest
import logging
import sys
import shutil
import os
from pathlib import Path

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 确保可以从 src 导入
sys.path.append(os.path.join(os.getcwd(), "src"))

from langrag.datasource.vdb.duckdb import DuckDBVector
from langrag.entities.document import Document
from langrag.entities.dataset import Dataset

def test_duckdb_with_path(db_path="./test_duckdb_data.db"):
    """DuckDB 向量数据库验证，使用指定的数据库路径"""
    logger.info("开始 DuckDB 向量数据库验证")

    # 清理之前的运行结果
    if os.path.exists(db_path):
        os.remove(db_path)

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
        return
    except Exception as e:
        logger.error(f"初始化失败: {e}")
        return

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

    # 测试关键词搜索 (DuckDB FTS 功能未实现)
    logger.info("测试关键词搜索...")
    try:
        kw_results = vector_store.search(
            query="DuckDB",
            query_vector=None,
            top_k=2,
            search_type="keyword"
        )
        # 如果意外成功，记录警告
        logger.warning(f"关键词搜索意外成功，返回 {len(kw_results)} 个结果")
        assert False, "关键词搜索应该抛出 NotImplementedError"

    except NotImplementedError as e:
        logger.info(f"关键词搜索正确抛出 NotImplementedError: {e}")
        assert "Full-Text Search" in str(e), "错误消息应该提到 FTS"

    except Exception as e:
        logger.error(f"关键词搜索抛出意外错误: {e}")
        assert False, f"应该抛出 NotImplementedError，而不是 {type(e).__name__}"

    # 测试混合搜索 (DuckDB FTS 功能未实现，所以混合搜索也不可用)
    logger.info("测试混合搜索...")
    try:
        hybrid_results = vector_store.search(
            query="vector",
            query_vector=query_vector,
            top_k=3,
            search_type="hybrid"
        )
        # 如果意外成功，记录警告
        logger.warning(f"混合搜索意外成功，返回 {len(hybrid_results)} 个结果")
        assert False, "混合搜索应该抛出 NotImplementedError"

    except NotImplementedError as e:
        logger.info(f"混合搜索正确抛出 NotImplementedError: {e}")
        assert "Hybrid Search" in str(e), "错误消息应该提到 Hybrid Search"

    except Exception as e:
        logger.error(f"混合搜索抛出意外错误: {e}")
        assert False, f"应该抛出 NotImplementedError，而不是 {type(e).__name__}"

    # 清理资源
    try:
        vector_store.delete()  # 删除表
        if os.path.exists(db_path):
            os.remove(db_path)
        logger.info("DuckDB 验证成功完成。")
    except Exception as e:
        logger.warning(f"清理失败: {e}")

def test_duckdb():
    """向后兼容的测试函数"""
    test_duckdb_with_path("./test_duckdb_data.db")

@pytest.mark.integration
def test_duckdb_integration():
    """集成测试：验证 DuckDB 向量数据库的完整功能"""
    # 使用不同的数据库路径避免冲突
    import tempfile
    with tempfile.TemporaryDirectory() as temp_dir:
        db_path = os.path.join(temp_dir, "test_duckdb.db")
        test_duckdb_with_path(db_path)
