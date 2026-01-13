
import pytest
import logging
import sys
import shutil
import os
import tempfile
from pathlib import Path

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 确保可以从 src 导入
sys.path.append(os.path.join(os.getcwd(), "src"))

# Check if pyseekdb is available and stable
try:
    import pyseekdb
    SEEKDB_AVAILABLE = True
except ImportError:
    SEEKDB_AVAILABLE = False

# Skip all tests if pyseekdb is not available
pytestmark = [
    pytest.mark.skipif(
        not SEEKDB_AVAILABLE,
        reason="pyseekdb not installed"
    ),
    pytest.mark.seekdb  # Mark for selective running: pytest -m seekdb
]

from langrag.datasource.vdb.seekdb import SeekDBVector
from langrag.entities.document import Document
from langrag.entities.dataset import Dataset


def _run_seekdb_test(db_path: str):
    """SeekDB 嵌入式模式验证的核心逻辑"""
    logger.info(f"开始 SeekDB 嵌入式模式验证，路径: {db_path}")

    # 创建虚拟数据集实体
    dataset = Dataset(
        id="test_ds",
        name="test_dataset",
        collection_name="test_collection"
    )

    logger.info("初始化 SeekDBVector...")
    try:
        # 初始化向量存储
        vector_store = SeekDBVector(
            dataset=dataset,
            mode="embedded",
            db_path=db_path
        )
        logger.info("SeekDBVector 初始化成功。")
    except ImportError:
        logger.error("pyseekdb 未安装或未找到。")
        pytest.skip("pyseekdb not installed")
        return
    except Exception as e:
        logger.error(f"初始化失败: {e}")
        raise

    # 创建虚拟文档
    docs = [
        Document(id="1", page_content="Hello world", metadata={"source": "test"}, vector=[0.1]*768),
        Document(id="2", page_content="SeekDB is great", metadata={"source": "test"}, vector=[0.2]*768),
        Document(id="3", page_content="Vector databases are useful", metadata={"source": "test"}, vector=[0.3]*768),
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
        logger.info(f"搜索返回 {len(results)} 个结果。")
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
        logger.error(f"搜索失败: {e}")
        raise e
    finally:
        # 确保关闭连接以释放锁
        if 'vector_store' in locals():
            logger.info("关闭 SeekDB 连接...")
            try:
                vector_store.close()
            except Exception as e:
                logger.error(f"关闭向量存储失败: {e}")

    logger.info("SeekDB 验证成功完成。")


@pytest.fixture
def temp_db_path():
    """创建临时目录用于测试，测试后自动清理"""
    with tempfile.TemporaryDirectory() as temp_dir:
        yield temp_dir


def test_seekdb_embedded_with_path(temp_db_path):
    """SeekDB 嵌入式模式验证，使用临时目录"""
    _run_seekdb_test(temp_db_path)


def test_seekdb_embedded(temp_db_path):
    """向后兼容的测试函数"""
    _run_seekdb_test(temp_db_path)


@pytest.mark.integration
def test_seekdb_embedded_integration(temp_db_path):
    """集成测试：验证 SeekDB 向量数据库的嵌入式模式功能"""
    _run_seekdb_test(temp_db_path)
