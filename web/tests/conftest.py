"""Pytest configuration for Web Demo tests."""

import pytest
import os
import tempfile
from pathlib import Path
from fastapi.testclient import TestClient


def pytest_configure(config):
    """Register custom markers."""
    config.addinivalue_line(
        "markers", "local_llm: Tests that require a local LLM model"
    )

# ==================== Fixtures ====================

@pytest.fixture(scope="module")
def test_client():
    """Create a test client for the FastAPI app."""
    # 设置测试环境变量
    os.environ.setdefault("LANGRAG_ENV", "test")

    from web.app import app
    with TestClient(app) as client:
        yield client


@pytest.fixture(scope="module")
def default_embedder_name(test_client):
    """Get or create a default embedder for testing."""
    # 检查是否有可用的 embedder
    response = test_client.get("/api/config/embedders")
    if response.status_code == 200:
        data = response.json()
        # API 可能返回 {'embedders': [...]} 或直接返回列表
        embedders = data.get("embedders", data) if isinstance(data, dict) else data
        if embedders and len(embedders) > 0:
            return embedders[0].get("name", "default")

    # 如果没有，使用默认名称（SeekDB embedder 通常是自动可用的）
    return "default"


@pytest.fixture(scope="module")
def test_kb_id(test_client, default_embedder_name):
    """Create a test knowledge base and return its ID."""
    response = test_client.post("/api/kb", json={
        "name": "Test KB",
        "description": "Test knowledge base for API testing",
        "vdb_type": "chroma",
        "embedder_name": default_embedder_name,
        "chunk_size": 500,
        "chunk_overlap": 50,
        "indexing_technique": "high_quality"
    })

    if response.status_code == 200:
        kb_id = response.json()["id"]
        yield kb_id
        # Cleanup
        test_client.delete(f"/api/kb/{kb_id}")
    else:
        pytest.skip(f"Failed to create test KB: {response.text}")


@pytest.fixture(scope="module")
def sample_document():
    """Create a sample document for testing."""
    content = """
# 分布式系统概述

分布式系统是由多台计算机通过网络连接组成的系统，这些计算机协同工作以完成共同的任务。

## 主要特点

1. **可扩展性**：可以通过添加更多节点来增加系统容量
2. **容错性**：单个节点故障不会导致整个系统崩溃
3. **并发性**：多个节点可以同时处理不同的任务

## CAP 定理

CAP 定理指出，分布式系统最多只能同时满足以下三个特性中的两个：
- 一致性 (Consistency)
- 可用性 (Availability)
- 分区容错性 (Partition tolerance)

## 常见应用

- 分布式数据库（如 Cassandra、MongoDB）
- 分布式缓存（如 Redis Cluster）
- 分布式消息队列（如 Kafka）
"""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.md', delete=False) as f:
        f.write(content)
        f.flush()
        yield Path(f.name)
    # Cleanup
    os.unlink(f.name)
