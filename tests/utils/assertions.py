"""Test utilities - 测试工具函数"""

from langrag.core.search_result import SearchResult
from langrag.core.chunk import Chunk


def assert_search_results_valid(results, min_results=0, max_results=float('inf')):
    """验证搜索结果的有效性

    Args:
        results: 搜索结果列表
        min_results: 最小结果数
        max_results: 最大结果数

    Raises:
        AssertionError: 如果结果无效
    """
    assert isinstance(results, list), "Results must be a list"
    assert min_results <= len(results) <= max_results, \
        f"Expected {min_results}-{max_results} results, got {len(results)}"

    for i, result in enumerate(results):
        assert isinstance(result, SearchResult), \
            f"Result {i} is not a SearchResult"
        assert isinstance(result.chunk, Chunk), \
            f"Result {i}.chunk is not a Chunk"
        assert 0 <= result.score <= 1.0, \
            f"Result {i} score {result.score} not in [0, 1]"


def assert_scores_descending(results):
    """验证搜索结果的分数是降序排列的

    Args:
        results: 搜索结果列表

    Raises:
        AssertionError: 如果分数不是降序
    """
    scores = [r.score for r in results]
    assert scores == sorted(scores, reverse=True), \
        f"Scores not in descending order: {scores}"


def assert_chunks_have_embeddings(chunks):
    """验证所有 chunks 都有 embeddings

    Args:
        chunks: Chunk 列表

    Raises:
        AssertionError: 如果有 chunk 缺少 embedding
    """
    for i, chunk in enumerate(chunks):
        assert chunk.embedding is not None, \
            f"Chunk {i} (id={chunk.id}) missing embedding"
        assert isinstance(chunk.embedding, list), \
            f"Chunk {i} embedding is not a list"
        assert len(chunk.embedding) > 0, \
            f"Chunk {i} embedding is empty"


def assert_file_exists(path):
    """验证文件存在

    Args:
        path: 文件路径

    Raises:
        AssertionError: 如果文件不存在
    """
    from pathlib import Path
    path = Path(path)
    assert path.exists(), f"File not found: {path}"
    assert path.is_file(), f"Path is not a file: {path}"


def assert_directory_exists(path):
    """验证目录存在

    Args:
        path: 目录路径

    Raises:
        AssertionError: 如果目录不存在
    """
    from pathlib import Path
    path = Path(path)
    assert path.exists(), f"Directory not found: {path}"
    assert path.is_dir(), f"Path is not a directory: {path}"
