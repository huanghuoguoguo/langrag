"""Qwen Reranker - 使用 DashScope API 进行重排序"""

from __future__ import annotations
from typing import TYPE_CHECKING
import httpx
from loguru import logger

from ..base import BaseReranker

if TYPE_CHECKING:
    from ...core.query import Query
    from ...core.search_result import SearchResult


class QwenReranker(BaseReranker):
    """Qwen 重排序器
    
    使用阿里云 DashScope API 的 Qwen 模型对检索结果进行重排序。
    
    参数:
        api_key: DashScope API 密钥
        model: 模型名称，默认 'qwen3-rerank'
        instruct: 重排序指令，可自定义
        timeout: API 超时时间（秒），默认 30.0
    
    使用示例:
        >>> reranker = QwenReranker(
        ...     api_key="your-api-key",
        ...     model="qwen3-rerank"
        ... )
        >>> reranked = reranker.rerank(query, results, top_k=5)
    """
    
    def __init__(
        self,
        api_key: str,
        model: str = "qwen3-rerank",
        instruct: str = "Given a web search query, retrieve relevant passages that answer the query.",
        timeout: float = 30.0
    ):
        """初始化 Qwen Reranker
        
        Args:
            api_key: DashScope API 密钥
            model: 模型名称
            instruct: 重排序指令
            timeout: API 超时时间
            
        Raises:
            ValueError: 如果 api_key 为空
        """
        if not api_key:
            raise ValueError("QwenReranker requires 'api_key' parameter")
        
        self.api_key = api_key
        self.model = model
        self.instruct = instruct
        self.timeout = timeout
        self.api_url = "https://dashscope.aliyuncs.com/compatible-api/v1/reranks"
        
        logger.info(
            f"Initialized QwenReranker with model='{model}', timeout={timeout}s"
        )
    
    def rerank(
        self,
        query: Query,
        results: list[SearchResult],
        top_k: int | None = None
    ) -> list[SearchResult]:
        """重排序检索结果（同步方法）
        
        Args:
            query: 查询对象
            results: 待重排序的结果列表
            top_k: 返回的结果数量，None 表示全部
            
        Returns:
            重排序后的结果列表
        """
        # 使用同步方式调用
        import asyncio
        
        try:
            # 在同步上下文中运行异步代码
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # 如果已经在事件循环中，使用 run_until_complete
                # 这可能会导致问题，但作为回退方案
                logger.warning("Cannot run async rerank in running event loop, using fallback")
                return results[:top_k] if top_k else results
            else:
                return loop.run_until_complete(self.rerank_async(query, results, top_k))
        except Exception as e:
            logger.error(f"Failed to run async rerank: {e}")
            return results[:top_k] if top_k else results
    
    async def rerank_async(
        self,
        query: Query,
        results: list[SearchResult],
        top_k: int | None = None
    ) -> list[SearchResult]:
        """重排序检索结果（异步方法）
        
        Args:
            query: 查询对象
            results: 待重排序的结果列表
            top_k: 返回的结果数量，None 表示全部
            
        Returns:
            重排序后的结果列表
        """
        if not results:
            logger.debug("Empty results, nothing to rerank")
            return []
        
        # 确定返回的结果数量
        k = min(top_k, len(results)) if top_k else len(results)
        
        logger.info(
            f"Reranking {len(results)} results for query: '{query.text[:50]}...', "
            f"returning top {k}"
        )
        
        try:
            # 提取文档内容
            documents = [result.chunk.content for result in results]
            
            # 准备 API 请求
            request_data = {
                "model": self.model,
                "query": query.text,
                "documents": documents,
                "top_n": k,
                "instruct": self.instruct
            }
            
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }
            
            # 发送 API 请求
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.post(
                    self.api_url,
                    json=request_data,
                    headers=headers
                )
                response.raise_for_status()
                api_response = response.json()
            
            # 解析响应并重排序
            reranked_results = self._parse_and_rerank(
                api_response, results, k
            )
            
            logger.info(
                f"Qwen reranker: {len(results)} → {len(reranked_results)} results"
            )
            return reranked_results
            
        except httpx.HTTPStatusError as e:
            logger.error(
                f"Qwen API error (status {e.response.status_code}): {e.response.text}"
            )
            # 降级：返回原始结果
            return results[:k]
            
        except httpx.TimeoutException:
            logger.error(f"Qwen API timeout after {self.timeout}s")
            return results[:k]
            
        except Exception as e:
            logger.error(f"Qwen reranker failed: {e}", exc_info=True)
            return results[:k]
    
    def _parse_and_rerank(
        self,
        api_response: dict,
        original_results: list[SearchResult],
        top_k: int
    ) -> list[SearchResult]:
        """解析 DashScope API 响应并重排序结果
        
        API 响应格式:
        {
            "object": "list",
            "results": [
                {"index": 1, "relevance_score": 0.6171875690986707},
                {"index": 0, "relevance_score": 0.6073028761000254},
                ...
            ],
            "model": "qwen3-rerank",
            "usage": {"total_tokens": 2064}
        }
        
        Args:
            api_response: API 响应
            original_results: 原始结果列表
            top_k: 返回的结果数量
            
        Returns:
            重排序后的结果列表
        """
        from ...core.search_result import SearchResult
        
        results_data = api_response.get('results', [])
        
        if not results_data:
            logger.warning("API returned empty results")
            return original_results[:top_k]
        
        reranked_results = []
        used_indices = set()
        
        # 按 API 返回的顺序创建重排序结果
        for result_item in results_data:
            index = result_item.get('index')
            relevance_score = result_item.get('relevance_score', 0.0)
            
            if index is None or not (0 <= index < len(original_results)):
                logger.warning(f"Invalid index {index} from API")
                continue
            
            used_indices.add(index)
            original_result = original_results[index]
            
            # relevance_score: 0-1，越高越相关
            # 直接作为 score（SearchResult 的 score 也是越高越好）
            score = max(0.0, min(1.0, relevance_score))
            
            # 创建新的 SearchResult（保持原 chunk，更新 score）
            new_result = SearchResult(
                chunk=original_result.chunk,
                score=score
            )
            
            reranked_results.append(new_result)
            
            if len(reranked_results) >= top_k:
                break
        
        # 如果 API 返回的结果不足 top_k，用原始结果填充
        if len(reranked_results) < top_k:
            logger.debug(
                f"API returned {len(reranked_results)} results, "
                f"filling to {top_k} with original results"
            )
            for i, result in enumerate(original_results):
                if i not in used_indices and len(reranked_results) < top_k:
                    reranked_results.append(result)
        
        logger.debug(
            f"Reranked results: "
            f"scores=[{', '.join(f'{r.score:.3f}' for r in reranked_results[:5])}...]"
        )
        
        return reranked_results[:top_k]

