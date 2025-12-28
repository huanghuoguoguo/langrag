"""Qwen-based context compressor using DashScope API."""

import asyncio
from typing import Any

from loguru import logger

from langrag.entities.search_result import SearchResult
from ..base import BaseCompressor

try:
    import httpx

    HTTPX_AVAILABLE = True
except ImportError:
    HTTPX_AVAILABLE = False
    logger.warning("httpx not installed. QwenCompressor will not be available.")


class QwenCompressor(BaseCompressor):
    """使用 Qwen API 进行上下文压缩
    
    通过调用 Qwen 的 Chat Completions API 来压缩每个检索结果的内容。
    
    参数：
        api_key: DashScope API Key
        model: 模型名称，默认 "qwen-plus"
        api_url: API 端点，默认使用 DashScope 兼容模式端点
        temperature: 生成温度，默认 0.3（更确定性的输出）
        timeout: 请求超时时间（秒），默认 30
        max_concurrent: 最大并发请求数，默认 5
    """

    def __init__(
        self,
        api_key: str,
        model: str = "qwen-plus",
        api_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions",
        temperature: float = 0.3,
        timeout: int = 30,
        max_concurrent: int = 5,
    ):
        """初始化 Qwen 压缩器
        
        Args:
            api_key: DashScope API Key
            model: 模型名称
            api_url: API 端点 URL
            temperature: 生成温度
            timeout: 请求超时时间
            max_concurrent: 最大并发请求数
        """
        if not HTTPX_AVAILABLE:
            raise ImportError(
                "httpx is required for QwenCompressor. Install it with: pip install httpx"
            )
        
        self.api_key = api_key
        self.model = model
        self.api_url = api_url
        self.temperature = temperature
        self.timeout = timeout
        self.max_concurrent = max_concurrent
        
        logger.info(f"Initialized QwenCompressor with model: {model}")

    def compress(
        self, query: str, results: list[SearchResult], target_ratio: float = 0.5
    ) -> list[SearchResult]:
        """压缩检索结果（同步接口）
        
        Args:
            query: 用户查询
            results: 检索结果列表
            target_ratio: 目标压缩比率（0-1）
            
        Returns:
            压缩后的检索结果列表
        """
        # 使用异步实现，然后在同步上下文中运行
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            # 没有运行中的事件循环，创建新的
            return asyncio.run(self.compress_async(query, results, target_ratio))
        else:
            # 已有事件循环，创建 task
            import nest_asyncio
            
            nest_asyncio.apply()
            return loop.run_until_complete(self.compress_async(query, results, target_ratio))

    async def compress_async(
        self, query: str, results: list[SearchResult], target_ratio: float = 0.5
    ) -> list[SearchResult]:
        """压缩检索结果（异步接口）
        
        Args:
            query: 用户查询
            results: 检索结果列表
            target_ratio: 目标压缩比率（0-1）
            
        Returns:
            压缩后的检索结果列表
        """
        if not results:
            return results
        
        logger.info(
            f"Compressing {len(results)} results with target ratio {target_ratio:.1%}"
        )
        
        # 创建信号量限制并发
        semaphore = asyncio.Semaphore(self.max_concurrent)
        
        # 并发压缩所有结果
        tasks = [
            self._compress_single_result(query, result, target_ratio, semaphore)
            for result in results
        ]
        
        compressed_results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # 处理异常，保留原始结果
        final_results = []
        for i, result in enumerate(compressed_results):
            if isinstance(result, Exception):
                logger.warning(f"Failed to compress result {i}: {result}, using original")
                final_results.append(results[i])
            else:
                final_results.append(result)
        
        # 计算压缩统计
        original_length = sum(len(r.chunk.page_content) for r in results)
        compressed_length = sum(len(r.chunk.page_content) for r in final_results)
        actual_ratio = compressed_length / original_length if original_length > 0 else 1.0
        
        logger.info(
            f"Compression complete: {original_length} → {compressed_length} chars "
            f"(actual ratio: {actual_ratio:.1%})"
        )
        
        return final_results

    async def _compress_single_result(
        self,
        query: str,
        result: SearchResult,
        target_ratio: float,
        semaphore: asyncio.Semaphore,
    ) -> SearchResult:
        """压缩单个检索结果
        
        Args:
            query: 用户查询
            result: 检索结果
            target_ratio: 目标压缩比率
            semaphore: 并发控制信号量
            
        Returns:
            压缩后的检索结果
        """
        async with semaphore:
            try:
                compressed_content = await self._call_qwen_api(
                    query, result.chunk.page_content, target_ratio
                )
                
                # 创建新的 Chunk 和 SearchResult（保持不可变性）
                from langrag.entities.document import Document
                
                compressed_chunk = Document(
                    id=result.chunk.id,
                    page_content=compressed_content,
                    vector=result.chunk.vector,
                    # source_doc_id=result.chunk.source_doc_id, # Document doesn't have source_doc_id field directly anymore, relies on metadata
                    metadata={
                        **result.chunk.metadata,
                        "source_doc_id": result.chunk.metadata.get("source_doc_id"),
                        "compressed": True,
                        "original_length": len(result.chunk.page_content),
                        "compressed_length": len(compressed_content),
                    },
                )
                
                return SearchResult(chunk=compressed_chunk, score=result.score)
            
            except Exception as e:
                logger.error(f"Failed to compress chunk {result.chunk.id}: {e}")
                raise

    async def _call_qwen_api(
        self, query: str, content: str, target_ratio: float
    ) -> str:
        """调用 Qwen API 进行内容压缩
        
        Args:
            query: 用户查询
            content: 要压缩的内容
            target_ratio: 目标压缩比率
            
        Returns:
            压缩后的内容
        """
        # 构建压缩提示词
        target_length_desc = f"{int(target_ratio * 100)}% 长度"
        if target_ratio <= 0.3:
            target_length_desc = "极简模式（保留核心要点）"
        elif target_ratio <= 0.5:
            target_length_desc = "精简模式（保留关键信息）"
        elif target_ratio <= 0.7:
            target_length_desc = "适度压缩（保留主要细节）"
        
        system_prompt = (
            "你是一个专业的文本压缩助手。你的任务是将给定的文本压缩到指定长度，"
            "同时保留与用户查询最相关的信息。"
            "压缩原则：1) 优先保留与查询相关的内容；2) 去除冗余和不重要的细节；"
            "3) 保持语义连贯性；4) 不添加原文没有的信息。"
        )
        
        user_prompt = f"""用户查询：{query}

原始文本：
{content}

请将上述文本压缩到 {target_length_desc}，重点保留与查询相关的内容。
直接输出压缩后的文本，不要添加任何解释或前缀。"""
        
        # 构建请求
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": self.temperature,
        }
        
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        
        # 发送请求
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            response = await client.post(
                self.api_url, headers=headers, json=payload
            )
            response.raise_for_status()
            
            result = response.json()
            
            # 解析响应
            if "choices" in result and len(result["choices"]) > 0:
                compressed_text = result["choices"][0]["message"]["content"]
                
                # 记录 token 使用情况
                if "usage" in result:
                    usage = result["usage"]
                    logger.debug(
                        f"Qwen API usage: "
                        f"prompt={usage.get('prompt_tokens', 0)}, "
                        f"completion={usage.get('completion_tokens', 0)}, "
                        f"total={usage.get('total_tokens', 0)}"
                    )
                
                return compressed_text.strip()
            else:
                raise ValueError(f"Unexpected API response format: {result}")

