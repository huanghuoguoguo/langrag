"""Factory for creating compressor instances."""

from loguru import logger

from .base import BaseCompressor


class CompressorFactory:
    """工厂类，用于创建上下文压缩器实例

    支持的压缩器类型：
    - noop: 不压缩，直接返回原始结果
    - qwen: 使用 Qwen API 进行智能压缩
    """

    @staticmethod
    def create(compressor_type: str, **kwargs) -> BaseCompressor:
        """创建压缩器实例

        Args:
            compressor_type: 压缩器类型 ("noop", "qwen")
            **kwargs: 压缩器特定的参数

        Returns:
            BaseCompressor 实例

        Raises:
            ValueError: 如果压缩器类型不支持

        Examples:
            >>> # 创建 NoOp 压缩器
            >>> compressor = CompressorFactory.create("noop")
            >>>
            >>> # 创建 Qwen 压缩器
            >>> compressor = CompressorFactory.create(
            ...     "qwen",
            ...     api_key="your-api-key",
            ...     model="qwen-plus"
            ... )
        """
        compressor_type = compressor_type.lower()

        if compressor_type == "noop":
            from .providers.noop import NoOpCompressor

            logger.debug("Creating NoOpCompressor")
            return NoOpCompressor()

        elif compressor_type == "qwen":
            from .providers.qwen import QwenCompressor

            # 检查必需参数
            if "api_key" not in kwargs:
                raise ValueError("api_key is required for Qwen compressor")

            logger.debug(f"Creating QwenCompressor with params: {list(kwargs.keys())}")
            return QwenCompressor(**kwargs)

        else:
            raise ValueError(
                f"Unknown compressor type: {compressor_type}. Supported types: noop, qwen"
            )
