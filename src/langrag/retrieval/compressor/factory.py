"""Factory for creating compressor instances."""

from loguru import logger

from .base import BaseCompressor


class CompressorFactory:
    """Factory class for creating context compressor instances.

    Supported compressor types:
    - noop: No compression, returns original results as-is
    - qwen: Uses Qwen API for intelligent compression
    """

    @staticmethod
    def create(compressor_type: str, **kwargs) -> BaseCompressor:
        """Create a compressor instance.

        Args:
            compressor_type: Compressor type ("noop", "qwen")
            **kwargs: Compressor-specific parameters

        Returns:
            BaseCompressor instance

        Raises:
            ValueError: If compressor type is not supported

        Examples:
            >>> # Create NoOp compressor
            >>> compressor = CompressorFactory.create("noop")
            >>>
            >>> # Create Qwen compressor
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

            # Check required parameters
            if "api_key" not in kwargs:
                raise ValueError("api_key is required for Qwen compressor")

            logger.debug(f"Creating QwenCompressor with params: {list(kwargs.keys())}")
            return QwenCompressor(**kwargs)

        else:
            raise ValueError(
                f"Unknown compressor type: {compressor_type}. Supported types: noop, qwen"
            )
