"""
LLM Configuration Module.

This module provides configuration classes for LLM operations including:
- Timeout settings for various operation types
- Retry configuration
- Rate limit handling
"""

import logging
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class TimeoutConfig:
    """
    Configuration for request timeouts.

    All timeout values are in seconds.

    Attributes:
        connect: Timeout for establishing connection
        read: Timeout for reading response
        total: Total timeout for entire request (None = unlimited)
        stream_chunk: Timeout between stream chunks (for streaming responses)

    Example:
        config = TimeoutConfig(connect=10.0, read=60.0, total=120.0)
    """

    connect: float = 10.0
    read: float = 60.0
    total: float | None = 120.0
    stream_chunk: float = 30.0

    def __post_init__(self):
        """Validate timeout values."""
        if self.connect <= 0:
            raise ValueError("connect timeout must be positive")
        if self.read <= 0:
            raise ValueError("read timeout must be positive")
        if self.total is not None and self.total <= 0:
            raise ValueError("total timeout must be positive or None")
        if self.stream_chunk <= 0:
            raise ValueError("stream_chunk timeout must be positive")

    @classmethod
    def fast(cls) -> "TimeoutConfig":
        """Preset for fast, low-latency operations."""
        return cls(connect=5.0, read=30.0, total=60.0, stream_chunk=15.0)

    @classmethod
    def standard(cls) -> "TimeoutConfig":
        """Standard preset for typical operations."""
        return cls(connect=10.0, read=60.0, total=120.0, stream_chunk=30.0)

    @classmethod
    def long_running(cls) -> "TimeoutConfig":
        """Preset for long-running operations (e.g., large embeddings)."""
        return cls(connect=15.0, read=300.0, total=600.0, stream_chunk=60.0)

    def to_httpx(self) -> tuple[float, float]:
        """Convert to httpx timeout tuple (connect, read)."""
        return (self.connect, self.read)


@dataclass
class LLMConfig:
    """
    Comprehensive configuration for LLM operations.

    Combines timeout, retry, and operational settings.

    Attributes:
        timeout: Timeout configuration
        max_retries: Maximum number of retry attempts
        retry_base_delay: Base delay between retries (seconds)
        retry_max_delay: Maximum delay between retries (seconds)
        retry_exponential_base: Exponential backoff multiplier

    Example:
        config = LLMConfig(
            timeout=TimeoutConfig.fast(),
            max_retries=3,
        )
    """

    timeout: TimeoutConfig = field(default_factory=TimeoutConfig.standard)
    max_retries: int = 3
    retry_base_delay: float = 1.0
    retry_max_delay: float = 60.0
    retry_exponential_base: float = 2.0

    def __post_init__(self):
        """Validate configuration."""
        if self.max_retries < 0:
            raise ValueError("max_retries must be non-negative")
        if self.retry_base_delay < 0:
            raise ValueError("retry_base_delay must be non-negative")
        if self.retry_max_delay < self.retry_base_delay:
            raise ValueError("retry_max_delay must be >= retry_base_delay")

    @classmethod
    def default(cls) -> "LLMConfig":
        """Create default configuration."""
        return cls()

    @classmethod
    def resilient(cls) -> "LLMConfig":
        """Configuration optimized for resilience (more retries, longer timeouts)."""
        return cls(
            timeout=TimeoutConfig.long_running(),
            max_retries=5,
            retry_base_delay=2.0,
            retry_max_delay=120.0,
        )

    @classmethod
    def fast(cls) -> "LLMConfig":
        """Configuration optimized for speed (fewer retries, shorter timeouts)."""
        return cls(
            timeout=TimeoutConfig.fast(),
            max_retries=2,
            retry_base_delay=0.5,
            retry_max_delay=10.0,
        )


# Default configurations
DEFAULT_TIMEOUT = TimeoutConfig.standard()
DEFAULT_LLM_CONFIG = LLMConfig.default()
