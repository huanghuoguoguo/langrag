"""
Tests for LLM Configuration Module.

These tests verify:
- TimeoutConfig validation and presets
- LLMConfig validation and presets
"""

import pytest

from langrag.llm.config import (
    DEFAULT_LLM_CONFIG,
    DEFAULT_TIMEOUT,
    LLMConfig,
    TimeoutConfig,
)


class TestTimeoutConfig:
    """Tests for TimeoutConfig dataclass."""

    def test_default_values(self):
        """Test default timeout values."""
        config = TimeoutConfig()
        assert config.connect == 10.0
        assert config.read == 60.0
        assert config.total == 120.0
        assert config.stream_chunk == 30.0

    def test_custom_values(self):
        """Test custom timeout values."""
        config = TimeoutConfig(
            connect=5.0,
            read=30.0,
            total=60.0,
            stream_chunk=15.0
        )
        assert config.connect == 5.0
        assert config.read == 30.0
        assert config.total == 60.0
        assert config.stream_chunk == 15.0

    def test_none_total_allowed(self):
        """Test that None total timeout is allowed (unlimited)."""
        config = TimeoutConfig(total=None)
        assert config.total is None

    def test_invalid_connect_timeout(self):
        """Test validation of connect timeout."""
        with pytest.raises(ValueError, match="connect"):
            TimeoutConfig(connect=0)
        with pytest.raises(ValueError, match="connect"):
            TimeoutConfig(connect=-1.0)

    def test_invalid_read_timeout(self):
        """Test validation of read timeout."""
        with pytest.raises(ValueError, match="read"):
            TimeoutConfig(read=0)

    def test_invalid_total_timeout(self):
        """Test validation of total timeout."""
        with pytest.raises(ValueError, match="total"):
            TimeoutConfig(total=0)

    def test_invalid_stream_chunk_timeout(self):
        """Test validation of stream_chunk timeout."""
        with pytest.raises(ValueError, match="stream_chunk"):
            TimeoutConfig(stream_chunk=0)

    def test_fast_preset(self):
        """Test fast preset configuration."""
        config = TimeoutConfig.fast()
        assert config.connect == 5.0
        assert config.read == 30.0
        assert config.total == 60.0

    def test_standard_preset(self):
        """Test standard preset configuration."""
        config = TimeoutConfig.standard()
        assert config.connect == 10.0
        assert config.read == 60.0
        assert config.total == 120.0

    def test_long_running_preset(self):
        """Test long_running preset configuration."""
        config = TimeoutConfig.long_running()
        assert config.connect == 15.0
        assert config.read == 300.0
        assert config.total == 600.0

    def test_to_httpx_conversion(self):
        """Test conversion to httpx timeout tuple."""
        config = TimeoutConfig(connect=5.0, read=30.0)
        httpx_tuple = config.to_httpx()
        assert httpx_tuple == (5.0, 30.0)


class TestLLMConfig:
    """Tests for LLMConfig dataclass."""

    def test_default_values(self):
        """Test default LLM configuration values."""
        config = LLMConfig()
        assert isinstance(config.timeout, TimeoutConfig)
        assert config.max_retries == 3
        assert config.retry_base_delay == 1.0
        assert config.retry_max_delay == 60.0
        assert config.retry_exponential_base == 2.0

    def test_custom_timeout(self):
        """Test custom timeout configuration."""
        timeout = TimeoutConfig.fast()
        config = LLMConfig(timeout=timeout)
        assert config.timeout.connect == 5.0

    def test_invalid_max_retries(self):
        """Test validation of max_retries."""
        with pytest.raises(ValueError, match="max_retries"):
            LLMConfig(max_retries=-1)

    def test_invalid_retry_base_delay(self):
        """Test validation of retry_base_delay."""
        with pytest.raises(ValueError, match="retry_base_delay"):
            LLMConfig(retry_base_delay=-1.0)

    def test_invalid_retry_max_delay(self):
        """Test retry_max_delay must be >= retry_base_delay."""
        with pytest.raises(ValueError, match="retry_max_delay"):
            LLMConfig(retry_base_delay=10.0, retry_max_delay=5.0)

    def test_default_preset(self):
        """Test default preset."""
        config = LLMConfig.default()
        assert config.max_retries == 3

    def test_resilient_preset(self):
        """Test resilient preset (more retries, longer timeouts)."""
        config = LLMConfig.resilient()
        assert config.max_retries == 5
        assert config.timeout.total == 600.0

    def test_fast_preset(self):
        """Test fast preset (fewer retries, shorter timeouts)."""
        config = LLMConfig.fast()
        assert config.max_retries == 2
        assert config.timeout.total == 60.0


class TestDefaultConfigs:
    """Tests for module-level default configurations."""

    def test_default_timeout(self):
        """Test DEFAULT_TIMEOUT is standard preset."""
        assert DEFAULT_TIMEOUT.connect == 10.0
        assert DEFAULT_TIMEOUT.read == 60.0

    def test_default_llm_config(self):
        """Test DEFAULT_LLM_CONFIG is default preset."""
        assert DEFAULT_LLM_CONFIG.max_retries == 3
