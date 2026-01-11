"""
Tests for Retry Utility with Exponential Backoff.

These tests verify:
- Basic retry functionality
- Exponential backoff calculation
- Jitter application
- Rate limit header handling
- Retry condition evaluation
- Context manager usage
"""

import time
from unittest.mock import MagicMock, patch

import pytest

from langrag.errors import (
    PermanentError,
    RateLimitError,
    RetryableError,
    TransientError,
)
from langrag.utils.retry import (
    RetryConfig,
    RetryContext,
    RetryState,
    calculate_delay,
    execute_with_retry,
    retry_with_backoff,
    should_retry,
)


class TestRetryConfig:
    """Tests for RetryConfig dataclass."""

    def test_default_config(self):
        """Test default configuration values."""
        config = RetryConfig()
        assert config.max_attempts == 3
        assert config.base_delay == 1.0
        assert config.max_delay == 60.0
        assert config.exponential_base == 2.0
        assert config.jitter == 0.1
        assert config.respect_retry_after is True

    def test_custom_config(self):
        """Test custom configuration values."""
        config = RetryConfig(
            max_attempts=5,
            base_delay=0.5,
            max_delay=30.0,
            exponential_base=3.0,
            jitter=0.2
        )
        assert config.max_attempts == 5
        assert config.base_delay == 0.5
        assert config.max_delay == 30.0

    def test_invalid_max_attempts(self):
        """Test validation of max_attempts."""
        with pytest.raises(ValueError, match="max_attempts"):
            RetryConfig(max_attempts=0)

    def test_invalid_base_delay(self):
        """Test validation of base_delay."""
        with pytest.raises(ValueError, match="base_delay"):
            RetryConfig(base_delay=-1.0)

    def test_invalid_max_delay(self):
        """Test max_delay must be >= base_delay."""
        with pytest.raises(ValueError, match="max_delay"):
            RetryConfig(base_delay=10.0, max_delay=5.0)

    def test_invalid_jitter(self):
        """Test jitter must be between 0 and 1."""
        with pytest.raises(ValueError, match="jitter"):
            RetryConfig(jitter=1.5)
        with pytest.raises(ValueError, match="jitter"):
            RetryConfig(jitter=-0.1)


class TestCalculateDelay:
    """Tests for delay calculation."""

    def test_first_attempt_delay(self):
        """Test delay for first attempt (no backoff)."""
        config = RetryConfig(base_delay=1.0, jitter=0.0)
        delay = calculate_delay(1, config)
        assert delay == 1.0

    def test_exponential_backoff(self):
        """Test exponential increase in delay."""
        config = RetryConfig(base_delay=1.0, exponential_base=2.0, jitter=0.0)

        delay1 = calculate_delay(1, config)
        delay2 = calculate_delay(2, config)
        delay3 = calculate_delay(3, config)

        assert delay1 == 1.0  # 1 * 2^0
        assert delay2 == 2.0  # 1 * 2^1
        assert delay3 == 4.0  # 1 * 2^2

    def test_max_delay_cap(self):
        """Test delay is capped at max_delay."""
        config = RetryConfig(
            base_delay=10.0,
            max_delay=30.0,
            exponential_base=2.0,
            jitter=0.0
        )

        # 10 * 2^4 = 160, but should be capped at 30
        delay = calculate_delay(5, config)
        assert delay == 30.0

    def test_jitter_applied(self):
        """Test jitter adds randomness to delay."""
        config = RetryConfig(base_delay=10.0, jitter=0.5)

        # Run multiple times and check variance
        delays = [calculate_delay(1, config) for _ in range(100)]

        # With 50% jitter, delays should vary between 5 and 15
        assert min(delays) >= 5.0
        assert max(delays) <= 15.0
        # And there should be some variance
        assert max(delays) - min(delays) > 1.0

    def test_retry_after_header_respected(self):
        """Test Retry-After from RateLimitError is used."""
        config = RetryConfig(base_delay=1.0, respect_retry_after=True)
        error = RateLimitError(retry_after=45.0)

        delay = calculate_delay(1, config, error)
        assert delay == 45.0

    def test_retry_after_capped_at_max(self):
        """Test Retry-After is capped at max_delay."""
        config = RetryConfig(base_delay=1.0, max_delay=30.0, respect_retry_after=True)
        error = RateLimitError(retry_after=60.0)

        delay = calculate_delay(1, config, error)
        assert delay == 30.0


class TestShouldRetry:
    """Tests for retry condition evaluation."""

    def test_max_attempts_reached(self):
        """Test no retry when max attempts reached."""
        config = RetryConfig(max_attempts=3)
        error = TransientError("temp failure")

        assert should_retry(error, 3, config) is False

    def test_retryable_error_within_attempts(self):
        """Test retry allowed for retryable errors."""
        config = RetryConfig(max_attempts=3)
        error = TransientError("temp failure")

        assert should_retry(error, 1, config) is True
        assert should_retry(error, 2, config) is True

    def test_permanent_error_never_retried(self):
        """Test permanent errors are never retried."""
        config = RetryConfig(max_attempts=5)
        error = PermanentError("permanent failure")

        assert should_retry(error, 1, config) is False

    def test_custom_retry_on_exceptions(self):
        """Test custom retry_on exception types."""
        config = RetryConfig(
            max_attempts=3,
            retry_on=(ValueError,)
        )

        assert should_retry(ValueError("test"), 1, config) is True
        assert should_retry(RuntimeError("test"), 1, config) is False

    def test_custom_stop_on_exceptions(self):
        """Test custom stop_on exception types."""
        config = RetryConfig(
            max_attempts=3,
            retry_on=(Exception,),
            stop_on=(KeyError,)
        )

        assert should_retry(ValueError("test"), 1, config) is True
        assert should_retry(KeyError("test"), 1, config) is False


class TestRetryWithBackoff:
    """Tests for retry_with_backoff decorator."""

    def test_success_on_first_attempt(self):
        """Test function succeeds on first attempt."""
        call_count = 0

        @retry_with_backoff(max_attempts=3)
        def always_succeed():
            nonlocal call_count
            call_count += 1
            return "success"

        result = always_succeed()
        assert result == "success"
        assert call_count == 1

    def test_retry_on_failure(self):
        """Test function is retried on failure."""
        call_count = 0

        @retry_with_backoff(max_attempts=3, base_delay=0.01)
        def fail_then_succeed():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise TransientError("temporary")
            return "success"

        result = fail_then_succeed()
        assert result == "success"
        assert call_count == 3

    def test_all_retries_exhausted(self):
        """Test exception raised when all retries fail."""
        call_count = 0

        @retry_with_backoff(max_attempts=3, base_delay=0.01)
        def always_fail():
            nonlocal call_count
            call_count += 1
            raise TransientError("always fails")

        with pytest.raises(TransientError, match="always fails"):
            always_fail()

        assert call_count == 3

    def test_no_retry_on_permanent_error(self):
        """Test permanent errors are not retried."""
        call_count = 0

        @retry_with_backoff(max_attempts=5, base_delay=0.01)
        def raise_permanent():
            nonlocal call_count
            call_count += 1
            raise PermanentError("permanent")

        with pytest.raises(PermanentError):
            raise_permanent()

        assert call_count == 1

    def test_on_retry_callback(self):
        """Test on_retry callback is invoked."""
        retries = []

        def track_retry(attempt, error, delay):
            retries.append({"attempt": attempt, "error": str(error), "delay": delay})

        @retry_with_backoff(max_attempts=3, base_delay=0.01, on_retry=track_retry)
        def fail_twice():
            if len(retries) < 2:
                raise TransientError("temporary")
            return "success"

        fail_twice()
        assert len(retries) == 2
        assert retries[0]["attempt"] == 1
        assert retries[1]["attempt"] == 2

    def test_decorator_without_parentheses(self):
        """Test decorator can be used without parentheses."""
        @retry_with_backoff
        def simple_func():
            return "result"

        assert simple_func() == "result"

    def test_decorator_with_config_object(self):
        """Test decorator accepts RetryConfig object."""
        config = RetryConfig(max_attempts=2, base_delay=0.01)

        call_count = 0

        @retry_with_backoff(config=config)
        def test_func():
            nonlocal call_count
            call_count += 1
            if call_count < 2:
                raise TransientError("retry me")
            return "done"

        result = test_func()
        assert result == "done"
        assert call_count == 2


class TestExecuteWithRetry:
    """Tests for execute_with_retry function."""

    def test_basic_execution(self):
        """Test basic function execution with retry."""
        def add(a, b):
            return a + b

        result = execute_with_retry(add, 2, 3)
        assert result == 5

    def test_with_custom_config(self):
        """Test execution with custom config."""
        call_count = 0

        def flaky():
            nonlocal call_count
            call_count += 1
            if call_count < 2:
                raise TransientError("flaky")
            return "success"

        config = RetryConfig(max_attempts=3, base_delay=0.01)
        result = execute_with_retry(flaky, config=config)

        assert result == "success"
        assert call_count == 2

    def test_with_kwargs(self):
        """Test execution with keyword arguments."""
        def greet(name, greeting="Hello"):
            return f"{greeting}, {name}!"

        result = execute_with_retry(greet, "World", greeting="Hi")
        assert result == "Hi, World!"


class TestRetryContext:
    """Tests for RetryContext context manager."""

    def test_basic_context_manager(self):
        """Test basic context manager usage."""
        with RetryContext(max_attempts=3) as ctx:
            assert ctx.state.attempt == 0
            assert ctx.attempts_remaining == 3

    def test_iterate_attempts(self):
        """Test iterating through attempts."""
        attempts_seen = []

        with RetryContext(max_attempts=3) as ctx:
            for attempt in ctx:
                attempts_seen.append(attempt)
                if attempt == 2:
                    break

        assert attempts_seen == [1, 2]

    def test_should_retry_tracking(self):
        """Test should_retry tracks errors."""
        with RetryContext(max_attempts=3) as ctx:
            error = TransientError("test")

            assert ctx.should_retry(error) is True
            assert len(ctx.state.errors) == 1

    def test_wait_updates_total_delay(self):
        """Test wait() updates total delay."""
        with RetryContext(max_attempts=3, base_delay=0.01) as ctx:
            ctx.state.attempt = 1
            delay = ctx.wait()

            assert delay > 0
            assert ctx.state.total_delay > 0

    def test_attempts_remaining(self):
        """Test attempts_remaining calculation."""
        with RetryContext(max_attempts=5) as ctx:
            ctx.state.attempt = 2
            assert ctx.attempts_remaining == 3


class TestRetryState:
    """Tests for RetryState dataclass."""

    def test_default_state(self):
        """Test default state values."""
        state = RetryState()
        assert state.attempt == 0
        assert state.total_delay == 0.0
        assert state.errors == []

    def test_elapsed_time(self):
        """Test elapsed time calculation."""
        state = RetryState()
        time.sleep(0.01)
        assert state.elapsed_time >= 0.01

    def test_errors_list(self):
        """Test error tracking."""
        state = RetryState()
        state.errors.append(ValueError("error 1"))
        state.errors.append(RuntimeError("error 2"))
        assert len(state.errors) == 2
