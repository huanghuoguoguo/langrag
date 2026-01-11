"""
Tests for LLM Resilience Scenarios.

These tests verify that the system handles various LLM failure scenarios:
- Rate limiting (429 errors)
- Timeouts
- Service unavailability
- Authentication errors
- Network errors

These are integration-style tests that verify the complete error handling flow.
"""

import time
from unittest.mock import MagicMock, patch

import pytest

from langrag.errors import (
    AuthenticationError,
    PermanentError,
    RateLimitError,
    ReadTimeoutError,
    RetryableError,
    ServiceUnavailableError,
    TimeoutError,
    TransientError,
    classify_http_error,
    is_retryable,
)
from langrag.utils.retry import RetryConfig, retry_with_backoff


class TestRateLimitHandling:
    """Tests for rate limit (429) error handling."""

    def test_rate_limit_is_retryable(self):
        """Verify rate limit errors are classified as retryable."""
        error = RateLimitError("Too many requests", retry_after=60.0)
        assert is_retryable(error) is True

    def test_rate_limit_from_http_status(self):
        """Verify 429 status code creates RateLimitError."""
        error = classify_http_error(429, "Rate limit exceeded")
        assert isinstance(error, RateLimitError)

    def test_rate_limit_respects_retry_after(self):
        """Verify Retry-After header is respected."""
        error = classify_http_error(
            429,
            "Too many requests",
            headers={"Retry-After": "30"}
        )
        assert error.retry_after == 30.0

    def test_retry_on_rate_limit(self):
        """Test function retries on rate limit error."""
        call_count = 0
        rate_limited_until = 2

        @retry_with_backoff(max_attempts=5, base_delay=0.01)
        def rate_limited_api():
            nonlocal call_count
            call_count += 1
            if call_count < rate_limited_until:
                raise RateLimitError("Rate limited", retry_after=0.01)
            return "success"

        result = rate_limited_api()
        assert result == "success"
        assert call_count == rate_limited_until


class TestTimeoutHandling:
    """Tests for timeout error handling."""

    def test_timeout_is_retryable(self):
        """Verify timeout errors are classified as retryable."""
        error = TimeoutError("Request timed out", timeout=30.0)
        assert is_retryable(error) is True

    def test_read_timeout_is_retryable(self):
        """Verify read timeout is retryable."""
        error = ReadTimeoutError("Read timed out", timeout=60.0)
        assert is_retryable(error) is True

    def test_retry_on_timeout(self):
        """Test function retries on timeout."""
        call_count = 0

        @retry_with_backoff(max_attempts=3, base_delay=0.01)
        def slow_api():
            nonlocal call_count
            call_count += 1
            if call_count < 2:
                raise TimeoutError("Timed out", timeout=30.0)
            return "success"

        result = slow_api()
        assert result == "success"
        assert call_count == 2


class TestServiceUnavailabilityHandling:
    """Tests for service unavailability (503) handling."""

    def test_503_is_retryable(self):
        """Verify 503 errors are classified as retryable."""
        error = classify_http_error(503, "Service unavailable")
        assert isinstance(error, ServiceUnavailableError)
        assert is_retryable(error) is True

    def test_500_is_retryable(self):
        """Verify 500 errors are classified as retryable."""
        error = classify_http_error(500, "Internal server error")
        assert isinstance(error, TransientError)
        assert is_retryable(error) is True

    def test_retry_on_service_unavailable(self):
        """Test function retries on service unavailable."""
        call_count = 0

        @retry_with_backoff(max_attempts=4, base_delay=0.01)
        def unavailable_service():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise ServiceUnavailableError("Service down", retry_after=0.01)
            return "recovered"

        result = unavailable_service()
        assert result == "recovered"
        assert call_count == 3


class TestAuthenticationErrorHandling:
    """Tests for authentication (401/403) error handling."""

    def test_401_not_retryable(self):
        """Verify 401 errors are NOT retryable."""
        error = classify_http_error(401, "Unauthorized")
        assert isinstance(error, AuthenticationError)
        assert is_retryable(error) is False

    def test_403_not_retryable(self):
        """Verify 403 errors are NOT retryable."""
        error = classify_http_error(403, "Forbidden")
        assert isinstance(error, AuthenticationError)
        assert is_retryable(error) is False

    def test_no_retry_on_auth_error(self):
        """Test function does NOT retry on auth error."""
        call_count = 0

        @retry_with_backoff(max_attempts=5, base_delay=0.01)
        def auth_failing_api():
            nonlocal call_count
            call_count += 1
            raise AuthenticationError("Invalid API key")

        with pytest.raises(AuthenticationError):
            auth_failing_api()

        # Should only be called once - no retries
        assert call_count == 1


class TestBadRequestErrorHandling:
    """Tests for bad request (400) error handling."""

    def test_400_not_retryable(self):
        """Verify 400 errors are NOT retryable."""
        error = classify_http_error(400, "Bad request")
        assert isinstance(error, PermanentError)
        assert is_retryable(error) is False

    def test_no_retry_on_bad_request(self):
        """Test function does NOT retry on bad request."""
        call_count = 0

        @retry_with_backoff(max_attempts=5, base_delay=0.01)
        def bad_request_api():
            nonlocal call_count
            call_count += 1
            error = classify_http_error(400, "Invalid parameters")
            raise error

        with pytest.raises(PermanentError):
            bad_request_api()

        assert call_count == 1


class TestExponentialBackoff:
    """Tests for exponential backoff behavior."""

    def test_delays_increase_exponentially(self):
        """Verify delays increase with each retry."""
        delays = []
        call_count = 0

        def track_delay(attempt, error, delay):
            delays.append(delay)

        @retry_with_backoff(
            max_attempts=4,
            base_delay=0.1,
            exponential_base=2.0,
            jitter=0.0,
            on_retry=track_delay
        )
        def failing_api():
            nonlocal call_count
            call_count += 1
            if call_count < 4:
                raise TransientError("Temporary failure")
            return "success"

        failing_api()

        # Delays should be approximately: 0.1, 0.2, 0.4
        assert len(delays) == 3
        assert delays[0] == pytest.approx(0.1, rel=0.01)
        assert delays[1] == pytest.approx(0.2, rel=0.01)
        assert delays[2] == pytest.approx(0.4, rel=0.01)

    def test_max_delay_cap(self):
        """Verify delays are capped at max_delay."""
        delays = []

        def track_delay(attempt, error, delay):
            delays.append(delay)

        @retry_with_backoff(
            max_attempts=5,
            base_delay=0.1,
            max_delay=0.2,
            exponential_base=2.0,
            jitter=0.0,
            on_retry=track_delay
        )
        def always_fail():
            raise TransientError("Always fails")

        with pytest.raises(TransientError):
            always_fail()

        # All delays after first should be capped at 0.2
        for delay in delays[1:]:
            assert delay <= 0.2


class TestPartialFailureRecovery:
    """Tests for graceful degradation on partial failures."""

    def test_recover_after_transient_failures(self):
        """Test system recovers after multiple transient failures."""
        call_count = 0
        failures_before_success = 3

        @retry_with_backoff(max_attempts=5, base_delay=0.01)
        def flaky_service():
            nonlocal call_count
            call_count += 1
            if call_count <= failures_before_success:
                raise TransientError(f"Failure {call_count}")
            return f"Success after {call_count} attempts"

        result = flaky_service()
        assert "Success" in result
        assert call_count == failures_before_success + 1

    def test_exhaust_retries_on_persistent_failure(self):
        """Test all retries are exhausted on persistent failure."""
        call_count = 0
        max_attempts = 4

        @retry_with_backoff(max_attempts=max_attempts, base_delay=0.01)
        def always_failing():
            nonlocal call_count
            call_count += 1
            raise TransientError("Persistent failure")

        with pytest.raises(TransientError):
            always_failing()

        assert call_count == max_attempts


class TestMixedErrorScenarios:
    """Tests for mixed error type scenarios."""

    def test_retryable_then_permanent_error(self):
        """Test retryable error followed by permanent stops retrying."""
        call_count = 0

        @retry_with_backoff(max_attempts=5, base_delay=0.01)
        def mixed_errors():
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise TransientError("Temporary")
            else:
                raise PermanentError("Permanent")

        with pytest.raises(PermanentError):
            mixed_errors()

        # Should stop after permanent error (2 calls: 1 transient + 1 permanent)
        assert call_count == 2

    def test_different_retryable_errors(self):
        """Test different retryable error types are all retried."""
        call_count = 0

        @retry_with_backoff(max_attempts=5, base_delay=0.01)
        def various_errors():
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise RateLimitError("Rate limited")
            elif call_count == 2:
                raise TimeoutError("Timed out")
            elif call_count == 3:
                raise ServiceUnavailableError("Unavailable")
            return "success"

        result = various_errors()
        assert result == "success"
        assert call_count == 4
