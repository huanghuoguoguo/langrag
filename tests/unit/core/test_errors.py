"""
Tests for LangRAG Error Classification System.

These tests verify the error hierarchy and helper functions for:
- Error classification based on HTTP status codes
- Retryable vs permanent error detection
- Error wrapping and context preservation
"""

import pytest

from langrag.errors import (
    AuthenticationError,
    ConfigurationError,
    ConnectionError,
    ConnectTimeoutError,
    EmbeddingError,
    IndexingError,
    InvalidRequestError,
    LangRAGError,
    NotFoundError,
    PermanentError,
    QuotaExceededError,
    RateLimitError,
    ReadTimeoutError,
    RetrievalError,
    RetryableError,
    ServiceUnavailableError,
    TimeoutError,
    TransientError,
    VectorStoreError,
    classify_http_error,
    is_retryable,
    wrap_exception,
)


class TestLangRAGError:
    """Tests for base LangRAGError class."""

    def test_basic_error(self):
        """Test basic error creation."""
        error = LangRAGError("Something went wrong")
        assert str(error) == "Something went wrong"
        assert error.message == "Something went wrong"
        assert error.details == {}
        assert error.original_error is None

    def test_error_with_details(self):
        """Test error with additional details."""
        error = LangRAGError(
            "Failed to process",
            details={"doc_id": "123", "status": "failed"}
        )
        assert "doc_id" in error.details
        assert error.details["doc_id"] == "123"
        assert "Details:" in str(error)

    def test_error_with_original_error(self):
        """Test error wrapping original exception."""
        original = ValueError("Original error")
        error = LangRAGError("Wrapped error", original_error=original)
        assert error.original_error is original
        assert "Caused by:" in str(error)
        assert "ValueError" in str(error)

    def test_to_dict(self):
        """Test error serialization to dict."""
        original = RuntimeError("Runtime issue")
        error = LangRAGError(
            "Test error",
            details={"key": "value"},
            original_error=original
        )
        d = error.to_dict()

        assert d["error_type"] == "LangRAGError"
        assert d["message"] == "Test error"
        assert d["details"] == {"key": "value"}
        assert "Runtime issue" in d["original_error"]


class TestRetryableErrors:
    """Tests for retryable error types."""

    def test_rate_limit_error(self):
        """Test RateLimitError creation and attributes."""
        error = RateLimitError(retry_after=60.0)
        assert error.retry_after == 60.0
        assert is_retryable(error) is True

    def test_rate_limit_error_default_message(self):
        """Test RateLimitError default message."""
        error = RateLimitError()
        assert "rate limit" in error.message.lower()

    def test_service_unavailable_error(self):
        """Test ServiceUnavailableError."""
        error = ServiceUnavailableError(retry_after=30.0)
        assert is_retryable(error) is True
        assert error.retry_after == 30.0

    def test_connection_error(self):
        """Test ConnectionError."""
        error = ConnectionError("DNS resolution failed")
        assert is_retryable(error) is True

    def test_transient_error(self):
        """Test generic TransientError."""
        error = TransientError("Temporary failure")
        assert is_retryable(error) is True


class TestTimeoutErrors:
    """Tests for timeout-related errors."""

    def test_timeout_error(self):
        """Test basic TimeoutError."""
        error = TimeoutError(timeout=30.0, timeout_type="total")
        assert error.timeout == 30.0
        assert error.timeout_type == "total"
        assert is_retryable(error) is True  # Timeouts are retryable

    def test_connect_timeout_error(self):
        """Test ConnectTimeoutError."""
        error = ConnectTimeoutError(timeout=10.0)
        assert error.timeout_type == "connect"
        assert is_retryable(error) is True

    def test_read_timeout_error(self):
        """Test ReadTimeoutError."""
        error = ReadTimeoutError(timeout=60.0)
        assert error.timeout_type == "read"
        assert is_retryable(error) is True


class TestPermanentErrors:
    """Tests for non-retryable permanent errors."""

    def test_authentication_error(self):
        """Test AuthenticationError."""
        error = AuthenticationError("Invalid API key")
        assert is_retryable(error) is False

    def test_invalid_request_error(self):
        """Test InvalidRequestError."""
        error = InvalidRequestError("Malformed JSON")
        assert is_retryable(error) is False

    def test_not_found_error(self):
        """Test NotFoundError."""
        error = NotFoundError("Model not found")
        assert is_retryable(error) is False

    def test_configuration_error(self):
        """Test ConfigurationError."""
        error = ConfigurationError("Missing API key")
        assert is_retryable(error) is False

    def test_quota_exceeded_error(self):
        """Test QuotaExceededError."""
        error = QuotaExceededError("Monthly limit reached")
        assert is_retryable(error) is False


class TestDomainErrors:
    """Tests for domain-specific errors."""

    def test_embedding_error(self):
        """Test EmbeddingError."""
        error = EmbeddingError("Embedding generation failed")
        assert isinstance(error, LangRAGError)

    def test_retrieval_error(self):
        """Test RetrievalError."""
        error = RetrievalError("Search failed")
        assert isinstance(error, LangRAGError)

    def test_indexing_error(self):
        """Test IndexingError."""
        error = IndexingError("Index creation failed")
        assert isinstance(error, LangRAGError)

    def test_vector_store_error(self):
        """Test VectorStoreError."""
        error = VectorStoreError("Connection to vector DB failed")
        assert isinstance(error, LangRAGError)


class TestClassifyHttpError:
    """Tests for HTTP error classification."""

    def test_classify_429_rate_limit(self):
        """Test 429 status code classification."""
        error = classify_http_error(429, "Too many requests")
        assert isinstance(error, RateLimitError)
        assert is_retryable(error) is True

    def test_classify_429_with_retry_after(self):
        """Test 429 with Retry-After header."""
        error = classify_http_error(
            429,
            "Rate limited",
            headers={"Retry-After": "60"}
        )
        assert isinstance(error, RateLimitError)
        assert error.retry_after == 60.0

    def test_classify_401_authentication(self):
        """Test 401 status code classification."""
        error = classify_http_error(401, "Unauthorized")
        assert isinstance(error, AuthenticationError)
        assert is_retryable(error) is False

    def test_classify_403_forbidden(self):
        """Test 403 status code classification."""
        error = classify_http_error(403, "Forbidden")
        assert isinstance(error, AuthenticationError)

    def test_classify_400_bad_request(self):
        """Test 400 status code classification."""
        error = classify_http_error(400, "Bad request")
        assert isinstance(error, InvalidRequestError)
        assert is_retryable(error) is False

    def test_classify_404_not_found(self):
        """Test 404 status code classification."""
        error = classify_http_error(404, "Not found")
        assert isinstance(error, NotFoundError)

    def test_classify_503_service_unavailable(self):
        """Test 503 status code classification."""
        error = classify_http_error(503, "Service unavailable")
        assert isinstance(error, ServiceUnavailableError)
        assert is_retryable(error) is True

    def test_classify_500_server_error(self):
        """Test 500 status code classification."""
        error = classify_http_error(500, "Internal server error")
        assert isinstance(error, TransientError)
        assert is_retryable(error) is True

    def test_classify_unknown_4xx(self):
        """Test unknown 4xx status code."""
        error = classify_http_error(418, "I'm a teapot")
        assert isinstance(error, PermanentError)


class TestWrapException:
    """Tests for exception wrapping utility."""

    def test_wrap_timeout_exception(self):
        """Test wrapping timeout-related exceptions."""
        original = Exception("Connection timed out")
        error = wrap_exception(original, context="API call")
        assert isinstance(error, TimeoutError)
        assert error.original_error is original

    def test_wrap_connection_exception(self):
        """Test wrapping connection-related exceptions."""
        original = Exception("Connection refused")
        error = wrap_exception(original, context="Connecting")
        assert isinstance(error, ConnectionError)

    def test_wrap_rate_limit_from_message(self):
        """Test detecting rate limit from error message."""
        original = Exception("Rate limit exceeded, try again later")
        error = wrap_exception(original)
        assert isinstance(error, RateLimitError)

    def test_wrap_auth_from_message(self):
        """Test detecting auth error from message."""
        original = Exception("Invalid API key provided")
        error = wrap_exception(original)
        assert isinstance(error, AuthenticationError)

    def test_wrap_explicit_retryable(self):
        """Test explicit retryable flag."""
        original = Exception("Unknown error")
        error = wrap_exception(original, retryable=True)
        assert isinstance(error, TransientError)
        assert is_retryable(error) is True

    def test_wrap_explicit_permanent(self):
        """Test explicit permanent flag."""
        original = Exception("Unknown error")
        error = wrap_exception(original, retryable=False)
        assert isinstance(error, PermanentError)
        assert is_retryable(error) is False

    def test_wrap_unknown_defaults_to_permanent(self):
        """Test unknown errors default to permanent (safe default)."""
        original = Exception("Completely unknown error")
        error = wrap_exception(original)
        assert isinstance(error, PermanentError)

    def test_wrap_preserves_context(self):
        """Test that context is preserved in error message."""
        original = ValueError("Bad value")
        error = wrap_exception(original, context="Processing document")
        assert "Processing document" in error.message


class TestIsRetryable:
    """Tests for is_retryable helper function."""

    def test_retryable_errors_return_true(self):
        """Test retryable error types return True."""
        retryable_errors = [
            RateLimitError("rate limited"),
            ServiceUnavailableError("service down"),
            ConnectionError("connection failed"),
            TimeoutError("timed out"),
            TransientError("transient failure"),
        ]
        for error in retryable_errors:
            assert is_retryable(error) is True, f"{type(error).__name__} should be retryable"

    def test_permanent_errors_return_false(self):
        """Test permanent error types return False."""
        permanent_errors = [
            AuthenticationError("auth failed"),
            InvalidRequestError("bad request"),
            NotFoundError("not found"),
            ConfigurationError("config error"),
            QuotaExceededError("quota exceeded"),
            PermanentError("permanent error"),
        ]
        for error in permanent_errors:
            assert is_retryable(error) is False, f"{type(error).__name__} should not be retryable"

    def test_non_langrag_errors_return_false(self):
        """Test non-LangRAG errors return False."""
        assert is_retryable(ValueError("test")) is False
        assert is_retryable(RuntimeError("test")) is False
        assert is_retryable(Exception("test")) is False
