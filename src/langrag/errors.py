"""
LangRAG Unified Error Classification System.

This module provides a hierarchy of exceptions for handling different types
of errors that can occur during LLM and retrieval operations.

Error Categories:
-----------------
1. Retryable Errors: Transient failures that may succeed on retry
   - Rate limiting (HTTP 429)
   - Service unavailable (HTTP 503)
   - Connection timeouts
   - Network errors

2. Permanent Errors: Failures that won't succeed on retry
   - Authentication errors (HTTP 401)
   - Invalid parameters (HTTP 400)
   - Not found (HTTP 404)
   - Configuration errors

3. Timeout Errors: Request exceeded time limits
   - Connection timeout
   - Read timeout
   - Total request timeout

Usage:
------
    from langrag.errors import (
        LangRAGError,
        RetryableError,
        RateLimitError,
        TimeoutError,
        PermanentError,
        AuthenticationError,
    )

    try:
        result = llm.chat(messages)
    except RateLimitError as e:
        # Wait and retry with backoff
        logger.warning(f"Rate limited, retry after {e.retry_after}s")
    except TimeoutError as e:
        # Log and possibly retry with longer timeout
        logger.error(f"Request timed out after {e.timeout}s")
    except PermanentError as e:
        # Don't retry, fix the underlying issue
        logger.error(f"Permanent error: {e}")
"""

from typing import Any


class LangRAGError(Exception):
    """
    Base exception for all LangRAG errors.

    All custom exceptions in LangRAG should inherit from this class
    to allow catching all LangRAG-specific errors with a single except clause.

    Attributes:
        message: Human-readable error description
        details: Additional context about the error (optional)
        original_error: The underlying exception that caused this error (optional)
    """

    def __init__(
        self,
        message: str,
        details: dict[str, Any] | None = None,
        original_error: Exception | None = None
    ):
        super().__init__(message)
        self.message = message
        self.details = details or {}
        self.original_error = original_error

    def __str__(self) -> str:
        base = self.message
        if self.details:
            base += f" | Details: {self.details}"
        if self.original_error:
            base += f" | Caused by: {type(self.original_error).__name__}: {self.original_error}"
        return base

    def to_dict(self) -> dict[str, Any]:
        """Convert error to dictionary for logging/serialization."""
        return {
            "error_type": type(self).__name__,
            "message": self.message,
            "details": self.details,
            "original_error": str(self.original_error) if self.original_error else None
        }


# =============================================================================
# Retryable Errors - Transient failures that may succeed on retry
# =============================================================================

class RetryableError(LangRAGError):
    """
    Base class for errors that may succeed on retry.

    These are transient failures such as:
    - Network connectivity issues
    - Rate limiting
    - Service temporarily unavailable

    Attributes:
        retry_after: Suggested wait time before retry (seconds), if known
    """

    def __init__(
        self,
        message: str,
        retry_after: float | None = None,
        details: dict[str, Any] | None = None,
        original_error: Exception | None = None
    ):
        super().__init__(message, details, original_error)
        self.retry_after = retry_after


class RateLimitError(RetryableError):
    """
    Raised when API rate limit is exceeded (HTTP 429).

    This error indicates that too many requests have been made in a given
    time period. The retry_after attribute contains the recommended wait
    time from the Retry-After header, if provided by the server.

    Example:
        try:
            response = llm.chat(messages)
        except RateLimitError as e:
            if e.retry_after:
                time.sleep(e.retry_after)
            else:
                time.sleep(60)  # Default backoff
    """

    def __init__(
        self,
        message: str = "API rate limit exceeded",
        retry_after: float | None = None,
        details: dict[str, Any] | None = None,
        original_error: Exception | None = None
    ):
        super().__init__(message, retry_after, details, original_error)


class ServiceUnavailableError(RetryableError):
    """
    Raised when service is temporarily unavailable (HTTP 503).

    The service might be overloaded or undergoing maintenance.
    Retry with exponential backoff is recommended.
    """

    def __init__(
        self,
        message: str = "Service temporarily unavailable",
        retry_after: float | None = None,
        details: dict[str, Any] | None = None,
        original_error: Exception | None = None
    ):
        super().__init__(message, retry_after, details, original_error)


class ConnectionError(RetryableError):
    """
    Raised when connection to service fails.

    This includes:
    - DNS resolution failures
    - Connection refused
    - Network unreachable
    """

    def __init__(
        self,
        message: str = "Failed to connect to service",
        details: dict[str, Any] | None = None,
        original_error: Exception | None = None
    ):
        super().__init__(message, retry_after=None, details=details, original_error=original_error)


class TransientError(RetryableError):
    """
    Generic retryable error for unclassified transient failures.

    Use this when the error is known to be transient but doesn't fit
    other specific categories.
    """
    pass


# =============================================================================
# Timeout Errors - Request exceeded time limits
# =============================================================================

class TimeoutError(RetryableError):
    """
    Base class for timeout-related errors.

    Timeouts are generally retryable but may indicate:
    - Network latency issues
    - Server overload
    - Request too large/complex

    Attributes:
        timeout: The timeout value that was exceeded (seconds)
        timeout_type: Type of timeout ('connect', 'read', 'total')
    """

    def __init__(
        self,
        message: str = "Request timed out",
        timeout: float | None = None,
        timeout_type: str = "total",
        details: dict[str, Any] | None = None,
        original_error: Exception | None = None
    ):
        details = details or {}
        details["timeout"] = timeout
        details["timeout_type"] = timeout_type
        super().__init__(message, retry_after=None, details=details, original_error=original_error)
        self.timeout = timeout
        self.timeout_type = timeout_type


class ConnectTimeoutError(TimeoutError):
    """Raised when connection attempt times out."""

    def __init__(
        self,
        message: str = "Connection timed out",
        timeout: float | None = None,
        details: dict[str, Any] | None = None,
        original_error: Exception | None = None
    ):
        super().__init__(message, timeout, "connect", details, original_error)


class ReadTimeoutError(TimeoutError):
    """Raised when reading response times out."""

    def __init__(
        self,
        message: str = "Read timed out while waiting for response",
        timeout: float | None = None,
        details: dict[str, Any] | None = None,
        original_error: Exception | None = None
    ):
        super().__init__(message, timeout, "read", details, original_error)


# =============================================================================
# Permanent Errors - Failures that won't succeed on retry
# =============================================================================

class PermanentError(LangRAGError):
    """
    Base class for errors that will not succeed on retry.

    These errors indicate issues that require user intervention:
    - Invalid credentials
    - Malformed requests
    - Resource not found
    - Configuration problems

    Retrying these errors is wasteful and may trigger rate limiting.
    """
    pass


class AuthenticationError(PermanentError):
    """
    Raised when authentication fails (HTTP 401/403).

    Common causes:
    - Invalid API key
    - Expired token
    - Insufficient permissions
    """

    def __init__(
        self,
        message: str = "Authentication failed",
        details: dict[str, Any] | None = None,
        original_error: Exception | None = None
    ):
        super().__init__(message, details, original_error)


class InvalidRequestError(PermanentError):
    """
    Raised when request parameters are invalid (HTTP 400).

    Common causes:
    - Malformed JSON
    - Invalid model name
    - Parameter out of range
    - Missing required field
    """

    def __init__(
        self,
        message: str = "Invalid request parameters",
        details: dict[str, Any] | None = None,
        original_error: Exception | None = None
    ):
        super().__init__(message, details, original_error)


class NotFoundError(PermanentError):
    """
    Raised when requested resource is not found (HTTP 404).

    Common causes:
    - Invalid model ID
    - Deleted resource
    - Wrong endpoint
    """

    def __init__(
        self,
        message: str = "Resource not found",
        details: dict[str, Any] | None = None,
        original_error: Exception | None = None
    ):
        super().__init__(message, details, original_error)


class ConfigurationError(PermanentError):
    """
    Raised when there's a configuration problem.

    Common causes:
    - Missing required configuration
    - Invalid configuration values
    - Incompatible settings
    """

    def __init__(
        self,
        message: str = "Configuration error",
        details: dict[str, Any] | None = None,
        original_error: Exception | None = None
    ):
        super().__init__(message, details, original_error)


class QuotaExceededError(PermanentError):
    """
    Raised when account quota is exceeded.

    Unlike RateLimitError, this indicates the account has exhausted
    its allocation (e.g., monthly token limit) and requires action
    like upgrading the plan or waiting for quota reset.
    """

    def __init__(
        self,
        message: str = "Account quota exceeded",
        details: dict[str, Any] | None = None,
        original_error: Exception | None = None
    ):
        super().__init__(message, details, original_error)


# =============================================================================
# Domain-Specific Errors
# =============================================================================

class EmbeddingError(LangRAGError):
    """Raised when embedding operation fails."""
    pass


class RetrievalError(LangRAGError):
    """Raised when retrieval operation fails."""
    pass


class IndexingError(LangRAGError):
    """Raised when indexing operation fails."""
    pass


class VectorStoreError(LangRAGError):
    """Raised when vector store operation fails."""
    pass


# =============================================================================
# Helper Functions
# =============================================================================

def is_retryable(error: Exception) -> bool:
    """
    Check if an error is retryable.

    Args:
        error: The exception to check

    Returns:
        True if the error is an instance of RetryableError

    Example:
        try:
            result = llm.chat(messages)
        except Exception as e:
            if is_retryable(e):
                # Apply retry logic
                pass
            else:
                # Don't retry, handle or propagate
                raise
    """
    return isinstance(error, RetryableError)


def classify_http_error(status_code: int, message: str = "", headers: dict | None = None) -> LangRAGError:
    """
    Classify an HTTP error based on status code.

    Args:
        status_code: HTTP status code
        message: Error message from response
        headers: Response headers (used to extract Retry-After)

    Returns:
        Appropriate LangRAGError subclass instance

    Example:
        if response.status_code >= 400:
            raise classify_http_error(
                response.status_code,
                response.text,
                dict(response.headers)
            )
    """
    headers = headers or {}
    retry_after = None

    # Extract Retry-After header if present
    if "Retry-After" in headers:
        try:
            retry_after = float(headers["Retry-After"])
        except (ValueError, TypeError):
            pass

    details = {"status_code": status_code}

    if status_code == 429:
        return RateLimitError(
            message=message or "API rate limit exceeded",
            retry_after=retry_after,
            details=details
        )
    elif status_code == 401:
        return AuthenticationError(
            message=message or "Authentication failed - invalid API key",
            details=details
        )
    elif status_code == 403:
        return AuthenticationError(
            message=message or "Access forbidden - insufficient permissions",
            details=details
        )
    elif status_code == 400:
        return InvalidRequestError(
            message=message or "Invalid request parameters",
            details=details
        )
    elif status_code == 404:
        return NotFoundError(
            message=message or "Resource not found",
            details=details
        )
    elif status_code == 503:
        return ServiceUnavailableError(
            message=message or "Service temporarily unavailable",
            retry_after=retry_after,
            details=details
        )
    elif status_code >= 500:
        return TransientError(
            message=message or f"Server error (HTTP {status_code})",
            details=details
        )
    else:
        return PermanentError(
            message=message or f"HTTP error {status_code}",
            details=details
        )


def wrap_exception(
    error: Exception,
    context: str = "",
    retryable: bool | None = None
) -> LangRAGError:
    """
    Wrap a generic exception in an appropriate LangRAGError.

    This function examines the exception type and message to determine
    the most appropriate LangRAGError subclass.

    Args:
        error: The original exception
        context: Additional context about where the error occurred
        retryable: Override retryability detection (None = auto-detect)

    Returns:
        LangRAGError instance wrapping the original error

    Example:
        try:
            response = httpx.post(url, json=data)
        except Exception as e:
            raise wrap_exception(e, context="LLM chat request")
    """
    error_str = str(error).lower()
    error_type = type(error).__name__

    message = f"{context}: {error}" if context else str(error)

    # Timeout detection
    if "timeout" in error_str or "timed out" in error_str:
        if "connect" in error_str:
            return ConnectTimeoutError(message=message, original_error=error)
        elif "read" in error_str:
            return ReadTimeoutError(message=message, original_error=error)
        else:
            return TimeoutError(message=message, original_error=error)

    # Connection detection
    if any(x in error_str for x in ["connection", "connect", "network", "dns"]):
        return ConnectionError(message=message, original_error=error)

    # Rate limit detection (from error messages)
    if any(x in error_str for x in ["rate limit", "too many requests", "429"]):
        return RateLimitError(message=message, original_error=error)

    # Auth detection
    if any(x in error_str for x in ["auth", "api key", "credential", "401", "403"]):
        return AuthenticationError(message=message, original_error=error)

    # If retryability is explicitly specified
    if retryable is True:
        return TransientError(message=message, original_error=error)
    elif retryable is False:
        return PermanentError(message=message, original_error=error)

    # Default to transient for common network libraries
    if error_type in ("ConnectionError", "ConnectError", "TimeoutException"):
        return TransientError(message=message, original_error=error)

    # Default to permanent for unknown errors (safer - avoids infinite retry)
    return PermanentError(message=message, original_error=error)
