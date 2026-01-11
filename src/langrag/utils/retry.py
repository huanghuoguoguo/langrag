"""
Retry Utility with Exponential Backoff.

This module provides a flexible retry mechanism for handling transient failures
in LLM and external service calls.

Features:
---------
- Exponential backoff with jitter
- Configurable retry conditions
- Detailed logging of retry attempts
- Support for both sync and async operations
- Respects Retry-After headers

Usage:
------
    from langrag.utils.retry import retry_with_backoff, RetryConfig

    # Using decorator
    @retry_with_backoff(max_attempts=3, base_delay=1.0)
    def call_llm(messages):
        return client.chat(messages)

    # Using context manager style
    config = RetryConfig(max_attempts=3)
    result = retry_with_backoff(call_llm, config=config)(messages)

    # With custom retry condition
    config = RetryConfig(
        max_attempts=5,
        retry_on=(RateLimitError, TimeoutError),
        on_retry=lambda attempt, error, delay: logger.warning(f"Retry {attempt}")
    )
"""

import logging
import random
import time
from dataclasses import dataclass, field
from functools import wraps
from typing import Any, Callable, TypeVar

from langrag.errors import (
    LangRAGError,
    PermanentError,
    RateLimitError,
    RetryableError,
    is_retryable,
)

logger = logging.getLogger(__name__)

T = TypeVar("T")

# Default exceptions that are considered retryable
DEFAULT_RETRYABLE_EXCEPTIONS: tuple[type[Exception], ...] = (
    RetryableError,
    ConnectionError,
    OSError,  # Includes network errors
)


@dataclass
class RetryConfig:
    """
    Configuration for retry behavior.

    Attributes:
        max_attempts: Maximum number of attempts (including initial)
        base_delay: Initial delay between retries (seconds)
        max_delay: Maximum delay between retries (seconds)
        exponential_base: Base for exponential backoff (delay = base_delay * exponential_base^attempt)
        jitter: Add random jitter to delays (0.0 to 1.0, fraction of delay)
        retry_on: Tuple of exception types to retry on
        stop_on: Tuple of exception types to never retry on
        on_retry: Callback called before each retry (attempt, error, delay) -> None
        respect_retry_after: Honor Retry-After from RateLimitError
    """

    max_attempts: int = 3
    base_delay: float = 1.0
    max_delay: float = 60.0
    exponential_base: float = 2.0
    jitter: float = 0.1
    retry_on: tuple[type[Exception], ...] = DEFAULT_RETRYABLE_EXCEPTIONS
    stop_on: tuple[type[Exception], ...] = (PermanentError,)
    on_retry: Callable[[int, Exception, float], None] | None = None
    respect_retry_after: bool = True

    def __post_init__(self):
        """Validate configuration values."""
        if self.max_attempts < 1:
            raise ValueError("max_attempts must be at least 1")
        if self.base_delay < 0:
            raise ValueError("base_delay must be non-negative")
        if self.max_delay < self.base_delay:
            raise ValueError("max_delay must be >= base_delay")
        if not 0 <= self.jitter <= 1:
            raise ValueError("jitter must be between 0 and 1")


@dataclass
class RetryState:
    """
    Tracks the state of a retry operation.

    Useful for logging and debugging retry behavior.
    """

    attempt: int = 0
    total_delay: float = 0.0
    errors: list[Exception] = field(default_factory=list)
    start_time: float = field(default_factory=time.time)

    @property
    def elapsed_time(self) -> float:
        """Total elapsed time since first attempt."""
        return time.time() - self.start_time


def calculate_delay(
    attempt: int,
    config: RetryConfig,
    error: Exception | None = None
) -> float:
    """
    Calculate delay before next retry attempt.

    Args:
        attempt: Current attempt number (1-based)
        config: Retry configuration
        error: The exception that triggered the retry

    Returns:
        Delay in seconds before next attempt
    """
    # Check for Retry-After from rate limit errors
    if config.respect_retry_after and isinstance(error, RateLimitError):
        if error.retry_after is not None and error.retry_after > 0:
            logger.debug(f"Using Retry-After header: {error.retry_after}s")
            return min(error.retry_after, config.max_delay)

    # Calculate exponential backoff
    delay = config.base_delay * (config.exponential_base ** (attempt - 1))

    # Apply jitter
    if config.jitter > 0:
        jitter_range = delay * config.jitter
        delay += random.uniform(-jitter_range, jitter_range)

    # Clamp to max_delay
    delay = min(delay, config.max_delay)
    delay = max(delay, 0)  # Ensure non-negative

    return delay


def should_retry(
    error: Exception,
    attempt: int,
    config: RetryConfig
) -> bool:
    """
    Determine if an error should trigger a retry.

    Args:
        error: The exception that was raised
        attempt: Current attempt number
        config: Retry configuration

    Returns:
        True if should retry, False otherwise
    """
    # Check max attempts
    if attempt >= config.max_attempts:
        logger.debug(f"Max attempts ({config.max_attempts}) reached, not retrying")
        return False

    # Check stop_on (never retry these)
    if isinstance(error, config.stop_on):
        logger.debug(f"Error type {type(error).__name__} in stop_on list, not retrying")
        return False

    # Check retry_on (only retry these)
    if isinstance(error, config.retry_on):
        return True

    # Check if it's a LangRAG retryable error
    if is_retryable(error):
        return True

    # Default: don't retry unknown errors
    logger.debug(f"Error type {type(error).__name__} not in retry_on list, not retrying")
    return False


def retry_with_backoff(
    func: Callable[..., T] | None = None,
    *,
    max_attempts: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 60.0,
    exponential_base: float = 2.0,
    jitter: float = 0.1,
    retry_on: tuple[type[Exception], ...] = DEFAULT_RETRYABLE_EXCEPTIONS,
    stop_on: tuple[type[Exception], ...] = (PermanentError,),
    on_retry: Callable[[int, Exception, float], None] | None = None,
    config: RetryConfig | None = None,
) -> Callable[..., T] | Callable[[Callable[..., T]], Callable[..., T]]:
    """
    Decorator that retries a function with exponential backoff.

    Can be used with or without arguments:

        @retry_with_backoff
        def my_func():
            pass

        @retry_with_backoff(max_attempts=5)
        def my_func():
            pass

    Args:
        func: The function to wrap (when used without arguments)
        max_attempts: Maximum retry attempts
        base_delay: Initial delay between retries
        max_delay: Maximum delay cap
        exponential_base: Multiplier for exponential backoff
        jitter: Random jitter factor (0-1)
        retry_on: Exception types to retry
        stop_on: Exception types to never retry
        on_retry: Callback before each retry
        config: Full RetryConfig (overrides other params if provided)

    Returns:
        Decorated function with retry logic

    Example:
        @retry_with_backoff(max_attempts=3, base_delay=1.0)
        def call_external_api():
            return requests.get(url)
    """
    # Build config from params if not provided
    if config is None:
        config = RetryConfig(
            max_attempts=max_attempts,
            base_delay=base_delay,
            max_delay=max_delay,
            exponential_base=exponential_base,
            jitter=jitter,
            retry_on=retry_on,
            stop_on=stop_on,
            on_retry=on_retry,
        )

    def decorator(fn: Callable[..., T]) -> Callable[..., T]:
        @wraps(fn)
        def wrapper(*args: Any, **kwargs: Any) -> T:
            state = RetryState()
            last_error: Exception | None = None

            for attempt in range(1, config.max_attempts + 1):
                state.attempt = attempt

                try:
                    logger.debug(
                        f"Attempt {attempt}/{config.max_attempts} for {fn.__name__}"
                    )
                    return fn(*args, **kwargs)

                except Exception as e:
                    last_error = e
                    state.errors.append(e)

                    # Log the error with context
                    _log_error(fn.__name__, attempt, config.max_attempts, e)

                    # Check if we should retry
                    if not should_retry(e, attempt, config):
                        logger.error(
                            f"[{fn.__name__}] Non-retryable error after {attempt} attempts: "
                            f"{type(e).__name__}: {e}"
                        )
                        raise

                    # Calculate delay
                    delay = calculate_delay(attempt, config, e)
                    state.total_delay += delay

                    # Call on_retry callback if provided
                    if config.on_retry:
                        try:
                            config.on_retry(attempt, e, delay)
                        except Exception as callback_error:
                            logger.warning(f"on_retry callback failed: {callback_error}")

                    # Log retry intention
                    logger.warning(
                        f"[{fn.__name__}] Retrying in {delay:.2f}s "
                        f"(attempt {attempt}/{config.max_attempts}) "
                        f"after {type(e).__name__}: {e}"
                    )

                    # Wait before retry
                    time.sleep(delay)

            # All attempts exhausted
            logger.error(
                f"[{fn.__name__}] All {config.max_attempts} attempts failed. "
                f"Total time: {state.elapsed_time:.2f}s, "
                f"Total delay: {state.total_delay:.2f}s"
            )

            # Raise the last error
            if last_error:
                raise last_error
            raise RuntimeError(f"Retry failed for {fn.__name__} with no error captured")

        return wrapper

    # Handle both @retry_with_backoff and @retry_with_backoff()
    if func is not None:
        return decorator(func)
    return decorator


def _log_error(func_name: str, attempt: int, max_attempts: int, error: Exception) -> None:
    """Log error with appropriate level based on attempt number."""
    error_info = {
        "function": func_name,
        "attempt": attempt,
        "max_attempts": max_attempts,
        "error_type": type(error).__name__,
        "error_message": str(error),
    }

    # Add LangRAG-specific error details
    if isinstance(error, LangRAGError):
        error_info["details"] = error.details
        if error.original_error:
            error_info["original_error"] = str(error.original_error)

    if attempt == max_attempts:
        logger.error(f"[{func_name}] Final attempt failed: {error_info}")
    else:
        logger.debug(f"[{func_name}] Attempt {attempt} failed: {error_info}")


def execute_with_retry(
    func: Callable[..., T],
    *args: Any,
    config: RetryConfig | None = None,
    **kwargs: Any
) -> T:
    """
    Execute a function with retry logic (non-decorator style).

    Useful when you can't use decorators or need dynamic retry config.

    Args:
        func: Function to execute
        *args: Positional arguments for func
        config: Retry configuration
        **kwargs: Keyword arguments for func

    Returns:
        Result of func

    Example:
        result = execute_with_retry(
            client.chat,
            messages,
            config=RetryConfig(max_attempts=5)
        )
    """
    config = config or RetryConfig()

    @retry_with_backoff(config=config)
    def _wrapper() -> T:
        return func(*args, **kwargs)

    return _wrapper()


class RetryContext:
    """
    Context manager for retry operations with state tracking.

    Provides more control over retry behavior and access to retry state.

    Example:
        with RetryContext(max_attempts=3) as ctx:
            for attempt in ctx:
                try:
                    result = risky_operation()
                    break
                except RetryableError as e:
                    if not ctx.should_retry(e):
                        raise
                    ctx.wait()
    """

    def __init__(
        self,
        max_attempts: int = 3,
        base_delay: float = 1.0,
        max_delay: float = 60.0,
        exponential_base: float = 2.0,
        jitter: float = 0.1,
    ):
        self.config = RetryConfig(
            max_attempts=max_attempts,
            base_delay=base_delay,
            max_delay=max_delay,
            exponential_base=exponential_base,
            jitter=jitter,
        )
        self.state = RetryState()
        self._current_error: Exception | None = None

    def __enter__(self) -> "RetryContext":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> bool:
        return False  # Don't suppress exceptions

    def __iter__(self):
        """Iterate through retry attempts."""
        for attempt in range(1, self.config.max_attempts + 1):
            self.state.attempt = attempt
            yield attempt

    def should_retry(self, error: Exception) -> bool:
        """Check if should retry after this error."""
        self._current_error = error
        self.state.errors.append(error)
        return should_retry(error, self.state.attempt, self.config)

    def wait(self) -> float:
        """Wait before next retry attempt. Returns actual delay."""
        delay = calculate_delay(self.state.attempt, self.config, self._current_error)
        self.state.total_delay += delay

        logger.debug(
            f"Waiting {delay:.2f}s before attempt {self.state.attempt + 1}"
        )
        time.sleep(delay)
        return delay

    @property
    def attempts_remaining(self) -> int:
        """Number of retry attempts remaining."""
        return self.config.max_attempts - self.state.attempt

    @property
    def total_elapsed(self) -> float:
        """Total time elapsed since start."""
        return self.state.elapsed_time
