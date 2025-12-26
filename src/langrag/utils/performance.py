"""Performance monitoring utilities."""

import time
from contextlib import contextmanager
from functools import wraps
from typing import Callable, Any
from loguru import logger


@contextmanager
def timer(operation: str, log_level: str = "INFO", threshold_ms: float = 0):
    """Context manager for timing operations.

    Args:
        operation: Description of the operation being timed
        log_level: Log level to use ("DEBUG", "INFO", "WARNING")
        threshold_ms: Only log if operation takes longer than this (in milliseconds)

    Example:
        >>> with timer("Embedding 100 chunks"):
        ...     embeddings = embedder.embed(texts)
    """
    start = time.perf_counter()
    try:
        yield
    finally:
        elapsed_ms = (time.perf_counter() - start) * 1000

        if elapsed_ms >= threshold_ms:
            log_func = getattr(logger, log_level.lower())
            log_func(f"{operation} took {elapsed_ms:.2f}ms")


def timed(operation: str = None, threshold_ms: float = 100):
    """Decorator for timing function execution.

    Args:
        operation: Description of the operation (defaults to function name)
        threshold_ms: Only log if operation takes longer than this (in milliseconds)

    Example:
        >>> @timed("Embedding", threshold_ms=50)
        ... def embed_texts(texts):
        ...     return embedder.embed(texts)
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            op_name = operation or f"{func.__module__}.{func.__name__}"

            start = time.perf_counter()
            try:
                result = func(*args, **kwargs)
                return result
            finally:
                elapsed_ms = (time.perf_counter() - start) * 1000

                if elapsed_ms >= threshold_ms:
                    if elapsed_ms > 1000:
                        logger.info(f"{op_name} took {elapsed_ms/1000:.2f}s")
                    else:
                        logger.debug(f"{op_name} took {elapsed_ms:.2f}ms")

        return wrapper
    return decorator
