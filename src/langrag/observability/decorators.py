"""
Tracing Decorators

Provides decorators for easy instrumentation of functions and methods.
"""

import functools
import logging
from collections.abc import Callable
from typing import Any

from langrag.observability.tracer import get_tracer, is_tracing_enabled

logger = logging.getLogger(__name__)


def trace_span(
    name: str | None = None,
    attributes: dict[str, Any] | None = None,
    record_exception: bool = True,
):
    """
    Decorator to trace a function execution as an OpenTelemetry span.

    Args:
        name: Span name. Defaults to function name if not provided.
        attributes: Static attributes to add to the span.
        record_exception: If True, record exceptions in the span.

    Example:
        @trace_span("my_operation", attributes={"component": "retrieval"})
        def my_function(arg1, arg2):
            ...
    """
    def decorator(func: Callable) -> Callable:
        span_name = name or func.__qualname__

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            if not is_tracing_enabled():
                return func(*args, **kwargs)

            tracer = get_tracer()
            with tracer.start_as_current_span(span_name) as span:
                # Add static attributes
                if attributes:
                    for key, value in attributes.items():
                        span.set_attribute(key, value)

                try:
                    result = func(*args, **kwargs)
                    return result
                except Exception as e:
                    if record_exception:
                        span.record_exception(e)
                        _set_error_status(span)
                    raise

        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            if not is_tracing_enabled():
                return await func(*args, **kwargs)

            tracer = get_tracer()
            with tracer.start_as_current_span(span_name) as span:
                # Add static attributes
                if attributes:
                    for key, value in attributes.items():
                        span.set_attribute(key, value)

                try:
                    result = await func(*args, **kwargs)
                    return result
                except Exception as e:
                    if record_exception:
                        span.record_exception(e)
                        _set_error_status(span)
                    raise

        # Return appropriate wrapper based on function type
        if _is_coroutine_function(func):
            return async_wrapper
        return wrapper

    return decorator


def _is_coroutine_function(func: Callable) -> bool:
    """Check if a function is a coroutine function."""
    import asyncio
    return asyncio.iscoroutinefunction(func)


def _set_error_status(span):
    """Set span status to error."""
    try:
        from opentelemetry.trace import Status, StatusCode
        span.set_status(Status(StatusCode.ERROR))
    except ImportError:
        pass
