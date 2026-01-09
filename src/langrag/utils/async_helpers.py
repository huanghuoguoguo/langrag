"""Async helper utilities - handling sync/async context conversion.

This module provides utility functions for calling async functions from sync code.
Primarily used for backward compatibility and running async code in environments
that don't support async.

Examples:
    >>> async def fetch_data():
    ...     return "data"
    >>>
    >>> # Call from sync code
    >>> result = run_async_in_sync_context(fetch_data())
    >>> print(result)
    data
"""

import asyncio
import concurrent.futures
from collections.abc import Coroutine
from typing import TypeVar

from loguru import logger

T = TypeVar("T")


def run_async_in_sync_context(coro: Coroutine[None, None, T]) -> T:
    """Run an async coroutine in a sync context.

    Automatically handles:
    1. Detecting existing event loops
    2. Trying nest_asyncio (for Jupyter and similar environments)
    3. Falling back to thread pool execution
    4. Using asyncio.run() when no loop exists

    Warning:
        In environments with existing event loops (Jupyter, FastAPI),
        this may cause performance degradation.
        It's recommended to use async APIs directly.

    Args:
        coro: The coroutine object to execute

    Returns:
        The return value of the coroutine

    Examples:
        >>> async def get_user(user_id: int):
        ...     # Simulate async database query
        ...     await asyncio.sleep(0.1)
        ...     return {"id": user_id, "name": "Alice"}
        >>>
        >>> # Use in sync code
        >>> user = run_async_in_sync_context(get_user(1))
        >>> print(user["name"])
        Alice
    """
    try:
        # Try to get the currently running event loop
        loop = asyncio.get_running_loop()

        # In a running event loop - needs special handling
        logger.debug(
            "Detected running event loop. Consider using async methods directly "
            "for better performance."
        )

        # Try using nest_asyncio to handle nested loops
        try:
            import nest_asyncio

            nest_asyncio.apply()
            return loop.run_until_complete(coro)

        except ImportError:
            # nest_asyncio not installed, fall back to thread pool
            logger.debug("nest_asyncio not available, using thread pool fallback")

            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(lambda: asyncio.run(coro))
                return future.result()

    except RuntimeError:
        # No running event loop - use asyncio.run() directly
        return asyncio.run(coro)
