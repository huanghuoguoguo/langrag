"""Tests for async helper utilities."""

import asyncio

import pytest

from langrag.utils.async_helpers import run_async_in_sync_context


class TestRunAsyncInSyncContext:
    """Tests for run_async_in_sync_context function."""

    def test_simple_coroutine(self):
        """Run a simple async function from sync context."""
        async def simple_coro():
            return "hello"

        result = run_async_in_sync_context(simple_coro())
        assert result == "hello"

    def test_coroutine_with_args(self):
        """Run async function with arguments."""
        async def add(a, b):
            return a + b

        result = run_async_in_sync_context(add(2, 3))
        assert result == 5

    def test_coroutine_with_await(self):
        """Run async function that uses await."""
        async def async_sleep():
            await asyncio.sleep(0.01)
            return "done"

        result = run_async_in_sync_context(async_sleep())
        assert result == "done"

    def test_coroutine_raises_exception(self):
        """Exception from coroutine is propagated."""
        async def failing_coro():
            raise ValueError("Test error")

        with pytest.raises(ValueError, match="Test error"):
            run_async_in_sync_context(failing_coro())

    def test_nested_coroutines(self):
        """Run nested async functions."""
        async def inner():
            return 42

        async def outer():
            return await inner()

        result = run_async_in_sync_context(outer())
        assert result == 42

    def test_coroutine_returns_none(self):
        """Coroutine returning None."""
        async def returns_none():
            pass

        result = run_async_in_sync_context(returns_none())
        assert result is None

    def test_coroutine_with_complex_return(self):
        """Coroutine returning complex data structure."""
        async def complex_return():
            return {"key": "value", "list": [1, 2, 3]}

        result = run_async_in_sync_context(complex_return())
        assert result == {"key": "value", "list": [1, 2, 3]}
