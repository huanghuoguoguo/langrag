"""Tests for observability decorators."""

import pytest

from langrag.observability.decorators import trace_span, _is_coroutine_function


class TestTraceSpanDecorator:
    """Tests for trace_span decorator."""

    def test_decorator_sync_function_no_tracing(self):
        """Sync function works when tracing is disabled."""
        @trace_span("test_operation")
        def my_func(x, y):
            return x + y

        result = my_func(1, 2)
        assert result == 3

    def test_decorator_preserves_function_name(self):
        """Decorator preserves function metadata."""
        @trace_span("test_operation")
        def my_func():
            """My docstring."""
            pass

        assert my_func.__name__ == "my_func"
        assert my_func.__doc__ == "My docstring."

    def test_decorator_with_default_name(self):
        """Decorator uses function name when name not provided."""
        @trace_span()
        def another_func():
            return "result"

        result = another_func()
        assert result == "result"

    def test_decorator_with_attributes(self):
        """Decorator accepts attributes parameter."""
        @trace_span("operation", attributes={"component": "test"})
        def func_with_attrs():
            return "ok"

        result = func_with_attrs()
        assert result == "ok"

    def test_decorator_with_exception(self):
        """Decorator handles exceptions correctly."""
        @trace_span("failing_operation")
        def failing_func():
            raise ValueError("Test error")

        with pytest.raises(ValueError, match="Test error"):
            failing_func()

    def test_decorator_record_exception_false(self):
        """Decorator respects record_exception=False."""
        @trace_span("operation", record_exception=False)
        def failing_func():
            raise RuntimeError("Error")

        with pytest.raises(RuntimeError):
            failing_func()

    @pytest.mark.asyncio
    async def test_decorator_async_function_no_tracing(self):
        """Async function works when tracing is disabled."""
        @trace_span("async_operation")
        async def async_func(x):
            return x * 2

        result = await async_func(5)
        assert result == 10

    @pytest.mark.asyncio
    async def test_decorator_async_with_exception(self):
        """Async decorator handles exceptions correctly."""
        @trace_span("async_failing")
        async def async_failing():
            raise ValueError("Async error")

        with pytest.raises(ValueError, match="Async error"):
            await async_failing()


class TestIsCoroutineFunction:
    """Tests for _is_coroutine_function helper."""

    def test_regular_function(self):
        """Regular function is not a coroutine."""
        def regular():
            pass

        assert _is_coroutine_function(regular) is False

    def test_async_function(self):
        """Async function is a coroutine."""
        async def async_func():
            pass

        assert _is_coroutine_function(async_func) is True

    def test_lambda(self):
        """Lambda is not a coroutine."""
        func = lambda x: x  # noqa: E731

        assert _is_coroutine_function(func) is False
