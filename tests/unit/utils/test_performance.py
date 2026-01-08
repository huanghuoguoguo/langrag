"""Tests for performance monitoring utilities."""

import time

import pytest

from langrag.utils.performance import timer, timed


class TestTimer:
    """Tests for timer context manager."""

    def test_timer_basic(self):
        """Timer works for basic operation."""
        with timer("Test operation"):
            time.sleep(0.01)  # 10ms

    def test_timer_with_threshold(self):
        """Timer respects threshold."""
        # This should not log (operation faster than threshold)
        with timer("Fast operation", threshold_ms=1000):
            pass

    def test_timer_exception_handling(self):
        """Timer handles exceptions correctly."""
        with pytest.raises(ValueError):
            with timer("Failing operation"):
                raise ValueError("Test error")

    def test_timer_different_log_levels(self):
        """Timer works with different log levels."""
        with timer("Debug operation", log_level="DEBUG"):
            pass

        with timer("Info operation", log_level="INFO"):
            pass

        with timer("Warning operation", log_level="WARNING"):
            pass


class TestTimed:
    """Tests for timed decorator."""

    def test_timed_basic(self):
        """Timed decorator works for basic function."""
        @timed()
        def fast_function():
            return "result"

        result = fast_function()
        assert result == "result"

    def test_timed_with_custom_operation(self):
        """Timed decorator accepts custom operation name."""
        @timed(operation="Custom Operation")
        def my_function():
            return 42

        result = my_function()
        assert result == 42

    def test_timed_with_threshold(self):
        """Timed decorator respects threshold."""
        @timed(threshold_ms=1000)
        def fast_func():
            return "fast"

        result = fast_func()
        assert result == "fast"

    def test_timed_preserves_function_name(self):
        """Timed decorator preserves function metadata."""
        @timed()
        def named_function():
            """Docstring."""
            pass

        assert named_function.__name__ == "named_function"
        assert named_function.__doc__ == "Docstring."

    def test_timed_with_args(self):
        """Timed decorator works with function arguments."""
        @timed()
        def add(a, b):
            return a + b

        result = add(2, 3)
        assert result == 5

    def test_timed_with_kwargs(self):
        """Timed decorator works with keyword arguments."""
        @timed()
        def greet(name, greeting="Hello"):
            return f"{greeting}, {name}!"

        result = greet("World", greeting="Hi")
        assert result == "Hi, World!"

    def test_timed_exception_handling(self):
        """Timed decorator handles exceptions correctly."""
        @timed()
        def failing_function():
            raise RuntimeError("Error")

        with pytest.raises(RuntimeError):
            failing_function()

    def test_timed_slow_operation(self):
        """Timed decorator logs slow operations."""
        @timed(threshold_ms=1)
        def slow_function():
            time.sleep(0.01)  # 10ms
            return "done"

        result = slow_function()
        assert result == "done"
