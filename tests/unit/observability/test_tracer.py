"""Tests for observability module."""

import pytest

from langrag.observability import (
    get_tracer,
    is_tracing_enabled,
    shutdown_tracer,
)
from langrag.observability.tracer import _NoOpSpan, _NoOpTracer


class TestTracerFunctions:
    """Tests for tracer functions."""

    def test_is_tracing_enabled_default(self):
        """Tracing is disabled by default."""
        # Ensure clean state
        shutdown_tracer()
        assert is_tracing_enabled() is False

    def test_get_tracer_returns_noop_when_disabled(self):
        """Returns NoOp tracer when tracing not initialized."""
        shutdown_tracer()
        tracer = get_tracer()

        # Should return a tracer that doesn't raise errors
        with tracer.start_as_current_span("test") as span:
            span.set_attribute("key", "value")

    def test_shutdown_tracer_clears_state(self):
        """Shutdown clears tracer state."""
        shutdown_tracer()
        assert is_tracing_enabled() is False


class TestNoOpSpan:
    """Tests for _NoOpSpan class."""

    def test_context_manager(self):
        """NoOpSpan works as context manager."""
        span = _NoOpSpan()
        with span as s:
            assert s is span

    def test_set_attribute_no_error(self):
        """set_attribute doesn't raise errors."""
        span = _NoOpSpan()
        span.set_attribute("key", "value")
        span.set_attribute("number", 123)
        span.set_attribute("list", [1, 2, 3])

    def test_set_status_no_error(self):
        """set_status doesn't raise errors."""
        span = _NoOpSpan()
        span.set_status("OK")
        span.set_status(None)

    def test_record_exception_no_error(self):
        """record_exception doesn't raise errors."""
        span = _NoOpSpan()
        span.record_exception(Exception("test"))
        span.record_exception(ValueError("test"))

    def test_add_event_no_error(self):
        """add_event doesn't raise errors."""
        span = _NoOpSpan()
        span.add_event("event_name")
        span.add_event("event_name", attributes={"key": "value"})


class TestNoOpTracer:
    """Tests for _NoOpTracer class."""

    def test_start_as_current_span(self):
        """start_as_current_span returns NoOpSpan."""
        tracer = _NoOpTracer()
        span = tracer.start_as_current_span("test_span")

        assert isinstance(span, _NoOpSpan)

    def test_start_span(self):
        """start_span returns NoOpSpan."""
        tracer = _NoOpTracer()
        span = tracer.start_span("test_span")

        assert isinstance(span, _NoOpSpan)

    def test_start_as_current_span_with_kwargs(self):
        """start_as_current_span accepts kwargs."""
        tracer = _NoOpTracer()
        span = tracer.start_as_current_span(
            "test_span",
            attributes={"key": "value"},
            kind=None,
        )

        assert isinstance(span, _NoOpSpan)

    def test_context_manager_usage(self):
        """NoOp tracer works with context manager pattern."""
        tracer = _NoOpTracer()

        with tracer.start_as_current_span("outer") as outer_span:
            outer_span.set_attribute("level", "outer")
            with tracer.start_as_current_span("inner") as inner_span:
                inner_span.set_attribute("level", "inner")
