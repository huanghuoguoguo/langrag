"""
LangRAG Observability Module

Provides distributed tracing support using OpenTelemetry.
"""

from langrag.observability.decorators import trace_span
from langrag.observability.tracer import (
    get_tracer,
    init_tracer,
    is_tracing_enabled,
    shutdown_tracer,
)

__all__ = [
    "init_tracer",
    "get_tracer",
    "shutdown_tracer",
    "is_tracing_enabled",
    "trace_span",
]
