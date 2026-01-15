"""
OpenTelemetry Tracer Configuration

Provides initialization and management of OpenTelemetry tracing.
"""

import logging
import os
import json
from typing import TextIO, Sequence, Any

logger = logging.getLogger(__name__)

# Global state
_tracer = None
_tracer_provider = None
_tracing_enabled = False
_log_file_handle: TextIO | None = None


class UnicodeConsoleSpanExporter:
    """
    A custom Span Exporter that behaves like ConsoleSpanExporter but preserves Unicode characters
    in the output JSON, making Chinese logs readable.
    """
    def __init__(self, out: TextIO | None = None):
        import sys
        self.out = out or sys.stdout

    def export(self, spans: Sequence[Any]) -> Any:
        from opentelemetry.sdk.trace.export import SpanExportResult
        for span in spans:
            try:
                # span.to_json() uses defaults which escape non-ASCII.
                # We need to parse it back and dump with ensure_ascii=False
                span_json = span.to_json()
                span_dict = json.loads(span_json)
                output = json.dumps(span_dict, indent=4, ensure_ascii=False)
                self.out.write(output + os.linesep)
            except Exception:
                # Fallback to default behavior if something goes wrong
                self.out.write(span.to_json(indent=4) + os.linesep)
            
            self.out.flush()
        return SpanExportResult.SUCCESS

    def shutdown(self) -> None:
        pass


def is_tracing_enabled() -> bool:
    """Check if tracing is currently enabled."""
    return _tracing_enabled


def init_tracer(
    service_name: str = "langrag",
    endpoint: str | None = None,
    enable_console_export: bool = False,
    log_file: str | None = None,
) -> bool:
    """
    Initialize OpenTelemetry tracer.

    Args:
        service_name: Name of the service for tracing.
        endpoint: OTLP endpoint URL (e.g., "http://localhost:4317").
                  If None, reads from OTEL_EXPORTER_OTLP_ENDPOINT env var.
        enable_console_export: If True, also export spans to console (for debugging).
        log_file: Path to a file to write traces to (using ConsoleSpanExporter).

    Returns:
        True if initialization succeeded, False otherwise.
    """
    global _tracer, _tracer_provider, _tracing_enabled, _log_file_handle

    try:
        from opentelemetry import trace
        from opentelemetry.sdk.resources import SERVICE_NAME, Resource
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import BatchSpanProcessor
    except ImportError:
        logger.warning(
            "OpenTelemetry packages not installed. "
            "Install with: pip install opentelemetry-api opentelemetry-sdk opentelemetry-exporter-otlp"
        )
        return False

    # Get endpoint from parameter or environment
    endpoint = endpoint or os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT")

    # Create resource with service name
    resource = Resource.create({SERVICE_NAME: service_name})

    # Create tracer provider
    _tracer_provider = TracerProvider(resource=resource)

    # Add OTLP exporter if endpoint is configured
    if endpoint:
        try:
            from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter

            otlp_exporter = OTLPSpanExporter(endpoint=endpoint)
            _tracer_provider.add_span_processor(BatchSpanProcessor(otlp_exporter))
            logger.info(f"OpenTelemetry OTLP exporter configured: {endpoint}")
        except ImportError:
            logger.warning(
                "OTLP exporter not available. "
                "Install with: pip install opentelemetry-exporter-otlp"
            )

    # Add console/file exporter for debugging/logging
    if enable_console_export or log_file:
        try:
            from opentelemetry.sdk.trace.export import ConsoleSpanExporter, SimpleSpanProcessor

            out_stream = None
            if log_file:
                try:
                    os.makedirs(os.path.dirname(log_file), exist_ok=True)
                    # We keep this file handle open globally
                    _log_file_handle = open(log_file, "a", encoding="utf-8")
                    out_stream = _log_file_handle
                except Exception as e:
                    logger.error(f"Failed to open log file {log_file} for tracing: {e}")

            if out_stream and log_file:
                # Use our custom exporter for file logging to support Unicode
                exporter = UnicodeConsoleSpanExporter(out=out_stream)
            else:
                # Use standard console exporter for stdout/stderr
                # If out_stream is None, ConsoleSpanExporter defaults to sys.stderr
                exporter = ConsoleSpanExporter(out=out_stream) if out_stream else ConsoleSpanExporter()
            
            _tracer_provider.add_span_processor(SimpleSpanProcessor(exporter))
            dest = log_file if log_file else "console"
            logger.info(f"OpenTelemetry exporter enabled: {dest}")
        except ImportError:
            pass

    # Set global tracer provider
    trace.set_tracer_provider(_tracer_provider)

    # Get tracer instance
    _tracer = trace.get_tracer(__name__)
    _tracing_enabled = True

    logger.info(f"OpenTelemetry tracer initialized for service: {service_name}")
    return True


def get_tracer():
    """
    Get the OpenTelemetry tracer instance.

    Returns a no-op tracer if tracing is not initialized.
    """
    global _tracer

    if _tracer is not None:
        return _tracer

    # Return no-op tracer if not initialized
    try:
        from opentelemetry import trace
        return trace.get_tracer(__name__)
    except ImportError:
        return _NoOpTracer()


def shutdown_tracer():
    """Shutdown the tracer and flush any pending spans."""
    global _tracer, _tracer_provider, _tracing_enabled, _log_file_handle

    if _tracer_provider is not None:
        try:
            _tracer_provider.shutdown()
            logger.info("OpenTelemetry tracer shut down")
        except Exception as e:
            logger.error(f"Error shutting down tracer: {e}")

    # Close log file if it was opened
    if _log_file_handle:
        try:
            _log_file_handle.close()
        except Exception:
            pass
        _log_file_handle = None

    _tracer = None
    _tracer_provider = None
    _tracing_enabled = False


class _NoOpSpan:
    """No-op span for when tracing is disabled."""

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass

    def set_attribute(self, key, value):
        pass

    def set_status(self, status):
        pass

    def record_exception(self, exception):
        pass

    def add_event(self, name, attributes=None):
        pass


class _NoOpTracer:
    """No-op tracer for when OpenTelemetry is not installed."""

    def start_as_current_span(self, _name, **_kwargs):
        return _NoOpSpan()

    def start_span(self, _name, **_kwargs):
        return _NoOpSpan()
