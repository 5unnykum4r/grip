"""OpenTelemetry tracing integration for grip.

Provides trace spans for agent loop iterations, LLM calls, and tool
executions. Falls back gracefully to no-op when opentelemetry packages
are not installed (the [observe] optional dependency group).

Usage:
    tracer = get_tracer()
    with tracer.start_span("agent.run") as span:
        span.set_attribute("session_key", key)
"""

from __future__ import annotations

import importlib.util
from contextlib import contextmanager
from typing import Any

from loguru import logger

_OTEL_AVAILABLE = (
    importlib.util.find_spec("opentelemetry") is not None
    and importlib.util.find_spec("opentelemetry.sdk") is not None
)

_tracer = None
_initialized = False


class NoOpSpan:
    """Span substitute when OpenTelemetry is not installed."""

    def set_attribute(self, key: str, value: Any) -> None:
        pass

    def set_status(self, *args: Any, **kwargs: Any) -> None:
        pass

    def record_exception(self, exc: Exception) -> None:
        pass

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass


class NoOpTracer:
    """Tracer substitute when OpenTelemetry is not installed."""

    def start_span(self, name: str, **kwargs: Any) -> NoOpSpan:
        return NoOpSpan()

    @contextmanager
    def start_as_current_span(self, name: str, **kwargs: Any):
        yield NoOpSpan()


def init_tracing(service_name: str = "grip", endpoint: str = "") -> bool:
    """Initialize OpenTelemetry tracing if the SDK is available.

    Returns True if tracing was successfully initialized, False if
    the otel packages are not installed.
    """
    global _tracer, _initialized  # noqa: PLW0603

    if _initialized:
        return _tracer is not None and not isinstance(_tracer, NoOpTracer)

    _initialized = True

    if not _OTEL_AVAILABLE:
        logger.debug("OpenTelemetry not installed, tracing disabled")
        _tracer = NoOpTracer()
        return False

    try:
        from opentelemetry import trace
        from opentelemetry.sdk.resources import Resource
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import (
            BatchSpanProcessor,
            ConsoleSpanExporter,
        )

        resource = Resource.create({"service.name": service_name})
        provider = TracerProvider(resource=resource)

        if endpoint:
            from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import (
                OTLPSpanExporter,
            )

            exporter = OTLPSpanExporter(endpoint=endpoint)
        else:
            exporter = ConsoleSpanExporter()

        provider.add_span_processor(BatchSpanProcessor(exporter))
        trace.set_tracer_provider(provider)
        _tracer = trace.get_tracer("grip")

        logger.info("OpenTelemetry tracing initialized (service={})", service_name)
        return True

    except Exception as exc:
        logger.warning("Failed to initialize tracing: {}", exc)
        _tracer = NoOpTracer()
        return False


def get_tracer():
    """Get the global tracer instance (real or no-op)."""
    global _tracer  # noqa: PLW0603
    if _tracer is None:
        _tracer = NoOpTracer()
    return _tracer
