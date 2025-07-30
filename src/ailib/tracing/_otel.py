"""OpenTelemetry integration for advanced tracing - optional module."""

import os

try:
    from opentelemetry import trace
    from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import (
        OTLPSpanExporter,
    )
    from opentelemetry.sdk.resources import SERVICE_NAME, Resource
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor

    OTEL_AVAILABLE = True
except ImportError:
    OTEL_AVAILABLE = False

from ._core import trace_step

# Global OTEL tracer
_otel_tracer = None
_otel_enabled = False


def enable_otel(
    endpoint: str | None = None,
    service_name: str = "ailib",
    insecure: bool = True,
    headers: dict[str, str] | None = None,
):
    """Enable OpenTelemetry tracing export.

    Args:
        endpoint: OTLP endpoint (defaults to OTEL_EXPORTER_OTLP_ENDPOINT env var)
        service_name: Service name for traces
        insecure: Whether to use insecure connection
        headers: Additional headers for authentication

    Example:
        # Basic setup
        enable_otel()

        # Custom endpoint
        enable_otel(endpoint="http://localhost:4318")

        # With authentication
        enable_otel(
            endpoint="https://otel-collector.example.com:4317",
            headers={"api-key": "your-key"},
            insecure=False
        )
    """
    if not OTEL_AVAILABLE:
        raise ImportError(
            "OpenTelemetry dependencies not installed. "
            "Install with: pip install ailib[tracing]"
        )

    global _otel_tracer, _otel_enabled

    # Use environment variable if endpoint not specified
    if endpoint is None:
        endpoint = os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT", "http://localhost:4317")

    # Create resource
    resource = Resource(attributes={SERVICE_NAME: service_name})

    # Create tracer provider
    provider = TracerProvider(resource=resource)

    # Create OTLP exporter
    otlp_exporter = OTLPSpanExporter(
        endpoint=endpoint, insecure=insecure, headers=headers
    )

    # Add span processor
    provider.add_span_processor(BatchSpanProcessor(otlp_exporter))

    # Set as global tracer provider
    trace.set_tracer_provider(provider)

    # Get tracer
    _otel_tracer = trace.get_tracer(__name__)
    _otel_enabled = True

    # Hook into our tracing system
    _install_otel_hooks()


def _install_otel_hooks():
    """Install hooks to export AILib traces to OpenTelemetry."""
    import atexit

    from ._core import get_all_traces

    # Export traces on exit
    def export_on_exit():
        if not _otel_enabled or not _otel_tracer:
            return

        for ailib_trace in get_all_traces():
            _export_trace_to_otel(ailib_trace)

    atexit.register(export_on_exit)


def _export_trace_to_otel(ailib_trace):
    """Convert and export an AILib trace to OpenTelemetry."""
    if not _otel_enabled or not _otel_tracer:
        return

    # Create root span for the trace
    with _otel_tracer.start_as_current_span(
        ailib_trace.name,
        start_time=int(ailib_trace.start_time * 1e9),  # Convert to nanoseconds
    ) as root_span:
        # Add trace metadata
        for key, value in ailib_trace.metadata.items():
            root_span.set_attribute(f"ailib.{key}", str(value))

        # Track parent spans for nested steps
        span_map = {None: root_span}

        # Export each step
        for step in ailib_trace.steps:
            parent_span = span_map.get(step.parent_id, root_span)

            # Create span for step
            with _otel_tracer.start_as_current_span(
                step.name,
                start_time=int(step.start_time * 1e9),
                context=trace.set_span_in_context(parent_span),
            ) as span:
                # Set span kind based on step type
                if step.step_type == "llm_call":
                    span.set_attribute("span.kind", "CLIENT")
                elif step.step_type == "tool_use":
                    span.set_attribute("span.kind", "INTERNAL")

                # Add step metadata
                span.set_attribute("ailib.step_type", step.step_type)
                for key, value in step.metadata.items():
                    span.set_attribute(f"ailib.{key}", str(value))

                # Set error if present
                if step.error:
                    span.set_status(trace.Status(trace.StatusCode.ERROR, step.error))

                # Store span for potential children
                span_map[step.step_id] = span

                # End span at correct time
                if step.end_time:
                    span.end(end_time=int(step.end_time * 1e9))


def get_otel_tracer():
    """Get the OpenTelemetry tracer if enabled."""
    return _otel_tracer if _otel_enabled else None


# Enhanced trace_step that also creates OTEL spans
def otel_trace_step(name: str, **kwargs):
    """Enhanced trace_step that creates both AILib and OTEL spans."""
    # Use regular trace_step
    with trace_step(name, **kwargs) as step:
        # Also create OTEL span if enabled
        if _otel_enabled and _otel_tracer:
            with _otel_tracer.start_as_current_span(name) as span:
                # Add metadata as attributes
                for key, value in kwargs.items():
                    span.set_attribute(f"ailib.{key}", str(value))
                yield step
        else:
            yield step
