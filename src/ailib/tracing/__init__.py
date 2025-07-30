"""Simple tracing for AILib - Vercel AI SDK style.

This module provides zero-configuration tracing for AI workflows.
Traces are collected automatically and can be exported for analysis.

Example:
    # Automatic tracing - just works!
    agent = create_agent("assistant")
    result = agent.run("What's 2+2?")

    # Get trace data
    trace = get_trace()
    print(f"Trace has {len(trace.steps)} steps")

    # Export for analysis
    export_trace("agent_trace.json")
"""

from ._core import (
    Trace,
    TraceStep,
    disable_tracing,
    enable_tracing,
    export_trace,
    get_trace,
    is_tracing_enabled,
    start_trace,
    trace_step,
    traced,
)

# Optional OpenTelemetry integration
try:
    from ._otel import enable_otel

    __all__ = [
        "Trace",
        "TraceStep",
        "start_trace",
        "trace_step",
        "get_trace",
        "export_trace",
        "traced",
        "enable_tracing",
        "disable_tracing",
        "is_tracing_enabled",
        "enable_otel",
    ]
except ImportError:
    # OpenTelemetry is optional
    __all__ = [
        "Trace",
        "TraceStep",
        "start_trace",
        "trace_step",
        "get_trace",
        "export_trace",
        "traced",
        "enable_tracing",
        "disable_tracing",
        "is_tracing_enabled",
    ]

    def enable_otel(*args, **kwargs):
        """OpenTelemetry support requires extra dependencies."""
        raise ImportError(
            "OpenTelemetry support requires additional dependencies. "
            "Install with: pip install ailib[tracing]"
        )
