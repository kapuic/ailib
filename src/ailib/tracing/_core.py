"""Core tracing functionality - internal module."""

import contextvars
import json
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from functools import wraps
from pathlib import Path
from typing import Any
from uuid import uuid4


@dataclass
class TraceStep:
    """Represents a single step in a trace."""

    name: str
    step_type: str  # "llm_call", "tool_use", "chain_step", "agent_think", etc.
    start_time: float
    end_time: float | None = None
    duration_ms: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    error: str | None = None
    parent_id: str | None = None
    step_id: str = field(default_factory=lambda: str(uuid4()))

    def complete(self, error: str | None = None):
        """Mark step as complete."""
        self.end_time = time.time()
        self.duration_ms = (self.end_time - self.start_time) * 1000
        if error:
            self.error = error

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "step_id": self.step_id,
            "name": self.name,
            "type": self.step_type,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "duration_ms": self.duration_ms,
            "metadata": self.metadata,
            "error": self.error,
            "parent_id": self.parent_id,
        }


@dataclass
class Trace:
    """Represents a complete trace of an operation."""

    trace_id: str = field(default_factory=lambda: str(uuid4()))
    name: str = "default"
    start_time: float = field(default_factory=time.time)
    end_time: float | None = None
    steps: list[TraceStep] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    _current_step_id: str | None = None

    def add_step(self, step: TraceStep):
        """Add a step to the trace."""
        self.steps.append(step)

    def complete(self):
        """Mark trace as complete."""
        self.end_time = time.time()

    @property
    def duration_ms(self) -> float | None:
        """Get total duration in milliseconds."""
        if self.end_time:
            return (self.end_time - self.start_time) * 1000
        return None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "trace_id": self.trace_id,
            "name": self.name,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "duration_ms": self.duration_ms,
            "metadata": self.metadata,
            "steps": [step.to_dict() for step in self.steps],
        }

    def to_json(self, indent: int = 2) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict(), indent=indent)


# Thread-local context for current trace
_current_trace: contextvars.ContextVar["Trace | None"] = contextvars.ContextVar(
    "_current_trace", default=None
)

# Global tracing enabled flag
_tracing_enabled = True

# Global trace storage (for simplicity, production would use proper storage)
_trace_storage: dict[str, "Trace"] = {}
_storage_lock = threading.Lock()

# Maximum traces to keep in memory
MAX_TRACES = 100


def is_tracing_enabled() -> bool:
    """Check if tracing is enabled."""
    return _tracing_enabled


def enable_tracing():
    """Enable tracing globally."""
    global _tracing_enabled
    _tracing_enabled = True


def disable_tracing():
    """Disable tracing globally."""
    global _tracing_enabled
    _tracing_enabled = False


def start_trace(name: str = "default", **metadata) -> Trace:
    """Start a new trace.

    Args:
        name: Name for the trace
        **metadata: Additional metadata

    Returns:
        New Trace instance
    """
    if not _tracing_enabled:
        return Trace(name=name)  # Return dummy trace

    trace = Trace(name=name, metadata=metadata)
    _current_trace.set(trace)

    # Store trace
    with _storage_lock:
        _trace_storage[trace.trace_id] = trace
        # Cleanup old traces
        if len(_trace_storage) > MAX_TRACES:
            oldest_ids = sorted(_trace_storage.keys())[
                : len(_trace_storage) - MAX_TRACES
            ]
            for trace_id in oldest_ids:
                del _trace_storage[trace_id]

    return trace


def get_trace() -> Trace | None:
    """Get the current active trace."""
    return _current_trace.get()


def get_all_traces() -> list[Trace]:
    """Get all stored traces."""
    with _storage_lock:
        return list(_trace_storage.values())


@contextmanager
def trace_step(
    name: str, step_type: str = "custom", parent_id: str | None = None, **metadata
):
    """Context manager for tracing a step.

    Args:
        name: Step name
        step_type: Type of step
        parent_id: Parent step ID for nested steps
        **metadata: Additional metadata

    Example:
        with trace_step("database_query", step_type="db"):
            result = db.query("SELECT * FROM users")
    """
    if not _tracing_enabled:
        yield
        return

    trace = get_trace()
    if not trace:
        # Auto-start trace if needed
        trace = start_trace("auto")

    # Use current step as parent if not specified
    if parent_id is None and trace._current_step_id:
        parent_id = trace._current_step_id

    step = TraceStep(
        name=name,
        step_type=step_type,
        start_time=time.time(),
        metadata=metadata,
        parent_id=parent_id,
    )

    # Set as current step
    old_step_id = trace._current_step_id
    trace._current_step_id = step.step_id

    try:
        yield step
        step.complete()
    except Exception as e:
        step.complete(error=str(e))
        raise
    finally:
        trace.add_step(step)
        trace._current_step_id = old_step_id


def traced(
    name: str | None = None, step_type: str = "function", include_args: bool = False
):
    """Decorator to trace function execution.

    Args:
        name: Step name (defaults to function name)
        step_type: Type of step
        include_args: Whether to include function arguments in metadata

    Example:
        @traced()
        def process_data(data):
            return data.upper()
    """

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            if not _tracing_enabled:
                return func(*args, **kwargs)

            step_name = name or func.__name__
            metadata = {"function": func.__name__}

            if include_args:
                # Safely serialize args/kwargs
                try:
                    metadata["args"] = [str(arg) for arg in args]
                    metadata["kwargs"] = {k: str(v) for k, v in kwargs.items()}
                except Exception:
                    metadata["args_error"] = "Could not serialize arguments"

            with trace_step(step_name, step_type=step_type, **metadata):
                return func(*args, **kwargs)

        return wrapper

    return decorator


def export_trace(
    output: str | Path | None = None,
    trace_id: str | None = None,
    format: str = "json",
) -> str | None:
    """Export trace data.

    Args:
        output: Output file path or None for string return
        trace_id: Specific trace ID or None for current trace
        format: Export format (currently only "json")

    Returns:
        Trace data as string if output is None
    """
    # Get trace
    if trace_id:
        with _storage_lock:
            trace = _trace_storage.get(trace_id)
    else:
        trace = get_trace()

    if not trace:
        raise ValueError("No trace found")

    # Complete trace if still running
    if trace.end_time is None:
        trace.complete()

    # Export based on format
    if format == "json":
        data = trace.to_json()
    else:
        raise ValueError(f"Unsupported format: {format}")

    # Write or return
    if output:
        Path(output).write_text(data)
        return None
    else:
        return data


# Internal helpers for automatic tracing
def _trace_llm_call(
    model: str,
    messages: list[dict[str, Any]],
    response: dict[str, Any] | None = None,
    error: str | None = None,
    **kwargs,
):
    """Internal helper to trace LLM calls."""
    if not _tracing_enabled:
        return

    trace = get_trace()
    if not trace:
        return

    metadata = {
        "model": model,
        "message_count": len(messages),
        "temperature": kwargs.get("temperature"),
        "max_tokens": kwargs.get("max_tokens"),
    }

    if response:
        metadata["response_tokens"] = response.get("usage", {}).get("completion_tokens")
        metadata["prompt_tokens"] = response.get("usage", {}).get("prompt_tokens")
        metadata["finish_reason"] = response.get("finish_reason")

    step = TraceStep(
        name=f"llm_call_{model}",
        step_type="llm_call",
        start_time=time.time(),
        metadata=metadata,
        error=error,
    )
    step.complete(error=error)
    trace.add_step(step)


def _trace_tool_use(
    tool_name: str,
    args: dict[str, Any],
    result: Any = None,
    error: str | None = None,
):
    """Internal helper to trace tool usage."""
    if not _tracing_enabled:
        return

    trace = get_trace()
    if not trace:
        return

    metadata = {
        "tool": tool_name,
        "args": args,
    }

    if result is not None:
        metadata["result_type"] = type(result).__name__

    step = TraceStep(
        name=f"tool_{tool_name}",
        step_type="tool_use",
        start_time=time.time(),
        metadata=metadata,
        error=error,
    )
    step.complete(error=error)
    trace.add_step(step)
