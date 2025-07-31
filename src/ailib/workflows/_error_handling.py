"""Error handling for workflows."""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from ailib.workflows._core import WorkflowContext, WorkflowStep


@dataclass
class ErrorHandler:
    """Defines how to handle specific errors."""

    error_type: type[Exception]
    handler: Callable | WorkflowStep | list[WorkflowStep]
    max_retries: int = 1
    backoff_factor: float = 2.0

    async def handle(self, error: Exception, context: WorkflowContext) -> Any:
        """Handle the error."""
        if isinstance(self.handler, WorkflowStep):
            return await self.handler.execute(context)
        elif isinstance(self.handler, list):
            # Execute multiple steps
            result = None
            for step in self.handler:
                result = await step.execute(context)
                context.set_result(step.name, result)
            return result
        elif callable(self.handler):
            if asyncio.iscoroutinefunction(self.handler):
                return await self.handler(error, context)
            return self.handler(error, context)
        else:
            raise ValueError(f"Invalid handler type: {type(self.handler)}")


class ErrorHandlingStep(WorkflowStep):
    """Wraps a step with error handling."""

    def __init__(
        self,
        wrapped_step: WorkflowStep,
        error_handlers: list[ErrorHandler],
        default_handler: ErrorHandler | None = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.wrapped_step = wrapped_step
        self.error_handlers = error_handlers
        self.default_handler = default_handler

    async def execute(self, context: WorkflowContext) -> Any:
        """Execute with error handling."""
        last_error = None

        # Try executing the wrapped step
        try:
            return await self.wrapped_step.execute(context)
        except Exception as e:
            last_error = e

            # Find matching error handler
            handler = None
            for eh in self.error_handlers:
                if isinstance(e, eh.error_type):
                    handler = eh
                    break

            # Use default handler if no specific match
            if not handler and self.default_handler:
                handler = self.default_handler

            if handler:
                # Try handling with retries
                for attempt in range(handler.max_retries):
                    try:
                        # Add error info to context
                        context.errors.append(e)
                        context.variables["_last_error"] = str(e)

                        # Execute handler
                        result = await handler.handle(e, context)

                        # If handler succeeds, return result
                        return result

                    except Exception as retry_error:
                        last_error = retry_error
                        if attempt < handler.max_retries - 1:
                            # Exponential backoff
                            await asyncio.sleep(handler.backoff_factor**attempt)
                        continue

            # Re-raise if no handler or all retries failed
            raise last_error from e


class CircuitBreaker:
    """Circuit breaker pattern for preventing cascading failures."""

    def __init__(
        self,
        failure_threshold: int = 5,
        recovery_timeout: float = 60.0,
        expected_exception: type[Exception] = Exception,
    ):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.expected_exception = expected_exception
        self.failure_count = 0
        self.last_failure_time = None
        self.state = "closed"  # closed, open, half-open

    async def call(self, func: Callable, *args, **kwargs) -> Any:
        """Call function with circuit breaker protection."""
        if self.state == "open":
            # Check if we should try half-open
            if self.last_failure_time:
                elapsed = asyncio.get_event_loop().time() - self.last_failure_time
                if elapsed >= self.recovery_timeout:
                    self.state = "half-open"
                else:
                    raise RuntimeError("Circuit breaker is open")

        try:
            # Try calling the function
            if asyncio.iscoroutinefunction(func):
                result = await func(*args, **kwargs)
            else:
                result = func(*args, **kwargs)

            # Success - reset on half-open or reduce failure count
            if self.state == "half-open":
                self.state = "closed"
                self.failure_count = 0
            elif self.failure_count > 0:
                self.failure_count -= 1

            return result

        except self.expected_exception as e:
            self.failure_count += 1
            self.last_failure_time = asyncio.get_event_loop().time()

            if self.failure_count >= self.failure_threshold:
                self.state = "open"

            raise e


def with_retry(
    func: Callable,
    max_attempts: int = 3,
    backoff_factor: float = 2.0,
    exceptions: tuple[type[Exception], ...] = (Exception,),
) -> Callable:
    """Decorator to add retry logic to a function."""

    async def async_wrapper(*args, **kwargs):
        last_error = None
        for attempt in range(max_attempts):
            try:
                return await func(*args, **kwargs)
            except exceptions as e:
                last_error = e
                if attempt < max_attempts - 1:
                    await asyncio.sleep(backoff_factor**attempt)
                continue
        raise last_error

    def sync_wrapper(*args, **kwargs):
        import time

        last_error = None
        for attempt in range(max_attempts):
            try:
                return func(*args, **kwargs)
            except exceptions as e:
                last_error = e
                if attempt < max_attempts - 1:
                    time.sleep(backoff_factor**attempt)
                continue
        raise last_error

    if asyncio.iscoroutinefunction(func):
        return async_wrapper
    return sync_wrapper
