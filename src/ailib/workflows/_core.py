"""Core workflow classes and interfaces."""

from __future__ import annotations

import asyncio
import time
from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from pydantic import BaseModel

from ailib.chains import Chain
from ailib.core.client_factory import create_client
from ailib.tracing import get_trace


class StepType(Enum):
    """Types of workflow steps."""

    SIMPLE = "simple"
    CONDITIONAL = "conditional"
    LOOP = "loop"
    PARALLEL = "parallel"
    HUMAN_APPROVAL = "human_approval"
    SUBWORKFLOW = "subworkflow"


@dataclass
class WorkflowContext:
    """Execution context for workflows."""

    # Current execution state
    variables: dict[str, Any] = field(default_factory=dict)
    results: dict[str, Any] = field(default_factory=dict)  # Named step results
    current_result: Any = None  # Result of the last step

    # Workflow state
    state: dict[str, Any] = field(default_factory=dict)

    # Execution metadata
    step_count: int = 0
    errors: list[Exception] = field(default_factory=list)

    # Configuration
    llm_client: Any | None = None

    def get(self, key: str, default: Any = None) -> Any:
        """Get a value from context (checks results, then state, then variables)."""
        if key in self.results:
            return self.results[key]
        if key in self.state:
            return self.state[key]
        return self.variables.get(key, default)

    def set_result(self, name: str | None, value: Any) -> None:
        """Set a step result."""
        self.current_result = value
        if name:
            self.results[name] = value

    def update_state(self, updates: dict[str, Any]) -> None:
        """Update workflow state."""
        self.state.update(updates)


class WorkflowStep(ABC):
    """Base class for all workflow steps."""

    def __init__(
        self,
        name: str | None = None,
        retry: int = 0,
        timeout: float | None = None,
        output_schema: type[BaseModel] | None = None,
    ):
        self.name = name
        self.retry = retry
        self.timeout = timeout
        self.output_schema = output_schema
        self.step_type = StepType.SIMPLE

    @abstractmethod
    async def execute(self, context: WorkflowContext) -> Any:
        """Execute the step."""
        pass

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name={self.name})"


class SimpleStep(WorkflowStep):
    """A simple LLM or function-based step."""

    def __init__(
        self,
        prompt_or_func: str | Callable,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.prompt_or_func = prompt_or_func

    async def execute(self, context: WorkflowContext) -> Any:
        """Execute the step."""
        if isinstance(self.prompt_or_func, str):
            # LLM-based step
            return await self._execute_llm(context)
        else:
            # Function-based step
            return await self._execute_function(context)

    async def _execute_llm(self, context: WorkflowContext) -> Any:
        """Execute an LLM-based step."""
        # Format the prompt with context
        # Include previous_result for easy reference in prompts
        format_vars = {
            **context.variables,
            **context.results,
            **context.state,  # Include state values for formatting
            "previous_result": context.current_result,
            "previous": context.current_result,  # Shorter alias
        }
        prompt = self.prompt_or_func.format(**format_vars)

        # Get LLM client
        client = context.llm_client or create_client("gpt-3.5-turbo")

        # Execute with retry logic
        last_error = None
        for attempt in range(self.retry + 1):
            try:
                # Create a simple chain for execution
                chain = Chain(client)
                chain.add_user(prompt)

                # Execute
                result = (
                    await chain.arun()
                    if asyncio.iscoroutinefunction(chain.run)
                    else chain.run()
                )

                # Validate output if schema provided
                if self.output_schema:
                    result = self._validate_output(result)

                return result

            except Exception as e:
                last_error = e
                if attempt < self.retry:
                    await asyncio.sleep(2**attempt)  # Exponential backoff
                    continue
                raise

        raise last_error

    async def _execute_function(self, context: WorkflowContext) -> Any:
        """Execute a function-based step."""
        if asyncio.iscoroutinefunction(self.prompt_or_func):
            return await self.prompt_or_func(context)
        return self.prompt_or_func(context)

    def _validate_output(self, output: Any) -> Any:
        """Validate output against schema."""
        if not self.output_schema:
            return output

        # If output is already the correct type, return it
        if isinstance(output, self.output_schema):
            return output

        # Try to parse and validate
        if isinstance(output, str):
            # Try to parse as JSON and validate
            import json

            try:
                # First try direct JSON parsing
                data = json.loads(output)
                return self.output_schema(**data)
            except json.JSONDecodeError:
                # If not valid JSON, try to extract JSON from the text
                # LLMs often wrap JSON in markdown code blocks
                import re

                json_match = re.search(
                    r"```(?:json)?\s*(\{.*?\}|\[.*?\])\s*```", output, re.DOTALL
                )
                if json_match:
                    try:
                        data = json.loads(json_match.group(1))
                        return self.output_schema(**data)
                    except Exception:
                        pass

                # If validation fails, will be caught by retry logic
                raise ValueError(
                    f"Output validation failed - expected "
                    f"{self.output_schema.__name__}, got: {output[:200]}..."
                ) from None
            except Exception as e:
                raise ValueError(f"Schema validation failed: {str(e)}") from e

        # Try to convert dict to schema
        if isinstance(output, dict):
            try:
                return self.output_schema(**output)
            except Exception as e:
                raise ValueError(f"Schema validation failed: {str(e)}") from e

        raise ValueError(
            f"Cannot validate output type {type(output)} against "
            f"schema {self.output_schema.__name__}"
        )


class ConditionalStep(WorkflowStep):
    """A conditional branching step."""

    def __init__(
        self,
        condition: Callable[[Any], bool],
        then_step: WorkflowStep,
        else_step: WorkflowStep | None = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.condition = condition
        self.then_step = then_step
        self.else_step = else_step
        self.step_type = StepType.CONDITIONAL

    async def execute(self, context: WorkflowContext) -> Any:
        """Execute the conditional step."""
        # Evaluate condition
        condition_result = self.condition(context.current_result)

        # Execute appropriate branch
        if condition_result:
            return await self.then_step.execute(context)
        elif self.else_step:
            return await self.else_step.execute(context)

        return None


class Workflow:
    """Main workflow class that executes steps."""

    def __init__(
        self,
        steps: list[WorkflowStep] | None = None,
        name: str | None = None,
        initial_state: dict[str, Any] | None = None,
    ):
        self.steps = steps or []
        self.name = name
        self.initial_state = initial_state or {}
        self._trace = None

    async def run(self, **kwargs) -> Any:
        """Run the workflow with given inputs."""
        # Create execution context
        context = WorkflowContext(
            variables=kwargs,
            state=self.initial_state.copy(),
        )

        # Start tracing
        trace = get_trace()
        if trace:
            from ailib.tracing._core import TraceStep

            trace.add_step(
                TraceStep(
                    name="workflow_start",
                    step_type="workflow",
                    start_time=time.time(),
                    metadata={"name": self.name, "inputs": kwargs},
                )
            )

        try:
            # Execute steps
            for step in self.steps:
                context.step_count += 1

                # Log step start
                if trace:
                    from ailib.tracing._core import TraceStep

                    trace.add_step(
                        TraceStep(
                            name=f"step_{context.step_count}_start",
                            step_type="workflow_step",
                            start_time=time.time(),
                            metadata={
                                "step": str(step),
                                "type": step.step_type.value,
                            },
                        )
                    )

                # Execute step
                result = await step.execute(context)

                # Update context
                context.set_result(step.name, result)

                # Log step end
                if trace:
                    from ailib.tracing._core import TraceStep

                    trace.add_step(
                        TraceStep(
                            name=f"step_{context.step_count}_end",
                            step_type="workflow_step",
                            start_time=time.time(),
                            metadata={
                                "result": str(result)[:200],  # Truncate for tracing
                            },
                        )
                    )

            # Store context for state propagation in sub-workflows
            self._last_context = context

            # Return final result
            return context.current_result

        except Exception as e:
            context.errors.append(e)
            if trace:
                from ailib.tracing._core import TraceStep

                trace.add_step(
                    TraceStep(
                        name="workflow_error",
                        step_type="workflow",
                        start_time=time.time(),
                        metadata={"error": str(e)},
                    )
                )
            raise

        finally:
            if trace:
                from ailib.tracing._core import TraceStep

                trace.add_step(
                    TraceStep(
                        name="workflow_end",
                        step_type="workflow",
                        start_time=time.time(),
                        metadata={
                            "total_steps": context.step_count,
                            "final_state": context.state,
                        },
                    )
                )
            self._trace = trace

    def get_trace(self):
        """Get execution trace."""
        return self._trace

    def add_step(self, step: WorkflowStep) -> Workflow:
        """Add a step to the workflow."""
        self.steps.append(step)
        return self
