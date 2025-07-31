"""Workflow builder with fluent API."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from pydantic import BaseModel

from ailib.workflows._core import (
    ConditionalStep,
    SimpleStep,
    Workflow,
    WorkflowContext,
    WorkflowStep,
)
from ailib.workflows._steps import (
    ForEachStep,
    ParallelStep,
    WhileStep,
)


class WorkflowBuilder:
    """Fluent API for building workflows."""

    def __init__(self, initial_steps: list[str | WorkflowStep] | None = None):
        self.steps: list[WorkflowStep] = []
        self._current_step: WorkflowStep | None = None
        self._pending_condition: Callable | None = None
        self._pending_then: WorkflowStep | None = None
        self._state: dict[str, Any] = {}

        # Handle initial steps
        if initial_steps:
            for step in initial_steps:
                if isinstance(step, str):
                    self.step(step)
                else:
                    self.steps.append(step)

    def step(
        self,
        prompt_or_func: str | Callable,
        name: str | None = None,
        retry: int = 0,
        timeout: float | None = None,
        output_schema: type[BaseModel] | None = None,
        **kwargs,
    ) -> WorkflowBuilder:
        """Add a simple step."""
        step = SimpleStep(
            prompt_or_func,
            name=name,
            retry=retry,
            timeout=timeout,
            output_schema=output_schema,
        )

        self._add_step(step)
        return self

    def if_(
        self,
        condition: Callable[[Any], bool] | str,
    ) -> WorkflowBuilder:
        """Start a conditional branch."""
        # Handle string conditions (for simpler syntax)
        if isinstance(condition, str):
            # Convert string like "result == 'yes'" to lambda
            def condition_func(result):
                return eval(condition, {"result": result})

        else:
            condition_func = condition

        self._pending_condition = condition_func
        return self

    # Alias for more natural syntax
    if__ = if_

    def then(
        self,
        step_or_prompt: str | WorkflowStep | list[str | WorkflowStep] | Workflow,
    ) -> WorkflowBuilder:
        """Define the 'then' branch of a condition."""
        if not self._pending_condition:
            raise ValueError("then() must be called after if_()")

        from ailib.workflows._core import Workflow
        from ailib.workflows._steps import SubWorkflowStep

        # Convert to step(s)
        if isinstance(step_or_prompt, str):
            then_step = SimpleStep(step_or_prompt)
        elif isinstance(step_or_prompt, list):
            # Create a sub-workflow for multiple steps
            sub_builder = WorkflowBuilder(step_or_prompt)
            workflow = sub_builder.build()
            then_step = SubWorkflowStep(workflow)
        elif isinstance(step_or_prompt, Workflow):
            then_step = SubWorkflowStep(step_or_prompt)
        else:
            then_step = step_or_prompt

        self._pending_then = then_step
        return self

    def else_(
        self,
        step_or_prompt: str | WorkflowStep | list[str | WorkflowStep] | Workflow,
    ) -> WorkflowBuilder:
        """Define the 'else' branch of a condition."""
        if not self._pending_then:
            raise ValueError("else_() must be called after then()")

        from ailib.workflows._core import Workflow
        from ailib.workflows._steps import SubWorkflowStep

        # Convert to step
        if isinstance(step_or_prompt, str):
            else_step = SimpleStep(step_or_prompt)
        elif isinstance(step_or_prompt, list):
            sub_builder = WorkflowBuilder(step_or_prompt)
            workflow = sub_builder.build()
            else_step = SubWorkflowStep(workflow)
        elif isinstance(step_or_prompt, Workflow):
            else_step = SubWorkflowStep(step_or_prompt)
        else:
            else_step = step_or_prompt

        # Create conditional step
        conditional = ConditionalStep(
            self._pending_condition,
            self._pending_then,
            else_step,
        )

        self._add_step(conditional)

        # Reset pending
        self._pending_condition = None
        self._pending_then = None

        return self

    def elif_(
        self,
        condition: Callable[[Any], bool] | str,
    ) -> WorkflowBuilder:
        """Add an elif branch."""
        # First complete the pending if-then as if-then-else(check next condition)
        if self._pending_then and not self._pending_condition:
            # We're in an elif chain
            pass
        else:
            # Need to close the current if-then first
            self.else_(lambda ctx: None)  # Empty else

        # Start new condition
        return self.if_(condition)

    def while_(
        self,
        condition: Callable[[WorkflowContext], bool],
    ) -> WhileBuilder:
        """Start a while loop."""
        return WhileBuilder(self, condition)

    def for_each(
        self,
        item_name: str = "item",
        items_from: str | None = None,
    ) -> ForEachBuilder:
        """Start a for-each loop."""
        return ForEachBuilder(self, item_name, items_from)

    def parallel(
        self,
        *steps: str | WorkflowStep,
    ) -> ParallelBuilder:
        """Execute steps in parallel."""
        return ParallelBuilder(self, list(steps))

    def with_state(
        self,
        initial_state: dict[str, Any],
    ) -> WorkflowBuilder:
        """Set initial workflow state."""
        self._state = initial_state
        return self

    def checkpoint_after(
        self,
        step_name: str,
    ) -> WorkflowBuilder:
        """Add checkpoint after a named step."""
        # This will be implemented with state persistence
        # For now, just mark the intent
        return self

    def on_error(
        self,
        error_type: type[Exception] | None = None,
    ) -> ErrorHandlerBuilder:
        """Add error handling."""
        return ErrorHandlerBuilder(self, error_type)

    def require_approval(
        self,
        **kwargs,
    ) -> WorkflowBuilder:
        """Add human approval step."""
        # Will implement with HITL functionality
        from ailib.workflows._steps import HumanApprovalStep

        step = HumanApprovalStep(**kwargs)
        self._add_step(step)
        return self

    def use(
        self,
        workflow: Workflow | WorkflowBuilder,
    ) -> WorkflowBuilder:
        """Compose with another workflow."""
        if isinstance(workflow, WorkflowBuilder):
            workflow = workflow.build()

        from ailib.workflows._steps import SubWorkflowStep

        step = SubWorkflowStep(workflow)
        self._add_step(step)
        return self

    def build(self) -> Workflow:
        """Build the workflow."""
        # Handle any pending conditions
        if self._pending_then:
            # Close with empty else
            self.else_(lambda ctx: None)

        return Workflow(
            steps=self.steps,
            initial_state=self._state,
        )

    def run(self, **kwargs) -> Any:
        """Build and run the workflow."""
        workflow = self.build()

        # Handle async execution
        import asyncio

        if asyncio.iscoroutinefunction(workflow.run):
            # If in async context, await it
            try:
                asyncio.get_running_loop()
                return asyncio.create_task(workflow.run(**kwargs))
            except RuntimeError:
                # Not in async context, run in new loop
                return asyncio.run(workflow.run(**kwargs))

        return workflow.run(**kwargs)

    def _add_step(self, step: WorkflowStep) -> None:
        """Add a step to the workflow."""
        self.steps.append(step)
        self._current_step = step


class WhileBuilder:
    """Builder for while loops."""

    def __init__(self, parent: WorkflowBuilder, condition: Callable):
        self.parent = parent
        self.condition = condition
        self.body_steps: list[WorkflowStep] = []

    def do(
        self,
        *steps: str | WorkflowStep,
    ) -> WhileBuilder:
        """Define loop body."""
        for step in steps:
            if isinstance(step, str):
                self.body_steps.append(SimpleStep(step))
            else:
                self.body_steps.append(step)
        return self

    def break_if(
        self,
        condition: Callable[[WorkflowContext], bool],
    ) -> WorkflowBuilder:
        """Add break condition and return to main builder."""
        while_step = WhileStep(
            self.condition,
            self.body_steps,
            break_condition=condition,
        )
        self.parent._add_step(while_step)
        return self.parent


class ForEachBuilder:
    """Builder for for-each loops."""

    def __init__(self, parent: WorkflowBuilder, item_name: str, items_from: str | None):
        self.parent = parent
        self.item_name = item_name
        self.items_from = items_from
        self.body_steps: list[WorkflowStep] = []
        self.parallel_config = None

    def do(
        self,
        *steps: str | WorkflowStep | Workflow,
    ) -> WorkflowBuilder:
        """Define loop body and return to main workflow."""
        from ailib.workflows._core import Workflow
        from ailib.workflows._steps import SubWorkflowStep

        for step in steps:
            if isinstance(step, str):
                self.body_steps.append(SimpleStep(step))
            elif isinstance(step, Workflow):
                # Wrap Workflow in SubWorkflowStep
                self.body_steps.append(SubWorkflowStep(step))
            else:
                self.body_steps.append(step)
        # Automatically build and return to parent
        return self._build_and_return()

    def parallel(
        self,
        max_concurrent: int = 5,
    ) -> WorkflowBuilder:
        """Execute iterations in parallel."""
        self.parallel_config = {"max_concurrent": max_concurrent}
        return self._build_and_return()

    def _build_and_return(self) -> WorkflowBuilder:
        """Build the for-each step and return to parent."""
        for_each_step = ForEachStep(
            self.item_name,
            self.items_from,
            self.body_steps,
            parallel_config=self.parallel_config,
        )
        self.parent._add_step(for_each_step)
        return self.parent


class ParallelBuilder:
    """Builder for parallel execution."""

    def __init__(self, parent: WorkflowBuilder, steps: list[str | WorkflowStep]):
        self.parent = parent
        self.steps = steps
        self.strategy = "all"  # Default strategy

    def all(self) -> WorkflowBuilder:
        """Wait for all steps to complete."""
        self.strategy = "all"
        return self._build_and_return()

    def race(self) -> WorkflowBuilder:
        """Return first to complete."""
        self.strategy = "race"
        return self._build_and_return()

    def any(self, count: int = 1) -> WorkflowBuilder:
        """Wait for any N to complete."""
        self.strategy = f"any:{count}"
        return self._build_and_return()

    def combine(self) -> WorkflowBuilder:
        """Combine all results (default)."""
        return self.all()

    def _build_and_return(self) -> WorkflowBuilder:
        """Build parallel step and return to parent."""
        # Convert strings to steps
        steps = []
        for step in self.steps:
            if isinstance(step, str):
                steps.append(SimpleStep(step))
            else:
                steps.append(step)

        parallel_step = ParallelStep(steps, strategy=self.strategy)
        self.parent._add_step(parallel_step)
        return self.parent


class ErrorHandlerBuilder:
    """Builder for error handling."""

    def __init__(self, parent: WorkflowBuilder, error_type: type[Exception] | None):
        self.parent = parent
        self.error_type = error_type or Exception
        self.handler_steps: list[WorkflowStep] = []
        self.retry_config = {"max_retries": 1, "backoff_factor": 2.0}

    def do(
        self,
        *steps: str | WorkflowStep,
    ) -> ErrorHandlerBuilder:
        """Define error handling steps."""
        for step in steps:
            if isinstance(step, str):
                self.handler_steps.append(SimpleStep(step))
            else:
                self.handler_steps.append(step)
        return self

    def retry(
        self,
        max_attempts: int = 3,
        backoff_factor: float = 2.0,
    ) -> ErrorHandlerBuilder:
        """Configure retry behavior."""
        self.retry_config["max_retries"] = max_attempts
        self.retry_config["backoff_factor"] = backoff_factor
        return self

    def finally_do(
        self,
        *steps: str | WorkflowStep,
    ) -> WorkflowBuilder:
        """Add finally block and return to workflow."""
        # Import here to avoid circular import
        from ailib.workflows._error_handling import ErrorHandler, ErrorHandlingStep

        # Get the last step from parent
        if not self.parent.steps:
            raise ValueError("No step to add error handling to")

        last_step = self.parent.steps.pop()

        # Create error handler
        handler = ErrorHandler(
            error_type=self.error_type,
            handler=(
                self.handler_steps
                if len(self.handler_steps) > 1
                else self.handler_steps[0] if self.handler_steps else None
            ),
            max_retries=self.retry_config["max_retries"],
            backoff_factor=self.retry_config["backoff_factor"],
        )

        # Wrap the step with error handling
        wrapped_step = ErrorHandlingStep(
            last_step,
            error_handlers=[handler] if handler.handler else [],
            default_handler=None,
        )

        # Add wrapped step back
        self.parent._add_step(wrapped_step)

        # Add finally steps if provided
        for step in steps:
            if isinstance(step, str):
                self.parent._add_step(SimpleStep(step))
            else:
                self.parent._add_step(step)

        return self.parent

    def then_continue(self) -> WorkflowBuilder:
        """Continue execution after handling error."""
        return self.finally_do()


def create_workflow(
    steps: str | list[str | WorkflowStep] | None = None,
) -> WorkflowBuilder:
    """Create a workflow with optional initial steps.

    Examples:
        # Simple one-liner
        result = create_workflow("Summarize this: {text}").run(text="...")

        # Multi-step
        workflow = create_workflow([
            "Extract key points",
            "Rank by importance"
        ])

        # Fluent API
        workflow = create_workflow()
            .step("Analyze sentiment")
            .if_(lambda r: "negative" in r)
                .then("Escalate to manager")
                .else_("Send standard response")
    """
    if isinstance(steps, str):
        # Single step workflow - return builder for immediate execution
        return WorkflowBuilder([steps])
    elif isinstance(steps, list):
        # Multi-step workflow
        return WorkflowBuilder(steps)
    else:
        # Empty builder for fluent API
        return WorkflowBuilder()


def create_workflow_template() -> WorkflowTemplateBuilder:
    """Create a reusable workflow template."""
    # Will implement template functionality
    return WorkflowTemplateBuilder()


class WorkflowTemplateBuilder:
    """Builder for workflow templates."""

    def __init__(self):
        self.template_steps = []

    def create(self, **params) -> Workflow:
        """Create workflow instance from template."""
        # Will implement template instantiation
        pass
