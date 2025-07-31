"""Additional workflow step implementations."""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from ailib.workflows._core import StepType, WorkflowContext, WorkflowStep

if TYPE_CHECKING:
    from ailib.workflows._core import Workflow


class WhileStep(WorkflowStep):
    """A while loop step."""

    def __init__(
        self,
        condition: Callable[[WorkflowContext], bool],
        body_steps: list[WorkflowStep],
        break_condition: Callable[[WorkflowContext], bool] | None = None,
        max_iterations: int = 1000,  # Safety limit
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.condition = condition
        self.body_steps = body_steps
        self.break_condition = break_condition
        self.max_iterations = max_iterations
        self.step_type = StepType.LOOP

    async def execute(self, context: WorkflowContext) -> Any:
        """Execute the while loop."""
        iterations = 0
        results = []

        while iterations < self.max_iterations:
            # Check main condition
            if not self.condition(context):
                break

            # Execute body
            for step in self.body_steps:
                result = await step.execute(context)
                context.set_result(step.name, result)

            results.append(context.current_result)

            # Check break condition
            if self.break_condition and self.break_condition(context):
                break

            iterations += 1

        return results


class ForEachStep(WorkflowStep):
    """A for-each loop step."""

    def __init__(
        self,
        item_name: str,
        items_from: str | None,
        body_steps: list[WorkflowStep],
        parallel_config: dict[str, Any] | None = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.item_name = item_name
        self.items_from = items_from
        self.body_steps = body_steps
        self.parallel_config = parallel_config
        self.step_type = StepType.LOOP

    async def execute(self, context: WorkflowContext) -> Any:
        """Execute the for-each loop."""
        # Get items to iterate over
        if self.items_from:
            items = context.get(self.items_from, [])
        else:
            # Use previous result
            items = context.current_result

        if not isinstance(items, list):
            items = [items]

        results = []

        if self.parallel_config:
            # Parallel execution
            max_concurrent = self.parallel_config.get("max_concurrent", 5)

            async def process_item(item):
                # Create a sub-context for this iteration
                item_context = WorkflowContext(
                    variables={**context.variables, self.item_name: item},
                    results=context.results.copy(),
                    state=context.state.copy(),
                    llm_client=context.llm_client,
                )

                # Execute body steps
                for step in self.body_steps:
                    result = await step.execute(item_context)
                    item_context.set_result(step.name, result)

                return item_context.current_result

            # Process in batches
            for i in range(0, len(items), max_concurrent):
                batch = items[i : i + max_concurrent]
                batch_results = await asyncio.gather(
                    *[process_item(item) for item in batch]
                )
                results.extend(batch_results)
        else:
            # Sequential execution
            for item in items:
                # Update context with current item
                context.variables[self.item_name] = item

                # Execute body steps
                for step in self.body_steps:
                    result = await step.execute(context)
                    context.set_result(step.name, result)

                results.append(context.current_result)

        return results


class ParallelStep(WorkflowStep):
    """Execute multiple steps in parallel."""

    def __init__(
        self,
        steps: list[WorkflowStep],
        strategy: str = "all",
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.steps = steps
        self.strategy = strategy
        self.step_type = StepType.PARALLEL

    async def execute(self, context: WorkflowContext) -> Any:
        """Execute steps in parallel based on strategy."""
        # Create tasks for all steps
        tasks = []
        for step in self.steps:
            # Create a sub-context for each parallel branch
            sub_context = WorkflowContext(
                variables=context.variables.copy(),
                results=context.results.copy(),
                state=context.state.copy(),
                llm_client=context.llm_client,
            )
            task = asyncio.create_task(step.execute(sub_context))
            tasks.append((task, step.name))

        if self.strategy == "race":
            # Return first to complete
            done, pending = await asyncio.wait(
                [t[0] for t in tasks], return_when=asyncio.FIRST_COMPLETED
            )

            # Cancel pending tasks
            for task in pending:
                task.cancel()

            # Return first result
            return done.pop().result()

        elif self.strategy.startswith("any:"):
            # Wait for N to complete
            count = int(self.strategy.split(":")[1])
            done, pending = await asyncio.wait(
                [t[0] for t in tasks], return_when=asyncio.FIRST_COMPLETED
            )

            results = []
            while len(results) < count and (done or pending):
                if done:
                    task = done.pop()
                    results.append(task.result())

                if len(results) < count and pending:
                    done, pending = await asyncio.wait(
                        pending, return_when=asyncio.FIRST_COMPLETED
                    )

            # Cancel remaining
            for task in pending:
                task.cancel()

            return results

        else:  # "all" - default
            # Wait for all to complete
            results = await asyncio.gather(*[t[0] for t in tasks])

            # If steps have names, return dict
            if any(t[1] for t in tasks):
                return {
                    name: result
                    for (_, name), result in zip(tasks, results, strict=False)
                    if name
                }

            return results


class SubWorkflowStep(WorkflowStep):
    """Execute a sub-workflow as a step."""

    def __init__(
        self,
        workflow: Workflow,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.workflow = workflow
        self.step_type = StepType.SUBWORKFLOW

    async def execute(self, context: WorkflowContext) -> Any:
        """Execute the sub-workflow."""
        # Pass current context variables and state to sub-workflow
        # The sub-workflow will inherit the parent's state
        original_state = getattr(self.workflow, "initial_state", {}).copy()

        # Set the sub-workflow's initial state to include parent's state
        self.workflow.initial_state = {**original_state, **context.state}

        try:
            # Run the sub-workflow
            result = await self.workflow.run(**context.variables)

            # After execution, merge any state changes back to parent context
            # This is important for stateful workflows
            if hasattr(self.workflow, "_last_context"):
                # Update parent's state with changes from sub-workflow
                context.state.update(self.workflow._last_context.state)

            return result
        finally:
            # Restore original state to avoid side effects
            self.workflow.initial_state = original_state


class HumanApprovalStep(WorkflowStep):
    """Human-in-the-loop approval step."""

    def __init__(
        self,
        notify: list[str] | None = None,
        timeout: str | None = None,
        message: str | None = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.notify = notify or []
        self.timeout = timeout
        self.message = message
        self.step_type = StepType.HUMAN_APPROVAL

    async def execute(self, context: WorkflowContext) -> Any:
        """Execute human approval step."""
        # For now, this is a placeholder
        # In a real implementation, this would:
        # 1. Send notifications to approvers
        # 2. Wait for approval/rejection
        # 3. Handle timeout

        # Simulate approval for testing
        await asyncio.sleep(0.1)  # Simulate waiting

        return {
            "approved": True,
            "approver": "simulated",
            "timestamp": "2024-01-01T00:00:00Z",
            "comments": "Auto-approved in test mode",
        }


class ErrorHandlerStep(WorkflowStep):
    """Error handling step."""

    def __init__(
        self,
        error_type: type | None = None,
        handler_steps: list[WorkflowStep] | None = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.error_type = error_type or Exception
        self.handler_steps = handler_steps or []

    async def execute(self, context: WorkflowContext) -> Any:
        """Execute error handling."""
        # This is typically wrapped around other steps
        # Rather than executed directly
        pass
