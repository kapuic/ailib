"""Tests for the workflow system."""

from unittest.mock import Mock, patch

import pytest
from ailib import create_workflow
from ailib.workflows import WorkflowState
from ailib.workflows._core import ConditionalStep
from ailib.workflows._steps import ForEachStep, ParallelStep, WhileStep


class TestBasicWorkflows:
    """Test basic workflow functionality."""

    def test_simple_workflow_creation(self):
        """Test creating a simple workflow."""
        # One-liner
        workflow = create_workflow("Test step")
        assert workflow is not None

        # Multi-step
        workflow = create_workflow(["Step 1", "Step 2", "Step 3"])
        assert len(workflow.steps) == 3

    def test_fluent_api(self):
        """Test fluent API for workflow building."""
        workflow = (
            create_workflow()
            .step("First step")
            .step("Second step", name="second")
            .step("Third step")
        )

        assert len(workflow.steps) == 3
        assert workflow.steps[1].name == "second"

    @pytest.mark.asyncio
    async def test_simple_execution(self):
        """Test executing a simple workflow."""
        # Mock the client creation
        with patch("ailib.workflows._core.create_client") as mock_create_client:
            # Create a mock client
            mock_client = Mock()
            mock_create_client.return_value = mock_client

            # Mock the chain execution
            with patch("ailib.chains.Chain.run") as mock_run:
                mock_run.return_value = "Test result"

                workflow = create_workflow("Test prompt: {input}")
                result = await workflow.build().run(input="test")

                assert result == "Test result"

    def test_state_management(self):
        """Test workflow state management."""
        workflow = (
            create_workflow()
            .with_state({"counter": 0})
            .step(lambda ctx: ctx.state["counter"] + 1)
        )

        wf = workflow.build()
        assert wf.initial_state["counter"] == 0


class TestConditionalBranching:
    """Test conditional branching in workflows."""

    def test_if_then_else(self):
        """Test if-then-else branching."""
        workflow = (
            create_workflow()
            .step("Check condition")
            .if_(lambda result: "yes" in result)
            .then("Positive branch")
            .else_("Negative branch")
        )

        wf = workflow.build()
        assert len(wf.steps) == 2
        assert isinstance(wf.steps[1], ConditionalStep)

    def test_multiple_conditions(self):
        """Test elif chains."""
        workflow = (
            create_workflow()
            .step("Get category")
            .if_(lambda r: r == "A")
            .then("Handle A")
            .elif_(lambda r: r == "B")
            .then("Handle B")
            .else_("Handle other")
        )

        # This creates nested conditionals
        wf = workflow.build()
        assert len(wf.steps) >= 2


class TestLoops:
    """Test loop constructs in workflows."""

    def test_while_loop(self):
        """Test while loop creation."""
        workflow = (
            create_workflow()
            .while_(lambda ctx: ctx.get("count", 0) < 3)
            .do("Process item")
            .break_if(lambda ctx: ctx.get("done", False))
        )

        wf = workflow.build()
        assert len(wf.steps) == 1
        assert isinstance(wf.steps[0], WhileStep)

    def test_for_each_loop(self):
        """Test for-each loop creation."""
        workflow = (
            create_workflow().step("Get items").for_each("item").do("Process {item}")
        )

        wf = workflow.build()
        assert len(wf.steps) == 2
        assert isinstance(wf.steps[1], ForEachStep)

    def test_parallel_for_each(self):
        """Test parallel for-each execution."""
        workflow = (
            create_workflow()
            .step("Get items")
            .for_each("item")
            .do("Process {item}")
            .parallel(max_concurrent=5)
        )

        wf = workflow.build()
        for_each_step = wf.steps[1]
        assert for_each_step.parallel_config["max_concurrent"] == 5


class TestParallelExecution:
    """Test parallel execution strategies."""

    def test_parallel_all(self):
        """Test parallel execution waiting for all."""
        workflow = create_workflow().parallel("Task 1", "Task 2", "Task 3").all()

        wf = workflow.build()
        assert len(wf.steps) == 1
        assert isinstance(wf.steps[0], ParallelStep)
        assert wf.steps[0].strategy == "all"

    def test_parallel_race(self):
        """Test parallel execution with race strategy."""
        workflow = create_workflow().parallel("Fast task", "Slow task").race()

        wf = workflow.build()
        assert wf.steps[0].strategy == "race"

    def test_parallel_any(self):
        """Test parallel execution waiting for N."""
        workflow = (
            create_workflow().parallel("Task 1", "Task 2", "Task 3", "Task 4").any(2)
        )

        wf = workflow.build()
        assert wf.steps[0].strategy == "any:2"


class TestErrorHandling:
    """Test error handling in workflows."""

    @pytest.mark.asyncio
    async def test_retry_on_error(self):
        """Test retry logic."""
        attempt_count = 0

        async def failing_step(ctx):
            nonlocal attempt_count
            attempt_count += 1
            if attempt_count < 3:
                raise ValueError("Test error")
            return "Success"

        workflow = create_workflow().step(failing_step, retry=3)

        wf = workflow.build()
        result = await wf.run()

        assert result == "Success"
        assert attempt_count == 3


class TestWorkflowComposition:
    """Test workflow composition and reusability."""

    def test_subworkflow(self):
        """Test using sub-workflows."""
        # Create a reusable workflow
        validation_workflow = create_workflow(
            ["Validate format", "Check constraints"]
        ).build()

        # Use in main workflow
        main_workflow = (
            create_workflow()
            .step("Load data")
            .use(validation_workflow)
            .step("Process data")
        )

        wf = main_workflow.build()
        assert len(wf.steps) == 3


class TestSchemaValidation:
    """Test schema validation with Pydantic."""

    def test_output_schema(self):
        """Test output schema validation."""
        from pydantic import BaseModel

        class OutputSchema(BaseModel):
            name: str
            age: int

        workflow = create_workflow().step(
            "Extract person info", output_schema=OutputSchema
        )

        wf = workflow.build()
        assert wf.steps[0].output_schema == OutputSchema


class TestStateManagement:
    """Test workflow state management."""

    def test_workflow_state(self):
        """Test WorkflowState class."""
        state = WorkflowState({"count": 0})

        # Test get/set
        assert state.get("count") == 0
        state.set("count", 1)
        assert state.get("count") == 1

        # Test update
        state.update({"count": 2, "name": "test"})
        assert state.get("count") == 2
        assert state.get("name") == "test"

        # Test checkpoint
        state.checkpoint("before_change")
        state.set("count", 10)
        assert state.get("count") == 10

        state.restore_checkpoint("before_change")
        assert state.get("count") == 2

        # Test rollback
        state.set("count", 5)
        state.rollback(1)
        assert state.get("count") == 2


class TestHumanInTheLoop:
    """Test human-in-the-loop functionality."""

    def test_approval_step(self):
        """Test adding approval steps."""
        workflow = (
            create_workflow()
            .step("Generate proposal")
            .require_approval(notify=["manager@example.com"], timeout="1 hour")
            .step("Execute approved plan")
        )

        wf = workflow.build()
        assert len(wf.steps) == 3

    def test_conditional_approval(self):
        """Test conditional approval."""
        workflow = (
            create_workflow()
            .step("Calculate risk")
            .if_(lambda r: r.get("risk_score", 0) > 0.8)
            .then(
                create_workflow()
                .require_approval(message="High risk!")
                .step("Proceed with caution")
                .build()
            )
            .else_("Proceed normally")
        )

        wf = workflow.build()
        assert len(wf.steps) == 2


# Example of integration test
class TestWorkflowIntegration:
    """Integration tests for complex workflows."""

    @pytest.mark.asyncio
    async def test_customer_support_workflow(self):
        """Test a complete customer support workflow."""
        with patch("ailib.chains.Chain.run") as mock_run:
            # Mock responses
            mock_run.side_effect = [
                {"category": "billing"},  # Categorization
                "Billing issue resolved",  # Resolution
            ]

            workflow = (
                create_workflow()
                .step("Categorize: {message}", name="category")
                .if_(lambda r: r.get("category") == "billing")
                .then("Resolve billing issue")
                .elif_(lambda r: r.get("category") == "technical")
                .then("Create tech ticket")
                .else_("Send to general support")
            )

            result = await workflow.build().run(message="I have a billing question")

            assert result == "Billing issue resolved"
