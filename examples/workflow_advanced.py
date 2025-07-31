"""Advanced workflow examples showcasing all features."""

import asyncio

from ailib import create_workflow
from pydantic import BaseModel, Field


# Schema definitions for structured outputs
class SentimentAnalysis(BaseModel):
    """Schema for sentiment analysis output."""

    sentiment: str = Field(description="positive, negative, or neutral")
    confidence: float = Field(ge=0, le=1, description="Confidence score")
    keywords: list[str] = Field(description="Key words/phrases")


class ProductIdea(BaseModel):
    """Schema for product ideas."""

    name: str
    description: str
    target_market: str
    unique_value: str


class CustomerTicket(BaseModel):
    """Schema for customer support tickets."""

    category: str = Field(description="billing, technical, or general")
    priority: str = Field(description="low, medium, or high")
    summary: str
    suggested_action: str


async def demo_schema_validation():
    """Demonstrate schema validation with automatic retry."""
    print("=== Schema Validation Demo ===\n")

    # Workflow with schema validation
    _workflow = create_workflow().step(
        "Analyze sentiment of: {text}",
        output_schema=SentimentAnalysis,
        retry=3,  # Will retry up to 3 times if output doesn't match schema
    )

    # In a real scenario, this would call an LLM
    # For demo, we'll show the expected structure
    print("Expected output schema:")
    print("- sentiment: str (positive/negative/neutral)")
    print("- confidence: float (0-1)")
    print("- keywords: List[str]")
    print(
        "\nThe workflow will automatically retry if LLM output doesn't match "
        "this schema."
    )


async def demo_error_handling():
    """Demonstrate comprehensive error handling."""
    print("\n=== Error Handling Demo ===\n")

    # Workflow with multiple error handlers
    _workflow = (
        create_workflow()
        .step("Call external API: {endpoint}")
        .on_error(ConnectionError)
        .do("Switch to backup API")
        .retry(max_attempts=3, backoff_factor=2)
        .then_continue()
        .on_error(ValueError)
        .do("Log error", "Use cached data")
        .then_continue()
        .on_error()  # Catch all other errors
        .do("Send alert to admin")
        .finally_do("Clean up resources")
    )

    print("Error handling structure:")
    print("- ConnectionError → Switch to backup API (with retry)")
    print("- ValueError → Log error and use cached data")
    print("- Any other error → Send alert and clean up")


async def demo_complex_customer_support():
    """Demonstrate a production-ready customer support workflow."""
    print("\n=== Complex Customer Support Workflow ===\n")

    # Define sub-workflows for different ticket types
    billing_workflow = (
        create_workflow()
        .step("Look up customer billing history: {customer_id}")
        .step("Identify billing issue pattern")
        .step("Generate resolution with refund amount if applicable")
        .require_approval()  # Human approval for refunds
        .step("Apply resolution and send confirmation")
    )

    technical_workflow = (
        create_workflow()
        .parallel(
            "Search knowledge base for: {issue_description}",
            "Check system status",
            "Look for similar resolved tickets",
        )
        .all()
        .step("Generate technical solution")
        .if_(lambda r: "critical" in str(r).lower())
        .then("Escalate to engineering team")
        .else_("Send solution to customer")
    )

    # Main workflow with composition
    _workflow = (
        create_workflow()
        .with_state({"tickets_processed": 0, "escalations": 0, "resolutions": []})
        # Categorize and validate ticket
        .step(
            "Categorize ticket: {message}",
            name="ticket_info",
            output_schema=CustomerTicket,
            retry=2,
        )
        # Update state
        .step(
            lambda ctx: ctx.update_state(
                {"tickets_processed": ctx.state["tickets_processed"] + 1}
            )
        )
        # Route based on category
        .if_(lambda r: r.category == "billing")
        .then(billing_workflow.build())
        .elif_(lambda r: r.category == "technical")
        .then(technical_workflow.build())
        .else_(
            create_workflow()
            .step("Generate general support response")
            .step("Add to FAQ suggestions: {message}")
            .build()
        )
        # Error handling for the entire workflow
        .on_error()
        .do("Log error to monitoring system")
        .do("Create manual review ticket")
        .finally_do("Update ticket status to 'requires attention'")
    )

    print("Workflow structure:")
    print("1. Categorize ticket with schema validation")
    print("2. Update processing statistics")
    print("3. Route to specialized sub-workflow:")
    print("   - Billing → History lookup → Resolution → Approval → Confirmation")
    print("   - Technical → Parallel search → Solution → Conditional escalation")
    print("   - General → Response generation → FAQ update")
    print("4. Comprehensive error handling with logging and manual review")


async def demo_data_pipeline():
    """Demonstrate a data processing pipeline with checkpoints."""
    print("\n=== Data Processing Pipeline ===\n")

    _workflow = (
        create_workflow()
        .with_state(
            {"total_records": 0, "processed": 0, "errors": [], "checkpoint": None}
        )
        # Load and validate data
        .step("Load data from: {source}")
        .step("Validate data format", output_schema=DataValidation)
        .checkpoint_after("data_validation")
        # Process in batches with parallel execution
        .step("Split into batches of 100")
        .for_each("batch")
        .do(
            "Clean and normalize batch",
            "Enrich with external data",
            "Apply transformations",
        )
        # Checkpoint after processing
        .checkpoint_after("batch_processing")
        # Aggregate and generate report
        .parallel(
            "Calculate statistics", "Identify anomalies", "Generate quality metrics"
        )
        .all()
        .step("Generate final report with {processed} records")
        # Error recovery
        .on_error()
        .do("Save partial results")
        .do("Generate error report: {errors}")
        .do("Restore from last checkpoint")
        .then_continue()
    )

    print("Pipeline features:")
    print("- State management with statistics tracking")
    print("- Checkpointing for failure recovery")
    print("- Parallel batch processing (10 concurrent)")
    print("- Error collection without stopping pipeline")
    print("- Automatic checkpoint restoration on failure")


async def demo_multi_agent_collaboration():
    """Demonstrate multiple agents working together."""
    print("\n=== Multi-Agent Collaboration ===\n")

    # Research agent
    research_workflow = (
        create_workflow()
        .parallel(
            "Search academic papers on: {topic}",
            "Search news articles on: {topic}",
            "Search patents on: {topic}",
        )
        .all()
        .step("Synthesize findings into research summary")
    )

    # Analysis agent
    analysis_workflow = (
        create_workflow()
        .step("Analyze market potential of: {research_summary}")
        .step("Identify key opportunities")
        .step("Assess risks and challenges")
    )

    # Strategy agent
    strategy_workflow = (
        create_workflow()
        .step("Generate business strategies based on: {analysis}")
        .step("Prioritize strategies by impact and feasibility")
        .step("Create implementation roadmap")
    )

    # Coordinator workflow
    _coordinator = (
        create_workflow()
        .step("Refine research topic: {initial_topic}", name="topic")
        # Research phase
        .use(research_workflow)
        .step("Review research quality")
        .if_(lambda r: "insufficient" in str(r).lower())
        .then("Expand research scope and retry")
        .else_("Proceed to analysis")
        # Analysis phase
        .use(analysis_workflow)
        # Strategy phase with human approval
        .use(strategy_workflow)
        .require_approval(
            notify=["strategy-team@company.com"],
            timeout="24 hours",
            message="Please review the proposed strategies",
        )
        # Final output generation
        .step("Generate executive summary")
        .step("Create presentation deck")
        .step("Schedule follow-up actions")
    )

    print("Multi-agent collaboration:")
    print("1. Coordinator refines the research topic")
    print("2. Research Agent performs parallel searches")
    print("3. Quality check with conditional re-research")
    print("4. Analysis Agent evaluates findings")
    print("5. Strategy Agent creates business plan")
    print("6. Human approval required for strategies")
    print("7. Final deliverables generated")


# Helper class for demo
class DataValidation(BaseModel):
    """Schema for data validation results."""

    is_valid: bool
    record_count: int
    issues: list[str] = Field(default_factory=list)


async def demo_advanced_features():
    """Demonstrate various advanced features."""
    print("\n=== Advanced Features Showcase ===\n")

    print("1. Dynamic Tool Selection:")
    _workflow = (
        create_workflow()
        .step("Analyze task: {task}")
        .step(
            lambda ctx: (
                "Use math tools"
                if "calculate" in ctx.get("task", "")
                else "Use search tools"
            )
        )
    )

    print("   ✓ Workflow dynamically selects tools based on task analysis\n")

    print("2. Conditional Parallel Execution:")
    _workflow = (
        create_workflow()
        .step("Assess urgency: {request}")
        .if_(lambda r: "urgent" in str(r).lower())
        .then(
            create_workflow()
            .parallel(  # Fast path for urgent requests
                "Quick solution", "Notify on-call team"
            )
            .race()  # Return first to complete
            .build()
        )
        .else_("Standard processing")
    )

    print("   ✓ Urgent requests use parallel fast-path with race condition\n")

    print("3. Recursive Workflows:")
    print("   ✓ Workflows can call themselves for recursive problems")
    print("   ✓ Built-in depth limiting prevents infinite recursion\n")

    print("4. Event-Driven Workflows:")
    print("   ✓ Workflows can be triggered by external events")
    print("   ✓ Support for webhooks and message queues")
    print("   ✓ Async execution with result callbacks\n")


def main():
    """Run all demos."""
    print("AILib Advanced Workflow Examples")
    print("=" * 60)
    print("\nThese examples demonstrate production-ready workflow patterns.")
    print("In real usage, workflows would execute with LLM/tool integration.\n")

    # Run demos synchronously for display purposes
    asyncio.run(demo_schema_validation())
    asyncio.run(demo_error_handling())
    asyncio.run(demo_complex_customer_support())
    asyncio.run(demo_data_pipeline())
    asyncio.run(demo_multi_agent_collaboration())
    asyncio.run(demo_advanced_features())

    print("\n" + "=" * 60)
    print("Key Takeaways:")
    print("- Workflows start simple but can handle complex scenarios")
    print("- Schema validation ensures reliable LLM outputs")
    print("- Comprehensive error handling prevents failures")
    print("- State management enables sophisticated logic")
    print("- Composition allows building complex from simple")
    print("- Human-in-the-loop for critical decisions")
    print("- Production-ready patterns for real applications")


if __name__ == "__main__":
    main()
