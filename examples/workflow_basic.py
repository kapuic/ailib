"""Basic workflow examples demonstrating progressive complexity."""

import os

from ailib import create_workflow


async def example_simple_workflow():
    """Example 1: The simplest workflow."""
    print("=== Example 1: Simple Workflow ===")

    # Simplest case - one step
    result = await create_workflow("Summarize: The sky is blue").run()
    print(f"Result: {result}")

    # Multi-step workflow
    workflow = create_workflow(
        [
            "Extract key points from: {text}",
            "Rank them by importance",
            "Format as bullet points",
        ]
    )

    result = await workflow.run(text="AI is transforming industries...")
    print(f"\nMulti-step result: {result}")


async def example_conditional_workflow():
    """Example 2: Conditional branching."""
    print("\n=== Example 2: Conditional Workflow ===")

    # Simple if-then-else
    workflow = (
        create_workflow()
        .step("Analyze sentiment: {text}")
        .if_(lambda result: "positive" in result.lower())
        .then("Generate enthusiastic response")
        .else_("Generate empathetic response")
    )

    # Test with positive sentiment
    result = await workflow.run(text="I love this product!")
    print(f"Positive sentiment result: {result}")

    # Test with negative sentiment
    result = await workflow.run(text="This is disappointing")
    print(f"Negative sentiment result: {result}")


async def example_loop_workflow():
    """Example 3: Loops and iteration."""
    print("\n=== Example 3: Loop Workflow ===")

    # For-each loop
    workflow = (
        create_workflow()
        .step("Generate 3 product names")
        .for_each("name")
        .do("Create tagline for: {name}")
    )

    result = await workflow.run()
    print(f"For-each result: {result}")

    # While loop with state
    workflow = (
        create_workflow()
        .with_state({"attempts": 0, "success": False})
        .while_(lambda ctx: ctx.state["attempts"] < 3 and not ctx.state["success"])
        .do(
            lambda ctx: {
                "attempts": ctx.state["attempts"] + 1,
                "success": ctx.state["attempts"] >= 2,
            }
        )
        .break_if(lambda ctx: ctx.get("success", False))
    )

    result = await workflow.run()
    print("While loop completed after attempts")


async def example_parallel_workflow():
    """Example 4: Parallel execution."""
    print("\n=== Example 4: Parallel Workflow ===")

    # Run multiple tasks in parallel
    workflow = (
        create_workflow()
        .parallel(
            "Search web for: {query}",
            "Search knowledge base for: {query}",
            "Search documentation for: {query}",
        )
        .all()
    )  # Wait for all to complete

    result = await workflow.run(query="workflow automation")
    print(f"Parallel search results: {result}")

    # Race condition - first to complete wins
    workflow = (
        create_workflow()
        .parallel("Quick analysis of: {topic}", "Detailed analysis of: {topic}")
        .race()
    )

    result = await workflow.run(topic="AI trends")
    print(f"Race result (first to complete): {result}")


async def example_complex_workflow():
    """Example 5: Complex real-world workflow."""
    print("\n=== Example 5: Complex Customer Support Workflow ===")

    # Customer support workflow with multiple branches
    workflow = (
        create_workflow()
        .step("Categorize issue: {message}", name="category")
        .if_(lambda r: "billing" in str(r).lower())
        .then(
            create_workflow()
            .step("Look up account: {customer_id}")
            .step("Check billing history")
            .step("Generate billing resolution")
            .build()
        )
        .elif_(lambda r: "technical" in str(r).lower())
        .then(
            create_workflow()
            .step("Identify technical issue")
            .step("Search knowledge base")
            .step("Generate technical solution")
            .build()
        )
        .else_("Forward to general support")
    )

    # Test billing issue
    result = await workflow.run(
        message="I was charged twice this month", customer_id="12345"
    )
    print(f"Billing issue result: {result}")

    # Test technical issue
    result = await workflow.run(message="My app keeps crashing", customer_id="12345")
    print(f"Technical issue result: {result}")


async def example_error_handling():
    """Example 6: Error handling and retry."""
    print("\n=== Example 6: Error Handling ===")

    # Workflow with retry logic - using a prompt that might need retry
    workflow = (
        create_workflow()
        .step(
            "Generate exactly 3 product names for a {product_type}. "
            "Return as a JSON array.",
            retry=2,  # Retry if output isn't valid JSON
        )
        .step("Format the names as a numbered list")
    )

    result = await workflow.run(product_type="smart home assistant")
    print(f"Result with retry handling: {result}")


async def example_schema_validation():
    """Example 7: Schema validation."""
    print("\n=== Example 7: Schema Validation ===")

    from pydantic import BaseModel

    class ProductIdea(BaseModel):
        name: str
        tagline: str
        target_market: str

    # Workflow with schema validation
    workflow = (
        create_workflow()
        .step(
            "Generate a product idea for the fitness industry",
            output_schema=ProductIdea,
        )
        .step("Create a 30-second pitch for: {name} - {tagline}")
    )

    result = await workflow.run()
    print(f"Generated pitch: {result}")


def main():
    """Run all examples."""
    # Check for API key
    if not os.getenv("OPENAI_API_KEY"):
        print("Error: OPENAI_API_KEY environment variable is required")
        print("Please set it to run the workflow examples.")
        exit(1)

    print("AILib Workflow Examples")
    print("=" * 50)

    # Run all examples
    import asyncio

    print("\nRunning workflow examples...\n")

    # Run each example
    asyncio.run(example_simple_workflow())
    asyncio.run(example_conditional_workflow())
    asyncio.run(example_loop_workflow())
    asyncio.run(example_parallel_workflow())
    asyncio.run(example_complex_workflow())
    asyncio.run(example_error_handling())
    asyncio.run(example_schema_validation())

    print("\n" + "=" * 50)
    print("✨ All workflow examples completed successfully!")


if __name__ == "__main__":
    main()
