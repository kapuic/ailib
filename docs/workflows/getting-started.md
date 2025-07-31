# Getting Started with Workflows

This guide will walk you through creating your first workflows, from simple to complex.

## Installation

Workflows are included in the core AILib package:

```bash
pip install ailib
```

## Your First Workflow

### The Simplest Workflow

```python
from ailib import create_workflow

# One-line workflow
result = await create_workflow("Write a haiku about Python").run()
print(result)
```

That's it! No configuration, no boilerplate.

### Multi-Step Workflow

Chain multiple operations together:

```python
# Simple chaining - each step can access the previous result
workflow = create_workflow([
    "Write a haiku about {topic}",
    "Translate this to Japanese: {previous}",  # {previous} contains the haiku
    "Explain the cultural significance of: {previous_result}"  # or use {previous_result}
])

result = await workflow.run(topic="cherry blossoms")

# For more control, use named steps
workflow = create_workflow()
    .step("Write a haiku about {topic}", name="haiku")
    .step("Translate to Japanese: {haiku}")  # Reference by name
    .step("Explain each line of: {haiku}")
```

## Adding Logic with Conditions

### Simple If-Then-Else

```python
workflow = (
    create_workflow()
    .step("Analyze the sentiment of: {text}")
    .if_(lambda result: "positive" in result.lower())
    .then("Write a thank you note")
    .else_("Write an apology and offer assistance")
)

# Test with different inputs
positive_result = await workflow.run(text="Your product is amazing!")
negative_result = await workflow.run(text="This is disappointing")
```

### Multiple Conditions

```python
workflow = (
    create_workflow()
    .step("Categorize the issue: {issue}")
    .if_(lambda r: "billing" in r.lower())
    .then("Route to billing department")
    .elif_(lambda r: "technical" in r.lower())
    .then("Route to tech support")
    .elif_(lambda r: "shipping" in r.lower())
    .then("Route to logistics")
    .else_("Route to general support")
)
```

## Working with Loops

### Process a List (For-Each)

```python
workflow = (
    create_workflow()
    .step("List 5 popular programming languages")
    .for_each("language")
    .do("Write a one-line description of {language}")
)

result = await workflow.run()
# Returns descriptions for each language
```

### Parallel Processing

Process items concurrently for better performance:

```python
workflow = (
    create_workflow()
    .step("List customer emails from the database")
    .for_each("email")
    .do("Generate personalized message for {email}")
    .parallel(max_concurrent=5)  # Process 5 at a time
)
```

### While Loops

Repeat until a condition is met:

```python
workflow = (
    create_workflow()
    .with_state({"attempts": 0, "found": False})
    .while_(lambda ctx: ctx.state["attempts"] < 3 and not ctx.state["found"])
    .do(
        create_workflow()
        .step("Search for the answer to: {question}")
        .step(lambda ctx: {
            "attempts": ctx.state["attempts"] + 1,
            "found": "definitive answer" in ctx.current_result
        })
    )
)

result = await workflow.run(question="What is the meaning of life?")
```

## Parallel Execution

### Run Multiple Tasks Concurrently

```python
workflow = (
    create_workflow()
    .parallel(
        "Search Google for: {query}",
        "Search Wikipedia for: {query}",
        "Search documentation for: {query}"
    )
    .all()  # Wait for all to complete
)

results = await workflow.run(query="Python async programming")
# Returns all three search results
```

### Race Condition - First Wins

```python
workflow = (
    create_workflow()
    .parallel(
        "Quick summary of: {text}",
        "Detailed analysis of: {text}"
    )
    .race()  # Return first to complete
)

# Gets whichever finishes first
result = await workflow.run(text=long_article)
```

## State Management

### Maintaining Context

```python
workflow = (
    create_workflow()
    .with_state({"items_processed": 0, "total_cost": 0})
    .step("Get list of orders")
    .for_each("order")
    .do(
        lambda ctx: {
            "items_processed": ctx.state["items_processed"] + 1,
            "total_cost": ctx.state["total_cost"] + ctx.order.amount
        }
    )
    .step("Generate summary report using state.total_cost and state.items_processed")
)
```

## Error Handling

### Automatic Retry

```python
workflow = (
    create_workflow()
    .step(
        "Call external API: {endpoint}",
        retry=3,  # Retry up to 3 times on failure
        timeout=5.0  # 5 second timeout
    )
    .step("Process the response")
)
```

### Custom Error Handling

```python
workflow = (
    create_workflow()
    .step("Risky operation")
    .on_error()
    .do("Log the error", "Send alert to admin")
    .then_continue()  # Continue with workflow
)
```

## Schema Validation

Ensure outputs match expected structure:

```python
from pydantic import BaseModel

class ProductInfo(BaseModel):
    name: str
    price: float
    description: str

workflow = (
    create_workflow()
    .step(
        "Extract product info from: {webpage}",
        output_schema=ProductInfo  # Validates and retries if needed
    )
    .step("Format product for catalog")
)

result = await workflow.run(webpage=product_html)
# result is guaranteed to be valid ProductInfo
```

## Human-in-the-Loop

### Approval Gates

```python
workflow = (
    create_workflow()
    .step("Generate marketing campaign for: {product}")
    .require_approval(
        notify=["marketing-team@company.com"],
        timeout="24 hours"
    )
    .step("Launch the approved campaign")
)
```

## Composing Workflows

### Reusable Sub-Workflows

```python
# Define reusable workflow
validation_workflow = (
    create_workflow()
    .step("Validate format")
    .step("Check for errors")
    .step("Verify completeness")
)

# Use in another workflow
main_workflow = (
    create_workflow()
    .step("Extract data from: {source}")
    .use(validation_workflow)  # Embed the validation workflow
    .step("Process validated data")
)
```

## Complete Example: Content Pipeline

Here's a real-world example that combines multiple features:

```python
from ailib import create_workflow

content_pipeline = (
    create_workflow()
    # Extract articles
    .step("Extract article URLs from RSS feed: {feed_url}")
    .for_each("url")
    .do(
        create_workflow()
        # Process each article
        .step("Fetch article content from {url}")
        .step("Extract main text")
        .step("Summarize in 3 paragraphs")

        # Quality check
        .if_(lambda r: len(r) < 100)
        .then("Mark as low quality: too short")
        .else_(
            create_workflow()
            .step("Generate SEO metadata")
            .step("Create social media posts")
            .step("Save to database")
        )
    )
    .parallel(max_concurrent=3)  # Process 3 articles at once

    # Final report
    .step("Generate summary report of all processed articles")
)

# Run the pipeline
result = await content_pipeline.run(
    feed_url="https://example.com/tech-news.rss"
)
```

## Running Workflows

### Async Execution (Recommended)

```python
import asyncio

async def main():
    workflow = create_workflow("Do something async")
    result = await workflow.run()
    print(result)

asyncio.run(main())
```

### Sync Execution

The workflow system automatically handles sync contexts:

```python
# In sync code (like Jupyter notebooks)
workflow = create_workflow("Do something")
result = workflow.run()  # Automatically runs in sync mode
```

## Best Practices

1. **Start Simple**: Begin with basic sequential workflows and add features as needed
2. **Use Descriptive Names**: Name your steps for better debugging
3. **Handle Errors**: Add retry and error handling for external calls
4. **Validate Output**: Use schemas for critical data transformations
5. **Monitor Performance**: Use parallel execution for independent operations
6. **Compose and Reuse**: Build a library of reusable sub-workflows

## Next Steps

-   Explore the [API Reference](./api-reference.md) for detailed documentation
-   Check out [Advanced Examples](../../examples/workflow_advanced.py)
-   Learn about [Workflow Patterns](./patterns.md) for common use cases
