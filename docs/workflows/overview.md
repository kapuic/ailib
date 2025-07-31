# Workflows Overview

AILib's workflow system provides a powerful yet simple way to orchestrate complex AI tasks. Built with the same design philosophy as Vercel AI SDK, workflows allow you to chain operations, handle conditional logic, manage state, and coordinate parallel tasks - all with minimal code.

## What are Workflows?

Workflows extend beyond simple sequential chains to provide:

-   **Conditional branching** - Make decisions based on outputs
-   **Loops and iteration** - Process lists and repeat operations
-   **State management** - Track data across workflow steps
-   **Parallel execution** - Run multiple operations concurrently
-   **Error handling** - Robust retry and recovery mechanisms
-   **Human-in-the-loop** - Request approvals and interventions

## Design Philosophy

Following our core principle of "simple by default, powerful when needed":

```python
# Simplest case - one line
result = await create_workflow("Summarize this: {text}").run(text=article)

# Add complexity progressively
workflow = create_workflow()
    .step("Analyze sentiment")
    .if_(lambda r: "negative" in r)
        .then("Escalate to support team")
        .else_("Send standard thank you")
```

## When to Use Workflows vs Chains

Use **Chains** when you need:

-   Simple sequential prompt execution
-   Fixed flow with no branching
-   Quick prototypes

Use **Workflows** when you need:

-   Conditional logic (if/then/else)
-   Loops or iteration over data
-   Parallel task execution
-   Error recovery strategies
-   State management across steps
-   Human approval gates

## Core Concepts

### Steps

The basic building block - can be a prompt, function, or sub-workflow:

```python
workflow.step("Generate product description for: {product}")
workflow.step(analyze_sentiment_function)
workflow.step(another_workflow)
```

### Conditions

Branch execution based on results:

```python
workflow
    .if_(lambda result: result.score > 0.8)
    .then("Mark as high quality")
    .elif_(lambda result: result.score > 0.5)
    .then("Queue for review")
    .else_("Reject and notify")
```

### Loops

Iterate over data or repeat until condition:

```python
# For each item
workflow
    .step("Get product list")
    .for_each("product")
    .do("Generate description for {product}")

# While condition
workflow
    .while_(lambda ctx: ctx.attempts < 3)
    .do("Try to extract data")
```

### State Management

Maintain data across workflow execution:

```python
workflow
    .with_state({"total": 0, "processed": []})
    .step(lambda ctx: ctx.state["total"] += 1)
```

### Parallel Execution

Run multiple operations concurrently:

```python
workflow
    .parallel(
        "Search web for: {query}",
        "Search docs for: {query}",
        "Query database for: {query}"
    )
    .all()  # Wait for all to complete
```

## Key Features

### 🚀 Progressive Complexity

Start with single-line workflows and add features as needed. No upfront configuration or boilerplate required.

### 🔄 Async-First

All workflows run asynchronously by default for optimal performance, with automatic sync adapters when needed.

### 📊 Built-in Observability

Automatic tracing and logging without configuration. Every step is tracked for debugging and monitoring.

### 🛡️ Safety by Default

-   Automatic retry on transient failures
-   Schema validation with Pydantic
-   Timeout protection
-   Circuit breakers for external calls

### 🔌 Extensible

-   Custom step types
-   Plugin system for new capabilities
-   Integration with existing tools and services

## Quick Example

Here's a complete customer support workflow:

```python
from ailib import create_workflow

support_workflow = (
    create_workflow()
    .step("Categorize issue: {message}")
    .if_(lambda r: "urgent" in r.lower())
    .then(
        create_workflow()
        .step("Extract key details")
        .step("Check knowledge base")
        .step("Draft priority response")
        .require_approval(notify=["support-lead@company.com"])
        .step("Send response")
    )
    .else_(
        create_workflow()
        .step("Generate automated response")
        .step("Send response")
        .step("Schedule follow-up")
    )
)

result = await support_workflow.run(
    message="My payment failed and I need this resolved urgently!"
)
```

## Next Steps

-   [Getting Started](./getting-started.md) - Build your first workflow
-   [API Reference](./api-reference.md) - Detailed API documentation
-   [Examples](../../examples/workflow_basic.py) - Common workflow patterns
-   [Advanced Patterns](../../examples/workflow_advanced.py) - Production-ready workflows
