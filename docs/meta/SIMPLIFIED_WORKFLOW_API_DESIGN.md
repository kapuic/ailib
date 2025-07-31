# Simplified Workflow API Design

## Design Philosophy

Keep ALL the power from AGENT_WORKFLOW_REQUIREMENTS.md but with an API that:

1. Starts simple - one line for basic workflows
2. Progressively reveals complexity
3. Uses familiar patterns (method chaining)
4. Provides escape hatches for advanced users

## Core API Design

### 1. The Simplest Case

```python
# One-liner for simple workflows
result = create_workflow("Summarize this: {text}").run(text="...")

# Multi-step workflow
workflow = create_workflow([
    "Extract key points from: {text}",
    "Rank by importance",
    "Format as bullet points"
])
result = workflow.run(text="...")
```

### 2. Conditional Branching

After careful consideration, method chaining is better than dictionaries because:

-   Better IDE support and autocomplete
-   More readable
-   Follows established patterns
-   Type-safe

```python
# Simple branching
workflow = create_workflow()
    .step("Analyze sentiment: {text}")
    .if(lambda result: "negative" in result)
        .then("Route to senior support")
        .else("Send automated response")

# Complex branching with multiple conditions
workflow = create_workflow()
    .step("Categorize issue: {text}")
    .if(lambda r: r.category == "billing")
        .then(billing_workflow)  # Reuse another workflow
    .elif(lambda r: r.category == "technical")
        .then("Create tech support ticket")
    .else("Send to general support")
```

### 3. Loops and Iteration

```python
# Simple retry
workflow = create_workflow()
    .step("Call API", retry=3)  # Built-in retry

# While loop
workflow = create_workflow()
    .while(lambda ctx: ctx.attempts < 3 and not ctx.success)
        .do("Try to fetch data")
        .break_if(lambda ctx: ctx.success)

# For each with automatic batching
workflow = create_workflow()
    .step("Load documents")
    .for_each("doc")  # Automatically iterates over previous result
        .do("Summarize document: {doc}")
        .parallel(max=5)  # Process 5 at a time
```

### 4. State Management

State should be implicit but accessible:

```python
# Implicit state (simple case)
workflow = create_workflow()
    .step("Extract entities", name="entities")  # Named steps
    .step("Analyze {entities}")  # Reference previous results

# Explicit state when needed
workflow = create_workflow()
    .with_state({"count": 0, "errors": []})
    .step("Process item")
    .update_state(lambda state, result: {
        **state,
        "count": state["count"] + 1
    })

# State persistence (checkpointing)
workflow = create_workflow()
    .checkpoint_after("data_processing")  # Save state
    .step("Expensive operation")
    # Can resume from checkpoint if interrupted
```

### 5. Schema Validation

```python
# Simple validation
from pydantic import BaseModel

class ResponseSchema(BaseModel):
    summary: str
    confidence: float

workflow = create_workflow()
    .step("Analyze text", output_schema=ResponseSchema)
    # Automatically retries if output doesn't match schema

# With custom retry behavior
workflow = create_workflow()
    .step(
        "Extract structured data",
        output_schema=DataSchema,
        on_validation_error="The output was invalid: {error}. Please fix and try again.",
        max_retries=5
    )
```

### 6. Error Handling

```python
# Simple error handling
workflow = create_workflow()
    .step("Call external API")
    .on_error("Use cached data")  # Simple fallback

# Advanced error handling
workflow = create_workflow()
    .step("Process payment")
    .on_error(PaymentError)
        .do("Retry with backup provider")
    .on_error(RateLimitError)
        .wait(60)  # Wait 60 seconds
        .retry()
    .on_any_error()
        .do("Log error")
        .fail()  # Propagate error
```

### 7. Human-in-the-Loop

```python
# Simple approval
workflow = create_workflow()
    .step("Generate proposal")
    .require_approval()  # Pauses for human input
    .step("Execute approved plan")

# Advanced HITL
workflow = create_workflow()
    .step("Analyze risk")
    .if(lambda r: r.risk_score > 0.8)
        .require_approval(
            notify=["manager@company.com"],
            timeout="2 hours",
            message="High risk operation needs approval"
        )
```

### 8. Parallel Execution

```python
# Simple parallel execution
workflow = create_workflow()
    .parallel(
        "Search web for: {query}",
        "Query database for: {query}",
        "Check cache for: {query}"
    )
    .combine()  # Combines all results

# Advanced parallel with strategies
workflow = create_workflow()
    .parallel(
        search_workflow,
        database_workflow,
        cache_workflow
    )
    .race()  # Returns first to complete
    # or .all() - waits for all
    # or .any(2) - waits for any 2 to complete
```

### 9. Composition and Reusability

```python
# Define reusable workflows
validation_workflow = create_workflow([
    "Check format",
    "Verify constraints"
])

main_workflow = create_workflow()
    .step("Load data")
    .use(validation_workflow)  # Compose workflows
    .step("Process validated data")

# Workflow templates
template = create_workflow_template()
    .step("Process {item_type}")
    .step("Validate against {schema}")

# Use template with different parameters
order_workflow = template.create(item_type="order", schema=OrderSchema)
user_workflow = template.create(item_type="user", schema=UserSchema)
```

### 10. Observability

```python
# Built-in tracing (zero config)
workflow = create_workflow([...])
result = workflow.run(data)
trace = workflow.get_trace()  # Automatic tracing

# Advanced observability
workflow = create_workflow()
    .trace_to("opentelemetry")  # Send to OTel
    .log_level("debug")
    .measure_cost()  # Track token usage
    .profile()  # Performance profiling
```

## Progressive Complexity Examples

### Example 1: Customer Support Bot

```python
# Level 1: Simplest version
bot = create_workflow("Answer customer question: {question}")

# Level 2: Add tools
bot = create_workflow()
    .with_tools(search_faq, check_order)
    .step("Answer using tools: {question}")

# Level 3: Add routing
bot = create_workflow()
    .with_tools(search_faq, check_order)
    .step("Categorize: {question}")
    .if(lambda r: r.category == "order")
        .then("Check order status using tools")
    .else("Search FAQ for answer")

# Level 4: Add validation and error handling
bot = create_workflow()
    .with_tools(search_faq, check_order)
    .step("Categorize: {question}", output_schema=Category)
    .if(lambda r: r.category == "order")
        .then("Check order status", retry=3)
        .on_error("Apologize and escalate to human")
    .else("Search FAQ")
    .step("Format response", output_schema=Response)
```

### Example 2: Data Processing Pipeline

```python
# Level 1: Simple pipeline
pipeline = create_workflow([
    "Load data from: {source}",
    "Clean data",
    "Generate report"
])

# Level 2: Add validation
pipeline = create_workflow()
    .step("Load data from: {source}")
    .step("Validate format", output_schema=DataFormat)
    .step("Clean data")
    .step("Generate report")

# Level 3: Add parallel processing
pipeline = create_workflow()
    .step("Load data from: {source}")
    .step("Split into chunks")
    .for_each("chunk")
        .do("Process chunk")
        .parallel(max=10)
    .step("Merge results")
    .step("Generate report")

# Level 4: Production-ready
pipeline = create_workflow()
    .with_state({"processed": 0, "errors": []})
    .checkpoint_every(100)  # Save progress
    .step("Load data from: {source}")
    .step("Validate", output_schema=DataFormat)
    .on_error()
        .update_state(lambda s, e: {...s, "errors": s["errors"] + [e]})
        .continue()  # Don't fail entire pipeline
    .for_each("batch", size=100)
        .do("Process batch")
        .parallel(max=10)
        .on_complete()
            .update_state(lambda s: {...s, "processed": s["processed"] + 100})
    .step("Generate report with {processed} items and {errors}")
```

## Key Design Decisions

1. **Method Chaining > Dictionaries**

    - More readable
    - Better IDE support
    - Follows familiar patterns

2. **Progressive Disclosure**

    - Start with strings/arrays
    - Add methods as needed
    - Full control available but not required

3. **Smart Defaults**

    - Automatic retries on validation errors
    - Built-in tracing
    - Sensible timeouts

4. **Escape Hatches**
    - Can drop down to low-level APIs
    - Custom executors
    - Plugin system for extensions

## Implementation Strategy

1. **Core Engine**: Build on existing Chain/Agent infrastructure
2. **Builder Pattern**: Use fluent interface for workflow construction
3. **Async First**: All operations async by default
4. **Plugin System**: Allow extensions without modifying core

This design achieves all requirements from AGENT_WORKFLOW_REQUIREMENTS.md while maintaining the simplicity that makes AILib approachable.
