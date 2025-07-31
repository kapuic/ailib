# Workflows API Reference

Complete API documentation for AILib's workflow system.

## Factory Functions

### `create_workflow()`

Creates a new workflow with optional initial steps.

```python
create_workflow(
    steps: str | list[str | WorkflowStep] | None = None
) -> WorkflowBuilder | Workflow
```

**Parameters:**

-   `steps`: Initial steps for the workflow
    -   `str`: Single step workflow
    -   `list`: Multi-step workflow
    -   `None`: Empty workflow for fluent API

**Returns:**

-   `WorkflowBuilder` for further configuration

**Examples:**

```python
# Single step
workflow = create_workflow("Summarize: {text}")

# Multiple steps
workflow = create_workflow([
    "Extract key points",
    "Rank by importance"
])

# Fluent API
workflow = create_workflow().step("First").step("Second")
```

### `create_workflow_template()`

Creates a reusable workflow template (coming soon).

```python
create_workflow_template() -> WorkflowTemplateBuilder
```

## WorkflowBuilder Methods

### Core Step Methods

#### `.step()`

Add a step to the workflow.

```python
.step(
    prompt_or_func: str | Callable,
    name: str | None = None,
    retry: int = 0,
    timeout: float | None = None,
    output_schema: type[BaseModel] | None = None
) -> WorkflowBuilder
```

**Parameters:**

-   `prompt_or_func`: Prompt string or callable function
-   `name`: Optional step name for reference
-   `retry`: Number of retry attempts (default: 0)
-   `timeout`: Step timeout in seconds
-   `output_schema`: Pydantic model for output validation

**Example:**

```python
workflow.step(
    "Generate product description",
    name="description",
    retry=3,
    output_schema=ProductSchema
)
```

### Conditional Methods

#### `.if_()`

Start a conditional branch.

```python
.if_(condition: Callable[[Any], bool] | str) -> WorkflowBuilder
```

**Parameters:**

-   `condition`: Function that returns bool or string condition

**Example:**

```python
workflow.if_(lambda result: "error" in result)
workflow.if_("result.score > 0.8")  # String conditions (planned)
```

#### `.then()`

Define the 'then' branch.

```python
.then(step_or_prompt: str | WorkflowStep | list) -> WorkflowBuilder
```

**Example:**

```python
workflow.if_(condition).then("Handle true case")
workflow.if_(condition).then([
    "Step 1 for true",
    "Step 2 for true"
])
```

#### `.else_()`

Define the 'else' branch.

```python
.else_(step_or_prompt: str | WorkflowStep | list) -> WorkflowBuilder
```

#### `.elif_()`

Add an else-if branch.

```python
.elif_(condition: Callable[[Any], bool] | str) -> WorkflowBuilder
```

**Example:**

```python
workflow
    .if_(lambda r: r.score > 0.8)
    .then("High quality")
    .elif_(lambda r: r.score > 0.5)
    .then("Medium quality")
    .else_("Low quality")
```

### Loop Methods

#### `.for_each()`

Iterate over items.

```python
.for_each(
    item_name: str = "item",
    items_from: str | None = None
) -> ForEachBuilder
```

**Parameters:**

-   `item_name`: Variable name for current item
-   `items_from`: Source of items (uses previous result if None)

**Example:**

```python
workflow
    .step("Get user list")
    .for_each("user")
    .do("Send email to {user}")
```

#### `.while_()`

Create a while loop.

```python
.while_(condition: Callable[[WorkflowContext], bool]) -> WhileBuilder
```

**Example:**

```python
workflow
    .while_(lambda ctx: ctx.state["retries"] < 3)
    .do("Try operation")
```

### Parallel Execution

#### `.parallel()`

Execute steps in parallel.

```python
.parallel(*steps: str | WorkflowStep) -> ParallelBuilder
```

**Example:**

```python
workflow
    .parallel(
        "Task 1",
        "Task 2",
        "Task 3"
    )
    .all()  # Wait for all
```

### State Management

#### `.with_state()`

Set initial workflow state.

```python
.with_state(initial_state: dict[str, Any]) -> WorkflowBuilder
```

**Example:**

```python
workflow.with_state({
    "counter": 0,
    "results": []
})
```

### Error Handling

#### `.on_error()`

Add error handling.

```python
.on_error(error_type: type[Exception] | None = None) -> ErrorHandlerBuilder
```

**Example:**

```python
workflow
    .step("Risky operation")
    .on_error(ConnectionError)
    .retry(max_attempts=3)
    .do("Log error", "Use fallback")
```

### Human-in-the-Loop

#### `.require_approval()`

Add human approval step.

```python
.require_approval(
    notify: list[str] | None = None,
    timeout: str | None = None,
    message: str | None = None
) -> WorkflowBuilder
```

**Example:**

```python
workflow.require_approval(
    notify=["manager@company.com"],
    timeout="2 hours",
    message="Please review the generated content"
)
```

### Composition

#### `.use()`

Embed another workflow.

```python
.use(workflow: Workflow | WorkflowBuilder) -> WorkflowBuilder
```

**Example:**

```python
validation_workflow = create_workflow(["Validate", "Verify"])
main_workflow.use(validation_workflow)
```

### Execution

#### `.build()`

Build the workflow without running.

```python
.build() -> Workflow
```

#### `.run()`

Build and run the workflow.

```python
.run(**kwargs) -> Any
```

**Parameters:**

-   `**kwargs`: Variables for template substitution

**Example:**

```python
result = await workflow.run(
    topic="AI",
    max_length=500
)
```

## Builder Classes

### ForEachBuilder

Returned by `.for_each()` for configuring iteration.

#### `.do()`

Define loop body.

```python
.do(*steps: str | WorkflowStep) -> WorkflowBuilder
```

#### `.parallel()`

Enable parallel iteration.

```python
.parallel(max_concurrent: int = 5) -> WorkflowBuilder
```

**Example:**

```python
workflow
    .for_each("item")
    .do("Process {item}")
    .parallel(max_concurrent=10)
```

### WhileBuilder

Returned by `.while_()` for configuring loops.

#### `.do()`

Define loop body.

```python
.do(*steps: str | WorkflowStep) -> WhileBuilder
```

#### `.break_if()`

Add break condition.

```python
.break_if(condition: Callable[[WorkflowContext], bool]) -> WorkflowBuilder
```

**Example:**

```python
workflow
    .while_(lambda ctx: True)  # Infinite loop
    .do("Check condition")
    .break_if(lambda ctx: ctx.found_answer)
```

### ParallelBuilder

Returned by `.parallel()` for configuring parallel execution.

#### `.all()`

Wait for all tasks to complete.

```python
.all() -> WorkflowBuilder
```

#### `.race()`

Return first to complete.

```python
.race() -> WorkflowBuilder
```

#### `.any()`

Wait for N tasks to complete.

```python
.any(count: int = 1) -> WorkflowBuilder
```

**Example:**

```python
workflow
    .parallel("Fast API", "Slow API", "Medium API")
    .any(2)  # Return when 2 complete
```

### ErrorHandlerBuilder

Returned by `.on_error()` for configuring error handling.

#### `.do()`

Define error handling steps.

```python
.do(*steps: str | WorkflowStep) -> ErrorHandlerBuilder
```

#### `.retry()`

Configure retry behavior.

```python
.retry(
    max_attempts: int = 3,
    backoff_factor: float = 2.0
) -> ErrorHandlerBuilder
```

#### `.finally_do()`

Add finally block.

```python
.finally_do(*steps: str | WorkflowStep) -> WorkflowBuilder
```

**Example:**

```python
workflow
    .on_error(APIError)
    .retry(max_attempts=5, backoff_factor=1.5)
    .do("Log error")
    .finally_do("Clean up resources")
```

## Context Objects

### WorkflowContext

Available in step functions for accessing workflow state.

**Properties:**

-   `state`: Current workflow state dict
-   `variables`: Template variables
-   `current_result`: Result of previous step
-   `results`: All step results by name
-   `errors`: List of errors encountered
-   `llm_client`: LLM client instance

**Available in Prompts:**
When using string prompts, these variables are available:

-   `{previous}` or `{previous_result}`: Result from the previous step
-   `{step_name}`: Any named step result (e.g., `{haiku}` if step was named "haiku")
-   Any variables passed to `run()` (e.g., `{topic}`, `{user_input}`)
-   Any values in workflow state (accessible via state management)

**Example:**

```python
def my_step(ctx: WorkflowContext):
    # Access state
    count = ctx.state.get("count", 0)

    # Access previous result
    previous = ctx.current_result

    # Access named results
    summary = ctx.results.get("summary_step")

    # Update state
    ctx.state["count"] = count + 1

    return f"Processed {count} items"

workflow.step(my_step)
```

## Step Types

### SimpleStep

Basic execution step for prompts or functions.

### ConditionalStep

Handles if/then/else branching logic.

### WhileStep

Executes body while condition is true.

### ForEachStep

Iterates over a collection.

### ParallelStep

Executes multiple steps concurrently.

### SubWorkflowStep

Embeds another workflow as a step.

### HumanApprovalStep

Waits for human approval before continuing.

## Schema Validation

Use Pydantic models for automatic output validation:

```python
from pydantic import BaseModel

class AnalysisResult(BaseModel):
    sentiment: str
    score: float
    keywords: list[str]

workflow.step(
    "Analyze the text: {text}",
    output_schema=AnalysisResult
)
```

The workflow will:

1. Parse LLM output as JSON
2. Validate against schema
3. Retry with error feedback if validation fails
4. Return validated object

## Complete Example

```python
from ailib import create_workflow
from pydantic import BaseModel

class Lead(BaseModel):
    name: str
    email: str
    score: float

# Build a lead qualification workflow
lead_workflow = (
    create_workflow()
    .with_state({"qualified_leads": []})

    # Get leads
    .step("Fetch leads from CRM", name="leads")

    # Process each lead
    .for_each("lead", items_from="leads")
    .do(
        create_workflow()
        .step(
            "Analyze lead quality: {lead}",
            output_schema=Lead,
            retry=2
        )
        .if_(lambda r: r.score > 0.7)
        .then(
            lambda ctx: ctx.state["qualified_leads"].append(ctx.current_result)
        )
    )
    .parallel(max_concurrent=5)

    # Send results
    .step("Generate report from qualified_leads")
    .require_approval(notify=["sales@company.com"])
    .step("Send qualified leads to sales team")
)

# Run it
result = await lead_workflow.run()
```
