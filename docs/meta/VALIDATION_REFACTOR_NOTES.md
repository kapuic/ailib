# Validation Refactor Notes

## Current Issue

The current Pydantic validation implementation follows LangChain's verbose, object-oriented pattern rather than Vercel AI SDK's minimal, functional approach.

## Philosophy Comparison

### Vercel AI SDK (What we want)

-   **Functional API** - Simple function calls like `generateText()`
-   **Minimal boilerplate** - One-call interfaces
-   **Quick integration** - Get started fast
-   **Implicit handling** - Framework does the work under the hood
-   **Invisible validation** - Parameters validated internally without user awareness

### LangChain (What we want to avoid)

-   **Object-oriented** - Classes and instances everywhere
-   **Explicit structure** - Message objects, configuration classes
-   **More verbose** - Requires more setup code
-   **Visible complexity** - Users must interact with config objects

### Current Implementation (Too LangChain-like)

```python
# Current approach - explicit config objects
config = LLMConfig(model="gpt-4", temperature=0.7)
agent_config = AgentConfig(name="assistant", model="gpt-4")
template_config = PromptTemplateConfig(template="...", input_variables=[...])

# User has to import and use validation classes
from ailib.validation import LLMConfig, AgentConfig
```

## Desired Implementation (Vercel-style)

### 1. Simple function calls with inline validation

```python
# Direct usage - validation happens invisibly
result = complete(
    model="gpt-4",
    prompt="Hello {name}",
    temperature=0.7  # Validated internally, user never sees Pydantic
)

# Agent creation - simple and clean
agent = Agent(
    name="assistant",
    tools=[search, calculate],
    model="gpt-4"
)  # Validation happens internally
```

### 2. Hide Pydantic models from public API

-   Move validation models to internal modules (`_validation.py`)
-   Use them only internally for parameter validation
-   Never expose config classes to users

### 3. Functional wrappers

```python
# Instead of:
chain = Chain(llm=client, config=ChainConfig(...))

# We want:
chain = create_chain(
    llm=client,
    max_iterations=5,
    retry_attempts=3
)  # Validation happens inside create_chain
```

## Refactoring Strategy

1. **Move validation to internal**

    - Rename `validation.py` to `_validation.py` (private)
    - Remove from public exports in `__init__.py`

2. **Update constructors**

    - Keep validation but make it invisible
    - Accept kwargs directly, validate internally
    - Don't require config objects

3. **Add functional factories**

    ```python
    # Factory functions that hide complexity
    def create_agent(name: str, model: str = "gpt-4", **kwargs) -> Agent:
        # Validate internally with Pydantic
        # Return clean Agent instance

    def create_chain(**kwargs) -> Chain:
        # Validate internally
        # Return Chain instance
    ```

4. **Maintain backward compatibility**
    - Keep existing validation for internal use
    - Just hide it from users

## Benefits of Refactoring

1. **Better DX** - Users can start coding immediately
2. **Less imports** - No need to import config classes
3. **Cleaner API** - Matches Vercel AI SDK simplicity
4. **Same safety** - Validation still happens, just invisibly
5. **Easier migration** - Users coming from Vercel AI SDK will feel at home

## Examples After Refactoring

### Before (current)

```python
from ailib import OpenAIClient, Agent, Chain
from ailib.validation import AgentConfig, ChainConfig

# Verbose setup
client = OpenAIClient(model="gpt-4")
agent_config = AgentConfig(
    name="assistant",
    model="gpt-4",
    tools=["search", "calculate"]
)
agent = Agent(llm=client, config=agent_config)
```

### After (Vercel-style)

```python
from ailib import create_agent, create_chain

# Simple and clean
agent = create_agent(
    name="assistant",
    model="gpt-4",
    tools=["search", "calculate"]
)

# Or even simpler with defaults
agent = create_agent("assistant")
```

## Implementation Priority

1. **High Priority**

    - Update Agent creation
    - Update Chain creation
    - Hide validation modules

2. **Medium Priority**

    - Add factory functions
    - Update documentation
    - Update examples

3. **Low Priority**
    - Deprecation warnings for old style
    - Migration guide

## Notes

-   Keep validation internally for robustness
-   Just hide the complexity from users
-   Make the "happy path" as simple as possible
-   Follow Vercel AI SDK's "it just works" philosophy
