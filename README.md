# AILib

A simple, intuitive Python SDK for building LLM-powered applications with chains, agents, and tools.

## Features

-   🔗 **Chains**: Sequential prompt execution with fluent API
-   🤖 **Agents**: ReAct-style autonomous agents with tool usage
-   🛠️ **Tools**: Easy tool creation with decorators and type safety
-   📝 **Templates**: Powerful prompt templating system
-   💾 **Sessions**: Conversation state and memory management
-   🔒 **Type Safety**: Full type hints and Pydantic integration
-   ⚡ **Async Support**: Both sync and async APIs

## Installation

```bash
pip install ailib
```

### Development Setup

For development, clone the repository and install with development dependencies:

```bash
# Clone the repository
git clone https://github.com/kapuic/ailib.git
cd ailib

# Create virtual environment with uv (recommended)
uv venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install in development mode with all dependencies
uv pip install -e ".[dev,test]"

# Install pre-commit hooks
pre-commit install

# Run formatters and linters
make format  # Format code with black and isort
make lint    # Check code style
```

## Quick Start

### Simple Completion

```python
from ailib import OpenAIClient, Prompt

# Initialize client
client = OpenAIClient(model="gpt-3.5-turbo")

# Create prompt
prompt = Prompt()
prompt.add_system("You are a helpful assistant.")
prompt.add_user("What is the capital of France?")

# Get completion
response = client.complete(prompt.build())
print(response.content)
```

### Using Chains

```python
from ailib import Chain, OpenAIClient

client = OpenAIClient()

# Create a multi-step chain
chain = (Chain(client)
    .add_system("You are a helpful assistant.")
    .add_user("What is the capital of {country}?", name="capital")
    .add_user("What is the population of {capital}?")
)

result = chain.run(country="France")
print(result)
```

### Creating Tools

```python
from ailib import tool

@tool
def weather(city: str) -> str:
    """Get the weather for a city."""
    return f"The weather in {city} is sunny and 72°F"

@tool
def calculator(expression: str) -> float:
    """Evaluate a mathematical expression."""
    return eval(expression)
```

### Using Agents

```python
from ailib import Agent, OpenAIClient

# Create agent with tools
client = OpenAIClient(model="gpt-4")
agent = Agent(llm=client)
agent.with_tools(weather, calculator)

# Run agent
result = agent.run("What's the weather in Paris? Also, what's 15% of 85?")
print(result)
```

### Session Management

```python
from ailib import Session, OpenAIClient

# Create session
session = Session()
client = OpenAIClient()

# Add messages
session.add_system_message("You are a helpful tutor.")
session.add_user_message("Explain quantum computing")

# Get response with context
response = client.complete(session.get_messages())
session.add_assistant_message(response.content)

# Store memory
session.set_memory("topic", "quantum computing")
session.set_memory("level", "beginner")
```

## Core Concepts

### LLM Clients

The SDK provides an abstract `LLMClient` interface with implementations for different providers:

-   `OpenAIClient`: OpenAI GPT models (GPT-4, GPT-3.5-turbo, etc.)
-   Easy to extend with custom implementations

### Prompt Templates

Templates support variable substitution and partial formatting:

```python
from ailib import PromptTemplate

template = PromptTemplate("Translate '{text}' to {language}")
result = template.format(text="Hello", language="French")

# Partial templates
partial = template.partial(language="Spanish")
result = partial.format(text="Goodbye")
```

### Chains

Chains allow sequential execution of prompts with context passing:

```python
chain = (Chain(client)
    .add_user("Generate a random number", name="number")
    .add_user("Double {number}", processor=lambda x: int(x) * 2)
)
```

### Tools and Agents

Tools are functions that agents can use. The `@tool` decorator automatically:

-   Extracts function documentation
-   Infers parameter types
-   Handles validation with Pydantic

```python
@tool
def search(query: str, max_results: int = 5) -> str:
    """Search the web for information."""
    # Implementation
    return results
```

## Advanced Features

### Async Support

All main components support async operations:

```python
async def example():
    response = await client.acomplete(messages)
    result = await chain.arun(context="value")
    answer = await agent.arun("Task description")
```

### Custom Processors

Add processing functions to chain steps:

```python
def extract_number(text: str) -> int:
    import re
    match = re.search(r'\d+', text)
    return int(match.group()) if match else 0

chain.add_user("How many apples?", processor=extract_number)
```

### Tool Registry

Manage tools programmatically:

```python
from ailib import ToolRegistry

registry = ToolRegistry()
registry.register(my_tool)

# Use with agent
agent = Agent(llm=client, tools=registry)
```

## Best Practices

1. **Use environment variables** for API keys:

    ```bash
    export OPENAI_API_KEY="your-key"
    ```

2. **Enable verbose mode** for debugging:

    ```python
    chain.verbose(True)
    agent = Agent(llm=client, verbose=True)
    ```

3. **Set appropriate max_steps** for agents to prevent infinite loops

4. **Use sessions** to maintain conversation context

5. **Type your tool functions** for better validation and documentation

## Requirements

-   Python >= 3.10
-   OpenAI API key (for OpenAI models)

## License

MIT License - see LICENSE file for details

## Contributing

Contributions are welcome! Please see CONTRIBUTING.md for guidelines.

## Credits

Created by Kapui Cheung as a demonstration of modern Python SDK design.
