# AILib

A simple, intuitive Python SDK for building LLM-powered applications with chains, agents, and tools.

**Philosophy**: Simplicity of Vercel AI SDK + Power of LangChain = AILib 🚀

## Features

-   🌐 **Multi-Provider Support**: Seamlessly switch between OpenAI, Anthropic Claude, and more (🆕)
-   🚀 **Simple API**: Inspired by Vercel AI SDK - minimal boilerplate, maximum productivity
-   🔗 **Chains**: Sequential prompt execution with fluent API
-   🤖 **Agents**: ReAct-style autonomous agents with tool usage
-   🛠️ **Tools**: Easy tool creation with decorators and type safety
-   📝 **Templates**: Powerful prompt templating system
-   💾 **Sessions**: Conversation state and memory management
-   🔒 **Type Safety**: Full type hints and optional Pydantic validation
-   🛡️ **Safety**: Built-in content moderation and safety hooks
-   📊 **Tracing**: Comprehensive observability and debugging support
-   ⚡ **Async Support**: Both sync and async APIs

## Installation

```bash
# Basic installation (includes OpenAI support)
pip install ailib

# Install specific LLM providers
pip install ailib[anthropic]      # For Claude support
pip install ailib[all-providers]  # Install all supported providers

# Development and testing
pip install ailib[dev,test]       # For development
pip install ailib[tracing]        # For advanced tracing
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

## Tutorials

Comprehensive tutorials are available in the `examples/tutorials/` directory:

1. **[Setup and Installation](examples/tutorials/01_setup_and_installation.ipynb)** - Getting started with AILib
2. **[Basic LLM Completions](examples/tutorials/02_basic_llm_completions.ipynb)** - Making your first API calls
3. **[Prompt Templates](examples/tutorials/03_prompt_templates.ipynb)** - Building dynamic prompts
4. **[Prompt Builder](examples/tutorials/04_prompt_builder.ipynb)** - Constructing conversations programmatically
5. **[Session Management](examples/tutorials/05_session_management.ipynb)** - Managing conversation state
6. **[Chains](examples/tutorials/06_chains.ipynb)** - Building sequential workflows
7. **[Tools and Decorators](examples/tutorials/07_tools_and_decorators.ipynb)** - Creating reusable tools
8. **[Agents](examples/tutorials/08_agents.ipynb)** - Building autonomous AI agents
9. **[Advanced Features](examples/tutorials/09_advanced_features.ipynb)** - Async, streaming, and optimization
10. **[Real-World Examples](examples/tutorials/10_real_world_examples.ipynb)** - Complete applications

Start with the **[Tutorial Index](examples/tutorials/00_index.ipynb)** for a guided learning path.

🆕 **New**: Check out our [simplified API examples](examples/simplified_api_example.py) showcasing the new factory functions!

## Quick Start

### Simple Completion

```python
from ailib import create_client, Prompt

# Initialize client - auto-detects provider from model name
client = create_client("gpt-3.5-turbo")  # OpenAI
# client = create_client("claude-3-opus-20240229")  # Anthropic

# Create prompt
prompt = Prompt()
prompt.add_system("You are a helpful assistant.")
prompt.add_user("What is the capital of France?")

# Get completion
response = client.complete(prompt.build())
print(response.content)
```

### Multi-Provider Support 🆕

AILib supports 15+ LLM providers through OpenAI-compatible APIs and custom implementations:

```python
from ailib import create_client, create_agent, list_providers

# Many providers work with just a base URL change!
client = create_client("gpt-4")  # OpenAI (default)
client = create_client("mistralai/Mixtral-8x7B-Instruct-v0.1")  # Together
client = create_client("llama-2-70b", provider="groq")  # Groq (fast inference)

# Local models
client = create_client(
    model="llama2",
    base_url="http://localhost:11434/v1"  # Ollama
)

# Create agents with any provider
agent = create_agent("assistant", model="gpt-4")
agent = create_agent("assistant", model="claude-3-opus-20240229")  # Anthropic
agent = create_agent("assistant", provider="together", model="llama-2-70b")
```

**Supported Providers:**

-   ✅ **OpenAI** - GPT-4, GPT-3.5
-   ✅ **Anthropic** - Claude 3 (Opus, Sonnet, Haiku)
-   ✅ **Local** - Ollama, LM Studio, llama.cpp
-   ✅ **Groq** - Fast inference for open models
-   ✅ **Perplexity** - Online models with web search
-   ✅ **DeepSeek** - DeepSeek-V2, DeepSeek-Coder
-   ✅ **Together** - Open models (Llama, Mixtral, etc.)
-   ✅ **Anyscale** - Scalable open model hosting
-   ✅ **Fireworks** - Fast open model inference
-   ✅ **Moonshot** - Kimi models
-   🔄 More coming soon...

### Using Chains - The Easy Way

```python
from ailib import create_chain

# Create a chain with the simplified API - no client needed!
chain = create_chain(
    "You are a helpful assistant.",
    "What is the capital of {country}?",
    "What is the population?"
)

result = chain.run(country="France")
print(result)
```

<details>
<summary>Alternative: Using direct instantiation for more control</summary>

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

</details>

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

### Using Agents - The Easy Way

```python
from ailib import create_agent

# Create agent with the simplified API
agent = create_agent(
    "assistant",
    tools=[weather, calculator],
    model="gpt-4"
)

# Run agent
result = agent.run("What's the weather in Paris? Also, what's 15% of 85?")
print(result)
```

<details>
<summary>Alternative: Using direct instantiation for more control</summary>

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

</details>

### Session Management - The Easy Way

```python
from ailib import create_session, OpenAIClient

# Create session with validation
session = create_session(
    session_id="tutorial-001",
    metadata={"user": "student"}
)

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

## Why AILib?

AILib follows the philosophy of **Vercel AI SDK** rather than LangChain:

-   **Simple by default**: Start with one line of code, not pages of configuration
-   **Progressive disclosure**: Complexity is available when you need it, hidden when you don't
-   **Multi-provider**: Switch between OpenAI, Anthropic, and more with a single parameter
-   **Type-safe**: Full TypeScript-style type hints and optional runtime validation
-   **Production-ready**: Built-in safety, tracing, and error handling

```python
# LangChain style (verbose)
from langchain import LLMChain, PromptTemplate
from langchain.llms import OpenAI

llm = OpenAI(temperature=0.7)
prompt = PromptTemplate(
    input_variables=["product"],
    template="What is a good name for a company that makes {product}?"
)
chain = LLMChain(llm=llm, prompt=prompt)
result = chain.run("colorful socks")

# AILib style (simple)
from ailib import create_chain

chain = create_chain("What is a good name for a company that makes {product}?")
result = chain.run(product="colorful socks")
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
-   Handles validation automatically

```python
@tool
def search(query: str, max_results: int = 5) -> str:
    """Search the web for information."""
    # Implementation
    return results
```

## Advanced Features

### Safety and Moderation

AILib includes built-in safety features to ensure responsible AI usage:

```python
from ailib.safety import enable_safety, with_moderation

# Enable global safety checks
enable_safety(
    block_harmful=True,
    max_length=4000,
    blocked_words=["violence", "hate"]
)

# Use with OpenAI moderation
pre_hook, post_hook = with_moderation()

# Check content directly
from ailib.safety import check_content
is_safe, violations = check_content("Some text to check")
```

### Tracing and Observability

Comprehensive tracing support for debugging and monitoring:

```python
from ailib.tracing import get_trace_manager

# Automatic tracing for agents and chains
agent = create_agent("assistant", verbose=True)
result = agent.run("Complex task")  # Automatically traced

# Access trace data
manager = get_trace_manager()
trace = manager.get_trace(trace_id)
print(trace.to_dict())  # Full execution history
```

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
agent = create_agent("assistant", tools=registry)
```

### Rate Limiting

Built-in rate limiting to prevent abuse:

```python
from ailib.safety import set_rate_limit, check_rate_limit

# Set rate limit: 10 requests per minute per user
set_rate_limit(max_requests=10, window_seconds=60)

# Check before making requests
if check_rate_limit("user-123"):
    result = agent.run("Query")
else:
    print("Rate limit exceeded")
```

## Factory Functions vs Direct Instantiation

AILib provides two ways to create objects:

1. **Factory Functions** (Recommended): Simple, validated, and safe

    ```python
    agent = create_agent("assistant", temperature=0.7)
    chain = create_chain("Prompt template")
    session = create_session(max_messages=100)
    ```

2. **Direct Instantiation**: More control, no validation
    ```python
    agent = Agent(llm=client, temperature=5.0)  # No validation!
    ```

Use factory functions for safety, direct instantiation for flexibility.

## Best Practices

1. **Use environment variables** for API keys:

    ```bash
    export OPENAI_API_KEY="your-key"
    ```

2. **Enable verbose mode** for debugging:

    ```python
    # With factory functions
    agent = create_agent("assistant", verbose=True)
    chain = create_chain("Template", verbose=True)

    # Or with fluent API
    chain.verbose(True)
    ```

3. **Set appropriate max_steps** for agents to prevent infinite loops

4. **Use sessions** to maintain conversation context

5. **Type your tool functions** for better validation and documentation

6. **Use safety features** in production environments

7. **Enable tracing** for debugging complex workflows

## Requirements

-   Python >= 3.10
-   OpenAI API key (for OpenAI models)

## License

MIT License - see LICENSE file for details

## Testing

### Running Tests

```bash
# Run unit tests
make test

# Run notebook validation tests
make test-notebooks-lax

# Test specific notebook
pytest --nbval-lax examples/tutorials/01_setup_and_installation.ipynb
```

### Notebook Validation

All tutorial notebooks are automatically tested to ensure they work correctly:

```bash
# Install test dependencies
pip install -e ".[test]"

# Validate notebooks (recommended - ignores output differences)
make test-notebooks-lax

# Strict validation (checks outputs match)
make test-notebooks
```

See [docs/notebook_testing.md](docs/notebook_testing.md) for detailed testing guidelines.

## Contributing

Contributions are welcome! Please see CONTRIBUTING.md for guidelines.

## Project Status

AILib is under active development. Current version includes:

-   ✅ Core LLM client abstractions
-   ✅ Multi-provider support (OpenAI, Anthropic) 🆕
-   ✅ Chain and agent implementations
-   ✅ Tool system with decorators
-   ✅ Session management
-   ✅ Safety and moderation hooks
-   ✅ Comprehensive tracing
-   ✅ Full async support
-   🔄 More LLM providers (Ollama, Google Gemini - coming soon)
-   🔄 Vector store integrations (coming soon)
-   🔄 Streaming support (coming soon)

See [ROADMAP.md](ROADMAP.md) for detailed development plans and upcoming features.

## Related Projects

-   [LangChain](https://github.com/langchain-ai/langchain) - Comprehensive but complex
-   [Vercel AI SDK](https://github.com/vercel/ai) - Our inspiration for simplicity
-   [AutoGen](https://github.com/microsoft/autogen) - Multi-agent conversations
-   [CrewAI](https://github.com/joaomdmoura/crewAI) - Agent collaboration

## Credits

Created by Kapui Cheung as a demonstration of modern Python SDK design, combining the simplicity of Vercel AI SDK with the power of LangChain.
