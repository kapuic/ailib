# AILib Quick Reference

## Testing Commands

### Unit Tests

```bash
make test                # Run all tests
make test-all           # Test on all Python versions
pytest tests/           # Run with pytest directly
```

### Notebook Validation

```bash
make test-notebooks-lax # Validate notebooks (recommended)
make test-notebooks     # Strict validation (checks outputs)

# Test specific notebooks
pytest --nbval-lax examples/tutorials/01_setup_and_installation.ipynb
pytest --nbval-lax examples/tutorials/*.ipynb

# Debug failures
pytest --nbval-lax -vv examples/tutorials/failing_notebook.ipynb

# Use validation script
python scripts/validate_notebooks.py --help
python scripts/validate_notebooks.py --mode lax
python scripts/validate_notebooks.py --single 01_setup_and_installation.ipynb
```

### Code Quality

```bash
make lint              # Run all linters
make format           # Format code with black/isort
make ruff             # Run ruff linter only
```

### Development Workflow

```bash
# Install in development mode
pip install -e ".[dev,test]"

# Set up pre-commit hooks
pre-commit install

# Run specific test file
pytest tests/test_agents.py

# Run tests with coverage
make coverage
```

## Common Patterns

### Creating New Components

#### Factory Functions (Recommended)

```python
from ailib import create_agent, create_chain, create_session

# Simple agent
agent = create_agent("assistant", model="gpt-4")

# Chain with templates
chain = create_chain(
    "You are a {role}",
    "Answer: {question}"
)

# Session with validation
session = create_session(session_id="test-001")
```

#### Direct Instantiation (Advanced)

```python
from ailib import Agent, Chain, Session, OpenAIClient

client = OpenAIClient()
agent = Agent(llm=client, max_steps=10)
chain = Chain().add_step("task", lambda: "result")
session = Session(session_id="test-001")
```

### Working with Tools

```python
from ailib import tool, ToolRegistry

@tool
def calculator(expression: str) -> float:
    """Evaluate a mathematical expression."""
    return eval(expression)

# Or with registry
registry = ToolRegistry()

@tool(registry=registry)
def weather(city: str) -> str:
    """Get weather for a city."""
    return f"Sunny in {city}"
```

## Environment Setup

### API Keys

```bash
# Set in environment
export OPENAI_API_KEY="your-key"

# Or use .env file
echo "OPENAI_API_KEY=your-key" > .env
```

### Virtual Environment

```bash
# Using uv (recommended)
uv venv
source .venv/bin/activate

# Using standard venv
python -m venv venv
source venv/bin/activate
```

## Troubleshooting

### Import Errors

```bash
# Ensure ailib is installed
pip install -e .

# Check installation
python -c "import ailib; print(ailib.__version__)"
```

### Notebook Failures

```bash
# Clear notebook outputs
jupyter nbconvert --clear-output --inplace examples/tutorials/*.ipynb

# Test with lax mode
pytest --nbval-lax examples/tutorials/problematic.ipynb

# Skip output validation
pytest --nbval-lax --nbval-cell-timeout=60 examples/tutorials/*.ipynb
```

### API Key Issues

```bash
# Check if key is set
python -c "import os; print('API key set:', bool(os.getenv('OPENAI_API_KEY')))"

# Load from .env
python -c "from dotenv import load_dotenv; load_dotenv(); import os; print(os.getenv('OPENAI_API_KEY')[:10] + '...')"
```

## See Also

-   [Contributing Guide](../CONTRIBUTING.md)
-   [Notebook Testing Guide](notebook_testing.md)
-   [API Documentation](modules.md)
-   [Tutorials](../examples/tutorials/00_index.ipynb)
