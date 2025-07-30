# Notebook Testing Guide

This guide explains how to test and validate Jupyter notebooks in the AILib project.

## Overview

AILib uses `nbval` - a pytest plugin that validates notebook execution and optionally checks outputs. This ensures that all tutorial notebooks remain functional as the library evolves.

## Quick Start

### Running Notebook Tests

```bash
# Install test dependencies
pip install -e ".[test]"

# Validate all notebooks (strict mode - checks outputs)
make test-notebooks

# Validate notebooks (lax mode - ignores output differences)
make test-notebooks-lax

# Validate a specific notebook
pytest --nbval examples/tutorials/01_setup_and_installation.ipynb

# Use the validation script
python scripts/validate_notebooks.py
```

### Common Commands

```bash
# Run with output sanitization
pytest --nbval --nbval-sanitize-with .nbval_sanitize.cfg examples/tutorials/

# Run specific notebooks
pytest --nbval examples/tutorials/0[1-5]*.ipynb

# Run with verbose output
pytest --nbval -v examples/tutorials/

# Skip output comparison entirely
pytest --nbval-lax examples/tutorials/
```

## Validation Modes

### 1. Strict Mode (`--nbval`)

-   Executes notebooks
-   Compares outputs with saved outputs
-   Fails if outputs don't match exactly
-   Use for notebooks with deterministic outputs

### 2. Lax Mode (`--nbval-lax`)

-   Executes notebooks
-   Ignores output differences
-   Only fails on execution errors
-   **Recommended for most cases**

### 3. Sanitized Mode

-   Uses `.nbval_sanitize.cfg` to ignore expected differences
-   Good for notebooks with timestamps, UUIDs, etc.

## Best Practices

### 1. Before Committing Notebooks

```bash
# Clear all outputs to avoid large diffs
jupyter nbconvert --clear-output --inplace examples/tutorials/*.ipynb

# Or use Jupyter menu: Cell → All Output → Clear
```

### 2. Writing Testable Notebooks

-   **Use deterministic examples** when possible
-   **Mock external APIs** or use stable test data
-   **Avoid time-dependent outputs** or use sanitizers
-   **Set random seeds** for reproducible results
-   **Handle missing API keys gracefully**

### 3. Handling API Keys

```python
import os
from ailib import OpenAIClient

# Gracefully handle missing API keys
api_key = os.getenv("OPENAI_API_KEY", "demo-key")
if api_key == "demo-key":
    print("Note: Using demo mode. Set OPENAI_API_KEY for full functionality.")
    # Use mock responses or skip API calls
else:
    client = OpenAIClient(api_key=api_key)
```

### 4. Cell Tags for Testing

You can tag cells to control testing:

```json
{
    "tags": [
        "nbval-skip" // Skip this cell during validation
    ]
}
```

Common tags:

-   `nbval-skip`: Skip cell execution
-   `nbval-ignore-output`: Execute but ignore output
-   `raises-exception`: Cell is expected to raise an exception

## Troubleshooting

### Common Issues

1. **Import Errors**

    ```
    ModuleNotFoundError: No module named 'ailib'
    ```

    Solution: Install ailib in development mode

    ```bash
    pip install -e .
    ```

2. **API Key Errors**

    ```
    OpenAI API key not found
    ```

    Solution: Set environment variable or use mock mode

    ```bash
    export OPENAI_API_KEY="your-key"
    ```

3. **Output Mismatches**

    ```
    AssertionError: Notebook output differs from expected
    ```

    Solution: Use lax mode or clear outputs

    ```bash
    pytest --nbval-lax examples/tutorials/
    ```

4. **Timeout Errors**
    ```
    Timeout waiting for cell execution
    ```
    Solution: Increase timeout
    ```bash
    pytest --nbval --nbval-timeout=60 examples/tutorials/
    ```

### Debugging Failed Tests

```bash
# Run with verbose output
pytest --nbval -vv examples/tutorials/failing_notebook.ipynb

# Run with Python warnings
pytest --nbval -W default examples/tutorials/

# Keep notebook outputs for inspection
pytest --nbval --nbval-current-env examples/tutorials/
```

## CI/CD Integration

Notebooks are automatically validated in CI:

1. **On Push/PR**: When notebooks or source code changes
2. **Python Versions**: Tests run on Python 3.10 and 3.12
3. **Mode**: Uses lax mode to avoid flaky failures
4. **Artifacts**: Failed notebooks are uploaded for debugging

### Running Locally Like CI

```bash
# Simulate CI environment
python -m pytest --nbval-lax examples/tutorials/
```

## Advanced Configuration

### Custom Sanitizers

Edit `.nbval_sanitize.cfg` to add patterns:

```ini
[regex_custom]
regex: Your pattern here
replace: <SANITIZED>
```

### Excluding Notebooks

```bash
# Using pytest
pytest --nbval --ignore=examples/tutorials/experimental.ipynb examples/

# Using the script
python scripts/validate_notebooks.py --exclude experimental.ipynb
```

### Parallel Execution

```bash
# Run notebooks in parallel (requires pytest-xdist)
pip install pytest-xdist
pytest --nbval -n auto examples/tutorials/
```

## Writing New Tutorials

When creating new tutorial notebooks:

1. **Follow the naming convention**: `NN_topic_name.ipynb`
2. **Include in the index**: Update `00_index.ipynb`
3. **Test locally first**: `pytest --nbval-lax your_notebook.ipynb`
4. **Clear outputs before committing**
5. **Add to validation suite**: Ensure it's in `examples/tutorials/`

## Summary

Notebook validation ensures that:

-   ✅ All tutorials remain functional
-   ✅ Examples work with the latest code
-   ✅ Users have a smooth learning experience
-   ✅ Breaking changes are caught early

For most development, use:

```bash
make test-notebooks-lax
```

This provides a good balance between catching errors and avoiding false positives.
