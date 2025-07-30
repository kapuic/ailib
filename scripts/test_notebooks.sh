#!/bin/bash
# Test notebook validation for AILib

set -e

echo "=== AILib Notebook Validation Test ==="
echo

# Check if AILib is installed
echo "1. Checking AILib installation..."
if python -c "import ailib; print(f'AILib version: {ailib.__version__}')" 2>/dev/null; then
    echo "✅ AILib is installed"
else
    echo "❌ AILib not found. Installing in editable mode..."
    pip install -e . --quiet
fi
echo

# Check if test dependencies are installed
echo "2. Checking test dependencies..."
if python -c "import nbval" 2>/dev/null; then
    echo "✅ nbval is installed"
else
    echo "❌ nbval not found. Installing test dependencies..."
    pip install -e ".[test]" --quiet
fi
echo

# Set API key if provided
if [ -n "$1" ]; then
    export OPENAI_API_KEY="$1"
    echo "3. Using provided API key"
elif [ -f ".env" ]; then
    export $(grep -v '^#' .env | xargs)
    echo "3. Loaded API key from .env file"
else
    echo "3. Warning: No API key found. Some tests may fail."
fi
echo

# Run tests
echo "4. Running notebook validation tests..."
echo

# Test with lax mode (recommended)
echo "Testing all tutorial notebooks (lax mode - ignores output differences):"
pytest --nbval-lax examples/tutorials/*.ipynb --tb=short || true

echo
echo "=== Summary ==="
echo "To run specific notebooks:"
echo "  pytest --nbval-lax examples/tutorials/01_setup_and_installation.ipynb"
echo
echo "To run with strict output checking:"
echo "  pytest --nbval examples/tutorials/*.ipynb"
echo
echo "To debug failures:"
echo "  pytest --nbval-lax -vv examples/tutorials/failing_notebook.ipynb"
