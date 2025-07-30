#!/usr/bin/env python3
"""
Validate Jupyter notebooks for the AILib project.

This script provides utilities for testing and validating tutorial notebooks.
"""

import argparse
import subprocess
import sys
from pathlib import Path


def find_notebooks(directory: Path) -> list[Path]:
    """Find all Jupyter notebooks in a directory."""
    return list(directory.glob("**/*.ipynb"))


def validate_notebook(notebook: Path, mode: str = "normal") -> tuple[bool, str]:
    """
    Validate a single notebook.

    Args:
        notebook: Path to the notebook
        mode: Validation mode - "normal", "lax", or "sanitize"

    Returns:
        Tuple of (success, output_message)
    """
    cmd = ["pytest", "--no-header", "-rN"]

    if mode == "lax":
        cmd.append("--nbval-lax")
    elif mode == "sanitize":
        cmd.extend(["--nbval", "--nbval-sanitize-with", ".nbval_sanitize.cfg"])
    else:
        cmd.append("--nbval")

    cmd.append(str(notebook))

    try:
        subprocess.run(cmd, capture_output=True, text=True, check=True)
        return True, f"✅ {notebook.name}: PASSED"
    except subprocess.CalledProcessError as e:
        error_msg = e.stderr or e.stdout
        # Extract relevant error information
        lines = error_msg.split("\n")
        error_summary = []
        for line in lines:
            if "FAILED" in line or "ERROR" in line or "Exception" in line:
                error_summary.append(line.strip())

        if error_summary:
            return False, f"❌ {notebook.name}: FAILED\n   " + "\n   ".join(
                error_summary[:3]
            )
        else:
            return False, f"❌ {notebook.name}: FAILED (run with -v for details)"


def validate_all_notebooks(
    directory: Path, mode: str = "normal", exclude: list[str] | None = None
) -> tuple[int, int]:
    """
    Validate all notebooks in a directory.

    Returns:
        Tuple of (passed_count, failed_count)
    """
    notebooks = find_notebooks(directory)

    if exclude:
        notebooks = [nb for nb in notebooks if nb.name not in exclude]

    print(f"Found {len(notebooks)} notebooks to validate")
    print(f"Validation mode: {mode}")
    print("=" * 60)

    passed = 0
    failed = 0

    for notebook in sorted(notebooks):
        success, message = validate_notebook(notebook, mode)
        print(message)

        if success:
            passed += 1
        else:
            failed += 1

    return passed, failed


def check_dependencies() -> bool:
    """Check if required dependencies are installed."""
    try:
        import nbval  # noqa: F401

        return True
    except ImportError:
        print("❌ nbval is not installed!")
        print("Install it with: pip install nbval")
        print("Or install all test dependencies: pip install -e '.[test]'")
        return False


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Validate AILib tutorial notebooks")
    parser.add_argument(
        "path",
        nargs="?",
        default="examples/tutorials",
        help="Path to notebooks directory (default: examples/tutorials)",
    )
    parser.add_argument(
        "--mode",
        choices=["normal", "lax", "sanitize"],
        default="lax",
        help="Validation mode: normal (strict), lax (ignore outputs), "
        "sanitize (use sanitizers)",
    )
    parser.add_argument(
        "--exclude", nargs="+", help="Notebook names to exclude from validation"
    )
    parser.add_argument("--single", help="Validate a single notebook by name or path")

    args = parser.parse_args()

    # Check dependencies
    if not check_dependencies():
        return 1

    # Handle single notebook validation
    if args.single:
        notebook_path = Path(args.single)
        if not notebook_path.exists():
            # Try to find it in the examples directory
            possible_path = Path("examples/tutorials") / args.single
            if possible_path.exists():
                notebook_path = possible_path
            else:
                print(f"❌ Notebook not found: {args.single}")
                return 1

        success, message = validate_notebook(notebook_path, args.mode)
        print(message)
        return 0 if success else 1

    # Validate all notebooks
    notebook_dir = Path(args.path)
    if not notebook_dir.exists():
        print(f"❌ Directory not found: {notebook_dir}")
        return 1

    passed, failed = validate_all_notebooks(
        notebook_dir, mode=args.mode, exclude=args.exclude
    )

    print("=" * 60)
    print(f"Summary: {passed} passed, {failed} failed")

    if failed > 0:
        print("\n💡 Tips for fixing failures:")
        print("1. Run notebooks locally to ensure they work")
        print("2. Clear outputs before committing (Cell → All Output → Clear)")
        print("3. Use --mode=lax to ignore output differences")
        print("4. Check for missing dependencies or API keys")
        return 1

    print("\n✅ All notebooks validated successfully!")
    return 0


if __name__ == "__main__":
    sys.exit(main())
