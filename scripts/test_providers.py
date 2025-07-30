#!/usr/bin/env python3
"""Test script to verify multi-provider support in AILib."""

import sys
from pathlib import Path

# Add src to path for development testing
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from ailib import create_agent, create_chain, create_client, list_providers
from dotenv import load_dotenv


def test_provider_detection():
    """Test automatic provider detection from model names."""
    print("Testing provider detection...")

    test_cases = [
        ("gpt-4", "openai"),
        ("gpt-3.5-turbo", "openai"),
        ("claude-3-opus-20240229", "anthropic"),
        ("claude-3-sonnet-20240229", "anthropic"),
        ("llama2", "ollama"),
        ("mistral", "ollama"),
    ]

    from ailib.core import detect_provider

    for model, expected_provider in test_cases:
        detected = detect_provider(model)
        status = "✓" if detected == expected_provider else "✗"
        print(f"  {status} {model} -> {detected} (expected: {expected_provider})")

    print()


def test_client_creation():
    """Test client creation with different providers."""
    print("Testing client creation...")

    providers = list_providers()
    print(f"Available providers: {providers}")
    print()

    # Test OpenAI client
    try:
        client = create_client("gpt-3.5-turbo")  # noqa: F841
        print("✓ Successfully created OpenAI client")
    except Exception as e:
        print(f"✗ Failed to create OpenAI client: {e}")

    # Test Anthropic client
    if providers.get("anthropic"):
        try:
            client = create_client("claude-3-opus-20240229")  # noqa: F841
            print("✓ Successfully created Anthropic client")
        except Exception as e:
            print(f"✗ Failed to create Anthropic client: {e}")
    else:
        print("- Anthropic not installed (pip install ailib[anthropic])")

    print()


def test_factory_functions():
    """Test that factory functions support provider parameter."""
    print("Testing factory functions...")

    # Test create_agent
    try:
        _ = create_agent("test", model="gpt-3.5-turbo", provider="openai")
        print("✓ create_agent supports provider parameter")
    except Exception as e:
        print(f"✗ create_agent error: {e}")

    # Test create_chain
    try:
        _ = create_chain("Test prompt", model="gpt-3.5-turbo", provider="openai")
        print("✓ create_chain supports provider parameter")
    except Exception as e:
        print(f"✗ create_chain error: {e}")

    print()


def main():
    """Run all tests."""
    print("=== AILib Multi-Provider Support Test ===\n")

    # Load environment variables
    load_dotenv()

    # Run tests
    test_provider_detection()
    test_client_creation()
    test_factory_functions()

    print("Test complete!")


if __name__ == "__main__":
    main()
