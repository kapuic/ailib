"""Demonstration of AILib's multi-provider support.

This example shows how to use different LLM providers with AILib.
"""

from ailib import create_agent, create_chain, create_client, list_providers
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


def main():
    """Demonstrate multi-provider capabilities."""

    print("=== AILib Multi-Provider Demo ===\n")

    # Show available providers
    print("Available providers:")
    providers = list_providers()
    for provider, available in providers.items():
        status = "✓ Installed" if available else "✗ Not installed"
        print(f"  {provider}: {status}")
    print()

    # Demo 1: Auto-detection based on model name
    print("1. Auto-detecting provider from model name:")
    print("-" * 40)

    # OpenAI model (auto-detected)
    try:
        client1 = create_client("gpt-4")  # noqa: F841
        print("✓ Created client for 'gpt-4' -> OpenAI provider")
    except Exception as e:
        print(f"✗ Failed to create OpenAI client: {e}")

    # Claude model (auto-detected)
    try:
        client2 = create_client("claude-3-opus-20240229")  # noqa: F841
        print("✓ Created client for 'claude-3-opus' -> Anthropic provider")
    except Exception as e:
        print(f"✗ Failed to create Anthropic client: {e}")

    print()

    # Demo 2: Using create_agent with different providers
    print("2. Creating agents with different providers:")
    print("-" * 40)

    # OpenAI agent
    try:
        openai_agent = create_agent(
            "assistant", model="gpt-4", instructions="You are a helpful assistant."
        )
        result = openai_agent.run("Say hello in one sentence.")
        print(f"OpenAI Agent: {result}")
    except Exception as e:
        print(f"OpenAI Agent error: {e}")

    # Claude agent (if Anthropic is installed)
    if providers.get("anthropic"):
        try:
            claude_agent = create_agent(
                "assistant",
                model="claude-3-opus-20240229",
                instructions="You are a helpful assistant.",
            )
            result = claude_agent.run("Say hello in one sentence.")
            print(f"Claude Agent: {result}")
        except Exception as e:
            print(f"Claude Agent error: {e}")
    else:
        print("Anthropic not installed. Install with: pip install ailib[anthropic]")

    print()

    # Demo 3: Using create_chain with different providers
    print("3. Creating chains with different providers:")
    print("-" * 40)

    # Chain with explicit provider
    try:
        chain = create_chain(
            "Translate to French: {text}",
            "Now make it more formal",
            provider="openai",  # Explicit provider
            model="gpt-3.5-turbo",
        )
        result = chain.run(text="Hello friend")
        print(f"Translation chain result: {result}")
    except Exception as e:
        print(f"Chain error: {e}")

    print()

    # Demo 4: Switching providers dynamically
    print("4. Dynamic provider switching:")
    print("-" * 40)

    providers_to_test = []
    if providers.get("openai"):
        providers_to_test.append(("openai", "gpt-3.5-turbo"))
    if providers.get("anthropic"):
        providers_to_test.append(("anthropic", "claude-3-haiku-20240307"))

    prompt = "What is 2+2? Answer in exactly one word."

    for provider_name, model in providers_to_test:
        try:
            client = create_client(model=model, provider=provider_name)
            agent = create_agent("calculator", llm=client)
            result = agent.run(prompt)
            print(f"{provider_name} ({model}): {result}")
        except Exception as e:
            print(f"{provider_name} error: {e}")


if __name__ == "__main__":
    main()
