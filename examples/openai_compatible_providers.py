"""Demonstration of using OpenAI-compatible providers with AILib.

Many LLM providers offer OpenAI-compatible APIs, making it easy to switch between them.
This example shows how to use various providers with AILib.
"""

from ailib import create_client, list_providers
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


def show_available_providers():
    """Display all available providers with their configuration."""
    print("=== Available Providers ===\n")

    providers = list_providers()

    # Group by compatibility
    openai_compatible = []
    custom_api = []

    for name, info in providers.items():
        if info["openai_compatible"]:
            openai_compatible.append((name, info))
        else:
            custom_api.append((name, info))

    print("OpenAI-Compatible Providers (use OpenAI client with base_url):")
    print("-" * 60)
    for name, info in openai_compatible:
        status = "✓" if info["available"] else "✗"
        print(f"{status} {name:15} - Base URL: {info['base_url'] or 'default'}")
        if info["default_model"]:
            print(f"  └─ Default model: {info['default_model']}")

    print("\nCustom API Providers (require separate implementation):")
    print("-" * 60)
    for name, info in custom_api:
        status = "✓" if info["available"] else "✗"
        print(f"{status} {name:15} - Available: {info['available']}")
        if info["default_model"]:
            print(f"  └─ Default model: {info['default_model']}")

    print()


def demo_provider_usage():
    """Demonstrate using different providers."""
    print("=== Provider Usage Examples ===\n")

    # Example 1: Default OpenAI
    print("1. Standard OpenAI:")
    try:
        client = create_client("gpt-3.5-turbo")  # noqa: F841
        print("   ✓ Created OpenAI client")
    except Exception as e:
        print(f"   ✗ Error: {e}")

    # Example 2: Together AI (OpenAI-compatible)
    print("\n2. Together AI (OpenAI-compatible):")
    print("   # Auto-detected from model name")
    print("   client = create_client('mistralai/Mixtral-8x7B-Instruct-v0.1')")
    print("   # Or explicit provider")
    print("   client = create_client(provider='together', model='llama-2-70b')")

    # Example 3: Local model with LM Studio
    print("\n3. Local model with LM Studio:")
    print("   client = create_client(")
    print("       model='local-model',")
    print("       base_url='http://localhost:1234/v1',")
    print("       api_key='not-needed'  # Local models usually don't need API keys")
    print("   )")

    # Example 4: Custom OpenAI-compatible endpoint
    print("\n4. Custom OpenAI-compatible endpoint:")
    print("   client = create_client(")
    print("       model='gpt-3.5-turbo',")
    print("       base_url='https://your-custom-endpoint.com/v1',")
    print("       api_key='your-api-key'")
    print("   )")

    print()


def demo_agent_with_providers():
    """Show how to use agents with different providers."""
    print("=== Using Agents with Different Providers ===\n")

    # Example agent configurations
    examples = [
        {
            "name": "OpenAI GPT-4 Agent",
            "code": """agent = create_agent(
    "assistant",
    model="gpt-4",
    instructions="You are a helpful assistant."
)""",
        },
        {
            "name": "Together AI Mixtral Agent",
            "code": """agent = create_agent(
    "assistant",
    model="mistralai/Mixtral-8x7B-Instruct-v0.1",
    api_key=os.getenv("TOGETHER_API_KEY"),
    instructions="You are a helpful assistant."
)""",
        },
        {
            "name": "Local Ollama Agent",
            "code": """agent = create_agent(
    "assistant",
    provider="ollama",
    model="llama2",
    instructions="You are a helpful assistant."
)""",
        },
        {
            "name": "Groq Agent (Fast inference)",
            "code": """agent = create_agent(
    "assistant",
    provider="groq",
    model="mixtral-8x7b-32768",
    api_key=os.getenv("GROQ_API_KEY"),
    instructions="You are a helpful assistant."
)""",
        },
    ]

    for example in examples:
        print(f"{example['name']}:")
        print(example["code"])
        print()


def main():
    """Run all demonstrations."""
    show_available_providers()
    demo_provider_usage()
    demo_agent_with_providers()

    print("=== Key Advantages ===\n")
    print(
        "1. Most providers work with the existing OpenAI client - just change base_url"
    )
    print("2. Auto-detection of providers from model names")
    print("3. Environment variable support for API keys")
    print("4. Easy switching between cloud and local models")
    print("5. No need for separate client implementations for each provider")
    print("\nSee the full list of providers in: ailib.core.PROVIDERS")


if __name__ == "__main__":
    main()
