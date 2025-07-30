"""Example demonstrating the simplified Vercel AI SDK-style API."""

from ailib import create_agent, create_chain, tool


def example_simple_agent():
    """Show how simple it is to create an agent."""
    print("=== Simple Agent Creation ===")

    # Create agent with zero configuration
    agent = create_agent("assistant")  # noqa: F841
    print("✓ Created agent with just a name!")

    # Create agent with custom model
    agent_35 = create_agent("budget_assistant", model="gpt-3.5-turbo")  # noqa: F841
    print("✓ Created agent with custom model")

    # Create agent with instructions
    agent_helpful = create_agent(  # noqa: F841
        "helper", instructions="You are a very helpful and friendly assistant."
    )
    print("✓ Created agent with custom instructions")


def example_agent_with_tools():
    """Show agent creation with tools."""
    print("\n=== Agent with Tools ===")

    # Define tools using decorator
    @tool
    def calculate(expression: str) -> str:
        """Evaluate a mathematical expression."""
        try:
            result = eval(expression)
            return f"The result is: {result}"
        except Exception as e:
            return f"Error: {e}"

    @tool
    def get_time() -> str:
        """Get the current time."""
        from datetime import datetime

        return f"Current time: {datetime.now().strftime('%H:%M:%S')}"

    # Create agent with tools - one line!
    agent = create_agent(  # noqa: F841
        "calculator",
        tools=[calculate, get_time],
        instructions="You are a helpful calculator assistant.",
    )
    print("✓ Created agent with tools in one line!")

    # Use the agent
    # result = agent.run("What's 25 * 4?")
    # print(result)


def example_simple_chain():
    """Show simplified chain creation."""
    print("\n=== Simple Chain Creation ===")

    # Create chain without explicit LLM client
    chain = create_chain(  # noqa: F841
        "Translate this to French: {text}",
        "Now make it more formal",
        "Finally, add a polite greeting",
    )
    print("✓ Created chain with auto-configured LLM!")

    # Chain with custom model
    chain_35 = create_chain(  # noqa: F841
        "Summarize this text: {text}", "Extract the key points", model="gpt-3.5-turbo"
    )
    print("✓ Created chain with custom model")


def example_vs_old_way():
    """Compare old verbose way vs new simple way."""
    print("\n=== Old Way vs New Way ===")

    print("Old way (verbose - LangChain style):")
    print(
        """
    from ailib import OpenAIClient, Agent
    from ailib.validation import AgentConfig

    # Had to create client first
    client = OpenAIClient(model="gpt-4")

    # Had to create config object
    config = AgentConfig(
        name="assistant",
        model="gpt-4",
        tools=["search", "calculate"]
    )

    # Finally create agent
    agent = Agent(llm=client, config=config)
    """
    )

    print("\nNew way (simple - Vercel AI SDK style):")
    print(
        """
    from ailib import create_agent

    # One line! Configuration is hidden
    agent = create_agent("assistant", tools=[search, calculate])
    """
    )

    print("\n✓ Much simpler and cleaner!")


def example_advanced_options():
    """Show that advanced options are still available."""
    print("\n=== Advanced Options Still Available ===")

    # Show the API signature (without actually creating)
    print("All options can be passed as kwargs:")
    print(
        """
    agent = create_agent(
        "advanced",
        model="gpt-4",
        temperature=0.2,
        max_steps=20,
        api_key="your-api-key",  # Can override API key
        verbose=True
    )
    """
    )
    print("✓ Advanced options work seamlessly as kwargs")

    print("\nCan still use custom LLM if needed:")
    print(
        """
    from ailib import OpenAIClient

    custom_llm = OpenAIClient(
        model="gpt-4",
        base_url="https://custom.openai.com",
        api_key="your-key"
    )
    agent = create_agent("custom", llm=custom_llm)
    """
    )
    print("✓ Custom LLM clients supported when needed")


def main():
    """Run all examples."""
    print("AILib Simplified API Examples")
    print("=" * 50)
    print("Following Vercel AI SDK philosophy:")
    print("- Simple, functional API")
    print("- Zero configuration to start")
    print("- Hide complexity (validation happens internally)")
    print("- Progressive disclosure (advanced options available)")
    print()

    try:
        example_simple_agent()
        example_agent_with_tools()
        example_simple_chain()
    except Exception as e:
        print(f"\n⚠️  Note: Some examples require an API key: {e}")
        print("Set OPENAI_API_KEY environment variable to run these examples.")

    # These examples don't require API key
    example_vs_old_way()
    example_advanced_options()

    print("\n" + "=" * 50)
    print("✅ The new API is much simpler and more intuitive!")
    print("\nKey Benefits:")
    print("- No need to import config classes")
    print("- No need to create LLM clients explicitly")
    print("- Clean, functional API like Vercel AI SDK")
    print("- Same power and flexibility, better developer experience")


if __name__ == "__main__":
    main()
