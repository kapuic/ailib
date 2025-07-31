"""Basic usage examples for AILib - Simple by default, powerful when needed."""

import os

from ailib import Message, Role, create_agent, create_chain, create_session, tool


def example_simplest_chain():
    """Example: The simplest way to use AILib."""
    print("=== Simplest Usage ===")

    # One line to create and run a chain!
    chain = create_chain("Translate to French: {text}")
    result = chain.run(text="Hello world")
    print(f"Translation: {result}")

    # Multi-step chain - still simple!
    story_chain = create_chain(
        "Write a one-sentence story about {topic}",
        "Now make it more dramatic",
        "Add a plot twist",
    )
    result = story_chain.run(topic="a lost cat")
    print(f"\nStory: {result}")


def example_simplest_agent():
    """Example: The simplest way to create an agent."""
    print("\n=== Simplest Agent ===")

    # Define a tool with a simple decorator
    @tool
    def weather(city: str) -> str:
        """Get the weather for a city."""
        return f"The weather in {city} is sunny and 72°F"

    # Create agent in one line
    agent = create_agent("weather assistant", tools=[weather])

    # Use it
    result = agent.run("What's the weather in Paris?")
    print(f"Agent response: {result}")


def example_session_simple():
    """Example: Simple conversation management."""
    print("\n=== Simple Session ===")

    # Create a session
    session = create_session()

    # Add messages naturally
    session.add_message(Message(role=Role.USER, content="Tell me a joke"))
    session.add_message(
        Message(
            role=Role.ASSISTANT,
            content="Why don't scientists trust atoms? They make up everything!",
        )
    )
    session.add_message(Message(role=Role.USER, content="That's funny! Another one?"))

    # Use with a chain
    chain = create_chain("You are a comedian. Continue this conversation: {history}")
    messages = "\n".join(
        [f"{m.role.value}: {m.content}" for m in session.get_messages()]
    )
    result = chain.run(history=messages)
    print(f"Next joke: {result}")


def example_progressive_complexity():
    """Example: Add complexity only when needed."""
    print("\n=== Progressive Complexity ===")

    # Start simple
    create_agent("assistant")
    print("✓ Basic agent created")

    # Add model when needed
    create_agent("assistant", model="gpt-3.5-turbo")
    print("✓ Agent with custom model")

    # Add instructions when needed
    create_agent(
        "assistant",
        model="gpt-3.5-turbo",
        instructions="You are a helpful coding assistant. Keep answers concise.",
    )
    print("✓ Agent with instructions")

    # Add tools when needed
    @tool
    def calculate(expression: str) -> float:
        """Evaluate a mathematical expression."""
        return eval(expression, {"__builtins__": {}}, {})

    create_agent(
        "calculator", tools=[calculate], instructions="Help users with math problems."
    )
    print("✓ Agent with tools")


def example_templates_when_needed():
    """Example: Templates are available but not required."""
    print("\n=== Templates (Optional) ===")

    # Most of the time, inline templates are enough
    chain = create_chain("Summarize in {style} style: {text}")
    result = chain.run(style="academic", text="AI is changing the world...")
    print(f"Inline template result: {result[:50]}...")

    # For complex cases, PromptTemplate is available
    print("\n✓ PromptTemplate is available for complex cases")
    print("  from ailib import PromptTemplate")
    print("  template = PromptTemplate(...)")
    print("  But most users won't need it!")


def example_real_world():
    """Example: Real-world usage pattern."""
    print("\n=== Real World Example ===")

    # Customer service bot in just a few lines
    @tool
    def check_order(order_id: str) -> str:
        """Check order status."""
        # In real app, this would query a database
        return f"Order {order_id} is being shipped"

    @tool
    def refund_policy() -> str:
        """Get refund policy."""
        return "30-day money back guarantee on all items"

    # Create the bot
    support_bot = create_agent(
        "customer support",
        tools=[check_order, refund_policy],
        instructions="You are a helpful customer service agent. Be polite.",
    )

    # Handle customer queries
    queries = ["What's the status of order #12345?", "What's your refund policy?"]

    for query in queries:
        print(f"\nCustomer: {query}")
        response = support_bot.run(query)
        print(f"Bot: {response}")


if __name__ == "__main__":
    # Check for API key
    if not os.getenv("OPENAI_API_KEY"):
        print("Please set OPENAI_API_KEY environment variable")
        print("\nBut you can still see how simple the API is!")
        print("Check the code examples above ☝️")
        exit(1)

    print("AILib Basic Usage Examples")
    print("=" * 50)
    print("Design Philosophy: Simple by default, powerful when needed\n")

    # Run examples
    example_simplest_chain()
    example_simplest_agent()
    example_session_simple()
    example_progressive_complexity()
    example_templates_when_needed()
    example_real_world()

    print("\n" + "=" * 50)
    print("✨ That's it! AILib makes AI development simple.")
    print("No boilerplate, no complexity - just results.")
