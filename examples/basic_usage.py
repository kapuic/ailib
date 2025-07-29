"""Basic usage examples for AILib."""

import os

from ailib import Agent, Chain, OpenAIClient, Prompt, PromptTemplate, Session, tool


def example_simple_completion():
    """Example: Simple LLM completion."""
    print("=== Simple Completion ===")

    # Initialize OpenAI client
    client = OpenAIClient(model="gpt-3.5-turbo")

    # Create messages
    prompt = Prompt()
    prompt.add_system("You are a helpful assistant.")
    prompt.add_user("What is the capital of France?")

    # Get completion
    response = client.complete(prompt.build())
    print(f"Response: {response.content}")
    print(f"Tokens used: {response.usage['total_tokens']}")


def example_prompt_templates():
    """Example: Using prompt templates."""
    print("\n=== Prompt Templates ===")

    # Create a template
    template = PromptTemplate(
        "Translate the following {language} text to English: {text}"
    )

    # Format the template
    formatted = template.format(language="French", text="Bonjour le monde")
    print(f"Formatted prompt: {formatted}")

    # Use with LLM
    client = OpenAIClient(model="gpt-3.5-turbo")
    messages = Prompt.from_template(
        template.template, language="Spanish", text="Hola mundo"
    )

    response = client.complete(messages)
    print(f"Translation: {response.content}")


def example_chains():
    """Example: Using chains for multi-step reasoning."""
    print("\n=== Chains ===")

    client = OpenAIClient(model="gpt-3.5-turbo")

    # Create a chain for a multi-step process
    chain = (
        Chain(client)
        .add_system("You are a helpful assistant.")
        .add_user("What is the capital of France?", name="capital_question")
        .add_user(
            "What is the population of {capital_question}?", name="population_question"
        )
    )

    # Run the chain
    result = chain.run()
    print(f"Final result: {result}")


def example_tools_and_agents():
    """Example: Creating tools and using agents."""
    print("\n=== Tools and Agents ===")

    # Define custom tools
    @tool
    def weather(city: str) -> str:
        """Get the weather for a city."""
        # In real implementation, this would call a weather API
        return f"The weather in {city} is sunny and 72°F"

    @tool
    def calculator(expression: str) -> float:
        """Evaluate a mathematical expression."""
        try:
            # Safe evaluation of simple math
            result = eval(expression, {"__builtins__": {}}, {})
            return float(result)
        except Exception as e:
            return f"Error: {str(e)}"

    # Create an agent with tools
    client = OpenAIClient(model="gpt-4")
    agent = Agent(llm=client, verbose=True)
    agent.with_tools(weather, calculator)

    # Run the agent
    result = agent.run(
        "What's the weather in Paris? Also, calculate 15% tip on a $85 bill."
    )
    print(f"\nFinal answer: {result}")


def example_session_management():
    """Example: Using sessions for conversation history."""
    print("\n=== Session Management ===")

    client = OpenAIClient(model="gpt-3.5-turbo")
    session = Session()

    # Add system message
    session.add_system_message("You are a helpful math tutor.")

    # First interaction
    session.add_user_message("What is the Pythagorean theorem?")
    response = client.complete(session.get_messages())
    session.add_assistant_message(response.content)
    print(f"Assistant: {response.content}")

    # Follow-up question (with context)
    session.add_user_message("Can you give me an example?")
    response = client.complete(session.get_messages())
    session.add_assistant_message(response.content)
    print(f"\nAssistant: {response.content}")

    # Store information in session memory
    session.set_memory("topic", "Pythagorean theorem")
    session.set_memory("examples_given", 1)

    print(f"\nSession info: {session}")


def example_chain_with_processing():
    """Example: Chain with custom processing functions."""
    print("\n=== Chain with Processing ===")

    client = OpenAIClient(model="gpt-3.5-turbo")

    # Define processors
    def extract_number(text: str) -> int:
        """Extract the first number from text."""
        import re

        match = re.search(r"\d+", text)
        return int(match.group()) if match else 0

    def to_uppercase(text: str) -> str:
        """Convert text to uppercase."""
        return text.upper()

    # Create chain with processors
    chain = (
        Chain(client)
        .add_system("You are a helpful assistant.")
        .add_user(
            "How many states are in the USA?",
            processor=extract_number,
            name="num_states",
        )
        .add_user(
            "Name any {num_states} US cities", processor=to_uppercase, name="cities"
        )
    )

    result = chain.run()
    print(f"Cities (uppercase): {result}")


if __name__ == "__main__":
    # Make sure OPENAI_API_KEY is set
    if not os.getenv("OPENAI_API_KEY"):
        print("Please set OPENAI_API_KEY environment variable")
        exit(1)

    # Run examples
    example_simple_completion()
    example_prompt_templates()
    example_chains()
    example_session_management()
    example_chain_with_processing()

    # Note: Agent example might use more tokens
    # example_tools_and_agents()
