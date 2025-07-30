"""Example demonstrating validation through factory functions in AILib.

Since validation is now internal, this example shows how validation
works through the simplified factory functions.
"""

from ailib import create_agent, create_chain, create_session


def demonstrate_agent_validation():
    """Demonstrate agent validation through factory function."""
    print("=== Agent Creation with Validation ===")

    # For demo purposes, use a dummy API key
    import os

    dummy_key = os.environ.get("OPENAI_API_KEY", "sk-dummy-key-for-validation-demo")

    # Valid agent creation
    try:
        agent = create_agent(
            "research_agent",
            model="gpt-4",
            temperature=0.8,
            max_steps=5,
            verbose=True,
            api_key=dummy_key,
        )
        print(f"✓ Created agent '{agent.name}' with model {agent.model}")
    except Exception as e:
        print(f"✗ Validation error: {e}")

    # Invalid - temperature out of bounds (will be caught by internal validation)
    try:
        agent = create_agent(
            "test_agent",
            temperature=3.0,  # Too high!
            api_key=dummy_key,
        )
    except Exception as e:
        # Extract the validation error message
        if "temperature" in str(e).lower():
            print(
                "✗ Expected error for invalid temperature: "
                "temperature must be between 0 and 2"
            )
        else:
            print(f"✗ Error: {e}")

    # Invalid - empty name (will be caught by internal validation)
    try:
        agent = create_agent(
            "",  # Empty name!
            model="gpt-4",
            api_key=dummy_key,
        )
    except Exception as e:
        # Extract the validation error message
        if "name" in str(e).lower() or "empty" in str(e).lower():
            print("✗ Expected error for empty name: name cannot be empty")
        else:
            print(f"✗ Error: {e}")


def demonstrate_chain_validation():
    """Demonstrate chain validation through factory function."""
    print("\n=== Chain Creation with Validation ===")

    # Valid chain creation
    try:
        chain = create_chain(
            "Translate to Spanish: {text}",
            "Now make it more formal",
            verbose=True,
            retry_attempts=3,
        )
        print(
            f"✓ Created chain with {len(chain)} steps and "
            f"{chain.retry_attempts} retry attempts"
        )
    except Exception as e:
        print(f"✗ Validation error: {e}")

    # Invalid - zero retry attempts (will be caught by internal validation)
    try:
        chain = create_chain(
            "Summarize: {text}",
            retry_attempts=0,  # Must be at least 1
        )
    except Exception as e:
        print(f"✗ Expected error for zero retry attempts: {e}")

    # The chain factory doesn't expose all validation errors directly,
    # but they're still enforced internally
    print("✓ Validation happens internally to maintain simplicity")


def demonstrate_session_validation():
    """Demonstrate session validation through factory function."""
    print("\n=== Session Creation with Validation ===")

    # Valid session creation
    try:
        session = create_session(
            session_id="user-123",
            max_messages=100,
            ttl=3600,
            metadata={"user_id": "123", "context": "support"},
        )
        print(f"✓ Created session '{session.session_id}' with TTL {session.ttl}s")
    except Exception as e:
        print(f"✗ Validation error: {e}")

    # Invalid - auto_save without path (will be caught by internal validation)
    try:
        session = create_session(
            auto_save=True,
            # Missing save_path!
        )
    except Exception as e:
        print(f"✗ Expected error for auto_save without path: {e}")

    # Valid - session with auto-save
    try:
        session = create_session(
            auto_save=True,
            save_path="/tmp/sessions",
        )
        print(f"✓ Created session with auto-save to {session.save_path}")
    except Exception as e:
        print(f"✗ Validation error: {e}")


def demonstrate_direct_instantiation():
    """Show that direct instantiation bypasses validation for simplicity."""
    print("\n=== Direct Instantiation (No Validation) ===")

    from ailib import Agent, Chain, Session

    # Direct instantiation doesn't validate - following Vercel AI SDK philosophy
    # This allows maximum flexibility but less safety
    agent = Agent(name="direct_agent", temperature=5.0)  # No validation!
    print(
        f"✓ Created agent directly with temperature {agent.temperature} (no validation)"
    )

    chain = Chain(verbose=True)  # noqa: F841
    print("✓ Created chain directly (no validation)")

    session = Session(session_id="direct-session")  # noqa: F841
    print("✓ Created session directly (no validation)")

    print("\nNote: Factory functions provide validation, direct instantiation doesn't.")
    print("This follows Vercel AI SDK's philosophy of progressive disclosure.")


def demonstrate_validation_benefits():
    """Show the benefits of validation through factory functions."""
    print("\n=== Benefits of Factory Function Validation ===")

    # For demo purposes
    import os

    dummy_key = os.environ.get("OPENAI_API_KEY", "sk-dummy-key-for-validation-demo")

    # 1. Type safety and early error detection
    print("1. Early error detection:")
    try:
        # This will fail with a clear error
        agent = create_agent(
            "helper", temperature="high", api_key=dummy_key
        )  # Wrong type!
    except Exception as e:
        print(f"   ✓ Caught type error: {type(e).__name__}")

    # 2. Sensible defaults
    print("\n2. Sensible defaults:")
    agent = create_agent("assistant", api_key=dummy_key)  # Minimal config
    print(f"   ✓ Model defaulted to: {agent.model}")
    print(f"   ✓ Temperature defaulted to: {agent.temperature}")
    print(f"   ✓ Max steps defaulted to: {agent.max_steps}")

    # 3. Configuration consistency
    print("\n3. Configuration consistency:")
    session = create_session(max_messages=50)
    print("   ✓ Session created with consistent defaults")
    print(f"   ✓ Max messages: {session.max_history}")
    print(f"   ✓ Auto-save: {session.auto_save}")


def demonstrate_simplified_api():
    """Show how the simplified API works compared to the old way."""
    print("\n=== Simplified API vs Old Approach ===")

    print("Old way (if we exposed validation):")
    print(
        """
    # Would need to import and use config classes
    from ailib.validation import AgentConfig
    config = AgentConfig(name="agent", model="gpt-4", ...)
    agent = Agent(config=config)
    """
    )

    print("\nNew way (Vercel AI SDK style):")
    print(
        """
    # Just use the factory function
    from ailib import create_agent
    agent = create_agent("agent", model="gpt-4")
    """
    )

    # Actual example
    agent = create_agent("demo", verbose=True)
    print(f"\n✓ Created agent '{agent.name}' with simple API")
    print("✓ Validation happened internally, complexity hidden from user")


def demonstrate_integration_with_tools():
    """Show how validation works with tools."""
    print("\n=== Validation with Tools ===")

    import os

    from ailib import tool

    dummy_key = os.environ.get("OPENAI_API_KEY", "sk-dummy-key-for-validation-demo")

    # Define a simple tool
    @tool
    def calculate(expression: str) -> str:
        """Calculate a mathematical expression."""
        try:
            return str(eval(expression))
        except Exception as e:
            return f"Error: {e}"

    # Create agent with tool
    try:
        agent = create_agent(
            "calculator",
            tools=[calculate],
            temperature=0.2,  # Lower temperature for accuracy
            api_key=dummy_key,
        )
        print(f"✓ Created agent with tool: {agent.tool_registry.list_tools()}")
    except Exception as e:
        print(f"✗ Error creating agent with tools: {e}")

    print("✓ Tools are validated as part of agent creation")


if __name__ == "__main__":
    print("AILib Validation Examples (Through Factory Functions)")
    print("=" * 50)
    print("Validation is now internal - following Vercel AI SDK philosophy")
    print("Factory functions provide safety, direct instantiation provides flexibility")
    print()

    demonstrate_agent_validation()
    demonstrate_chain_validation()
    demonstrate_session_validation()
    demonstrate_direct_instantiation()
    demonstrate_validation_benefits()
    demonstrate_simplified_api()
    demonstrate_integration_with_tools()

    print("\n✅ All validation examples completed!")
    print("\nKey Takeaways:")
    print(
        "- Use factory functions (create_agent, create_chain, create_session) "
        "for validated creation"
    )
    print("- Direct instantiation bypasses validation for maximum flexibility")
    print("- Validation complexity is hidden internally")
    print("- This follows Vercel AI SDK's principle of progressive disclosure")
