"""Example demonstrating Pydantic validation in AILib."""

from ailib.validation import (
    AgentConfig,
    ChainConfig,
    LLMConfig,
    PromptTemplateConfig,
    SessionConfig,
    create_dynamic_model,
)
from pydantic import ValidationError


def demonstrate_llm_validation():
    """Demonstrate LLM configuration validation."""
    print("=== LLM Configuration Validation ===")

    # Valid configuration
    try:
        config = LLMConfig(model="gpt-4", temperature=0.8, max_tokens=1000, top_p=0.95)
        print(f"✓ Valid config: {config.model} with temperature {config.temperature}")
    except ValidationError as e:
        print(f"✗ Validation error: {e}")

    # Invalid configuration - temperature out of bounds
    try:
        config = LLMConfig(model="gpt-4", temperature=3.0)  # Too high!
    except ValidationError as e:
        print(f"✗ Expected error for invalid temperature: {e.errors()[0]['msg']}")

    # Invalid configuration - empty model
    try:
        config = LLMConfig(model="")
    except ValidationError as e:
        print(f"✗ Expected error for empty model: {e.errors()[0]['msg']}")


def demonstrate_prompt_validation():
    """Demonstrate prompt template validation."""
    print("\n=== Prompt Template Validation ===")

    # Valid template
    try:
        config = PromptTemplateConfig(
            template="Hello {name}, you have {count} new messages!",
            input_variables=["name", "count"],
        )
        print(f"✓ Valid template with variables: {config.input_variables}")
    except ValidationError as e:
        print(f"✗ Validation error: {e}")

    # Invalid - missing variable in template
    try:
        config = PromptTemplateConfig(
            template="Hello {name}!",
            input_variables=["name", "age"],  # 'age' not in template
        )
    except ValidationError as e:
        print(f"✗ Expected error for missing variable: {e.errors()[0]['msg']}")

    # Invalid - empty template
    try:
        config = PromptTemplateConfig(template="   ", input_variables=[])
    except ValidationError as e:
        print(f"✗ Expected error for empty template: {e.errors()[0]['msg']}")


def demonstrate_agent_validation():
    """Demonstrate agent configuration validation."""
    print("\n=== Agent Configuration Validation ===")

    # Valid configuration
    try:
        config = AgentConfig(
            name="research_agent",
            model="gpt-4",
            description="An agent for research tasks",
            tools=["web_search", "calculator"],
            max_iterations=5,
            temperature=0.3,
            verbose=True,
        )
        print(f"✓ Valid agent '{config.name}' with {len(config.tools)} tools")
    except ValidationError as e:
        print(f"✗ Validation error: {e}")

    # Invalid - empty name
    try:
        config = AgentConfig(name="", model="gpt-4")
    except ValidationError as e:
        print(f"✗ Expected error for empty name: {e.errors()[0]['msg']}")

    # Invalid - negative memory size
    try:
        config = AgentConfig(name="test_agent", model="gpt-4", memory_size=-5)
    except ValidationError as e:
        print(f"✗ Expected error for negative memory: {e.errors()[0]['msg']}")


def demonstrate_chain_validation():
    """Demonstrate chain configuration validation."""
    print("\n=== Chain Configuration Validation ===")

    # Valid configuration
    try:
        config = ChainConfig(
            name="analysis_chain",
            description="Chain for data analysis",
            max_iterations=10,
            early_stopping=True,
            retry_attempts=3,
            retry_delay=2.0,
            timeout=30.0,
        )
        print(
            f"✓ Valid chain '{config.name}' with {config.retry_attempts} retry attempts"
        )
    except ValidationError as e:
        print(f"✗ Validation error: {e}")

    # Invalid - zero iterations
    try:
        config = ChainConfig(name="test_chain", max_iterations=0)
    except ValidationError as e:
        print(f"✗ Expected error for zero iterations: {e.errors()[0]['msg']}")


def demonstrate_session_validation():
    """Demonstrate session configuration validation."""
    print("\n=== Session Configuration Validation ===")

    # Valid configuration
    try:
        config = SessionConfig(
            session_id="user-123-session",
            max_messages=100,
            ttl=3600,
            metadata={"user_id": "123", "context": "support"},
            auto_save=True,
            save_path="/tmp/sessions",
        )
        print(
            f"✓ Valid session with TTL {config.ttl}s "
            f"and auto-save to {config.save_path}"
        )
    except ValidationError as e:
        print(f"✗ Validation error: {e}")

    # Invalid - auto_save without path
    try:
        config = SessionConfig(
            auto_save=True
            # Missing save_path!
        )
    except ValidationError as e:
        print(f"✗ Expected error for auto_save without path: {e.errors()[0]['msg']}")


def demonstrate_dynamic_models():
    """Demonstrate dynamic model creation."""
    print("\n=== Dynamic Model Creation ===")

    # Create a custom configuration model
    CustomToolConfig = create_dynamic_model(
        "CustomToolConfig",
        {
            "tool_id": (str, ...),  # Required
            "api_key": (str, ...),  # Required
            "timeout": (int, 30),  # Optional with default
            "retries": (int, 3),  # Optional with default
            "base_url": (str | None, None),  # Optional
        },
    )

    # Valid configuration
    try:
        config = CustomToolConfig(
            tool_id="weather_api", api_key="sk-123456", timeout=60
        )
        print(
            f"✓ Created custom config for '{config.tool_id}' "
            f"with timeout {config.timeout}s"
        )
    except ValidationError as e:
        print(f"✗ Validation error: {e}")

    # Invalid - missing required field
    try:
        config = CustomToolConfig(
            tool_id="weather_api"
            # Missing api_key!
        )
    except ValidationError as e:
        print(f"✗ Expected error for missing api_key: {e.errors()[0]['msg']}")


def demonstrate_integration_with_ailib():
    """Demonstrate how validation integrates with AILib components."""
    print("\n=== Integration with AILib ===")

    # The validation happens automatically when creating AILib objects
    from ailib.core import PromptTemplate

    try:
        # This will trigger validation
        template = PromptTemplate("Ask {question} about {topic}")
        print(f"✓ Created template with variables: {template.variables}")
    except ValidationError as e:
        print(f"✗ Template validation failed: {e}")

    # Invalid template
    try:
        template = PromptTemplate("")  # Empty template
    except ValueError as e:  # PromptTemplate raises ValueError based on ValidationError
        print(f"✗ Expected error for empty template: {e}")


if __name__ == "__main__":
    print("AILib Pydantic Validation Examples")
    print("=" * 50)

    demonstrate_llm_validation()
    demonstrate_prompt_validation()
    demonstrate_agent_validation()
    demonstrate_chain_validation()
    demonstrate_session_validation()
    demonstrate_dynamic_models()
    demonstrate_integration_with_ailib()

    print("\n✅ All validation examples completed!")
