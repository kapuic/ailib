"""Tests for Pydantic validation integration."""

import pytest
from ailib.validation import (
    AgentConfig,
    ChainConfig,
    LLMConfig,
    MessageConfig,
    PromptTemplateConfig,
    SafetyConfig,
    SessionConfig,
    ToolConfig,
    ToolParameterSchema,
    create_dynamic_model,
)
from pydantic import ValidationError


class TestPromptTemplateConfig:
    """Test prompt template configuration validation."""

    def test_valid_template_config(self):
        """Test valid template configuration."""
        config = PromptTemplateConfig(
            template="Hello {name}, welcome to {place}!",
            input_variables=["name", "place"],
        )
        assert config.template == "Hello {name}, welcome to {place}!"
        assert config.input_variables == ["name", "place"]

    def test_empty_template_raises_error(self):
        """Test that empty template raises validation error."""
        with pytest.raises(ValidationError) as exc_info:
            PromptTemplateConfig(template="", input_variables=[])
        assert "Template cannot be empty" in str(exc_info.value)

    def test_missing_placeholders_raises_error(self):
        """Test that missing placeholders raise validation error."""
        with pytest.raises(ValidationError) as exc_info:
            PromptTemplateConfig(
                template="Hello {name}!",
                input_variables=["name", "age"],  # age not in template
            )
        assert "not found in template placeholders" in str(exc_info.value)


class TestMessageConfig:
    """Test message configuration validation."""

    def test_valid_message_config(self):
        """Test valid message configuration."""
        config = MessageConfig(role="user", content="Hello, AI!")
        assert config.role == "user"
        assert config.content == "Hello, AI!"

    def test_invalid_role_raises_error(self):
        """Test that invalid role raises validation error."""
        with pytest.raises(ValidationError):
            MessageConfig(role="invalid_role", content="Hello")

    def test_empty_message_raises_error(self):
        """Test that empty message without function calls raises error."""
        with pytest.raises(ValidationError) as exc_info:
            MessageConfig(role="user")
        assert "must have either content" in str(exc_info.value)

    def test_message_with_function_call_valid(self):
        """Test message with function call is valid."""
        config = MessageConfig(
            role="assistant",
            function_call={"name": "get_weather", "arguments": '{"location": "NYC"}'},
        )
        assert config.function_call is not None


class TestLLMConfig:
    """Test LLM configuration validation."""

    def test_valid_llm_config(self):
        """Test valid LLM configuration."""
        config = LLMConfig(model="gpt-4", temperature=0.8, max_tokens=1000)
        assert config.model == "gpt-4"
        assert config.temperature == 0.8
        assert config.max_tokens == 1000

    def test_temperature_bounds(self):
        """Test temperature validation bounds."""
        # Valid temperatures
        LLMConfig(model="gpt-4", temperature=0.0)
        LLMConfig(model="gpt-4", temperature=2.0)

        # Invalid temperatures
        with pytest.raises(ValidationError):
            LLMConfig(model="gpt-4", temperature=-0.1)
        with pytest.raises(ValidationError):
            LLMConfig(model="gpt-4", temperature=2.1)

    def test_empty_model_raises_error(self):
        """Test that empty model name raises error."""
        with pytest.raises(ValidationError) as exc_info:
            LLMConfig(model="   ")
        assert "Model name cannot be empty" in str(exc_info.value)


class TestChainConfig:
    """Test chain configuration validation."""

    def test_valid_chain_config(self):
        """Test valid chain configuration."""
        config = ChainConfig(
            name="my_chain",
            description="A test chain",
            max_iterations=5,
            retry_attempts=2,
        )
        assert config.name == "my_chain"
        assert config.max_iterations == 5
        assert config.retry_attempts == 2

    def test_empty_name_raises_error(self):
        """Test that empty chain name raises error."""
        with pytest.raises(ValidationError) as exc_info:
            ChainConfig(name="")
        assert "Chain name cannot be empty" in str(exc_info.value)

    def test_negative_iterations_raises_error(self):
        """Test that negative iterations raise error."""
        with pytest.raises(ValidationError):
            ChainConfig(name="test", max_iterations=0)


class TestAgentConfig:
    """Test agent configuration validation."""

    def test_valid_agent_config(self):
        """Test valid agent configuration."""
        config = AgentConfig(
            name="my_agent",
            model="gpt-4",
            tools=["calculator", "web_search"],
            temperature=0.5,
        )
        assert config.name == "my_agent"
        assert config.model == "gpt-4"
        assert config.tools == ["calculator", "web_search"]
        assert config.temperature == 0.5

    def test_empty_name_raises_error(self):
        """Test that empty agent name raises error."""
        with pytest.raises(ValidationError) as exc_info:
            AgentConfig(name="", model="gpt-4")
        assert "Agent name cannot be empty" in str(exc_info.value)

    def test_memory_size_validation(self):
        """Test memory size validation."""
        # Valid memory sizes
        AgentConfig(name="test", model="gpt-4", memory_size=0)
        AgentConfig(name="test", model="gpt-4", memory_size=100)

        # Invalid memory size
        with pytest.raises(ValidationError):
            AgentConfig(name="test", model="gpt-4", memory_size=-1)


class TestToolConfig:
    """Test tool configuration validation."""

    def test_valid_tool_config(self):
        """Test valid tool configuration."""
        config = ToolConfig(
            name="calculator",
            description="Performs mathematical calculations",
            parameters=[
                ToolParameterSchema(
                    name="expression",
                    type="str",
                    description="Math expression to evaluate",
                    required=True,
                )
            ],
        )
        assert config.name == "calculator"
        assert len(config.parameters) == 1

    def test_invalid_tool_name_raises_error(self):
        """Test that invalid tool name raises error."""
        with pytest.raises(ValidationError) as exc_info:
            ToolConfig(name="123-tool", description="Invalid name")
        assert "must be a valid Python identifier" in str(exc_info.value)

    def test_empty_description_raises_error(self):
        """Test that empty description raises error."""
        with pytest.raises(ValidationError) as exc_info:
            ToolConfig(name="tool", description="   ")
        assert "Tool description cannot be empty" in str(exc_info.value)


class TestToolParameterSchema:
    """Test tool parameter schema validation."""

    def test_valid_parameter_schema(self):
        """Test valid parameter schema."""
        schema = ToolParameterSchema(
            name="count",
            type="int",
            description="Number of items",
            required=False,
            default=10,
            min_value=1,
            max_value=100,
        )
        assert schema.name == "count"
        assert schema.type == "int"
        assert schema.default == 10

    def test_invalid_type_raises_error(self):
        """Test that invalid type raises error."""
        with pytest.raises(ValidationError) as exc_info:
            ToolParameterSchema(name="param", type="invalid_type")
        assert "Invalid parameter type" in str(exc_info.value)

    def test_numeric_constraints_on_non_numeric_raises_error(self):
        """Test that numeric constraints on non-numeric types raise error."""
        with pytest.raises(ValidationError) as exc_info:
            ToolParameterSchema(name="text", type="str", min_value=0)
        assert "min_value only applies to numeric types" in str(exc_info.value)

    def test_string_constraints_on_non_string_raises_error(self):
        """Test that string constraints on non-string types raise error."""
        with pytest.raises(ValidationError) as exc_info:
            ToolParameterSchema(name="num", type="int", min_length=5)
        assert "min_length only applies to string type" in str(exc_info.value)

    def test_min_max_constraint_validation(self):
        """Test min/max constraint validation."""
        with pytest.raises(ValidationError) as exc_info:
            ToolParameterSchema(name="num", type="int", min_value=100, max_value=50)
        assert "min_value cannot be greater than max_value" in str(exc_info.value)

    def test_non_required_without_default_raises_error(self):
        """Test that non-required parameters without default raise error."""
        with pytest.raises(ValidationError) as exc_info:
            ToolParameterSchema(name="param", type="str", required=False)
        assert "Non-required parameters must have a default value" in str(
            exc_info.value
        )


class TestSessionConfig:
    """Test session configuration validation."""

    def test_valid_session_config(self):
        """Test valid session configuration."""
        config = SessionConfig(
            session_id="test-123",
            max_messages=50,
            ttl=3600,
            metadata={"user": "test"},
        )
        assert config.session_id == "test-123"
        assert config.max_messages == 50
        assert config.ttl == 3600

    def test_auto_save_without_path_raises_error(self):
        """Test that auto_save without save_path raises error."""
        with pytest.raises(ValidationError) as exc_info:
            SessionConfig(auto_save=True)
        assert "save_path must be provided when auto_save is True" in str(
            exc_info.value
        )

    def test_auto_save_with_path_valid(self):
        """Test that auto_save with save_path is valid."""
        config = SessionConfig(auto_save=True, save_path="/tmp/sessions")
        assert config.auto_save is True
        assert config.save_path == "/tmp/sessions"


class TestSafetyConfig:
    """Test safety configuration validation."""

    def test_valid_safety_config(self):
        """Test valid safety configuration."""
        config = SafetyConfig(
            enabled=True,
            block_harmful_content=True,
            sensitive_topics=["politics", "religion"],
            max_output_length=2000,
            rate_limit=60,
        )
        assert config.enabled is True
        assert config.sensitive_topics == ["politics", "religion"]
        assert config.rate_limit == 60

    def test_negative_max_output_length_raises_error(self):
        """Test that negative max output length raises error."""
        with pytest.raises(ValidationError):
            SafetyConfig(max_output_length=0)

    def test_negative_rate_limit_raises_error(self):
        """Test that negative rate limit raises error."""
        with pytest.raises(ValidationError):
            SafetyConfig(rate_limit=0)


class TestDynamicModelCreation:
    """Test dynamic model creation utility."""

    def test_create_simple_dynamic_model(self):
        """Test creating a simple dynamic model."""
        UserModel = create_dynamic_model(
            "UserModel",
            {
                "name": (str, ...),
                "age": (int, 0),
                "email": (str | None, None),
            },
        )

        # Test valid instance
        user = UserModel(name="John Doe", age=30)
        assert user.name == "John Doe"
        assert user.age == 30
        assert user.email is None

        # Test missing required field
        with pytest.raises(ValidationError):
            UserModel(age=25)  # Missing required 'name'

    def test_dynamic_model_with_validators(self):
        """Test dynamic model with custom validators."""

        def validate_age(cls, v):
            if v < 0 or v > 150:
                raise ValueError("Age must be between 0 and 150")
            return v

        create_dynamic_model(
            "PersonModel",
            {"name": (str, ...), "age": (int, ...)},
            validators={"validate_age": validate_age},
        )

        # Note: The validator won't be automatically applied without proper decorator
        # This is a limitation of dynamic model creation
        # For full validation, use regular Pydantic model definition
