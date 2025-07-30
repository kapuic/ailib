"""Validation utilities using Pydantic for AILib."""

from collections.abc import Callable
from typing import Any, Literal

from pydantic import BaseModel, Field, create_model, field_validator, model_validator


class PromptTemplateConfig(BaseModel):
    """Configuration for prompt templates."""

    template: str = Field(..., description="The template string with placeholders")
    input_variables: list[str] = Field(
        default_factory=list, description="List of required input variables"
    )
    prefix: str = Field(default="", description="Optional prefix for the prompt")
    suffix: str = Field(default="", description="Optional suffix for the prompt")

    @field_validator("template")
    @classmethod
    def validate_template(cls, v: str) -> str:
        """Validate that template is not empty."""
        if not v.strip():
            raise ValueError("Template cannot be empty")
        return v

    @model_validator(mode="after")
    def validate_placeholders(self) -> "PromptTemplateConfig":
        """Validate that all input variables exist in template."""
        import re

        # Find all placeholders in template
        placeholders = set(re.findall(r"\{(\w+)\}", self.template))

        # Check if all input variables are in template
        missing = set(self.input_variables) - placeholders
        if missing:
            raise ValueError(
                f"Input variables {missing} not found in template placeholders"
            )

        return self


class MessageConfig(BaseModel):
    """Configuration for chat messages."""

    role: Literal["system", "user", "assistant", "function", "tool"] = Field(
        ..., description="The role of the message sender"
    )
    content: str | None = Field(None, description="The content of the message")
    name: str | None = Field(None, description="Optional name of the sender")
    function_call: dict[str, Any] | None = Field(
        None, description="Function call details"
    )
    tool_calls: list[dict[str, Any]] | None = Field(
        None, description="Tool call details"
    )

    @model_validator(mode="after")
    def validate_content_or_function(self) -> "MessageConfig":
        """Validate that message has either content or function/tool calls."""
        if not self.content and not self.function_call and not self.tool_calls:
            raise ValueError(
                "Message must have either content, function_call, or tool_calls"
            )
        return self


class LLMConfig(BaseModel):
    """Configuration for LLM clients."""

    model: str = Field(..., description="Model identifier")
    temperature: float = Field(
        default=0.7, ge=0.0, le=2.0, description="Sampling temperature"
    )
    max_tokens: int | None = Field(
        default=None, gt=0, description="Maximum tokens to generate"
    )
    top_p: float = Field(
        default=1.0, ge=0.0, le=1.0, description="Top-p sampling parameter"
    )
    frequency_penalty: float = Field(
        default=0.0, ge=-2.0, le=2.0, description="Frequency penalty"
    )
    presence_penalty: float = Field(
        default=0.0, ge=-2.0, le=2.0, description="Presence penalty"
    )
    stop: list[str] | None = Field(default=None, description="Stop sequences")
    n: int = Field(default=1, ge=1, description="Number of completions")
    stream: bool = Field(default=False, description="Whether to stream responses")
    logit_bias: dict[str, float] | None = Field(
        default=None, description="Token logit biases"
    )
    user: str | None = Field(default=None, description="User identifier")

    @field_validator("model")
    @classmethod
    def validate_model(cls, v: str) -> str:
        """Validate model name is not empty."""
        if not v.strip():
            raise ValueError("Model name cannot be empty")
        return v


class ChainConfig(BaseModel):
    """Configuration for chains."""

    name: str = Field(..., description="Name of the chain")
    description: str = Field(default="", description="Description of the chain")
    max_iterations: int = Field(
        default=10, gt=0, description="Maximum iterations for loops"
    )
    early_stopping: bool = Field(
        default=True, description="Whether to stop early on errors"
    )
    retry_attempts: int = Field(
        default=3, ge=0, description="Number of retry attempts on failure"
    )
    retry_delay: float = Field(
        default=1.0, ge=0.0, description="Delay between retries in seconds"
    )
    timeout: float | None = Field(
        default=None, gt=0, description="Overall timeout in seconds"
    )
    verbose: bool = Field(default=False, description="Whether to log verbose output")

    @field_validator("name")
    @classmethod
    def validate_name(cls, v: str) -> str:
        """Validate chain name."""
        if not v.strip():
            raise ValueError("Chain name cannot be empty")
        return v


class AgentConfig(BaseModel):
    """Configuration for agents."""

    name: str = Field(..., description="Name of the agent")
    description: str = Field(default="", description="Description of the agent")
    model: str = Field(..., description="LLM model to use")
    system_prompt: str | None = Field(
        default=None, description="System prompt for the agent"
    )
    tools: list[str] = Field(
        default_factory=list, description="List of tool names available to agent"
    )
    max_iterations: int = Field(
        default=10, gt=0, description="Maximum reasoning iterations"
    )
    temperature: float = Field(
        default=0.7, ge=0.0, le=2.0, description="Sampling temperature"
    )
    verbose: bool = Field(default=False, description="Whether to log verbose output")
    memory_size: int = Field(
        default=10, ge=0, description="Number of messages to keep in memory"
    )
    return_intermediate_steps: bool = Field(
        default=False, description="Whether to return intermediate reasoning steps"
    )

    @field_validator("name")
    @classmethod
    def validate_name(cls, v: str) -> str:
        """Validate agent name."""
        if not v.strip():
            raise ValueError("Agent name cannot be empty")
        return v

    @field_validator("model")
    @classmethod
    def validate_model(cls, v: str) -> str:
        """Validate model name."""
        if not v.strip():
            raise ValueError("Model name cannot be empty")
        return v


class ToolParameterSchema(BaseModel):
    """Schema for tool parameters."""

    name: str = Field(..., description="Parameter name")
    type: str = Field(..., description="Parameter type (str, int, float, bool, etc)")
    description: str = Field(default="", description="Parameter description")
    required: bool = Field(default=True, description="Whether parameter is required")
    default: Any = Field(default=None, description="Default value if not required")
    enum: list[Any] | None = Field(
        default=None, description="Allowed values for parameter"
    )
    min_value: float | None = Field(
        default=None, description="Minimum value for numeric parameters"
    )
    max_value: float | None = Field(
        default=None, description="Maximum value for numeric parameters"
    )
    min_length: int | None = Field(
        default=None, description="Minimum length for string parameters"
    )
    max_length: int | None = Field(
        default=None, description="Maximum length for string parameters"
    )

    @model_validator(mode="after")
    def validate_constraints(self) -> "ToolParameterSchema":
        """Validate parameter constraints."""
        if self.type not in ["str", "int", "float", "bool", "list", "dict", "Any"]:
            raise ValueError(f"Invalid parameter type: {self.type}")

        if self.min_value is not None and self.type not in ["int", "float"]:
            raise ValueError("min_value only applies to numeric types")

        if self.max_value is not None and self.type not in ["int", "float"]:
            raise ValueError("max_value only applies to numeric types")

        if self.min_length is not None and self.type != "str":
            raise ValueError("min_length only applies to string type")

        if self.max_length is not None and self.type != "str":
            raise ValueError("max_length only applies to string type")

        if self.min_value is not None and self.max_value is not None:
            if self.min_value > self.max_value:
                raise ValueError("min_value cannot be greater than max_value")

        if self.min_length is not None and self.max_length is not None:
            if self.min_length > self.max_length:
                raise ValueError("min_length cannot be greater than max_length")

        if not self.required and self.default is None and self.type != "Any":
            raise ValueError("Non-required parameters must have a default value")

        return self


class ToolConfig(BaseModel):
    """Configuration for tools."""

    name: str = Field(..., description="Tool name")
    description: str = Field(..., description="Tool description")
    parameters: list[ToolParameterSchema] = Field(
        default_factory=list, description="Tool parameters"
    )
    returns: str = Field(default="Any", description="Return type")
    examples: list[dict[str, Any]] = Field(
        default_factory=list, description="Example usages"
    )
    tags: list[str] = Field(default_factory=list, description="Tool tags/categories")

    @field_validator("name")
    @classmethod
    def validate_name(cls, v: str) -> str:
        """Validate tool name."""
        if not v.strip():
            raise ValueError("Tool name cannot be empty")
        # Check for valid Python identifier
        if not v.replace("_", "").isalnum():
            raise ValueError("Tool name must be a valid Python identifier")
        return v

    @field_validator("description")
    @classmethod
    def validate_description(cls, v: str) -> str:
        """Validate tool description."""
        if not v.strip():
            raise ValueError("Tool description cannot be empty")
        return v


class SessionConfig(BaseModel):
    """Configuration for sessions."""

    session_id: str | None = Field(default=None, description="Session identifier")
    max_messages: int = Field(
        default=100, gt=0, description="Maximum messages to store"
    )
    ttl: int | None = Field(default=None, gt=0, description="Time to live in seconds")
    metadata: dict[str, Any] = Field(
        default_factory=dict, description="Session metadata"
    )
    auto_save: bool = Field(default=False, description="Whether to auto-save session")
    save_path: str | None = Field(default=None, description="Path to save session data")

    @model_validator(mode="after")
    def validate_auto_save(self) -> "SessionConfig":
        """Validate auto-save configuration."""
        if self.auto_save and not self.save_path:
            raise ValueError("save_path must be provided when auto_save is True")
        return self


class SafetyConfig(BaseModel):
    """Configuration for safety and moderation."""

    enabled: bool = Field(default=True, description="Whether safety checks are enabled")
    block_harmful_content: bool = Field(
        default=True, description="Block potentially harmful content"
    )
    sensitive_topics: list[str] = Field(
        default_factory=list, description="List of sensitive topics to monitor"
    )
    max_output_length: int = Field(
        default=4000, gt=0, description="Maximum output length"
    )
    allowed_domains: list[str] | None = Field(
        default=None, description="Allowed domains for web access"
    )
    blocked_domains: list[str] = Field(
        default_factory=list, description="Blocked domains"
    )
    rate_limit: int | None = Field(
        default=None, gt=0, description="Rate limit per minute"
    )
    custom_filters: list[str] = Field(
        default_factory=list, description="Custom regex filters"
    )


def create_dynamic_model(
    name: str,
    fields: dict[str, tuple[type, Any]],
    validators: dict[str, Callable] | None = None,
) -> type[BaseModel]:
    """Create a dynamic Pydantic model.

    Args:
        name: Model name
        fields: Dictionary of field names to (type, default) tuples
        validators: Optional dictionary of validators

    Returns:
        Dynamic Pydantic model class
    """
    field_definitions = {}
    for field_name, (field_type, default_value) in fields.items():
        if default_value is ...:
            field_definitions[field_name] = (field_type, Field(...))
        else:
            field_definitions[field_name] = (field_type, Field(default=default_value))

    # Create the model
    model = create_model(name, **field_definitions)

    # Add validators if provided
    if validators:
        for validator_name, validator_func in validators.items():
            setattr(model, validator_name, validator_func)

    return model
