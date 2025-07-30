"""Abstract base class for LLM clients.

This module defines the interface that all LLM clients must implement.
"""

from abc import ABC, abstractmethod
from collections.abc import AsyncIterator, Iterator
from dataclasses import dataclass
from enum import Enum
from typing import Any

from .._validation import LLMConfig


class Role(Enum):
    """Message roles in a conversation."""

    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"


@dataclass
class Message:
    """Represents a message in a conversation."""

    role: Role
    content: str
    name: str | None = None
    tool_calls: list[dict[str, Any]] | None = None
    tool_call_id: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert message to dictionary format."""
        data = {"role": self.role.value, "content": self.content}
        if self.name:
            data["name"] = self.name
        if self.tool_calls:
            data["tool_calls"] = self.tool_calls
        if self.tool_call_id:
            data["tool_call_id"] = self.tool_call_id
        return data


@dataclass
class CompletionResponse:
    """Response from LLM completion."""

    content: str
    model: str
    usage: dict[str, int]
    finish_reason: str | None = None
    tool_calls: list[dict[str, Any]] | None = None


class LLMClient(ABC):
    """Abstract base class for LLM clients."""

    def __init__(self, model: str, **kwargs):
        """Initialize the LLM client.

        Args:
            model: Model name/identifier
            **kwargs: Additional client-specific parameters
        """
        # Validate configuration
        config = LLMConfig(model=model, **kwargs)
        self.model = config.model
        self.config = config.model_dump(exclude={"model"})

    def complete(
        self,
        messages: list[Message],
        temperature: float = 0.7,
        max_tokens: int | None = None,
        tools: list[dict[str, Any]] | None = None,
        tool_choice: str | dict[str, Any] | None = None,
        **kwargs,
    ) -> CompletionResponse:
        """Generate a completion for the given messages.

        Args:
            messages: List of messages in the conversation
            temperature: Sampling temperature (0-2)
            max_tokens: Maximum tokens to generate
            tools: Available tools for function calling
            tool_choice: Tool selection strategy
            **kwargs: Additional model-specific parameters

        Returns:
            CompletionResponse with generated content
        """
        # Apply safety hooks if available
        from ..safety.hooks import _internal_post_hook, _internal_pre_hook

        # Import tracing
        from ..tracing._core import trace_step

        # Pre-process the prompt
        if messages:
            last_message = messages[-1]
            if last_message.content:
                try:
                    checked_content = _internal_pre_hook(
                        last_message.content, user_id=kwargs.get("user_id")
                    )
                    # Update the content if modified
                    if checked_content != last_message.content:
                        messages = messages[:-1] + [
                            Message(role=last_message.role, content=checked_content)
                        ]
                except ValueError as e:
                    # Safety check failed
                    return CompletionResponse(
                        content=str(e),
                        model=self.model,
                        usage={
                            "prompt_tokens": 0,
                            "completion_tokens": 0,
                            "total_tokens": 0,
                        },
                        finish_reason="safety",
                    )

        # Trace the LLM call
        with trace_step(
            f"llm_{self.model}",
            step_type="llm_call",
            model=self.model,
            message_count=len(messages),
            temperature=temperature,
            max_tokens=max_tokens,
            has_tools=bool(tools),
        ) as step:
            try:
                # Call the actual implementation
                response = self._complete_impl(
                    messages, temperature, max_tokens, tools, tool_choice, **kwargs
                )

                # Add response metadata to trace
                step.metadata.update(
                    {
                        "response_tokens": response.usage.get("completion_tokens"),
                        "prompt_tokens": response.usage.get("prompt_tokens"),
                        "total_tokens": response.usage.get("total_tokens"),
                        "finish_reason": response.finish_reason,
                    }
                )

                # Post-process the response
                return _internal_post_hook(response, **kwargs)
            except Exception:
                # Trace will capture the error
                raise

    @abstractmethod
    def _complete_impl(
        self,
        messages: list[Message],
        temperature: float = 0.7,
        max_tokens: int | None = None,
        tools: list[dict[str, Any]] | None = None,
        tool_choice: str | dict[str, Any] | None = None,
        **kwargs,
    ) -> CompletionResponse:
        """Actual implementation of completion - to be overridden by subclasses."""
        pass

    @abstractmethod
    async def acomplete(
        self,
        messages: list[Message],
        temperature: float = 0.7,
        max_tokens: int | None = None,
        tools: list[dict[str, Any]] | None = None,
        tool_choice: str | dict[str, Any] | None = None,
        **kwargs,
    ) -> CompletionResponse:
        """Async version of complete."""
        pass

    @abstractmethod
    def stream(
        self,
        messages: list[Message],
        temperature: float = 0.7,
        max_tokens: int | None = None,
        tools: list[dict[str, Any]] | None = None,
        tool_choice: str | dict[str, Any] | None = None,
        **kwargs,
    ) -> Iterator[str]:
        """Stream completions token by token.

        Args:
            messages: List of messages in the conversation
            temperature: Sampling temperature (0-2)
            max_tokens: Maximum tokens to generate
            tools: Available tools for function calling
            tool_choice: Tool selection strategy
            **kwargs: Additional model-specific parameters

        Yields:
            Generated tokens as they become available
        """
        pass

    @abstractmethod
    async def astream(
        self,
        messages: list[Message],
        temperature: float = 0.7,
        max_tokens: int | None = None,
        tools: list[dict[str, Any]] | None = None,
        tool_choice: str | dict[str, Any] | None = None,
        **kwargs,
    ) -> AsyncIterator[str]:
        """Async version of stream."""
        pass

    def count_tokens(self, text: str) -> int:
        """Count tokens in text.

        Args:
            text: Text to count tokens for

        Returns:
            Number of tokens
        """
        # Basic estimation - subclasses should override with model-specific counting
        return len(text.split()) * 1.3  # Rough estimate

    def validate_messages(self, messages: list[Message]) -> None:
        """Validate message format.

        Args:
            messages: Messages to validate

        Raises:
            ValueError: If messages are invalid
        """
        if not messages:
            raise ValueError("Messages list cannot be empty")

        # Check for proper role sequence
        for i, msg in enumerate(messages):
            if not isinstance(msg, Message):
                raise ValueError(f"Message at index {i} must be a Message instance")
            if msg.role == Role.TOOL and not msg.tool_call_id:
                raise ValueError(f"Tool message at index {i} must have tool_call_id")
