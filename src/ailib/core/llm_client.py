"""Abstract base class for LLM clients.

This module defines the interface that all LLM clients must implement.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Any, AsyncIterator, Dict, Iterator, List, Optional, Union


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
    name: Optional[str] = None
    tool_calls: Optional[List[Dict[str, Any]]] = None
    tool_call_id: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
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
    usage: Dict[str, int]
    finish_reason: Optional[str] = None
    tool_calls: Optional[List[Dict[str, Any]]] = None


class LLMClient(ABC):
    """Abstract base class for LLM clients."""

    def __init__(self, model: str, **kwargs):
        """Initialize the LLM client.
        
        Args:
            model: Model name/identifier
            **kwargs: Additional client-specific parameters
        """
        self.model = model
        self.config = kwargs

    @abstractmethod
    def complete(
        self,
        messages: List[Message],
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
        tool_choice: Optional[Union[str, Dict[str, Any]]] = None,
        **kwargs
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
        pass

    @abstractmethod
    async def acomplete(
        self,
        messages: List[Message],
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
        tool_choice: Optional[Union[str, Dict[str, Any]]] = None,
        **kwargs
    ) -> CompletionResponse:
        """Async version of complete."""
        pass

    @abstractmethod
    def stream(
        self,
        messages: List[Message],
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
        tool_choice: Optional[Union[str, Dict[str, Any]]] = None,
        **kwargs
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
        messages: List[Message],
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
        tool_choice: Optional[Union[str, Dict[str, Any]]] = None,
        **kwargs
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

    def validate_messages(self, messages: List[Message]) -> None:
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