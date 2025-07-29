"""OpenAI client implementation."""

import os
from collections.abc import AsyncIterator, Iterator
from typing import Any

import tiktoken
from openai import AsyncOpenAI, OpenAI

from .llm_client import CompletionResponse, LLMClient, Message


class OpenAIClient(LLMClient):
    """OpenAI API client implementation."""

    def __init__(
        self,
        model: str = "gpt-4",
        api_key: str | None = None,
        organization: str | None = None,
        base_url: str | None = None,
        **kwargs,
    ):
        """Initialize OpenAI client.

        Args:
            model: Model name (e.g., 'gpt-4', 'gpt-3.5-turbo')
            api_key: OpenAI API key (defaults to OPENAI_API_KEY env var)
            organization: OpenAI organization ID
            base_url: Custom API base URL
            **kwargs: Additional client parameters
        """
        super().__init__(model, **kwargs)

        # Initialize clients
        self.client = OpenAI(
            api_key=api_key or os.getenv("OPENAI_API_KEY"),
            organization=organization,
            base_url=base_url,
        )
        self.async_client = AsyncOpenAI(
            api_key=api_key or os.getenv("OPENAI_API_KEY"),
            organization=organization,
            base_url=base_url,
        )

        # Initialize tokenizer for accurate token counting
        try:
            self._encoding = tiktoken.encoding_for_model(model)
        except KeyError:
            # Fallback for unknown models
            self._encoding = tiktoken.get_encoding("cl100k_base")

    def complete(
        self,
        messages: list[Message],
        temperature: float = 0.7,
        max_tokens: int | None = None,
        tools: list[dict[str, Any]] | None = None,
        tool_choice: str | dict[str, Any] | None = None,
        **kwargs,
    ) -> CompletionResponse:
        """Generate a completion using OpenAI API.

        Args:
            messages: Conversation messages
            temperature: Sampling temperature (0-2)
            max_tokens: Maximum tokens to generate
            tools: Available function tools
            tool_choice: Tool selection strategy
            **kwargs: Additional OpenAI-specific parameters

        Returns:
            CompletionResponse with generated content
        """
        self.validate_messages(messages)

        # Convert messages to OpenAI format
        openai_messages = [msg.to_dict() for msg in messages]

        # Prepare request parameters
        params = {
            "model": self.model,
            "messages": openai_messages,
            "temperature": temperature,
            **kwargs,
        }

        if max_tokens:
            params["max_tokens"] = max_tokens
        if tools:
            params["tools"] = tools
        if tool_choice:
            params["tool_choice"] = tool_choice

        # Make API call
        response = self.client.chat.completions.create(**params)

        # Extract response data
        choice = response.choices[0]
        message = choice.message

        return CompletionResponse(
            content=message.content or "",
            model=response.model,
            usage={
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
                "total_tokens": response.usage.total_tokens,
            },
            finish_reason=choice.finish_reason,
            tool_calls=(
                [tc.model_dump() for tc in message.tool_calls]
                if message.tool_calls
                else None
            ),
        )

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
        self.validate_messages(messages)

        # Convert messages to OpenAI format
        openai_messages = [msg.to_dict() for msg in messages]

        # Prepare request parameters
        params = {
            "model": self.model,
            "messages": openai_messages,
            "temperature": temperature,
            **kwargs,
        }

        if max_tokens:
            params["max_tokens"] = max_tokens
        if tools:
            params["tools"] = tools
        if tool_choice:
            params["tool_choice"] = tool_choice

        # Make async API call
        response = await self.async_client.chat.completions.create(**params)

        # Extract response data
        choice = response.choices[0]
        message = choice.message

        return CompletionResponse(
            content=message.content or "",
            model=response.model,
            usage={
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
                "total_tokens": response.usage.total_tokens,
            },
            finish_reason=choice.finish_reason,
            tool_calls=(
                [tc.model_dump() for tc in message.tool_calls]
                if message.tool_calls
                else None
            ),
        )

    def stream(
        self,
        messages: list[Message],
        temperature: float = 0.7,
        max_tokens: int | None = None,
        tools: list[dict[str, Any]] | None = None,
        tool_choice: str | dict[str, Any] | None = None,
        **kwargs,
    ) -> Iterator[str]:
        """Stream completions token by token."""
        self.validate_messages(messages)

        # Convert messages to OpenAI format
        openai_messages = [msg.to_dict() for msg in messages]

        # Prepare request parameters
        params = {
            "model": self.model,
            "messages": openai_messages,
            "temperature": temperature,
            "stream": True,
            **kwargs,
        }

        if max_tokens:
            params["max_tokens"] = max_tokens
        if tools:
            params["tools"] = tools
        if tool_choice:
            params["tool_choice"] = tool_choice

        # Stream response
        stream = self.client.chat.completions.create(**params)

        for chunk in stream:
            if chunk.choices and chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content

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
        self.validate_messages(messages)

        # Convert messages to OpenAI format
        openai_messages = [msg.to_dict() for msg in messages]

        # Prepare request parameters
        params = {
            "model": self.model,
            "messages": openai_messages,
            "temperature": temperature,
            "stream": True,
            **kwargs,
        }

        if max_tokens:
            params["max_tokens"] = max_tokens
        if tools:
            params["tools"] = tools
        if tool_choice:
            params["tool_choice"] = tool_choice

        # Async stream response
        stream = await self.async_client.chat.completions.create(**params)

        async for chunk in stream:
            if chunk.choices and chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content

    def count_tokens(self, text: str) -> int:
        """Count tokens using tiktoken for accurate OpenAI token counting.

        Args:
            text: Text to count tokens for

        Returns:
            Number of tokens
        """
        return len(self._encoding.encode(text))
