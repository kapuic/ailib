"""Anthropic Claude client implementation."""

import os
from collections.abc import AsyncIterator, Iterator
from typing import Any

from anthropic import Anthropic, AsyncAnthropic

from .llm_client import CompletionResponse, LLMClient, Message, Role


class AnthropicClient(LLMClient):
    """Anthropic Claude API client implementation."""

    def __init__(
        self,
        model: str = "claude-3-opus-20240229",
        api_key: str | None = None,
        base_url: str | None = None,
        **kwargs,
    ):
        """Initialize Anthropic client.

        Args:
            model: Model name (e.g., 'claude-3-opus-20240229')
            api_key: Anthropic API key (defaults to ANTHROPIC_API_KEY env var)
            base_url: Custom API base URL
            **kwargs: Additional client parameters
        """
        super().__init__(model, **kwargs)

        # Initialize clients
        self.client = Anthropic(
            api_key=api_key or os.getenv("ANTHROPIC_API_KEY"),
            base_url=base_url,
        )
        self.async_client = AsyncAnthropic(
            api_key=api_key or os.getenv("ANTHROPIC_API_KEY"),
            base_url=base_url,
        )

    def _convert_messages(
        self, messages: list[Message]
    ) -> tuple[str | None, list[dict]]:
        """Convert messages to Anthropic format.

        Anthropic requires system messages to be separate from the conversation.

        Returns:
            Tuple of (system_prompt, messages)
        """
        system_prompt = None
        anthropic_messages = []

        for msg in messages:
            if msg.role == Role.SYSTEM:
                # Anthropic expects only one system message at the start
                if system_prompt is None:
                    system_prompt = msg.content
                else:
                    # Append additional system messages to the first one
                    system_prompt += "\n\n" + msg.content
            else:
                # Convert role to Anthropic format
                role = "user" if msg.role == Role.USER else "assistant"
                anthropic_messages.append({"role": role, "content": msg.content})

        return system_prompt, anthropic_messages

    def _convert_tools(
        self, tools: list[dict[str, Any]] | None
    ) -> list[dict[str, Any]] | None:
        """Convert OpenAI tool format to Anthropic format."""
        if not tools:
            return None

        anthropic_tools = []
        for tool in tools:
            if tool["type"] == "function":
                func = tool["function"]
                anthropic_tools.append(
                    {
                        "name": func["name"],
                        "description": func.get("description", ""),
                        "input_schema": func.get("parameters", {}),
                    }
                )

        return anthropic_tools if anthropic_tools else None

    def _complete_impl(
        self,
        messages: list[Message],
        temperature: float = 0.7,
        max_tokens: int | None = None,
        tools: list[dict[str, Any]] | None = None,
        tool_choice: str | dict[str, Any] | None = None,
        **kwargs,
    ) -> CompletionResponse:
        """Generate a completion using Anthropic API.

        Args:
            messages: Conversation messages
            temperature: Sampling temperature (0-1)
            max_tokens: Maximum tokens to generate
            tools: Available function tools
            tool_choice: Tool selection strategy
            **kwargs: Additional Anthropic-specific parameters

        Returns:
            CompletionResponse with generated content
        """
        self.validate_messages(messages)

        # Convert messages to Anthropic format
        system_prompt, anthropic_messages = self._convert_messages(messages)

        # Prepare request parameters
        params = {
            "model": self.model,
            "messages": anthropic_messages,
            "temperature": temperature,
            "max_tokens": max_tokens or 4096,  # Anthropic requires max_tokens
            **kwargs,
        }

        if system_prompt:
            params["system"] = system_prompt

        # Convert and add tools if provided
        anthropic_tools = self._convert_tools(tools)
        if anthropic_tools:
            params["tools"] = anthropic_tools

            # Handle tool_choice
            if tool_choice:
                if isinstance(tool_choice, str):
                    if tool_choice == "auto":
                        params["tool_choice"] = {"type": "auto"}
                    elif tool_choice == "none":
                        params["tool_choice"] = {"type": "none"}
                else:
                    # Specific tool choice
                    params["tool_choice"] = tool_choice

        # Make API call
        response = self.client.messages.create(**params)

        # Extract response data and convert tool calls if present
        tool_calls = None
        if hasattr(response.content[0], "name"):  # Tool use response
            tool_calls = []
            for content in response.content:
                if hasattr(content, "name"):
                    tool_calls.append(
                        {
                            "id": content.id,
                            "type": "function",
                            "function": {
                                "name": content.name,
                                "arguments": str(content.input),
                            },
                        }
                    )

        # Get text content
        text_content = ""
        for content in response.content:
            if hasattr(content, "text"):
                text_content = content.text
                break

        return CompletionResponse(
            content=text_content,
            model=response.model,
            usage={
                "prompt_tokens": response.usage.input_tokens,
                "completion_tokens": response.usage.output_tokens,
                "total_tokens": response.usage.input_tokens
                + response.usage.output_tokens,
            },
            finish_reason=response.stop_reason,
            tool_calls=tool_calls,
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

        # Convert messages to Anthropic format
        system_prompt, anthropic_messages = self._convert_messages(messages)

        # Prepare request parameters
        params = {
            "model": self.model,
            "messages": anthropic_messages,
            "temperature": temperature,
            "max_tokens": max_tokens or 4096,
            **kwargs,
        }

        if system_prompt:
            params["system"] = system_prompt

        # Convert and add tools if provided
        anthropic_tools = self._convert_tools(tools)
        if anthropic_tools:
            params["tools"] = anthropic_tools

            if tool_choice:
                if isinstance(tool_choice, str):
                    if tool_choice == "auto":
                        params["tool_choice"] = {"type": "auto"}
                    elif tool_choice == "none":
                        params["tool_choice"] = {"type": "none"}
                else:
                    params["tool_choice"] = tool_choice

        # Make async API call
        response = await self.async_client.messages.create(**params)

        # Extract response data
        tool_calls = None
        if hasattr(response.content[0], "name"):
            tool_calls = []
            for content in response.content:
                if hasattr(content, "name"):
                    tool_calls.append(
                        {
                            "id": content.id,
                            "type": "function",
                            "function": {
                                "name": content.name,
                                "arguments": str(content.input),
                            },
                        }
                    )

        text_content = ""
        for content in response.content:
            if hasattr(content, "text"):
                text_content = content.text
                break

        return CompletionResponse(
            content=text_content,
            model=response.model,
            usage={
                "prompt_tokens": response.usage.input_tokens,
                "completion_tokens": response.usage.output_tokens,
                "total_tokens": response.usage.input_tokens
                + response.usage.output_tokens,
            },
            finish_reason=response.stop_reason,
            tool_calls=tool_calls,
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

        # Convert messages to Anthropic format
        system_prompt, anthropic_messages = self._convert_messages(messages)

        # Prepare request parameters
        params = {
            "model": self.model,
            "messages": anthropic_messages,
            "temperature": temperature,
            "max_tokens": max_tokens or 4096,
            "stream": True,
            **kwargs,
        }

        if system_prompt:
            params["system"] = system_prompt

        # Note: Anthropic streaming doesn't support tools yet
        if tools:
            raise NotImplementedError("Anthropic streaming doesn't support tools yet")

        # Stream response
        with self.client.messages.stream(**params) as stream:
            yield from stream.text_stream

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

        # Convert messages to Anthropic format
        system_prompt, anthropic_messages = self._convert_messages(messages)

        # Prepare request parameters
        params = {
            "model": self.model,
            "messages": anthropic_messages,
            "temperature": temperature,
            "max_tokens": max_tokens or 4096,
            "stream": True,
            **kwargs,
        }

        if system_prompt:
            params["system"] = system_prompt

        if tools:
            raise NotImplementedError("Anthropic streaming doesn't support tools yet")

        # Async stream response
        async with self.async_client.messages.stream(**params) as stream:
            async for text in stream.text_stream:
                yield text  # noqa: UP028

    def count_tokens(self, text: str) -> int:
        """Count tokens in text.

        Note: This is an approximation as Anthropic doesn't provide a public tokenizer.
        """
        # Rough approximation for Claude models
        # Claude typically uses ~1.2 tokens per word
        return int(len(text.split()) * 1.2)
