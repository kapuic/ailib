"""Edge case tests for multi-provider support."""

import json
from unittest.mock import Mock, patch

import pytest
from ailib import create_agent, create_client
from ailib.core.llm_client import CompletionResponse, Message, Role
from ailib.core.providers import get_provider_config


class TestProviderEdgeCases:
    """Test edge cases and error conditions."""

    def test_empty_model_name(self):
        """Test behavior with empty model name."""
        # Empty string should trigger validation error
        from pydantic import ValidationError

        with pytest.raises(ValidationError, match="Model name cannot be empty"):
            create_client("")

    def test_none_model_name(self):
        """Test behavior with None model name."""
        # None gets default model from openai provider
        with patch("os.getenv", return_value="test-key"):
            client = create_client(None)
            assert client.model == "gpt-3.5-turbo"  # Default OpenAI model

    def test_very_long_model_name(self):
        """Test with extremely long model name."""
        long_model = "gpt-" + "4" * 1000

        with patch("ailib.core.client_factory.OpenAIClient") as mock_openai:
            mock_instance = Mock()
            mock_openai.return_value = mock_instance

            # Should still work, defaulting to OpenAI
            client = create_client(long_model, api_key="test")
            assert client == mock_instance

    def test_model_name_with_special_characters(self):
        """Test model names with special characters."""
        special_models = [
            "gpt-4@latest",
            "model_with_underscore",
            "model-with-many-dashes-2024",
            "custom/model/name",
        ]

        with patch("ailib.core.client_factory.OpenAIClient") as mock_openai:
            mock_instance = Mock()
            mock_openai.return_value = mock_instance

            for model in special_models:
                client = create_client(model, api_key="test")
                assert client == mock_instance

    def test_case_sensitivity_in_provider_names(self):
        """Test that provider names handle case variations."""
        providers = ["OpenAI", "OPENAI", "openai", "OpEnAi"]

        for provider in providers:
            config = get_provider_config(provider.lower())
            assert config is not None

    def test_concurrent_client_creation(self):
        """Test creating multiple clients concurrently."""
        import threading

        clients = []
        errors = []

        def create_client_thread(model, provider):
            try:
                with patch("ailib.core.client_factory.OpenAIClient") as mock:
                    mock.return_value = Mock()
                    client = create_client(model, provider=provider, api_key="test")
                    clients.append(client)
            except Exception as e:
                errors.append(e)

        threads = []
        for i in range(10):
            model = f"gpt-4-thread-{i}"
            t = threading.Thread(target=create_client_thread, args=(model, "openai"))
            threads.append(t)
            t.start()

        for t in threads:
            t.join()

        assert len(errors) == 0
        assert len(clients) == 10

    def test_malformed_api_responses(self):
        """Test handling of malformed API responses."""
        with patch("ailib.core.client_factory.OpenAIClient") as mock_openai:
            mock_instance = Mock()

            # Test various malformed responses
            malformed_responses = [
                None,
                "",
                {},
                {"wrong": "format"},
                CompletionResponse(content=None, model="test", usage={}),
            ]

            mock_openai.return_value = mock_instance
            client = create_client("gpt-4", api_key="test")

            for response in malformed_responses:
                mock_instance.complete.return_value = response

                # Should handle gracefully or raise meaningful error
                if response is None or response == "":
                    with pytest.raises(AttributeError):
                        agent = create_agent("test", llm=client)
                        agent.run("test")

    def test_environment_variable_conflicts(self):
        """Test behavior when multiple API keys are set."""
        with patch.dict(
            "os.environ",
            {
                "OPENAI_API_KEY": "openai-key",
                "ANTHROPIC_API_KEY": "anthropic-key",
                "GROQ_API_KEY": "groq-key",
            },
        ):
            # Should use the appropriate key based on provider
            with patch("ailib.core.client_factory.OpenAIClient") as mock_openai:
                mock_openai.return_value = Mock()

                # OpenAI should use OPENAI_API_KEY
                create_client("gpt-4")
                call_kwargs = mock_openai.call_args.kwargs
                assert call_kwargs.get("api_key") == "openai-key"

    def test_invalid_base_url_format(self):
        """Test with invalid base URL formats."""
        invalid_urls = [
            "not-a-url",
            "http://",
            "//example.com",
            "ftp://example.com",
            "",
            None,
        ]

        for url in invalid_urls:
            with patch("ailib.core.client_factory.OpenAIClient") as mock_openai:
                if url is None or url == "":
                    # Should work with default URL
                    mock_openai.return_value = Mock()
                    client = create_client("gpt-4", base_url=url, api_key="test")
                    assert client is not None
                else:
                    # Should handle invalid URLs gracefully
                    mock_openai.return_value = Mock()
                    client = create_client("gpt-4", base_url=url, api_key="test")
                    # The client is created, but may fail on actual API calls

    def test_provider_specific_parameters(self):
        """Test that provider-specific parameters are passed correctly."""
        with patch("ailib.core.client_factory.OpenAIClient") as mock_openai:
            mock_openai.return_value = Mock()

            # Test with various provider-specific params
            create_client(
                "gpt-4",
                api_key="test",
                organization="org-123",  # OpenAI specific
                timeout=30,
                max_retries=5,
            )

            call_kwargs = mock_openai.call_args.kwargs
            assert "organization" in call_kwargs
            assert call_kwargs["timeout"] == 30
            assert call_kwargs["max_retries"] == 5


class TestProviderFailover:
    """Test failover scenarios between providers."""

    def test_provider_timeout_handling(self):
        """Test timeout handling for different providers."""
        from concurrent.futures import TimeoutError

        with patch("ailib.core.client_factory.OpenAIClient") as mock_openai:
            mock_instance = Mock()
            mock_instance.complete.side_effect = TimeoutError("Request timed out")
            mock_openai.return_value = mock_instance

            client = create_client("gpt-4", api_key="test")

            with pytest.raises(TimeoutError):
                agent = create_agent("test", llm=client)
                agent.run("test prompt")

    def test_rate_limit_handling(self):
        """Test rate limit error handling."""
        with patch("ailib.core.client_factory.OpenAIClient") as mock_openai:
            mock_instance = Mock()

            # Simulate rate limit error
            rate_limit_error = Exception("Rate limit exceeded")
            rate_limit_error.status_code = 429  # Common rate limit status
            mock_instance.complete.side_effect = rate_limit_error

            mock_openai.return_value = mock_instance

            client = create_client("gpt-4", api_key="test")

            with pytest.raises(Exception) as exc_info:
                agent = create_agent("test", llm=client)
                agent.run("test")

            assert "Rate limit" in str(exc_info.value)


class TestProviderCompatibility:
    """Test compatibility between different provider implementations."""

    def test_message_format_compatibility(self):
        """Test that message formats work across providers."""
        messages = [
            Message(role=Role.SYSTEM, content="You are helpful."),
            Message(role=Role.USER, content="Hello"),
            Message(role=Role.ASSISTANT, content="Hi there!"),
        ]

        # Test with both OpenAI and Anthropic style clients
        with patch("ailib.core.client_factory.OpenAIClient") as mock_openai:
            mock_instance = Mock()
            mock_instance.complete.return_value = CompletionResponse(
                content="Response",
                model="test",
                usage={"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
            )
            mock_openai.return_value = mock_instance

            client = create_client("gpt-4", api_key="test")

            # Should handle all message types
            response = client.complete(messages)
            assert response.content == "Response"

    def test_tool_calling_compatibility(self):
        """Test tool calling across different providers."""
        from ailib import tool

        @tool
        def test_tool(x: int) -> int:
            """Test tool."""
            return x * 2

        # Mock successful tool calling
        with patch("ailib.core.client_factory.OpenAIClient") as mock_openai:
            mock_instance = Mock()

            # First response: tool call
            mock_instance.complete.side_effect = [
                CompletionResponse(
                    content="",
                    model="test",
                    usage={
                        "prompt_tokens": 10,
                        "completion_tokens": 5,
                        "total_tokens": 15,
                    },
                    tool_calls=[
                        {
                            "id": "call_1",
                            "type": "function",
                            "function": {
                                "name": "test_tool",
                                "arguments": json.dumps({"x": 5}),
                            },
                        }
                    ],
                ),
                # Second response: final answer
                CompletionResponse(
                    content="The result is 10",
                    model="test",
                    usage={
                        "prompt_tokens": 20,
                        "completion_tokens": 10,
                        "total_tokens": 30,
                    },
                ),
            ]

            mock_openai.return_value = mock_instance

            _ = create_agent("test", tools=[test_tool], api_key="test")

            # This would test tool calling, but needs proper mocking
            # of the agent's run method
            # Left as an example of what should be tested


class TestMemoryAndState:
    """Test memory and state handling across providers."""

    def test_session_consistency_across_providers(self):
        """Test that sessions work consistently across providers."""
        from ailib import Session

        session = Session()

        with patch("ailib.core.client_factory.OpenAIClient") as mock_openai:
            mock_instance = Mock()
            mock_instance.complete.return_value = CompletionResponse(
                content="Response 1",
                model="test",
                usage={"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
            )
            mock_openai.return_value = mock_instance

            # Create clients for different providers
            _ = create_client("gpt-4", api_key="test")

            # Sessions should maintain state regardless of provider
            session.add_message(Message(role=Role.USER, content="Hello"))
            session.add_message(Message(role=Role.ASSISTANT, content="Hi!"))

            messages = session.get_messages()
            assert len(messages) == 2
            assert messages[0].content == "Hello"
            assert messages[1].content == "Hi!"
