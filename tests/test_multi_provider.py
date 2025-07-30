"""Comprehensive tests for multi-provider support."""

from unittest.mock import Mock, patch

import pytest
from ailib import create_agent, create_chain, create_client, list_providers
from ailib.core import detect_provider
from ailib.core.llm_client import CompletionResponse, Message, Role
from ailib.core.providers import get_provider_config, is_openai_compatible


class TestProviderDetection:
    """Test automatic provider detection from model names."""

    @pytest.mark.parametrize(
        "model,expected_provider",
        [
            # OpenAI models
            ("gpt-4", "openai"),
            ("gpt-4-turbo", "openai"),
            ("gpt-3.5-turbo", "openai"),
            ("gpt-4o", "openai"),
            # Anthropic models
            ("claude-3-opus-20240229", "anthropic"),
            ("claude-3-sonnet-20240229", "anthropic"),
            ("claude-3-haiku-20240307", "anthropic"),
            ("claude-2.1", "anthropic"),
            # Groq models
            ("mixtral-8x7b-32768", "groq"),
            ("llama2-70b-4096", "groq"),
            # Perplexity models
            ("pplx-7b-online", "perplexity"),
            ("pplx-70b-online", "perplexity"),
            # DeepSeek models
            ("deepseek-chat", "deepseek"),
            ("deepseek-coder", "deepseek"),
            # Moonshot models
            ("moonshot-v1-8k", "moonshot"),
            ("moonshot-v1-32k", "moonshot"),
            # Local models
            ("llama2", "ollama"),
            ("mistral", "ollama"),
            ("codellama", "ollama"),
            # Unknown models
            ("unknown-model-xyz", "openai"),  # Default fallback
        ],
    )
    def test_detect_provider_from_model(self, model, expected_provider):
        """Test provider detection for various model names."""
        assert detect_provider(model) == expected_provider


class TestProviderConfiguration:
    """Test provider configuration and settings."""

    def test_all_providers_have_config(self):
        """Ensure all providers have valid configuration."""
        providers = list_providers()

        for provider_name in providers:
            if provider_name in ["openai", "anthropic"]:  # Built-in providers
                config = get_provider_config(provider_name)
                assert config is not None
                assert config.name == provider_name
                assert hasattr(config, "api_key_env")
                assert hasattr(config, "default_model")

    def test_openai_compatible_providers(self):
        """Test OpenAI-compatible provider detection."""
        # OpenAI-compatible providers
        assert is_openai_compatible("groq")
        assert is_openai_compatible("together")
        assert is_openai_compatible("perplexity")
        assert is_openai_compatible("deepseek")
        assert is_openai_compatible("moonshot")
        assert is_openai_compatible("ollama")

        # Non-OpenAI-compatible providers
        assert not is_openai_compatible("anthropic")
        assert not is_openai_compatible("openai")  # OpenAI itself is not "compatible"

    def test_provider_base_urls(self):
        """Test that OpenAI-compatible providers have base URLs."""
        from ailib.core.providers import PROVIDERS

        openai_compatible = []
        for name, config in PROVIDERS.items():
            if name != "openai" and not config.requires_separate_client:
                openai_compatible.append(name)

        assert len(openai_compatible) > 0  # Make sure we have some to test

        for provider in openai_compatible:
            config = get_provider_config(provider)
            assert config is not None
            assert config.base_url is not None
            assert config.base_url.startswith("http")


class TestClientCreation:
    """Test client creation with different providers."""

    @patch("ailib.core.client_factory.OpenAIClient")
    def test_create_openai_client(self, mock_openai_class):
        """Test creating OpenAI client."""
        mock_instance = Mock()
        mock_openai_class.return_value = mock_instance

        client = create_client("gpt-4", api_key="test-key")

        mock_openai_class.assert_called_once()
        assert client == mock_instance

    @patch("ailib.core.client_factory.AnthropicClient")
    @patch("ailib.core.client_factory.HAS_ANTHROPIC", True)
    def test_create_anthropic_client(self, mock_anthropic_class):
        """Test creating Anthropic client."""
        mock_instance = Mock()
        mock_anthropic_class.return_value = mock_instance

        client = create_client("claude-3-opus-20240229", api_key="test-key")

        mock_anthropic_class.assert_called_once()
        assert client == mock_instance

    @patch("ailib.core.client_factory.HAS_ANTHROPIC", False)
    def test_anthropic_not_installed(self):
        """Test error when Anthropic is not installed."""
        with pytest.raises(ValueError, match="anthropic.*package"):
            create_client("claude-3-opus-20240229")

    @patch("ailib.core.client_factory.OpenAIClient")
    def test_create_openai_compatible_client(self, mock_openai_class):
        """Test creating clients for OpenAI-compatible providers."""
        mock_instance = Mock()
        mock_openai_class.return_value = mock_instance

        # Test Groq
        _ = create_client("mixtral-8x7b-32768", api_key="test-key")
        mock_openai_class.assert_called()
        call_kwargs = mock_openai_class.call_args.kwargs
        assert "base_url" in call_kwargs
        assert "groq.com" in call_kwargs["base_url"]

        # Test Ollama (no API key needed)
        mock_openai_class.reset_mock()
        _ = create_client("llama2")
        mock_openai_class.assert_called()
        call_kwargs = mock_openai_class.call_args.kwargs
        assert "base_url" in call_kwargs
        assert "localhost:11434" in call_kwargs["base_url"]

    def test_explicit_provider_override(self):
        """Test explicit provider overrides auto-detection."""
        with patch("ailib.core.client_factory.OpenAIClient") as mock_openai:
            # Model suggests Anthropic, but we explicitly set OpenAI
            create_client(
                model="claude-3-opus-20240229", provider="openai", api_key="test-key"
            )
            mock_openai.assert_called_once()


class TestFactoryFunctions:
    """Test factory functions with multi-provider support."""

    def test_create_agent_with_provider(self):
        """Test create_agent with explicit provider."""
        with patch("ailib.core.client_factory.OpenAIClient") as mock_openai_class:
            mock_llm = Mock()
            mock_llm.model = "gpt-4"
            mock_openai_class.return_value = mock_llm

            agent = create_agent(
                "test-agent", model="gpt-4", provider="openai", api_key="test-key"
            )

            assert agent.llm == mock_llm
            assert agent.name == "test-agent"

    def test_create_chain_with_provider(self):
        """Test create_chain with explicit provider."""
        with patch("ailib.core.client_factory.AnthropicClient") as mock_anthropic_class:
            with patch("ailib.core.client_factory.HAS_ANTHROPIC", True):
                mock_llm = Mock()
                mock_llm.model = "claude-3-opus-20240229"
                mock_llm.complete.return_value = CompletionResponse(
                    content="Test response",
                    model="claude-3-opus-20240229",
                    usage={
                        "prompt_tokens": 10,
                        "completion_tokens": 5,
                        "total_tokens": 15,
                    },
                )
                mock_anthropic_class.return_value = mock_llm

                chain = create_chain(
                    "Test prompt",
                    model="claude-3-opus-20240229",
                    provider="anthropic",
                    api_key="test-key",
                )

                result = chain.run()
                assert result == "Test response"


class TestErrorHandling:
    """Test error handling across providers."""

    def test_missing_api_key_error(self):
        """Test appropriate error when API key is missing."""
        with patch.dict("os.environ", {}, clear=True):
            with pytest.raises(Exception) as exc_info:
                create_client("gpt-4")

            # Should mention API key in error
            assert (
                "api_key" in str(exc_info.value).lower()
                or "api key" in str(exc_info.value).lower()
            )

    def test_invalid_provider_error(self):
        """Test error with invalid provider name."""
        with pytest.raises(ValueError, match="Unknown provider"):
            create_client("gpt-4", provider="invalid-provider")

    @patch("ailib.core.client_factory.OpenAIClient")
    def test_network_error_handling(self, mock_openai_class):
        """Test handling of network errors."""
        mock_instance = Mock()
        mock_instance.complete.side_effect = Exception("Network error")
        mock_openai_class.return_value = mock_instance

        client = create_client("gpt-4", api_key="test-key")

        with pytest.raises(Exception, match="Network error"):
            client.complete([Message(role=Role.USER, content="Test")])


class TestProviderSpecificFeatures:
    """Test provider-specific features and differences."""

    @patch("ailib.core.client_factory.AnthropicClient")
    @patch("ailib.core.client_factory.HAS_ANTHROPIC", True)
    def test_anthropic_system_message_handling(self, mock_anthropic_class):
        """Test that Anthropic handles system messages differently."""
        mock_instance = Mock()
        mock_anthropic_class.return_value = mock_instance

        client = create_client("claude-3-opus-20240229", api_key="test-key")

        # Anthropic should be created with the special client
        mock_anthropic_class.assert_called_once()
        assert client == mock_instance

    @patch("ailib.core.client_factory.OpenAIClient")
    def test_ollama_no_api_key_required(self, mock_openai_class):
        """Test that Ollama doesn't require an API key."""
        mock_instance = Mock()
        mock_openai_class.return_value = mock_instance

        # Should work without API key
        with patch.dict("os.environ", {}, clear=True):
            _ = create_client("llama2", provider="ollama")

            # Should be called without api_key requirement
            mock_openai_class.assert_called_once()
            call_kwargs = mock_openai_class.call_args.kwargs
            assert "localhost:11434" in call_kwargs.get("base_url", "")


class TestIntegrationScenarios:
    """Test real-world integration scenarios."""

    def test_switching_providers_in_session(self):
        """Test switching between providers in the same session."""
        # Mock responses that follow ReAct format
        openai_responses = [
            CompletionResponse(
                content=(
                    "Thought: I'll provide a response\n"
                    "Action: Final Answer\n"
                    "Action Input: OpenAI response"
                ),
                model="gpt-4",
                usage={"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
            )
        ]

        claude_responses = [
            CompletionResponse(
                content=(
                    "Thought: I'll provide a response\n"
                    "Action: Final Answer\n"
                    "Action Input: Claude response"
                ),
                model="claude-3-opus-20240229",
                usage={"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
            )
        ]

        with patch("ailib.core.client_factory.OpenAIClient") as mock_openai:
            with patch("ailib.core.client_factory.AnthropicClient") as mock_anthropic:
                with patch("ailib.core.client_factory.HAS_ANTHROPIC", True):
                    # Setup mocks
                    mock_openai_instance = Mock()
                    mock_openai_instance.complete.side_effect = openai_responses
                    mock_openai.return_value = mock_openai_instance

                    mock_anthropic_instance = Mock()
                    mock_anthropic_instance.complete.side_effect = claude_responses
                    mock_anthropic.return_value = mock_anthropic_instance

                    # Test OpenAI
                    agent1 = create_agent("assistant", model="gpt-4", api_key="test")
                    result1 = agent1.run("Test")
                    assert "OpenAI" in result1

                    # Test Anthropic
                    agent2 = create_agent(
                        "assistant", model="claude-3-opus-20240229", api_key="test"
                    )
                    result2 = agent2.run("Test")
                    assert "Claude" in result2

    def test_fallback_behavior(self):
        """Test fallback behavior when preferred provider fails."""
        # This would test graceful degradation - left as example
        pass


class TestProviderList:
    """Test the list_providers functionality."""

    def test_list_providers_returns_dict(self):
        """Test that list_providers returns a dictionary."""
        providers = list_providers()
        assert isinstance(providers, dict)
        assert "openai" in providers
        assert "anthropic" in providers

    @patch("ailib.core.client_factory.HAS_ANTHROPIC", False)
    def test_list_providers_anthropic_not_installed(self):
        """Test provider list when optional dependencies are missing."""
        providers = list_providers()
        assert providers["openai"]["available"] is True  # Always available
        assert providers["anthropic"]["available"] is False  # Not installed
