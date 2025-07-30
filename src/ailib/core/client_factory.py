"""Factory for creating appropriate LLM clients based on model or provider."""

import os
from typing import Any

from .llm_client import LLMClient
from .openai_client import OpenAIClient
from .providers import (
    detect_provider_from_model,
    get_provider_config,
    is_openai_compatible,
)

# Try to import optional providers
try:
    from .anthropic_client import AnthropicClient

    HAS_ANTHROPIC = True
except ImportError:
    HAS_ANTHROPIC = False
    AnthropicClient = None


def detect_provider(model: str) -> str:
    """Detect the provider based on the model name.

    Args:
        model: Model name

    Returns:
        Provider name (openai, anthropic, together, etc.)
    """
    return detect_provider_from_model(model)


def create_client(
    model: str | None = None,
    provider: str | None = None,
    api_key: str | None = None,
    base_url: str | None = None,
    **kwargs,
) -> LLMClient:
    """Create an appropriate LLM client based on model or provider.

    This function intelligently handles multiple providers:
    - OpenAI-compatible providers (Groq, etc.) use OpenAIClient with base_url
    - Providers with custom APIs (Anthropic) use their own clients
    - Automatic provider detection from model names

    Args:
        model: Model name (optional if provider has a default)
        provider: Explicit provider name (optional - auto-detected from model)
        api_key: API key (optional, defaults to provider's environment variable)
        base_url: Custom API endpoint (overrides provider's default)
        **kwargs: Additional client-specific parameters

    Returns:
        LLMClient instance

    Raises:
        ValueError: If provider is not supported or not installed

    Examples:
        # Auto-detect provider from model
        client = create_client("gpt-4")  # OpenAI
        client = create_client("claude-3-opus-20240229")  # Anthropic

        # Explicit provider
        client = create_client(provider="groq", model="mixtral-8x7b")
        client = create_client(provider="ollama", model="llama2")

        # Custom endpoint
        client = create_client(
            model="gpt-3.5-turbo",
            base_url="http://localhost:1234/v1"  # Local LM Studio
        )
    """
    # Handle provider detection
    if provider is None:
        if model:
            provider = detect_provider(model)
        else:
            provider = "openai"  # Default provider

    provider = provider.lower()

    # Get provider configuration
    config = get_provider_config(provider)
    if not config:
        raise ValueError(f"Unknown provider: {provider}")

    # Use default model if not specified
    if model is None:
        model = config.default_model
        if not model:
            raise ValueError(
                f"No model specified and provider '{provider}' has no default model"
            )

    # Determine base_url
    if base_url is None and config.base_url:
        base_url = config.base_url

    # Determine API key
    if api_key is None and config.api_key_env:
        api_key = os.getenv(config.api_key_env)

    # Create appropriate client
    if provider == "openai" or is_openai_compatible(provider):
        # Use OpenAI client for OpenAI and compatible providers
        return OpenAIClient(model=model, api_key=api_key, base_url=base_url, **kwargs)
    else:
        # Use provider-specific client
        if provider == "anthropic":
            if not HAS_ANTHROPIC:
                raise ValueError(
                    "Anthropic provider requires the 'anthropic' package. "
                    "Install it with: pip install ailib[anthropic]"
                )
            return AnthropicClient(model=model, api_key=api_key, **kwargs)
        else:
            raise ValueError(
                f"Provider '{provider}' requires a separate implementation "
                f"that is not yet available. Currently supported: openai, "
                f"anthropic, and OpenAI-compatible providers."
            )


def list_providers() -> dict[str, dict[str, Any]]:
    """List available providers and their configuration.

    Returns:
        Dictionary with provider information including availability and configuration
    """
    from .providers import PROVIDERS

    result = {}
    for name, config in PROVIDERS.items():
        if config.requires_separate_client:
            # Check if special client is available
            if name == "anthropic":
                available = HAS_ANTHROPIC
            else:
                available = False
        else:
            # OpenAI-compatible providers are always available
            available = True

        result[name] = {
            "available": available,
            "base_url": config.base_url,
            "default_model": config.default_model,
            "openai_compatible": not config.requires_separate_client,
        }

    return result


def get_provider_models(provider: str) -> list[str]:
    """Get commonly used models for a provider.

    Args:
        provider: Provider name

    Returns:
        List of model names
    """
    models = {
        "openai": [
            "gpt-4",
            "gpt-4-turbo",
            "gpt-4-turbo-preview",
            "gpt-3.5-turbo",
            "gpt-3.5-turbo-16k",
        ],
        "anthropic": [
            "claude-3-opus-20240229",
            "claude-3-sonnet-20240229",
            "claude-3-haiku-20240307",
            "claude-2.1",
            "claude-2.0",
        ],
        "ollama": [
            "llama2",
            "llama2:70b",
            "mistral",
            "phi",
            "gemma",
        ],
    }

    return models.get(provider.lower(), [])
