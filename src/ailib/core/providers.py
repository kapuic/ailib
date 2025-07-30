"""Provider configurations and utilities for AILib.

This module handles the configuration for various LLM providers,
including OpenAI-compatible endpoints.
"""

from dataclasses import dataclass


@dataclass
class ProviderConfig:
    """Configuration for an LLM provider."""

    name: str
    base_url: str | None = None
    api_key_env: str | None = None
    default_model: str | None = None
    requires_separate_client: bool = False


# Provider configurations
PROVIDERS = {
    # OpenAI and compatible providers
    "openai": ProviderConfig(
        name="openai", api_key_env="OPENAI_API_KEY", default_model="gpt-3.5-turbo"
    ),
    "together": ProviderConfig(
        name="together",
        base_url="https://api.together.xyz/v1",
        api_key_env="TOGETHER_API_KEY",
        default_model="mistralai/Mixtral-8x7B-Instruct-v0.1",
    ),
    "anyscale": ProviderConfig(
        name="anyscale",
        base_url="https://api.endpoints.anyscale.com/v1",
        api_key_env="ANYSCALE_API_KEY",
        default_model="meta-llama/Llama-2-70b-chat-hf",
    ),
    "perplexity": ProviderConfig(
        name="perplexity",
        base_url="https://api.perplexity.ai",
        api_key_env="PERPLEXITY_API_KEY",
        default_model="llama-3-sonar-large-32k-online",
    ),
    "groq": ProviderConfig(
        name="groq",
        base_url="https://api.groq.com/openai/v1",
        api_key_env="GROQ_API_KEY",
        default_model="mixtral-8x7b-32768",
    ),
    "deepinfra": ProviderConfig(
        name="deepinfra",
        base_url="https://api.deepinfra.com/v1/openai",
        api_key_env="DEEPINFRA_API_KEY",
        default_model="meta-llama/Llama-2-70b-chat-hf",
    ),
    "fireworks": ProviderConfig(
        name="fireworks",
        base_url="https://api.fireworks.ai/inference/v1",
        api_key_env="FIREWORKS_API_KEY",
        default_model="accounts/fireworks/models/llama-v2-7b-chat",
    ),
    "deepseek": ProviderConfig(
        name="deepseek",
        base_url="https://api.deepseek.com/v1",
        api_key_env="DEEPSEEK_API_KEY",
        default_model="deepseek-chat",
    ),
    "moonshot": ProviderConfig(
        name="moonshot",
        base_url="https://api.moonshot.cn/v1",
        api_key_env="MOONSHOT_API_KEY",
        default_model="moonshot-v1-8k",
    ),
    # Local providers
    "local": ProviderConfig(
        name="local",
        base_url="http://localhost:1234/v1",
        api_key_env=None,  # Usually not required for local
        default_model="local-model",
    ),
    "lmstudio": ProviderConfig(
        name="lmstudio",
        base_url="http://localhost:1234/v1",
        api_key_env=None,
        default_model="local-model",
    ),
    "ollama": ProviderConfig(
        name="ollama",
        base_url="http://localhost:11434/v1",
        api_key_env=None,
        default_model="llama2",
    ),
    # Providers that need separate implementations
    "anthropic": ProviderConfig(
        name="anthropic",
        api_key_env="ANTHROPIC_API_KEY",
        default_model="claude-3-opus-20240229",
        requires_separate_client=True,
    ),
    "cohere": ProviderConfig(
        name="cohere",
        api_key_env="COHERE_API_KEY",
        default_model="command",
        requires_separate_client=True,
    ),
    "google": ProviderConfig(
        name="google",
        api_key_env="GOOGLE_API_KEY",
        default_model="gemini-pro",
        requires_separate_client=True,
    ),
}


# Model to provider mapping for auto-detection
MODEL_PROVIDERS = {
    # OpenAI models
    "gpt-4": "openai",
    "gpt-3.5": "openai",
    "o1": "openai",
    # Anthropic models
    "claude": "anthropic",
    # Together models
    "mistralai/": "together",
    "meta-llama/": "together",
    "NousResearch/": "together",
    "teknium/": "together",
    # Groq models
    "groq/": "groq",
    "mixtral-8x7b": "groq",  # More specific
    "llama2-70b-4096": "groq",  # More specific
    "gemma-7b-it": "groq",  # More specific
    # Perplexity models
    "sonar": "perplexity",
    "pplx": "perplexity",
    # DeepSeek models
    "deepseek": "deepseek",
    # Moonshot models
    "moonshot": "moonshot",
    # Local models
    "local": "local",
    "gguf": "ollama",
    "ggml": "ollama",
    "llama2": "ollama",
    "mistral": "ollama",
    "codellama": "ollama",
    "phi": "ollama",
    "neural-chat": "ollama",
    "starling": "ollama",
    "orca": "ollama",
    "vicuna": "ollama",
    "llava": "ollama",
    "gemma:": "ollama",  # Note: gemma without colon goes to groq
}


def get_provider_config(provider: str) -> ProviderConfig | None:
    """Get configuration for a provider.

    Args:
        provider: Provider name

    Returns:
        ProviderConfig if found, None otherwise
    """
    return PROVIDERS.get(provider.lower())


def detect_provider_from_model(model: str) -> str:
    """Detect provider from model name.

    Args:
        model: Model name

    Returns:
        Provider name (defaults to 'openai' if unknown)
    """
    model_lower = model.lower()

    # Check prefixes
    for prefix, provider in MODEL_PROVIDERS.items():
        if model_lower.startswith(prefix):
            return provider

    # Default to OpenAI
    return "openai"


def is_openai_compatible(provider: str) -> bool:
    """Check if a provider uses OpenAI-compatible API.

    Args:
        provider: Provider name

    Returns:
        True if OpenAI-compatible, False otherwise
    """
    if provider == "openai":
        return False  # OpenAI is the original, not "compatible"

    config = get_provider_config(provider)
    return config and not config.requires_separate_client
