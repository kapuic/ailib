"""Core module for AI Library.

This module provides the fundamental components for interacting with LLMs.
"""

from .client_factory import (
    create_client,
    detect_provider,
    get_provider_models,
    list_providers,
)
from .llm_client import CompletionResponse, LLMClient, Message, Role
from .openai_client import OpenAIClient
from .prompt import Prompt, PromptTemplate
from .providers import PROVIDERS, get_provider_config
from .session import Session, create_session

# Import AnthropicClient only if anthropic is installed
try:
    from .anthropic_client import AnthropicClient

    _has_anthropic = True
except ImportError:
    _has_anthropic = False
    AnthropicClient = None

__all__ = [
    "CompletionResponse",
    "LLMClient",
    "Message",
    "Role",
    "OpenAIClient",
    "Prompt",
    "PromptTemplate",
    "Session",
    "create_session",
    "create_client",
    "detect_provider",
    "list_providers",
    "get_provider_models",
    "PROVIDERS",
    "get_provider_config",
]

if _has_anthropic:
    __all__.append("AnthropicClient")
