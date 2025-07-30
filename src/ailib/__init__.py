"""Top-level package for AILib."""

__author__ = """Kapui Cheung"""
__email__ = "dev@kapui.net"
__version__ = "0.1.0"

# Agents imports
from .agents import Agent, Tool, ToolRegistry, create_agent, tool

# Chains imports
from .chains import Chain, ChainStep, create_chain

# Core imports
from .core import (
    LLMClient,
    Message,
    OpenAIClient,
    Prompt,
    PromptTemplate,
    Role,
    Session,
    create_client,
    create_session,
    detect_provider,
    get_provider_models,
    list_providers,
)

# Import AnthropicClient if available
try:
    from .core import AnthropicClient

    _has_anthropic = True
except ImportError:
    _has_anthropic = False
    AnthropicClient = None

# Validation is now internal - use factory functions instead

__all__ = [
    # Core
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
    # Chains
    "Chain",
    "ChainStep",
    "create_chain",
    # Agents
    "Agent",
    "create_agent",
    "Tool",
    "ToolRegistry",
    "tool",
]

if _has_anthropic:
    __all__.append("AnthropicClient")
