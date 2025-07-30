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
)

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
