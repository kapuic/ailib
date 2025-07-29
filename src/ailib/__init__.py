"""Top-level package for AILib."""

__author__ = """Kapui Cheung"""
__email__ = "dev@kapui.net"
__version__ = "0.1.0"

# Agents imports
from .agents import Agent, Tool, ToolRegistry, tool

# Chains imports
from .chains import Chain, ChainStep

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
    # Agents
    "Agent",
    "Tool",
    "ToolRegistry",
    "tool",
]
