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

# Validation imports
from .validation import (
    AgentConfig,
    ChainConfig,
    LLMConfig,
    MessageConfig,
    PromptTemplateConfig,
    SafetyConfig,
    SessionConfig,
    ToolConfig,
    ToolParameterSchema,
    create_dynamic_model,
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
    # Validation
    "AgentConfig",
    "ChainConfig",
    "LLMConfig",
    "MessageConfig",
    "PromptTemplateConfig",
    "SafetyConfig",
    "SessionConfig",
    "ToolConfig",
    "ToolParameterSchema",
    "create_dynamic_model",
]
