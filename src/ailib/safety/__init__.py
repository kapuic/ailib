"""Safety and moderation features for AILib.

Simple, functional API following Vercel AI SDK philosophy.
Safety is enabled by default with sensible settings.
"""

from .hooks import (
    add_custom_filter,
    check_content,
    check_rate_limit,
    disable_safety,
    enable_safety,
    with_moderation,
)

__all__ = [
    "enable_safety",
    "disable_safety",
    "add_custom_filter",
    "check_content",
    "check_rate_limit",
    "with_moderation",
]
