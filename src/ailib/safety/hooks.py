"""Simple safety hooks for AILib - Vercel AI SDK style."""

from collections.abc import Callable

from ..core import CompletionResponse
from ._core import RateLimiter, SafetyChecker, SafetyConfig
from ._moderation import create_moderation_hook

# Global safety settings (can be modified by users)
_global_safety = SafetyChecker()
_global_rate_limiter: RateLimiter | None = None


def enable_safety(
    block_harmful: bool = True,
    max_length: int = 4000,
    blocked_words: list[str] | None = None,
    custom_filters: list[str] | None = None,
    rate_limit: int | None = None,
):
    """Enable safety features with simple configuration.

    This is the main entry point for safety configuration.
    Called automatically with defaults, but can be customized.

    Args:
        block_harmful: Block potentially harmful content
        max_length: Maximum output length
        blocked_words: Topics to monitor/block
        custom_filters: Regex patterns to block
        rate_limit: Requests per minute limit

    Example:
        # Use defaults (already enabled)
        agent = create_agent("assistant")

        # Customize safety
        enable_safety(
            blocked_words=["medical", "legal"],
            rate_limit=30
        )
    """
    global _global_safety, _global_rate_limiter

    config = SafetyConfig(
        enabled=True,
        block_harmful_content=block_harmful,
        max_output_length=max_length,
        blocked_words=blocked_words or [],
        custom_filters=custom_filters or [],
    )

    _global_safety = SafetyChecker(config)

    if rate_limit:
        _global_rate_limiter = RateLimiter(rate_limit)


def disable_safety():
    """Disable all safety features."""
    global _global_safety
    _global_safety.config.enabled = False


def add_custom_filter(filter_func: Callable[[str], bool]):
    """Add a custom content filter.

    Args:
        filter_func: Function that returns True if content should be blocked

    Example:
        # Block content mentioning specific terms
        add_custom_filter(lambda text: "confidential" in text.lower())
    """
    _global_safety.add_custom_filter(filter_func)


def check_content(content: str) -> tuple[bool, list[str]]:
    """Check content for safety violations.

    Args:
        content: Text to check

    Returns:
        Tuple of (is_safe, list_of_violations)

    Example:
        is_safe, violations = check_content(response_text)
        if not is_safe:
            print(f"Safety violations: {violations}")
    """
    violations = _global_safety.check_content(content)

    if not violations:
        return True, []

    should_block = _global_safety.should_block(violations)
    violation_messages = [v.message for v in violations]

    # Return is_safe (not should_block) and violations
    return not should_block, violation_messages


def check_rate_limit(user_id: str) -> bool:
    """Check if user is within rate limit.

    Args:
        user_id: Unique user identifier

    Returns:
        True if within limit, False if exceeded

    Example:
        if not check_rate_limit(user_id):
            raise Exception("Rate limit exceeded")
    """
    if not _global_rate_limiter:
        return True

    return _global_rate_limiter.check_limit(user_id)


# Simple moderation wrapper
def with_moderation(api_key: str | None = None) -> tuple[Callable, Callable]:
    """Enable OpenAI moderation with one line.

    Args:
        api_key: OpenAI API key (uses env if not provided)

    Returns:
        Tuple of (pre_hook, post_hook) for moderation

    Example:
        pre_hook, post_hook = with_moderation()

        # Use with agent
        agent = create_agent(
            "assistant",
            pre_hook=pre_hook,
            post_hook=post_hook
        )
    """
    return create_moderation_hook(api_key)


# Internal hooks used by the framework
def _internal_pre_hook(prompt: str, user_id: str | None = None, **kwargs) -> str:
    """Internal pre-processing hook."""
    # Check rate limit
    if user_id and not check_rate_limit(user_id):
        raise ValueError("Rate limit exceeded")

    # Check input safety
    is_safe, violations = check_content(prompt)
    if not is_safe:
        raise ValueError(f"Input blocked: {', '.join(violations)}")

    return prompt


def _internal_post_hook(response: CompletionResponse, **kwargs) -> CompletionResponse:
    """Internal post-processing hook."""
    # Check output safety
    is_safe, violations = check_content(response.content)

    if not is_safe:
        # Replace with safe content
        response.content = (
            "I cannot provide that response as it may violate content policies. "
            f"Reasons: {', '.join(violations)}"
        )

    return response


# Initialize with sensible defaults
enable_safety()
