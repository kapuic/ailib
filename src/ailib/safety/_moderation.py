"""OpenAI moderation integration - internal module."""

from collections.abc import Callable
from typing import Any

from ..core import CompletionResponse


class OpenAIModerator:
    """OpenAI moderation API wrapper."""

    def __init__(self, api_key: str | None = None):
        """Initialize moderator.

        Args:
            api_key: OpenAI API key (uses environment variable if not provided)
        """
        try:
            import openai

            self._client = openai.OpenAI(api_key=api_key)
            self._available = True
        except (ImportError, Exception):
            # Handle both missing library and missing API key
            self._available = False
            self._client = None

    def check(self, content: str) -> dict[str, Any]:
        """Check content with OpenAI moderation.

        Args:
            content: Text to check

        Returns:
            Moderation results dict
        """
        if not self._available or not self._client:
            return {"error": "OpenAI client not available"}

        try:
            response = self._client.moderations.create(input=content)
            return response.model_dump()
        except Exception as e:
            return {"error": str(e)}

    def is_flagged(self, content: str) -> bool:
        """Check if content is flagged by OpenAI.

        Args:
            content: Text to check

        Returns:
            True if content is flagged
        """
        result = self.check(content)

        if "error" in result:
            # On error, don't block by default
            return False

        # Check if any result is flagged
        for item in result.get("results", []):
            if item.get("flagged", False):
                return True

        return False

    def get_categories(self, content: str) -> dict[str, bool]:
        """Get flagged categories for content.

        Args:
            content: Text to check

        Returns:
            Dict of category names to flagged status
        """
        result = self.check(content)

        if "error" in result or not result.get("results"):
            return {}

        # Extract categories from first result
        categories = result["results"][0].get("categories", {})
        return {k: v for k, v in categories.items() if v}


def create_moderation_hook(
    api_key: str | None = None, categories: list[str] | None = None
) -> tuple[Callable, Callable]:
    """Create moderation hooks for pre/post processing.

    Args:
        api_key: OpenAI API key
        categories: Specific categories to check (None = all)

    Returns:
        Tuple of (pre_hook, post_hook) functions
    """
    moderator = OpenAIModerator(api_key)

    def pre_hook(prompt: str, **kwargs) -> str:
        """Check prompt before sending to LLM."""
        if moderator.is_flagged(prompt):
            raise ValueError("Input contains inappropriate content")
        return prompt

    def post_hook(response: CompletionResponse, **kwargs) -> CompletionResponse:
        """Check response after LLM generation."""
        if moderator.is_flagged(response.content):
            # Replace content with safe message
            response.content = "I cannot provide that information as it may contain inappropriate content."  # noqa: E501
        return response

    return pre_hook, post_hook
