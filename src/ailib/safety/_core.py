"""Core safety functionality - internal module."""

import re
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from ..validation import SafetyConfig


@dataclass
class SafetyViolation:
    """Represents a safety violation."""

    type: str
    severity: str  # "low", "medium", "high"
    message: str
    details: dict[str, Any] | None = None


class SafetyChecker:
    """Internal safety checker implementation."""

    def __init__(self, config: SafetyConfig | None = None):
        """Initialize safety checker with configuration."""
        self.config = config or SafetyConfig()
        self._custom_filters: list[Callable[[str], bool]] = []
        self._compile_patterns()

    def _compile_patterns(self):
        """Compile regex patterns for efficiency."""
        self._compiled_filters = []
        for pattern in self.config.custom_filters:
            try:
                self._compiled_filters.append(re.compile(pattern, re.IGNORECASE))
            except re.error:
                # Skip invalid patterns
                pass

    def check_content(self, content: str) -> list[SafetyViolation]:
        """Check content for safety violations.

        Args:
            content: Text to check

        Returns:
            List of violations found
        """
        if not self.config.enabled:
            return []

        violations = []

        # Check length
        if len(content) > self.config.max_output_length:
            violations.append(
                SafetyViolation(
                    type="length_exceeded",
                    severity="medium",  # Changed from "low" to "medium"
                    message=f"Content exceeds maximum length of {self.config.max_output_length}",  # noqa: E501
                    details={
                        "length": len(content),
                        "max": self.config.max_output_length,
                    },
                )
            )

        # Check custom regex filters
        for pattern in self._compiled_filters:
            if pattern.search(content):
                violations.append(
                    SafetyViolation(
                        type="pattern_match",
                        severity="medium",
                        message=f"Content matches blocked pattern: {pattern.pattern}",
                        details={"pattern": pattern.pattern},
                    )
                )

        # Check sensitive topics
        content_lower = content.lower()
        for topic in self.config.sensitive_topics:
            if topic.lower() in content_lower:
                violations.append(
                    SafetyViolation(
                        type="sensitive_topic",
                        severity="medium",
                        message=f"Content contains sensitive topic: {topic}",
                        details={"topic": topic},
                    )
                )

        # Run custom filters
        for filter_func in self._custom_filters:
            try:
                if filter_func(content):
                    violations.append(
                        SafetyViolation(
                            type="custom_filter",
                            severity="medium",
                            message="Content blocked by custom filter",
                        )
                    )
            except Exception:
                # Ignore failing filters
                pass

        return violations

    def add_custom_filter(self, filter_func: Callable[[str], bool]):
        """Add a custom filter function.

        Args:
            filter_func: Function that returns True if content should be blocked
        """
        self._custom_filters.append(filter_func)

    def should_block(self, violations: list[SafetyViolation]) -> bool:
        """Determine if content should be blocked based on violations.

        Args:
            violations: List of violations found

        Returns:
            True if content should be blocked
        """
        if not violations:
            return False

        # Block if any high severity violations
        if any(v.severity == "high" for v in violations):
            return True

        # Block if configured to block harmful content and medium severity found
        if self.config.block_harmful_content:
            if any(v.severity == "medium" for v in violations):
                return True

        return False


class RateLimiter:
    """Simple in-memory rate limiter."""

    def __init__(self, limit: int, window_seconds: int = 60):
        """Initialize rate limiter.

        Args:
            limit: Maximum requests per window
            window_seconds: Time window in seconds
        """
        self.limit = limit
        self.window_seconds = window_seconds
        self._requests: dict[str, list[float]] = {}

    def check_limit(self, key: str) -> bool:
        """Check if request is within rate limit.

        Args:
            key: Unique key for rate limiting (e.g., user ID)

        Returns:
            True if within limit, False if exceeded
        """
        import time

        now = time.time()

        # Clean old requests
        if key in self._requests:
            self._requests[key] = [
                t for t in self._requests[key] if now - t < self.window_seconds
            ]
        else:
            self._requests[key] = []

        # Check limit
        if len(self._requests[key]) >= self.limit:
            return False

        # Record request
        self._requests[key].append(now)
        return True

    def reset(self, key: str | None = None):
        """Reset rate limit tracking.

        Args:
            key: Specific key to reset, or None to reset all
        """
        if key:
            self._requests.pop(key, None)
        else:
            self._requests.clear()
