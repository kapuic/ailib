"""Tests for safety and moderation features."""

from ailib.safety import (
    add_custom_filter,
    check_content,
    check_rate_limit,
    disable_safety,
    enable_safety,
)
from ailib.safety._core import RateLimiter, SafetyChecker, SafetyConfig


class TestSafetyChecker:
    """Test the internal SafetyChecker."""

    def test_default_safety_checker(self):
        """Test safety checker with default config."""
        checker = SafetyChecker()

        # Normal content should pass
        violations = checker.check_content("Hello, world!")
        assert len(violations) == 0

    def test_length_checking(self):
        """Test content length validation."""
        config = SafetyConfig(max_output_length=10)
        checker = SafetyChecker(config)

        # Short content passes
        violations = checker.check_content("Hello")
        assert len(violations) == 0

        # Long content fails
        violations = checker.check_content("This is a very long message")
        assert len(violations) == 1
        assert violations[0].type == "length_exceeded"

    def test_sensitive_topics(self):
        """Test sensitive topic detection."""
        config = SafetyConfig(sensitive_topics=["medical", "legal"])
        checker = SafetyChecker(config)

        # Normal content passes
        violations = checker.check_content("Let's talk about Python")
        assert len(violations) == 0

        # Sensitive topic detected
        violations = checker.check_content("I need medical advice")
        assert len(violations) == 1
        assert violations[0].type == "sensitive_topic"

    def test_custom_regex_filters(self):
        """Test custom regex pattern matching."""
        config = SafetyConfig(custom_filters=[r"\b(password|secret)\b"])
        checker = SafetyChecker(config)

        # Normal content passes
        violations = checker.check_content("Hello there")
        assert len(violations) == 0

        # Pattern matched
        violations = checker.check_content("My password is 123")
        assert len(violations) == 1
        assert violations[0].type == "pattern_match"

    def test_disabled_safety(self):
        """Test that disabled safety returns no violations."""
        config = SafetyConfig(enabled=False, sensitive_topics=["test"])
        checker = SafetyChecker(config)

        # Even "unsafe" content passes when disabled
        violations = checker.check_content("test content")
        assert len(violations) == 0


class TestRateLimiter:
    """Test the rate limiter."""

    def test_basic_rate_limiting(self):
        """Test basic rate limit functionality."""
        limiter = RateLimiter(limit=3, window_seconds=60)

        # First 3 requests should pass
        assert limiter.check_limit("user1") is True
        assert limiter.check_limit("user1") is True
        assert limiter.check_limit("user1") is True

        # 4th request should fail
        assert limiter.check_limit("user1") is False

        # Different user should pass
        assert limiter.check_limit("user2") is True

    def test_rate_limit_reset(self):
        """Test rate limit reset functionality."""
        limiter = RateLimiter(limit=1, window_seconds=60)

        # Use up the limit
        assert limiter.check_limit("user1") is True
        assert limiter.check_limit("user1") is False

        # Reset specific user
        limiter.reset("user1")
        assert limiter.check_limit("user1") is True

        # Reset all
        limiter.check_limit("user2")
        limiter.reset()
        assert len(limiter._requests) == 0


class TestPublicAPI:
    """Test the public safety API."""

    def test_enable_disable_safety(self):
        """Test enabling and disabling safety."""
        # Enable with custom settings
        enable_safety(max_length=100, sensitive_topics=["test"])

        # Long content should be flagged
        is_safe, violations = check_content("x" * 200)
        assert not is_safe
        assert any("exceeds maximum length" in v for v in violations)

        # Disable safety
        disable_safety()

        # Same content should pass
        is_safe, violations = check_content("x" * 200)
        assert is_safe
        assert len(violations) == 0

        # Re-enable defaults
        enable_safety()

    def test_custom_filters(self):
        """Test adding custom filters."""
        enable_safety()

        # Add custom filter
        add_custom_filter(lambda text: "blocked" in text.lower())

        # Test filter
        is_safe, _ = check_content("This is blocked content")
        assert not is_safe

        is_safe, _ = check_content("This is safe content")
        assert is_safe

    def test_rate_limiting_api(self):
        """Test rate limiting through public API."""
        # Enable with rate limit
        enable_safety(rate_limit=2)

        # First 2 requests pass
        assert check_rate_limit("test_user") is True
        assert check_rate_limit("test_user") is True

        # 3rd request fails
        assert check_rate_limit("test_user") is False

        # Different user passes
        assert check_rate_limit("other_user") is True


class TestIntegration:
    """Test safety integration with other components."""

    def test_safety_with_content_check(self):
        """Test comprehensive content checking."""
        enable_safety(
            max_length=50,
            sensitive_topics=["confidential"],
            custom_filters=[r"\d{3}-\d{3}-\d{4}"],  # Phone pattern
            rate_limit=10,
        )

        # Test various unsafe content
        test_cases = [
            ("x" * 100, False, "length"),  # Too long
            ("This is confidential", False, "sensitive"),  # Sensitive topic
            ("Call me at 555-123-4567", False, "pattern"),  # Phone number
            ("Hello world", True, None),  # Safe
        ]

        for content, expected_safe, reason in test_cases:
            is_safe, violations = check_content(content)
            assert is_safe == expected_safe, f"Failed for {reason}: {violations}"

    def test_combined_violations(self):
        """Test content with multiple violations."""
        enable_safety(
            max_length=20, sensitive_topics=["secret"], custom_filters=[r"test\d+"]
        )

        # Content with multiple issues
        content = "This secret test123 message is way too long"
        is_safe, violations = check_content(content)

        assert not is_safe
        assert len(violations) >= 3  # Length, topic, and pattern
