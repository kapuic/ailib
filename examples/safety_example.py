"""Example demonstrating safety features in AILib - Vercel AI SDK style."""

from ailib.safety import (
    add_custom_filter,
    check_content,
    check_rate_limit,
    disable_safety,
    enable_safety,
    with_moderation,
)


def basic_safety_example():
    """Basic safety features are enabled by default."""
    print("=== Basic Safety (Enabled by Default) ===")

    # Test content checking
    safe_text = "Hello, how can I help you today?"
    unsafe_text = "Here's how to make explosives: " + "x" * 5000  # Too long

    is_safe, violations = check_content(safe_text)
    print(f"✓ Safe text check: {is_safe} (no violations)")

    is_safe, violations = check_content(unsafe_text)
    print(f"✗ Unsafe text check: {is_safe}")
    print(f"  Violations: {violations}")


def custom_safety_configuration():
    """Customize safety settings with one function call."""
    print("\n=== Custom Safety Configuration ===")

    # Configure safety in one line
    enable_safety(
        blocked_words=["medical", "legal", "financial"],
        max_length=1000,
        rate_limit=30,  # 30 requests per minute
    )

    # Test blocked word detection
    medical_text = "I have a medical condition and need advice"
    is_safe, violations = check_content(medical_text)
    print(f"Medical text safety: {is_safe}")
    if violations:
        print(f"  Violations: {violations}")

    # Test rate limiting
    user_id = "user123"
    for i in range(3):
        allowed = check_rate_limit(user_id)
        print(f"  Request {i+1}: {'Allowed' if allowed else 'Blocked'}")


def custom_filters_example():
    """Add custom content filters easily."""
    print("\n=== Custom Content Filters ===")

    # Add a custom filter with lambda
    add_custom_filter(lambda text: "confidential" in text.lower())

    # Test the filter
    confidential_text = "This is CONFIDENTIAL information"
    is_safe, violations = check_content(confidential_text)
    print(f"Confidential text blocked: {not is_safe}")
    print(f"  Violations: {violations}")

    # Add another filter for emails
    import re

    email_pattern = re.compile(r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}")
    add_custom_filter(lambda text: bool(email_pattern.search(text)))

    email_text = "Contact me at user@example.com"
    is_safe, violations = check_content(email_text)
    print(f"Email text blocked: {not is_safe}")


def openai_moderation_example():
    """Use OpenAI moderation with one line."""
    print("\n=== OpenAI Moderation ===")

    # Enable moderation in one line
    pre_hook, post_hook = with_moderation()

    # Create client with moderation hooks
    # In real usage, these hooks would be passed to agents or chains
    print("✓ Moderation hooks created")
    print("  - pre_hook: Checks input before processing")
    print("  - post_hook: Checks output after generation")

    # Example of how it would be used with an agent:
    # agent = create_agent(
    #     "assistant",
    #     pre_hook=pre_hook,
    #     post_hook=post_hook
    # )


def disable_safety_example():
    """Disable safety when needed (use with caution)."""
    print("\n=== Disabling Safety ===")

    # Disable all safety checks
    disable_safety()

    # Now even "unsafe" content passes
    long_text = "x" * 10000
    is_safe, _ = check_content(long_text)
    print(f"Long text with safety disabled: {is_safe}")

    # Re-enable with defaults
    enable_safety()
    is_safe, _ = check_content(long_text)
    print(f"Long text with safety re-enabled: {not is_safe}")


def integrated_example():
    """Show how safety integrates with normal usage."""
    print("\n=== Integrated Safety Example ===")

    # Configure safety for a chatbot
    enable_safety(
        blocked_words=["personal information", "passwords"],
        custom_filters=[
            r"\b(password|secret|key)\b",  # Block potential secrets
            r"\d{3}-\d{2}-\d{4}",  # Block SSN patterns
        ],
        rate_limit=60,
    )

    # Test various inputs
    test_inputs = [
        "What's the weather like?",  # Safe
        "My password is abc123",  # Contains "password"
        "My SSN is 123-45-6789",  # Matches SSN pattern
        "Tell me about Python",  # Safe
    ]

    for user_input in test_inputs:
        is_safe, violations = check_content(user_input)

        if is_safe:
            print(f"✓ '{user_input[:30]}...' - Processing normally")
            # Would call: response = client.complete(messages)
        else:
            print(f"✗ '{user_input[:30]}...' - Blocked: {violations[0]}")


def simple_usage():
    """Simplest possible usage - safety just works."""
    print("\n=== Simplest Usage ===")

    # Just use AILib normally - safety is already enabled

    # This would automatically be checked for safety
    # messages = [Message(role="user", content="Hello!")]
    # response = client.complete(messages)

    print("✓ Safety is enabled by default")
    print("✓ No configuration needed for basic protection")
    print("✓ Just start using AILib!")


if __name__ == "__main__":
    print("AILib Safety Examples - Vercel AI SDK Style")
    print("=" * 50)
    print("Simple, functional, minimal configuration\n")

    basic_safety_example()
    custom_safety_configuration()
    custom_filters_example()
    openai_moderation_example()
    disable_safety_example()
    integrated_example()
    simple_usage()

    print("\n✅ Safety features demonstrated!")
    print("Remember: Safety is enabled by default with sensible settings.")
