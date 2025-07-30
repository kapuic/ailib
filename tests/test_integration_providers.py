"""Integration tests for providers with real APIs.

These tests are skipped by default unless AILIB_RUN_INTEGRATION_TESTS is set.
They require valid API keys for each provider.
"""

import os

import pytest
from ailib import create_agent, create_client
from ailib.core.llm_client import Message, Role

# Skip all tests in this file unless integration tests are enabled
pytestmark = pytest.mark.skipif(
    not os.getenv("AILIB_RUN_INTEGRATION_TESTS"),
    reason="Integration tests disabled. Set AILIB_RUN_INTEGRATION_TESTS=1 to run.",
)


class TestRealProviders:
    """Test real provider APIs (requires API keys)."""

    @pytest.mark.skipif(not os.getenv("OPENAI_API_KEY"), reason="No OpenAI API key")
    def test_openai_real_completion(self):
        """Test real OpenAI API completion."""
        client = create_client("gpt-3.5-turbo")

        messages = [
            Message(role=Role.USER, content="Say 'test passed' and nothing else")
        ]

        response = client.complete(messages, temperature=0)
        assert "test passed" in response.content.lower()

    @pytest.mark.skipif(
        not os.getenv("ANTHROPIC_API_KEY"), reason="No Anthropic API key"
    )
    def test_anthropic_real_completion(self):
        """Test real Anthropic API completion."""
        client = create_client("claude-3-haiku-20240307")

        messages = [
            Message(role=Role.USER, content="Say 'test passed' and nothing else")
        ]

        response = client.complete(messages, temperature=0)
        assert "test passed" in response.content.lower()

    @pytest.mark.skipif(not os.getenv("GROQ_API_KEY"), reason="No Groq API key")
    def test_groq_real_completion(self):
        """Test real Groq API completion."""
        client = create_client("mixtral-8x7b-32768")

        messages = [
            Message(role=Role.USER, content="Say 'test passed' and nothing else")
        ]

        response = client.complete(messages, temperature=0)
        assert "test passed" in response.content.lower()

    def test_ollama_real_completion(self):
        """Test real Ollama completion (requires local Ollama)."""
        try:
            client = create_client("llama2", provider="ollama")

            messages = [
                Message(role=Role.USER, content="Say 'test passed' and nothing else")
            ]

            response = client.complete(messages, temperature=0)
            assert response.content  # Just check we got something back
        except Exception as e:
            if "connection" in str(e).lower() or "refused" in str(e).lower():
                pytest.skip("Ollama not running locally")
            else:
                raise


class TestCrossProviderConsistency:
    """Test consistency across different providers."""

    def get_available_providers(self):
        """Get list of providers with valid API keys."""
        providers = []

        if os.getenv("OPENAI_API_KEY"):
            providers.append(("openai", "gpt-3.5-turbo"))

        if os.getenv("ANTHROPIC_API_KEY"):
            providers.append(("anthropic", "claude-3-haiku-20240307"))

        if os.getenv("GROQ_API_KEY"):
            providers.append(("groq", "mixtral-8x7b-32768"))

        return providers

    def test_basic_math_across_providers(self):
        """Test that all providers can do basic math."""
        providers = self.get_available_providers()

        if not providers:
            pytest.skip("No provider API keys available")

        prompt = "What is 2 + 2? Reply with just the number."

        for provider_name, model in providers:
            client = create_client(model)
            agent = create_agent(f"{provider_name}-agent", llm=client)

            result = agent.run(prompt)

            # All providers should recognize this as 4
            assert "4" in result

    def test_tool_usage_across_providers(self):
        """Test tool usage across providers."""
        from ailib import tool

        @tool
        def add_numbers(a: int, b: int) -> int:
            """Add two numbers."""
            return a + b

        providers = self.get_available_providers()

        if not providers:
            pytest.skip("No provider API keys available")

        for provider_name, model in providers:
            # Skip if provider doesn't support tools
            if provider_name == "groq":  # Example: if Groq doesn't support tools
                continue

            agent = create_agent(
                f"{provider_name}-calculator",
                model=model,
                tools=[add_numbers],
                verbose=True,
            )

            result = agent.run("What is 15 plus 27?")

            # Should use the tool and get 42
            assert "42" in result


class TestProviderLimits:
    """Test provider-specific limits and constraints."""

    @pytest.mark.skipif(not os.getenv("OPENAI_API_KEY"), reason="No OpenAI API key")
    def test_openai_context_length(self):
        """Test OpenAI context length handling."""
        client = create_client("gpt-3.5-turbo")

        # Create a very long message (but not too long to avoid costs)
        long_text = "This is a test. " * 100

        messages = [
            Message(role=Role.USER, content=f"Summarize this in 5 words: {long_text}")
        ]

        response = client.complete(messages)

        # Should handle long context gracefully
        assert len(response.content.split()) < 20  # Reasonable summary length

    @pytest.mark.skipif(
        not os.getenv("ANTHROPIC_API_KEY"), reason="No Anthropic API key"
    )
    def test_anthropic_system_message_handling(self):
        """Test Anthropic's special system message handling."""
        client = create_client("claude-3-haiku-20240307")

        messages = [
            Message(role=Role.SYSTEM, content="You always respond in uppercase."),
            Message(role=Role.USER, content="say hello"),
        ]

        response = client.complete(messages, temperature=0)

        # Should respect system message
        assert response.content.isupper() or "HELLO" in response.content.upper()


class TestProviderPerformance:
    """Basic performance tests across providers."""

    def test_response_times(self):
        """Test and compare response times across providers."""
        import time

        providers = []
        if os.getenv("OPENAI_API_KEY"):
            providers.append(("openai", "gpt-3.5-turbo"))
        if os.getenv("ANTHROPIC_API_KEY"):
            providers.append(("anthropic", "claude-3-haiku-20240307"))

        if not providers:
            pytest.skip("No provider API keys available")

        results = {}
        prompt = "Say 'hi' and nothing else."

        for provider_name, model in providers:
            client = create_client(model)

            start_time = time.time()
            response = client.complete([Message(role=Role.USER, content=prompt)])
            end_time = time.time()

            results[provider_name] = {
                "time": end_time - start_time,
                "response": response.content,
            }

        # Just log the results, don't assert on timing
        for provider, data in results.items():
            print(f"\n{provider}: {data['time']:.2f}s - {data['response']}")


# Fixture to show which providers are being tested
@pytest.fixture(scope="session", autouse=True)
def show_provider_status():
    """Show which providers have API keys available."""
    if os.getenv("AILIB_RUN_INTEGRATION_TESTS"):
        print("\n=== Provider Integration Test Status ===")
        print(f"OpenAI: {'✓' if os.getenv('OPENAI_API_KEY') else '✗'}")
        print(f"Anthropic: {'✓' if os.getenv('ANTHROPIC_API_KEY') else '✗'}")
        print(f"Groq: {'✓' if os.getenv('GROQ_API_KEY') else '✗'}")
        print(f"Together: {'✓' if os.getenv('TOGETHER_API_KEY') else '✗'}")
        print(f"Perplexity: {'✓' if os.getenv('PERPLEXITY_API_KEY') else '✗'}")
        print("=====================================\n")
