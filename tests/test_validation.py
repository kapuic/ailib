"""Tests for validation through factory functions.

Since validation is now internal, we test it through the public API.
"""

import os

import pytest
from ailib import create_agent, create_chain, create_session
from ailib.core import PromptTemplate

# Set dummy API key for tests
os.environ["OPENAI_API_KEY"] = "sk-test-key-for-validation"


class TestPromptTemplate:
    """Test prompt template validation through the public API."""

    def test_valid_template(self):
        """Test valid template creation."""
        template = PromptTemplate("Hello {name}, welcome to {place}!")
        assert template.template == "Hello {name}, welcome to {place}!"
        assert set(template.variables) == {"name", "place"}

    def test_invalid_empty_template(self):
        """Test that empty template is rejected."""
        with pytest.raises(ValueError) as exc_info:
            PromptTemplate("")
        assert "empty" in str(exc_info.value).lower()

    def test_template_with_role(self):
        """Test template with custom role."""
        from ailib.core import Role

        template = PromptTemplate("You are a {role}", Role.SYSTEM)
        assert template.role == Role.SYSTEM
        assert template.variables == ["role"]  # Only one variable, order doesn't matter


class TestAgentFactory:
    """Test agent validation through factory function."""

    def test_valid_agent_creation(self):
        """Test valid agent creation."""
        agent = create_agent(
            "test_agent",
            model="gpt-4",
            temperature=0.5,
            verbose=True,
        )
        assert agent.name == "test_agent"
        assert agent.model == "gpt-4"
        assert agent.temperature == 0.5
        assert agent.verbose is True

    def test_invalid_empty_name(self):
        """Test that empty name is rejected."""
        with pytest.raises(Exception) as exc_info:
            create_agent("", model="gpt-4")
        assert "empty" in str(exc_info.value).lower()

    def test_invalid_temperature(self):
        """Test temperature validation."""
        with pytest.raises(Exception) as exc_info:
            create_agent("test", temperature=3.0)
        assert "less than or equal to 2" in str(exc_info.value).lower()

    def test_default_values(self):
        """Test default values are set correctly."""
        agent = create_agent("test")
        assert agent.temperature == 0.7
        assert agent.max_steps == 10
        assert agent.verbose is False


class TestChainFactory:
    """Test chain validation through factory function."""

    def test_valid_chain_creation(self):
        """Test valid chain creation."""
        chain = create_chain(
            "Translate to Spanish: {text}",
            "Make it more formal",
            verbose=True,
        )
        assert chain._verbose is True
        assert len(chain) == 2  # Two prompts added

    def test_chain_with_custom_retry(self):
        """Test chain with custom retry configuration."""
        chain = create_chain(
            "Summarize: {text}",
            retry_attempts=5,
            retry_delay=2.0,
        )
        assert chain.retry_attempts == 5
        assert chain.retry_delay == 2.0

    def test_default_values(self):
        """Test default values."""
        chain = create_chain("Test prompt")
        assert chain.max_iterations == 10
        assert chain.early_stopping is True
        assert chain.retry_attempts == 3


class TestSessionFactory:
    """Test session validation through factory function."""

    def test_valid_session_creation(self):
        """Test valid session creation."""
        session = create_session(
            session_id="test-123",
            max_messages=50,
            ttl=3600,
            metadata={"user": "test"},
        )
        assert session.session_id == "test-123"
        assert session.max_history == 50
        assert session.ttl == 3600
        assert session.metadata["user"] == "test"

    def test_auto_save_requires_path(self):
        """Test that auto_save requires save_path."""
        with pytest.raises(Exception) as exc_info:
            create_session(auto_save=True)
        assert "save_path must be provided" in str(exc_info.value).lower()

    def test_default_values(self):
        """Test default values."""
        session = create_session()
        assert session.max_history == 100
        assert session.auto_save is False
        assert session.metadata == {}


class TestDirectInstantiation:
    """Test that direct instantiation bypasses validation."""

    def test_agent_direct_instantiation(self):
        """Test agent can be created directly without validation."""
        from ailib import Agent

        # This should work even with invalid temperature
        agent = Agent(name="test", temperature=5.0)
        assert agent.temperature == 5.0  # No validation!

    def test_chain_direct_instantiation(self):
        """Test chain can be created directly without validation."""
        from ailib import Chain

        # This should work even with zero iterations
        chain = Chain(max_iterations=0)
        assert chain.max_iterations == 0  # No validation!

    def test_session_direct_instantiation(self):
        """Test session can be created directly without validation."""
        from ailib import Session

        # This should work even with auto_save and no path
        session = Session(auto_save=True)  # No validation error!
        assert session.auto_save is True


class TestFactoryValidation:
    """Test that factory functions properly validate inputs."""

    def test_agent_factory_validates_types(self):
        """Test agent factory validates parameter types."""
        # String temperature should fail
        with pytest.raises(Exception):  # noqa: B017
            create_agent("test", temperature="high")

    def test_chain_factory_validates_config(self):
        """Test chain factory validates configuration."""
        # Negative retry attempts should fail validation
        with pytest.raises(Exception) as exc_info:
            create_chain("Test", retry_attempts=-1)
        assert "greater than or equal to 0" in str(exc_info.value).lower()

    def test_session_factory_validates_consistency(self):
        """Test session factory validates configuration consistency."""
        # auto_save without path should fail
        with pytest.raises(Exception) as exc_info:
            create_session(auto_save=True)
        assert "save_path" in str(exc_info.value).lower()


class TestIntegration:
    """Test integration between components."""

    def test_agent_with_custom_llm(self):
        """Test agent creation with custom LLM client."""
        from ailib import OpenAIClient

        llm = OpenAIClient(model="gpt-3.5-turbo")
        agent = create_agent("test", llm=llm)
        assert agent.llm == llm
        assert agent.model == "gpt-3.5-turbo"

    def test_chain_with_session(self):
        """Test chain with custom session."""
        session = create_session(session_id="test-session")
        chain = create_chain("Test prompt")
        chain.with_session(session)
        assert chain.session == session


class TestValidationPhilosophy:
    """Test that our validation follows Vercel AI SDK philosophy."""

    def test_progressive_disclosure(self):
        """Test progressive disclosure - simple by default, complex when needed."""
        # Simple case - just works
        agent = create_agent("simple")
        assert agent is not None

        # Complex case - all options available
        agent = create_agent(
            "complex",
            model="gpt-4",
            temperature=0.2,
            max_steps=20,
            verbose=True,
            memory_size=50,
            return_intermediate_steps=True,
        )
        assert agent.temperature == 0.2
        assert agent.max_steps == 20

    def test_factory_vs_direct(self):
        """Test factory provides safety, direct provides flexibility."""
        # Factory enforces validation
        with pytest.raises(Exception):  # noqa: B017
            create_agent("test", temperature=10.0)

        # Direct instantiation allows anything
        from ailib import Agent

        agent = Agent(temperature=10.0)  # Works!
        assert agent.temperature == 10.0
