"""Tests for chains module."""

from unittest.mock import Mock

import pytest
from ailib import Chain
from ailib.core import CompletionResponse


class TestChain:
    """Test Chain class."""

    def test_chain_creation(self):
        """Test creating a chain."""
        chain = Chain()
        assert len(chain) == 0
        assert chain.llm is None

    def test_chain_builder(self):
        """Test chain builder methods."""
        mock_llm = Mock()
        chain = (
            Chain()
            .with_llm(mock_llm)
            .verbose(True)
            .add_system("You are helpful")
            .add_user("Hello")
            .set_context(name="Alice")
        )

        assert chain.llm == mock_llm
        assert chain._verbose is True
        assert len(chain) == 2
        assert "name" in chain._context

    def test_chain_execution(self):
        """Test executing a chain."""
        # Mock LLM
        mock_llm = Mock()
        mock_response = CompletionResponse(
            content="Hello Alice!",
            model="test",
            usage={"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
        )
        mock_llm.complete.return_value = mock_response

        # Create and run chain
        chain = Chain(mock_llm).add_user("Say hello to {name}")

        result = chain.run(name="Alice")

        assert result == "Hello Alice!"
        mock_llm.complete.assert_called_once()

    def test_chain_with_processor(self):
        """Test chain with result processor."""
        # Mock LLM
        mock_llm = Mock()
        mock_response = CompletionResponse(
            content="42",
            model="test",
            usage={"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
        )
        mock_llm.complete.return_value = mock_response

        # Processor to convert to int
        def to_int(text: str) -> int:
            return int(text.strip())

        # Create and run chain
        chain = Chain(mock_llm).add_user("What is 40 + 2?", processor=to_int)

        result = chain.run()

        assert result == 42
        assert isinstance(result, int)

    def test_chain_multi_step(self):
        """Test multi-step chain."""
        # Mock LLM
        mock_llm = Mock()
        responses = [
            CompletionResponse(
                content="The capital is Paris",
                model="test",
                usage={"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
            ),
            CompletionResponse(
                content="Paris has about 2.2 million people",
                model="test",
                usage={
                    "prompt_tokens": 20,
                    "completion_tokens": 10,
                    "total_tokens": 30,
                },
            ),
        ]
        mock_llm.complete.side_effect = responses

        # Create multi-step chain
        chain = (
            Chain(mock_llm)
            .add_user("What is the capital of France?", name="capital_query")
            .add_user(
                "How many people live in {capital_query}?", name="population_query"
            )
        )

        result = chain.run()

        assert "2.2 million" in result
        assert mock_llm.complete.call_count == 2

    def test_chain_without_llm(self):
        """Test chain execution without LLM."""
        chain = Chain().add_user("Hello")

        with pytest.raises(ValueError, match="No LLM client set"):
            chain.run()

    def test_chain_reset(self):
        """Test resetting chain."""
        chain = Chain().add_user("Step 1").add_user("Step 2").set_context(key="value")

        assert len(chain) == 2
        assert "key" in chain._context

        chain.reset()

        assert len(chain) == 0
        assert len(chain._context) == 0
