"""Tests for core module."""

from unittest.mock import Mock, patch

import pytest
from ailib.core import Message, OpenAIClient, Prompt, PromptTemplate, Role, Session


class TestMessage:
    """Test Message dataclass."""

    def test_message_creation(self):
        """Test creating a message."""
        msg = Message(role=Role.USER, content="Hello")
        assert msg.role == Role.USER
        assert msg.content == "Hello"
        assert msg.name is None

    def test_message_to_dict(self):
        """Test converting message to dict."""
        msg = Message(role=Role.ASSISTANT, content="Hi there")
        data = msg.to_dict()
        assert data["role"] == "assistant"
        assert data["content"] == "Hi there"
        assert "name" not in data


class TestPromptTemplate:
    """Test PromptTemplate class."""

    def test_simple_template(self):
        """Test simple template formatting."""
        template = PromptTemplate("Hello {name}!")
        result = template.format(name="Alice")
        assert result == "Hello Alice!"

    def test_template_variables(self):
        """Test extracting variables."""
        template = PromptTemplate("Hello {name}, you are {age} years old")
        assert set(template.variables) == {"name", "age"}

    def test_missing_variable(self):
        """Test error on missing variable."""
        template = PromptTemplate("Hello {name}!")
        with pytest.raises(KeyError):
            template.format()

    def test_partial_template(self):
        """Test partial template substitution."""
        template = PromptTemplate("Hello {name} from {city}")
        partial = template.partial(city="Paris")
        assert partial.variables == ["name"]
        result = partial.format(name="Bob")
        assert result == "Hello Bob from Paris"

    def test_create_message(self):
        """Test creating message from template."""
        template = PromptTemplate("Hello {name}", role=Role.SYSTEM)
        msg = template.create_message(name="Alice")
        assert msg.role == Role.SYSTEM
        assert msg.content == "Hello Alice"


class TestPrompt:
    """Test Prompt builder."""

    def test_prompt_builder(self):
        """Test building prompts."""
        prompt = Prompt()
        prompt.add_system("You are helpful")
        prompt.add_user("Hello")
        prompt.add_assistant("Hi there")

        messages = prompt.build()
        assert len(messages) == 3
        assert messages[0].role == Role.SYSTEM
        assert messages[1].role == Role.USER
        assert messages[2].role == Role.ASSISTANT

    def test_prompt_with_template(self):
        """Test adding templated messages."""
        prompt = Prompt()
        prompt.add_template("Hello {name}", Role.USER, name="Alice")

        messages = prompt.build()
        assert len(messages) == 1
        assert messages[0].content == "Hello Alice"

    def test_from_template_static(self):
        """Test static method for single template."""
        messages = Prompt.from_template("Hello {name}", name="Bob")
        assert len(messages) == 1
        assert messages[0].content == "Hello Bob"


class TestSession:
    """Test Session class."""

    def test_session_creation(self):
        """Test creating a session."""
        session = Session()
        assert session.session_id is not None
        assert len(session) == 0

    def test_add_messages(self):
        """Test adding messages to session."""
        session = Session()
        session.add_user_message("Hello")
        session.add_assistant_message("Hi")

        assert len(session) == 2
        messages = session.get_messages()
        assert messages[0].role == Role.USER
        assert messages[1].role == Role.ASSISTANT

    def test_max_history(self):
        """Test history trimming."""
        session = Session(max_history=3)
        session.add_system_message("System")
        session.add_user_message("1")
        session.add_assistant_message("2")
        session.add_user_message("3")
        session.add_assistant_message("4")

        messages = session.get_messages()
        assert len(messages) == 3
        # System message should be kept
        assert messages[0].role == Role.SYSTEM

    def test_memory_storage(self):
        """Test session memory."""
        session = Session()
        session.set_memory("key1", "value1")
        session.update_memory({"key2": "value2", "key3": 123})

        assert session.get_memory("key1") == "value1"
        assert session.get_memory("key2") == "value2"
        assert session.get_memory("key3") == 123
        assert session.get_memory("missing", "default") == "default"

    def test_session_serialization(self):
        """Test session to/from dict."""
        session1 = Session()
        session1.add_user_message("Hello")
        session1.set_memory("key", "value")
        session1.metadata["test"] = True

        data = session1.to_dict()
        session2 = Session.from_dict(data)

        assert session2.session_id == session1.session_id
        assert len(session2) == 1
        assert session2.get_memory("key") == "value"
        assert session2.metadata["test"] is True


@patch("ailib.core.openai_client.OpenAI")
@patch("ailib.core.openai_client.AsyncOpenAI")
class TestOpenAIClient:
    """Test OpenAI client implementation."""

    def test_client_initialization(self, mock_async_openai, mock_openai):
        """Test initializing OpenAI client."""
        client = OpenAIClient(model="gpt-4", api_key="test-key")
        assert client.model == "gpt-4"
        mock_openai.assert_called_once()
        mock_async_openai.assert_called_once()

    def test_complete(self, mock_async_openai, mock_openai):
        """Test completion method."""
        # Mock response
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = "Test response"
        mock_response.choices[0].message.tool_calls = None
        mock_response.choices[0].finish_reason = "stop"
        mock_response.model = "gpt-4"
        mock_response.usage.prompt_tokens = 10
        mock_response.usage.completion_tokens = 5
        mock_response.usage.total_tokens = 15

        mock_openai.return_value.chat.completions.create.return_value = mock_response

        client = OpenAIClient(model="gpt-4")
        messages = [Message(role=Role.USER, content="Hello")]

        response = client.complete(messages)

        assert response.content == "Test response"
        assert response.model == "gpt-4"
        assert response.usage["total_tokens"] == 15
