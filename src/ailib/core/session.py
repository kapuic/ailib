"""Session management for maintaining conversation state and memory."""

from datetime import datetime, timezone
from typing import Any
from uuid import uuid4

from .llm_client import Message, Role


class Session:
    """Manages conversation state and memory across interactions."""

    def __init__(
        self, session_id: str | None = None, max_history: int | None = None, **kwargs
    ):
        """Initialize a new session.

        Args:
            session_id: Unique session identifier (auto-generated if not provided)
            max_history: Maximum number of messages to keep in history
            **kwargs: Additional session configuration
        """
        # Direct initialization - no validation here
        self.session_id = session_id or str(uuid4())
        self.max_history = max_history or kwargs.get("max_messages", 100)
        self.created_at = datetime.now(timezone.utc)

        # Store configuration values directly
        self.ttl = kwargs.get("ttl")
        self.auto_save = kwargs.get("auto_save", False)
        self.save_path = kwargs.get("save_path")
        self.metadata: dict[str, Any] = kwargs.get("metadata", {})

        # Conversation history
        self._messages: list[Message] = []

        # Memory storage for key-value pairs
        self._memory: dict[str, Any] = {}

        # Trace of all operations
        self._trace: list[dict[str, Any]] = []

    def add_message(self, message: Message) -> None:
        """Add a message to the conversation history.

        Args:
            message: Message to add
        """
        self._messages.append(message)

        # Trim history if needed
        if self.max_history and len(self._messages) > self.max_history:
            # Keep system messages and trim old messages
            system_messages = [m for m in self._messages if m.role == Role.SYSTEM]
            other_messages = [m for m in self._messages if m.role != Role.SYSTEM]

            # Keep last N messages plus system messages
            keep_count = self.max_history - len(system_messages)
            self._messages = system_messages + other_messages[-keep_count:]

    def add_user_message(self, content: str, **kwargs) -> None:
        """Add a user message to history.

        Args:
            content: Message content
            **kwargs: Additional message attributes
        """
        message = Message(role=Role.USER, content=content, **kwargs)
        self.add_message(message)

    def add_assistant_message(self, content: str, **kwargs) -> None:
        """Add an assistant message to history.

        Args:
            content: Message content
            **kwargs: Additional message attributes
        """
        message = Message(role=Role.ASSISTANT, content=content, **kwargs)
        self.add_message(message)

    def add_system_message(self, content: str, **kwargs) -> None:
        """Add a system message to history.

        Args:
            content: Message content
            **kwargs: Additional message attributes
        """
        message = Message(role=Role.SYSTEM, content=content, **kwargs)
        self.add_message(message)

    def get_messages(self, include_system: bool = True) -> list[Message]:
        """Get conversation history.

        Args:
            include_system: Whether to include system messages

        Returns:
            List of messages
        """
        if include_system:
            return self._messages.copy()
        return [m for m in self._messages if m.role != Role.SYSTEM]

    def clear_messages(self, keep_system: bool = True) -> None:
        """Clear conversation history.

        Args:
            keep_system: Whether to keep system messages
        """
        if keep_system:
            self._messages = [m for m in self._messages if m.role == Role.SYSTEM]
        else:
            self._messages = []

    def set_memory(self, key: str, value: Any) -> None:
        """Store a value in session memory.

        Args:
            key: Memory key
            value: Value to store
        """
        self._memory[key] = value

    def get_memory(self, key: str, default: Any = None) -> Any:
        """Retrieve a value from session memory.

        Args:
            key: Memory key
            default: Default value if key not found

        Returns:
            Stored value or default
        """
        return self._memory.get(key, default)

    def update_memory(self, updates: dict[str, Any]) -> None:
        """Update multiple memory values.

        Args:
            updates: Dictionary of key-value pairs to update
        """
        self._memory.update(updates)

    def clear_memory(self) -> None:
        """Clear all session memory."""
        self._memory.clear()

    def add_trace(self, event_type: str, data: dict[str, Any]) -> None:
        """Add an event to the trace log.

        Args:
            event_type: Type of event (e.g., 'llm_call', 'tool_use')
            data: Event data
        """
        trace_entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "event_type": event_type,
            "data": data,
        }
        self._trace.append(trace_entry)

    def get_trace(self) -> list[dict[str, Any]]:
        """Get the full trace log.

        Returns:
            List of trace entries
        """
        return self._trace.copy()

    def to_dict(self) -> dict[str, Any]:
        """Convert session to dictionary format.

        Returns:
            Dictionary representation of session
        """
        return {
            "session_id": self.session_id,
            "created_at": self.created_at.isoformat(),
            "messages": [msg.to_dict() for msg in self._messages],
            "memory": self._memory.copy(),
            "metadata": self.metadata.copy(),
            "trace": self._trace.copy(),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Session":
        """Create session from dictionary.

        Args:
            data: Dictionary representation

        Returns:
            Restored Session instance
        """
        session = cls(session_id=data["session_id"])
        session.created_at = datetime.fromisoformat(data["created_at"])

        # Restore messages
        for msg_data in data.get("messages", []):
            role = Role(msg_data["role"])
            message = Message(
                role=role,
                content=msg_data["content"],
                name=msg_data.get("name"),
                tool_calls=msg_data.get("tool_calls"),
                tool_call_id=msg_data.get("tool_call_id"),
            )
            session._messages.append(message)

        # Restore memory and metadata
        session._memory = data.get("memory", {})
        session.metadata = data.get("metadata", {})
        session._trace = data.get("trace", [])

        return session

    def __len__(self) -> int:
        """Get number of messages in session."""
        return len(self._messages)

    def __repr__(self) -> str:
        """String representation of session."""
        return (
            f"Session(id={self.session_id}, "
            f"messages={len(self._messages)}, "
            f"memory_keys={list(self._memory.keys())})"
        )


# Factory function for simplified session creation
def create_session(
    session_id: str | None = None,
    max_messages: int = 100,
    ttl: int | None = None,
    auto_save: bool = False,
    save_path: str | None = None,
    **kwargs,
) -> Session:
    """Create a session with simplified configuration.

    This is the recommended way to create sessions - simple and functional.

    Args:
        session_id: Unique session ID (auto-generated if not provided)
        max_messages: Maximum messages to keep in history (default: 100)
        ttl: Time-to-live in seconds
        auto_save: Enable automatic session saving
        save_path: Path for session persistence (required if auto_save=True)
        **kwargs: Additional options (metadata, etc.)

    Returns:
        Configured Session instance ready to use

    Example:
        # Simple session
        session = create_session()

        # Session with auto-save
        session = create_session(
            auto_save=True,
            save_path="/tmp/sessions",
            ttl=3600
        )

        # Named session with metadata
        session = create_session(
            session_id="user-123",
            metadata={"user_id": "123", "context": "support"}
        )
    """
    from .._validation import SessionConfig

    # Validate using internal config
    config = SessionConfig(
        session_id=session_id or str(uuid4()),
        max_messages=max_messages,
        ttl=ttl,
        metadata=kwargs.get("metadata", {}),
        auto_save=auto_save,
        save_path=save_path,
    )

    # Create session with validated values
    return Session(
        session_id=config.session_id,
        max_history=config.max_messages,
        ttl=config.ttl,
        auto_save=config.auto_save,
        save_path=config.save_path,
        metadata=config.metadata,
    )
