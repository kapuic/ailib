"""Workflow state management."""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from datetime import datetime
from pathlib import Path
from typing import Any


class StateStore(ABC):
    """Abstract base class for state persistence."""

    @abstractmethod
    def save(self, workflow_id: str, state: dict[str, Any]) -> None:
        """Save workflow state."""
        pass

    @abstractmethod
    def load(self, workflow_id: str) -> dict[str, Any] | None:
        """Load workflow state."""
        pass

    @abstractmethod
    def delete(self, workflow_id: str) -> None:
        """Delete workflow state."""
        pass


class MemoryStateStore(StateStore):
    """In-memory state store (for testing)."""

    def __init__(self):
        self._store: dict[str, dict[str, Any]] = {}

    def save(self, workflow_id: str, state: dict[str, Any]) -> None:
        """Save state in memory."""
        self._store[workflow_id] = state.copy()

    def load(self, workflow_id: str) -> dict[str, Any] | None:
        """Load state from memory."""
        return self._store.get(workflow_id, {}).copy()

    def delete(self, workflow_id: str) -> None:
        """Delete state from memory."""
        self._store.pop(workflow_id, None)


class FileStateStore(StateStore):
    """File-based state store."""

    def __init__(self, directory: str = ".workflow_state"):
        self.directory = Path(directory)
        self.directory.mkdir(exist_ok=True)

    def save(self, workflow_id: str, state: dict[str, Any]) -> None:
        """Save state to file."""
        file_path = self.directory / f"{workflow_id}.json"

        # Add metadata
        state_with_meta = {
            "state": state,
            "saved_at": datetime.now().isoformat(),
            "workflow_id": workflow_id,
        }

        with open(file_path, "w") as f:
            json.dump(state_with_meta, f, indent=2, default=str)

    def load(self, workflow_id: str) -> dict[str, Any] | None:
        """Load state from file."""
        file_path = self.directory / f"{workflow_id}.json"

        if not file_path.exists():
            return None

        with open(file_path) as f:
            state_with_meta = json.load(f)

        return state_with_meta.get("state", {})

    def delete(self, workflow_id: str) -> None:
        """Delete state file."""
        file_path = self.directory / f"{workflow_id}.json"
        if file_path.exists():
            file_path.unlink()


class WorkflowState:
    """Manages workflow state with versioning and rollback."""

    def __init__(
        self,
        initial_state: dict[str, Any] | None = None,
        store: StateStore | None = None,
    ):
        self._current_state = initial_state or {}
        self._history: list[dict[str, Any]] = [self._current_state.copy()]
        self._store = store or MemoryStateStore()
        self._checkpoints: dict[str, dict[str, Any]] = {}

    @property
    def current(self) -> dict[str, Any]:
        """Get current state."""
        return self._current_state.copy()

    def get(self, key: str, default: Any = None) -> Any:
        """Get value from state."""
        return self._current_state.get(key, default)

    def set(self, key: str, value: Any) -> None:
        """Set value in state."""
        self._current_state[key] = value
        self._add_to_history()

    def update(self, updates: dict[str, Any]) -> None:
        """Update multiple values."""
        self._current_state.update(updates)
        self._add_to_history()

    def checkpoint(self, name: str) -> None:
        """Create a named checkpoint."""
        self._checkpoints[name] = self._current_state.copy()

    def restore_checkpoint(self, name: str) -> None:
        """Restore from checkpoint."""
        if name not in self._checkpoints:
            raise ValueError(f"Checkpoint '{name}' not found")

        self._current_state = self._checkpoints[name].copy()
        self._add_to_history()

    def rollback(self, steps: int = 1) -> None:
        """Rollback to previous state."""
        if steps >= len(self._history):
            steps = len(self._history) - 1

        if steps > 0:
            self._current_state = self._history[-steps - 1].copy()
            # Don't add to history to avoid loops

    def save(self, workflow_id: str) -> None:
        """Persist state to store."""
        self._store.save(
            workflow_id,
            {
                "current": self._current_state,
                "history": self._history,
                "checkpoints": self._checkpoints,
            },
        )

    def load(self, workflow_id: str) -> bool:
        """Load state from store."""
        data = self._store.load(workflow_id)
        if data:
            self._current_state = data.get("current", {})
            self._history = data.get("history", [self._current_state.copy()])
            self._checkpoints = data.get("checkpoints", {})
            return True
        return False

    def _add_to_history(self) -> None:
        """Add current state to history."""
        self._history.append(self._current_state.copy())

        # Limit history size
        if len(self._history) > 100:
            self._history = self._history[-50:]
