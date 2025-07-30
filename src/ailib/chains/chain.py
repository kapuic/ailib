"""Chain implementation for sequential prompt execution."""

import asyncio
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from ..core import LLMClient, Message, PromptTemplate, Role, Session
from ..validation import ChainConfig


@dataclass
class ChainStep:
    """Represents a single step in a chain."""

    name: str
    prompt: str | PromptTemplate | list[Message]
    role: Role = Role.USER
    processor: Callable[[str], Any] | None = None
    temperature: float = 0.7
    max_tokens: int | None = None


class Chain:
    """Fluent API for chaining LLM calls."""

    def __init__(
        self, llm: LLMClient | None = None, session: Session | None = None, **kwargs
    ):
        """Initialize a new chain.

        Args:
            llm: LLM client to use (can be set later)
            session: Session for state management
            **kwargs: Additional chain configuration
        """
        # Validate configuration
        config = ChainConfig(
            name=kwargs.get("name", "chain"),
            description=kwargs.get("description", ""),
            max_iterations=kwargs.get("max_iterations", 10),
            early_stopping=kwargs.get("early_stopping", True),
            retry_attempts=kwargs.get("retry_attempts", 3),
            retry_delay=kwargs.get("retry_delay", 1.0),
            timeout=kwargs.get("timeout"),
            verbose=kwargs.get("verbose", False),
        )

        self.llm = llm
        self.session = session or Session()
        self._steps: list[ChainStep] = []
        self._context: dict[str, Any] = {}
        self._config = config
        self._verbose = config.verbose

    def with_llm(self, llm: LLMClient) -> "Chain":
        """Set the LLM client.

        Args:
            llm: LLM client instance

        Returns:
            Self for chaining
        """
        self.llm = llm
        return self

    def with_session(self, session: Session) -> "Chain":
        """Set the session.

        Args:
            session: Session instance

        Returns:
            Self for chaining
        """
        self.session = session
        return self

    def verbose(self, enabled: bool = True) -> "Chain":
        """Enable verbose mode for debugging.

        Args:
            enabled: Whether to enable verbose output

        Returns:
            Self for chaining
        """
        self._verbose = enabled
        return self

    def add_prompt(
        self,
        prompt: str | PromptTemplate,
        name: str | None = None,
        role: Role = Role.USER,
        processor: Callable[[str], Any] | None = None,
        temperature: float = 0.7,
        max_tokens: int | None = None,
    ) -> "Chain":
        """Add a prompt step to the chain.

        Args:
            prompt: Prompt template or string
            name: Step name (auto-generated if not provided)
            role: Message role
            processor: Function to process the response
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate

        Returns:
            Self for chaining
        """
        if name is None:
            name = f"step_{len(self._steps) + 1}"

        step = ChainStep(
            name=name,
            prompt=prompt,
            role=role,
            processor=processor,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        self._steps.append(step)
        return self

    def add_system(self, content: str, name: str | None = None) -> "Chain":
        """Add a system message step.

        Args:
            content: System message content
            name: Step name

        Returns:
            Self for chaining
        """
        return self.add_prompt(content, name=name, role=Role.SYSTEM)

    def add_user(self, content: str, name: str | None = None, **kwargs) -> "Chain":
        """Add a user message step.

        Args:
            content: User message content
            name: Step name
            **kwargs: Additional step parameters

        Returns:
            Self for chaining
        """
        return self.add_prompt(content, name=name, role=Role.USER, **kwargs)

    def set_context(self, **kwargs) -> "Chain":
        """Set context variables for template substitution.

        Args:
            **kwargs: Context variables

        Returns:
            Self for chaining
        """
        self._context.update(kwargs)
        return self

    def _prepare_messages(self, step: ChainStep) -> list[Message]:
        """Prepare messages for a step.

        Args:
            step: Chain step

        Returns:
            List of messages
        """
        # Get conversation history from session
        messages = self.session.get_messages()

        # Handle different prompt types
        if isinstance(step.prompt, list):
            # Already a list of messages
            messages.extend(step.prompt)
        elif isinstance(step.prompt, PromptTemplate):
            # Template that needs formatting
            message = step.prompt.create_message(**self._context)
            messages.append(message)
        else:
            # String prompt
            if "{" in step.prompt and "}" in step.prompt:
                # Looks like a template
                template = PromptTemplate(step.prompt, step.role)
                message = template.create_message(**self._context)
            else:
                # Plain string
                message = Message(role=step.role, content=step.prompt)
            messages.append(message)

        return messages

    def run(self, **kwargs) -> Any:
        """Execute the chain synchronously.

        Args:
            **kwargs: Additional context variables

        Returns:
            Result of the last step (or processed result)

        Raises:
            ValueError: If no LLM client is set
        """
        if not self.llm:
            raise ValueError("No LLM client set. Use with_llm() to set one.")

        # Import tracing
        from ..tracing._core import trace_step

        # Update context with runtime variables
        self._context.update(kwargs)

        last_result = None

        # Trace the entire chain execution
        with trace_step(
            f"chain_{self._config.name}",
            step_type="chain",
            total_steps=len(self._steps),
        ):
            for step in self._steps:
                # Trace individual chain step
                with trace_step(
                    f"chain_step_{step.name}",
                    step_type="chain_step",
                    step_name=step.name,
                    temperature=step.temperature,
                ):
                    # Prepare messages
                    messages = self._prepare_messages(step)

                    if self._verbose:
                        print(f"\n[{step.name}] Sending {len(messages)} messages...")
                        if messages:
                            print(f"Last message: {messages[-1].content[:100]}...")

                    # Call LLM
                    response = self.llm.complete(
                        messages=messages,
                        temperature=step.temperature,
                        max_tokens=step.max_tokens,
                    )

                    # Extract content
                    result = response.content

                    if self._verbose:
                        print(f"[{step.name}] Response: {result[:200]}...")

                    # Add to session history
                    if messages and messages[-1] not in self.session.get_messages():
                        self.session.add_message(messages[-1])
                    self.session.add_assistant_message(result)

                    # Process result if processor provided
                    if step.processor:
                        with trace_step(
                            f"processor_{step.name}",
                            step_type="processor",
                        ):
                            result = step.processor(result)

                    # Store result in context for next steps
                    self._context[step.name] = result
                    last_result = result

        return last_result

    async def arun(self, **kwargs) -> Any:
        """Execute the chain asynchronously.

        Args:
            **kwargs: Additional context variables

        Returns:
            Result of the last step (or processed result)

        Raises:
            ValueError: If no LLM client is set
        """
        if not self.llm:
            raise ValueError("No LLM client set. Use with_llm() to set one.")

        # Update context with runtime variables
        self._context.update(kwargs)

        last_result = None

        for step in self._steps:
            # Prepare messages
            messages = self._prepare_messages(step)

            if self._verbose:
                print(f"\n[{step.name}] Sending {len(messages)} messages...")
                if messages:
                    print(f"Last message: {messages[-1].content[:100]}...")

            # Call LLM
            response = await self.llm.acomplete(
                messages=messages,
                temperature=step.temperature,
                max_tokens=step.max_tokens,
            )

            # Extract content
            result = response.content

            if self._verbose:
                print(f"[{step.name}] Response: {result[:200]}...")

            # Add to session history
            if messages and messages[-1] not in self.session.get_messages():
                self.session.add_message(messages[-1])
            self.session.add_assistant_message(result)

            # Process result if processor provided
            if step.processor:
                # Handle async processors
                if asyncio.iscoroutinefunction(step.processor):
                    result = await step.processor(result)
                else:
                    result = step.processor(result)

            # Store result in context for next steps
            self._context[step.name] = result
            last_result = result

        return last_result

    def reset(self) -> "Chain":
        """Reset the chain state.

        Returns:
            Self for chaining
        """
        self._steps.clear()
        self._context.clear()
        return self

    def __len__(self) -> int:
        """Get number of steps in chain."""
        return len(self._steps)

    def __repr__(self) -> str:
        """String representation of chain."""
        return (
            f"Chain(steps={len(self._steps)}, "
            f"context_keys={list(self._context.keys())})"
        )


# Convenience function for quick chain creation
def create_chain(llm: LLMClient, *prompts: str) -> Chain:
    """Create a simple chain from a sequence of prompts.

    Args:
        llm: LLM client to use
        *prompts: Sequence of prompts

    Returns:
        Configured Chain instance
    """
    chain = Chain(llm)
    for prompt in prompts:
        chain.add_user(prompt)
    return chain
