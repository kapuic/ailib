"""Chain implementation for sequential prompt execution."""

import asyncio
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from ..core import LLMClient, Message, PromptTemplate, Role, Session


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
        # Direct initialization - no validation here
        self.llm = llm
        self.session = session or Session()
        self._steps: list[ChainStep] = []
        self._context: dict[str, Any] = {}

        # Store configuration values directly
        self.name = kwargs.get("name", "chain")
        self.description = kwargs.get("description", "")
        self.max_iterations = kwargs.get("max_iterations", 10)
        self.early_stopping = kwargs.get("early_stopping", True)
        self.retry_attempts = kwargs.get("retry_attempts", 3)
        self.retry_delay = kwargs.get("retry_delay", 1.0)
        self.timeout = kwargs.get("timeout")
        self._verbose = kwargs.get("verbose", False)

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
            f"chain_{self.name}",
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
def create_chain(
    *prompts: str,
    llm: LLMClient | None = None,
    model: str = "gpt-4",
    provider: str | None = None,
    **kwargs,
) -> Chain:
    """Create a simple chain from a sequence of prompts.

    This is the recommended way to create chains - simple and functional.
    Now supports multiple LLM providers!

    Args:
        *prompts: Sequence of prompts
        llm: LLM client to use (optional - will create one if not provided)
        model: Model to use if creating LLM client (default: gpt-4)
        provider: LLM provider (optional - auto-detected from model name)
        **kwargs: Additional options for chain or LLM client

    Returns:
        Configured Chain instance

    Example:
        # Simple chain with auto-created OpenAI client
        chain = create_chain(
            "Translate to Spanish: {text}",
            "Now make it more formal"
        )
        result = chain.run(text="Hello friend")

        # Chain with Claude
        chain = create_chain(
            "Summarize this text: {text}",
            "List 3 key points",
            model="claude-3-opus-20240229",
            api_key="your-anthropic-key"  # or set ANTHROPIC_API_KEY
        )

        # Explicit provider
        chain = create_chain(
            "Analyze: {text}",
            model="gpt-4",
            provider="openai"
        )

        # With custom client
        client = OpenAIClient(model="gpt-3.5-turbo")
        chain = create_chain("Summarize: {text}", llm=client)
    """
    # Create LLM client if not provided
    if llm is None:
        from ..core import create_client

        # Extract LLM-specific kwargs
        llm_kwargs = {}
        for key in ["api_key", "base_url", "timeout", "max_retries"]:
            if key in kwargs:
                llm_kwargs[key] = kwargs.pop(key)

        llm = create_client(model=model, provider=provider, **llm_kwargs)

    # Validate configuration before creating chain
    from .._validation import ChainConfig

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

    # Create chain with validated values
    chain = Chain(
        llm,
        name=config.name,
        description=config.description,
        max_iterations=config.max_iterations,
        early_stopping=config.early_stopping,
        retry_attempts=config.retry_attempts,
        retry_delay=config.retry_delay,
        timeout=config.timeout,
        verbose=config.verbose,
    )

    # Add prompts
    for prompt in prompts:
        chain.add_user(prompt)

    return chain
