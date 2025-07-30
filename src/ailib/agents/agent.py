"""ReAct-style agent implementation."""

import json
import re
from collections.abc import Callable
from typing import Any

from ..core import LLMClient, Message, Role, Session
from .tools import Tool, ToolRegistry, get_global_registry


class Agent:
    """Autonomous agent that can use tools to accomplish tasks."""

    def __init__(
        self,
        llm: LLMClient | None = None,
        tools: list[Tool] | ToolRegistry | None = None,
        max_steps: int = 10,
        verbose: bool = False,
        **kwargs,
    ):
        """Initialize agent.

        Args:
            llm: LLM client to use
            tools: List of tools or ToolRegistry
            max_steps: Maximum number of reasoning steps
            verbose: Enable verbose output
            **kwargs: Additional agent configuration
        """
        # Direct initialization - no validation here
        self.llm = llm
        self.max_steps = max_steps
        self.verbose = verbose

        # Store configuration values directly
        self.name = kwargs.get("name", "agent")
        self.description = kwargs.get("description", "")
        self.model = llm.model if llm else kwargs.get("model", "gpt-4")
        self.system_prompt = kwargs.get("system_prompt")
        self.temperature = kwargs.get("temperature", 0.7)
        self.memory_size = kwargs.get("memory_size", 10)
        self.return_intermediate_steps = kwargs.get("return_intermediate_steps", False)

        # Set up tool registry
        if tools is None:
            self.tool_registry = get_global_registry()
        elif isinstance(tools, ToolRegistry):
            self.tool_registry = tools
        else:
            self.tool_registry = ToolRegistry()
            for tool in tools:
                self.tool_registry.register(tool)

    def with_tools(self, *tools: Tool | Callable) -> "Agent":
        """Add tools to the agent.

        Args:
            *tools: Tools to add (Tool instances or decorated functions)

        Returns:
            Self for chaining
        """
        for tool in tools:
            if hasattr(tool, "_tool"):
                # Decorated function
                self.tool_registry.register(tool._tool)
            elif isinstance(tool, Tool):
                self.tool_registry.register(tool)
            else:
                raise ValueError(f"Invalid tool type: {type(tool)}")
        return self

    def _create_system_prompt(self) -> str:
        """Create ReAct system prompt.

        Returns:
            System prompt string
        """
        # Get tool descriptions
        tool_descriptions = []
        for tool in self.tool_registry.get_all_tools():
            tool_descriptions.append(f"- {tool.name}: {tool.description}")

        tools_text = (
            "\n".join(tool_descriptions) if tool_descriptions else "No tools available."
        )

        return (
            "You are a helpful AI assistant that can use tools "
            "to answer questions and complete tasks.\n\n"
            f"Available tools:\n{tools_text}\n\n"
            "You should follow this format for each step:\n\n"
            "Thought: [Your reasoning about what to do next]\n"
            "Action: [Tool name to use, or 'Final Answer' when done]\n"
            "Action Input: [Input for the tool as JSON, or final answer text]\n\n"
            "The user will provide the observation after each tool use.\n"
            "Continue this process until you have enough information to provide "
            "a final answer.\n\n"
            "Important:\n"
            "- Always start with a Thought\n"
            "- Action must be either a tool name or 'Final Answer'\n"
            "- Action Input for tools must be valid JSON\n"
            "- When you have the answer, use Action: Final Answer"
        )

    def _parse_response(self, response: str) -> dict[str, Any]:
        """Parse agent response to extract thought, action, and input.

        Args:
            response: LLM response

        Returns:
            Dictionary with parsed components
        """
        # Extract components using regex
        thought_match = re.search(r"Thought:\s*(.+?)(?=Action:|$)", response, re.DOTALL)
        action_match = re.search(
            r"Action:\s*(.+?)(?=Action Input:|$)", response, re.DOTALL
        )
        input_match = re.search(
            r"Action Input:\s*(.+?)(?=Thought:|$)", response, re.DOTALL
        )

        result = {
            "thought": thought_match.group(1).strip() if thought_match else "",
            "action": action_match.group(1).strip() if action_match else "",
            "action_input": input_match.group(1).strip() if input_match else "",
        }

        return result

    def _execute_tool(self, tool_name: str, tool_input: str) -> str:
        """Execute a tool and return the result.

        Args:
            tool_name: Name of the tool
            tool_input: JSON string of tool arguments

        Returns:
            Tool execution result as string
        """
        try:
            # Parse JSON input
            if tool_input.strip():
                try:
                    args = json.loads(tool_input)
                except json.JSONDecodeError:
                    # Try to parse as a simple string argument
                    args = {"input": tool_input}
            else:
                args = {}

            # Execute tool
            result = self.tool_registry.execute_tool(tool_name, **args)
            return str(result)

        except Exception as e:
            return f"Error executing tool '{tool_name}': {str(e)}"

    def run(self, task: str, session: Session | None = None) -> str:
        """Run the agent to complete a task.

        Args:
            task: Task or question to complete
            session: Optional session for conversation history

        Returns:
            Final answer from the agent

        Raises:
            ValueError: If no LLM client is set
        """
        if not self.llm:
            raise ValueError("No LLM client set. Initialize with llm parameter.")

        # Import tracing
        from ..tracing._core import start_trace, trace_step

        # Create session if not provided
        if session is None:
            session = Session()

        # Start trace for agent execution
        start_trace(
            f"agent_{self.name}",
            task=task,
            tools=self.tool_registry.list_tools(),
        )

        # Add system prompt
        messages = [Message(role=Role.SYSTEM, content=self._create_system_prompt())]

        # Add task
        messages.append(Message(role=Role.USER, content=task))

        # Track steps
        steps = 0

        while steps < self.max_steps:
            steps += 1

            with trace_step(
                f"agent_step_{steps}",
                step_type="agent_reasoning",
                step_number=steps,
            ):
                if self.verbose:
                    print(f"\n--- Step {steps} ---")

                # Get LLM response
                response = self.llm.complete(messages=messages)
                content = response.content

                if self.verbose:
                    print(f"Assistant: {content}")

                # Add assistant response to messages
                messages.append(Message(role=Role.ASSISTANT, content=content))

                # Parse response
                parsed = self._parse_response(content)

                # Check if agent wants to give final answer
                if parsed["action"].lower() == "final answer":
                    with trace_step(
                        "agent_final_answer",
                        step_type="agent_output",
                        answer=parsed["action_input"][:200],  # Truncate for trace
                    ):
                        return parsed["action_input"]

                # Execute tool if action is specified
                if parsed["action"]:
                    tool_name = parsed["action"]
                    tool_input = parsed["action_input"]

                    if self.verbose:
                        print(f"\nExecuting tool: {tool_name}")
                        print(f"Input: {tool_input}")

                    # Execute tool (already traced by Tool.execute)
                    observation = self._execute_tool(tool_name, tool_input)

                    if self.verbose:
                        print(f"Observation: {observation}")

                    # Add observation to messages
                    observation_msg = f"Observation: {observation}"
                    messages.append(Message(role=Role.USER, content=observation_msg))
                else:
                    # No action specified, prompt for action
                    messages.append(
                        Message(
                            role=Role.USER,
                            content=(
                                "Please specify an Action (tool name or "
                                "'Final Answer') and Action Input."
                            ),
                        )
                    )

        # Max steps reached
        return (
            "I couldn't complete the task within the maximum number of steps. "
            "Please try breaking it down into smaller tasks."
        )


# Factory function for simplified agent creation
def create_agent(
    name: str = "assistant",
    model: str = "gpt-4",
    provider: str | None = None,
    instructions: str | None = None,
    tools: list[Tool | Callable] | None = None,
    verbose: bool = False,
    **kwargs,
) -> Agent:
    """Create an agent with simplified configuration.

    This is the recommended way to create agents - simple and functional.
    Now supports multiple LLM providers!

    Args:
        name: Agent name/identifier
        model: LLM model to use (default: gpt-4)
        provider: LLM provider (optional - auto-detected from model name)
        instructions: Custom system instructions
        tools: List of tools (Tool instances or @tool decorated functions)
        verbose: Enable verbose output
        **kwargs: Additional options (temperature, max_steps, api_key, etc.)

    Returns:
        Configured Agent instance ready to use

    Example:
        # Simple agent with OpenAI (auto-detected)
        agent = create_agent("assistant")
        result = agent.run("What's the weather?")

        # Agent with Claude
        agent = create_agent(
            "assistant",
            model="claude-3-opus-20240229",
            api_key="your-anthropic-key"  # or set ANTHROPIC_API_KEY
        )

        # Explicit provider
        agent = create_agent(
            "assistant",
            model="gpt-4",
            provider="openai"
        )

        # With tools
        @tool
        def search(query: str) -> str:
            return f"Results for: {query}"

        agent = create_agent(
            "researcher",
            tools=[search],
            instructions="You are a helpful research assistant."
        )
    """
    from ..core import create_client

    # Extract LLM-specific kwargs
    llm_kwargs = {}
    agent_kwargs = {"name": name}

    # Common LLM parameters
    for key in ["api_key", "base_url", "timeout", "max_retries"]:
        if key in kwargs:
            llm_kwargs[key] = kwargs.pop(key)

    # Agent-specific parameters
    for key in ["max_steps", "temperature", "memory_size", "return_intermediate_steps"]:
        if key in kwargs:
            agent_kwargs[key] = kwargs.pop(key)

    # Create LLM client if not provided
    llm = kwargs.pop("llm", None)
    if llm is None:
        llm = create_client(model=model, provider=provider, **llm_kwargs)

    # Set custom instructions if provided
    if instructions:
        agent_kwargs["system_prompt"] = instructions

    # Any remaining kwargs go to agent
    agent_kwargs.update(kwargs)

    # Validate configuration before creating agent
    from .._validation import AgentConfig

    config = AgentConfig(
        name=agent_kwargs.get("name", "assistant"),
        description=agent_kwargs.get("description", ""),
        model=model,
        system_prompt=agent_kwargs.get("system_prompt"),
        tools=[],  # Validated separately
        max_iterations=agent_kwargs.get("max_steps", 10),
        temperature=agent_kwargs.get("temperature", 0.7),
        verbose=verbose,
        memory_size=agent_kwargs.get("memory_size", 10),
        return_intermediate_steps=agent_kwargs.get("return_intermediate_steps", False),
    )

    # Create agent with validated values
    agent = Agent(
        llm=llm,
        tools=tools,
        max_steps=config.max_iterations,
        verbose=config.verbose,
        name=config.name,
        description=config.description,
        temperature=config.temperature,
        memory_size=config.memory_size,
        return_intermediate_steps=config.return_intermediate_steps,
        system_prompt=config.system_prompt,
    )

    return agent
