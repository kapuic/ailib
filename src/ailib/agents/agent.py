"""ReAct-style agent implementation."""

import json
import re
from collections.abc import Callable
from typing import Any

from ..core import LLMClient, Message, Role, Session
from ..validation import AgentConfig
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
        # Validate configuration
        config = AgentConfig(
            name=kwargs.get("name", "agent"),
            description=kwargs.get("description", ""),
            model=llm.model if llm else kwargs.get("model", "gpt-4"),
            system_prompt=kwargs.get("system_prompt"),
            tools=[],  # Will be filled based on actual tools
            max_iterations=max_steps,
            temperature=kwargs.get("temperature", 0.7),
            verbose=verbose,
            memory_size=kwargs.get("memory_size", 10),
            return_intermediate_steps=kwargs.get("return_intermediate_steps", False),
        )

        self.llm = llm
        self.max_steps = config.max_iterations
        self.verbose = config.verbose
        self._config = config

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

        # Create session if not provided
        if session is None:
            session = Session()

        # Add system prompt
        messages = [Message(role=Role.SYSTEM, content=self._create_system_prompt())]

        # Add task
        messages.append(Message(role=Role.USER, content=task))

        # Track steps
        steps = 0

        while steps < self.max_steps:
            steps += 1

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
                return parsed["action_input"]

            # Execute tool if action is specified
            if parsed["action"]:
                tool_name = parsed["action"]
                tool_input = parsed["action_input"]

                if self.verbose:
                    print(f"\nExecuting tool: {tool_name}")
                    print(f"Input: {tool_input}")

                # Execute tool
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

    async def arun(self, task: str, session: Session | None = None) -> str:
        """Async version of run.

        Args:
            task: Task or question to complete
            session: Optional session for conversation history

        Returns:
            Final answer from the agent
        """
        if not self.llm:
            raise ValueError("No LLM client set. Initialize with llm parameter.")

        # Create session if not provided
        if session is None:
            session = Session()

        # Add system prompt
        messages = [Message(role=Role.SYSTEM, content=self._create_system_prompt())]

        # Add task
        messages.append(Message(role=Role.USER, content=task))

        # Track steps
        steps = 0

        while steps < self.max_steps:
            steps += 1

            if self.verbose:
                print(f"\n--- Step {steps} ---")

            # Get LLM response
            response = await self.llm.acomplete(messages=messages)
            content = response.content

            if self.verbose:
                print(f"Assistant: {content}")

            # Add assistant response to messages
            messages.append(Message(role=Role.ASSISTANT, content=content))

            # Parse response
            parsed = self._parse_response(content)

            # Check if agent wants to give final answer
            if parsed["action"].lower() == "final answer":
                return parsed["action_input"]

            # Execute tool if action is specified
            if parsed["action"]:
                tool_name = parsed["action"]
                tool_input = parsed["action_input"]

                if self.verbose:
                    print(f"\nExecuting tool: {tool_name}")
                    print(f"Input: {tool_input}")

                # Execute tool (sync for now - could be made async)
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
