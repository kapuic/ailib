"""Tests for agents module."""

from unittest.mock import Mock

import pytest
from ailib import Agent, Tool, ToolRegistry, tool
from ailib.core import CompletionResponse


class TestTool:
    """Test Tool class."""

    def test_tool_creation(self):
        """Test creating a tool."""

        def my_func(x: int) -> int:
            return x * 2

        tool = Tool(name="double", description="Double a number", func=my_func)

        assert tool.name == "double"
        assert tool.description == "Double a number"
        assert tool.execute(x=5) == 10
        assert tool(x=5) == 10  # Test callable

    def test_tool_to_openai_function(self):
        """Test converting tool to OpenAI format."""

        def search(query: str) -> str:
            return f"Results for {query}"

        tool = Tool(name="search", description="Search the web", func=search)

        func_def = tool.to_openai_function()
        assert func_def["name"] == "search"
        assert func_def["description"] == "Search the web"
        assert "parameters" in func_def


class TestToolDecorator:
    """Test @tool decorator."""

    def test_simple_decorator(self):
        """Test simple tool decorator."""

        @tool
        def my_calc(expression: str) -> float:
            """Calculate math expression."""
            return eval(expression)

        # Check function still works
        assert my_calc("2 + 2") == 4

        # Check tool was created
        assert hasattr(my_calc, "_tool")
        tool_obj = my_calc._tool
        assert tool_obj.name == "my_calc"
        assert "Calculate math expression" in tool_obj.description

    def test_decorator_with_params(self):
        """Test decorator with custom parameters."""

        @tool(name="web_search", description="Search the internet")
        def search(query: str) -> str:
            return f"Results: {query}"

        tool_obj = search._tool
        assert tool_obj.name == "web_search"
        assert tool_obj.description == "Search the internet"

    def test_decorator_with_default_args(self):
        """Test decorator with function having default arguments."""

        @tool
        def greet(name: str, greeting: str = "Hello") -> str:
            """Greet someone."""
            return f"{greeting} {name}!"

        tool_obj = greet._tool
        assert tool_obj.parameters is not None

        # Test execution with default
        result = tool_obj.execute(name="Alice")
        assert result == "Hello Alice!"

        # Test execution with custom greeting
        result = tool_obj.execute(name="Bob", greeting="Hi")
        assert result == "Hi Bob!"


class TestToolRegistry:
    """Test ToolRegistry class."""

    def test_registry_operations(self):
        """Test basic registry operations."""
        registry = ToolRegistry()

        # Create tools
        tool1 = Tool("tool1", "First tool", lambda: "1")
        tool2 = Tool("tool2", "Second tool", lambda: "2")

        # Register tools
        registry.register(tool1)
        registry.register(tool2)

        assert len(registry) == 2
        assert "tool1" in registry
        assert registry.get("tool1") == tool1
        assert registry.list_tools() == ["tool1", "tool2"]

    def test_duplicate_registration(self):
        """Test error on duplicate tool name."""
        registry = ToolRegistry()
        tool1 = Tool("test", "Test tool", lambda: None)

        registry.register(tool1)

        with pytest.raises(ValueError, match="already registered"):
            registry.register(tool1)

    def test_execute_tool(self):
        """Test executing tool through registry."""
        registry = ToolRegistry()

        def add(a: int, b: int) -> int:
            return a + b

        tool = Tool("add", "Add numbers", add)
        registry.register(tool)

        result = registry.execute_tool("add", a=5, b=3)
        assert result == 8

    def test_execute_missing_tool(self):
        """Test error when executing missing tool."""
        registry = ToolRegistry()

        with pytest.raises(ValueError, match="not found"):
            registry.execute_tool("missing")


class TestAgent:
    """Test Agent class."""

    def test_agent_creation(self):
        """Test creating an agent."""
        agent = Agent()
        assert agent.max_steps == 10
        assert agent.verbose is False
        assert agent.tool_registry is not None

    def test_agent_with_tools(self):
        """Test adding tools to agent."""
        # Create a tool manually
        search_tool = Tool(
            name="search_test",
            description="Search tool",
            func=lambda query: "search results",
        )

        # Create agent and add tool
        agent = Agent(tools=[search_tool])

        assert "search_test" in agent.tool_registry

    def test_agent_with_decorated_functions(self):
        """Test adding decorated functions as tools (README pattern)."""

        # Create decorated functions like in README
        @tool
        def weather_func(city: str) -> str:
            """Get the weather for a city."""
            return f"The weather in {city} is sunny and 72°F"

        @tool
        def calc_func(expression: str) -> float:
            """Evaluate a mathematical expression."""
            return eval(expression)

        # Test passing decorated functions directly (as shown in README)
        agent = Agent(tools=[weather_func, calc_func])

        # Verify tools were registered correctly
        assert len(agent.tool_registry.get_all_tools()) == 2
        assert "weather_func" in agent.tool_registry
        assert "calc_func" in agent.tool_registry

        # Verify tools work
        weather_tool = agent.tool_registry.get("weather_func")
        assert (
            weather_tool.execute(city="Paris")
            == "The weather in Paris is sunny and 72°F"
        )

        calc_tool = agent.tool_registry.get("calc_func")
        assert calc_tool.execute(expression="2 + 3") == 5

    def test_agent_run(self):
        """Test running agent with mocked LLM."""
        # Create a custom registry
        custom_registry = ToolRegistry()

        # Create a simple tool
        @tool(registry=custom_registry)
        def calc_tool(expression: str) -> float:
            """Calculate math."""
            return eval(expression)

        # Mock LLM
        mock_llm = Mock()

        # Mock responses for ReAct loop
        responses = [
            # First response - use calculator
            CompletionResponse(
                content="""Thought: I need to calculate 2 + 3
Action: calc_tool
Action Input: {"expression": "2 + 3"}""",
                model="test",
                usage={
                    "prompt_tokens": 50,
                    "completion_tokens": 20,
                    "total_tokens": 70,
                },
            ),
            # Second response - final answer
            CompletionResponse(
                content="""Thought: The calculation result is 5
Action: Final Answer
Action Input: The answer is 5""",
                model="test",
                usage={
                    "prompt_tokens": 80,
                    "completion_tokens": 15,
                    "total_tokens": 95,
                },
            ),
        ]
        mock_llm.complete.side_effect = responses

        # Create and run agent
        agent = Agent(llm=mock_llm, verbose=False, tools=custom_registry)

        result = agent.run("What is 2 + 3?")

        assert result == "The answer is 5"
        assert mock_llm.complete.call_count == 2

    def test_agent_parse_response(self):
        """Test parsing agent responses."""
        agent = Agent()

        response = """Thought: I need to search for information
Action: search
Action Input: {"query": "AI agents"}"""

        parsed = agent._parse_response(response)

        assert parsed["thought"] == "I need to search for information"
        assert parsed["action"] == "search"
        assert parsed["action_input"] == '{"query": "AI agents"}'

    def test_agent_without_llm(self):
        """Test error when running agent without LLM."""
        agent = Agent()

        with pytest.raises(ValueError, match="No LLM client set"):
            agent.run("Test task")

    def test_agent_max_steps(self):
        """Test agent respects max steps."""
        # Mock LLM that never gives final answer
        mock_llm = Mock()
        mock_llm.complete.return_value = CompletionResponse(
            content="Thought: Thinking...\nAction: search\nAction Input: {}",
            model="test",
            usage={"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
        )

        agent = Agent(llm=mock_llm, max_steps=3)
        result = agent.run("Test")

        assert "couldn't complete" in result
        assert mock_llm.complete.call_count == 3


class TestCreateAgent:
    """Test create_agent factory function."""

    def test_create_agent_with_decorated_functions(self):
        """Test create_agent with decorated functions (README pattern)."""
        from ailib import create_agent

        # Create decorated functions as shown in README
        @tool
        def weather(city: str) -> str:
            """Get the weather for a city."""
            return f"The weather in {city} is sunny and 72°F"

        @tool
        def math_eval(expression: str) -> float:
            """Evaluate a mathematical expression."""
            return eval(expression)

        # Mock LLM to avoid API key requirement
        mock_llm = Mock()
        mock_llm.model = "gpt-4"
        mock_llm.complete.return_value = CompletionResponse(
            content="Test",
            model="gpt-4",
            usage={"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
        )

        # Test the exact README pattern
        agent = create_agent(
            "assistant",
            tools=[weather, math_eval],
            model="gpt-4",
            llm=mock_llm,  # Pass mock to avoid API key issues
        )

        # Verify tools were registered
        assert len(agent.tool_registry.get_all_tools()) == 2
        assert "weather" in agent.tool_registry
        assert "math_eval" in agent.tool_registry
