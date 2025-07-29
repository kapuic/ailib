"""Tool system for agent capabilities."""

import inspect
import json
from dataclasses import dataclass, field
from functools import wraps
from typing import Any, Callable, Dict, List, Optional, Type, Union

from pydantic import BaseModel, Field, create_model


@dataclass
class Tool:
    """Represents a tool that agents can use."""
    
    name: str
    description: str
    func: Callable
    parameters: Optional[Type[BaseModel]] = None
    returns: Optional[Type] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_openai_function(self) -> Dict[str, Any]:
        """Convert tool to OpenAI function calling format.
        
        Returns:
            Dictionary in OpenAI function format
        """
        function_def = {
            "name": self.name,
            "description": self.description,
        }
        
        if self.parameters:
            # Convert Pydantic schema to OpenAI format
            schema = self.parameters.model_json_schema()
            # Remove title from schema
            schema.pop("title", None)
            function_def["parameters"] = schema
        else:
            function_def["parameters"] = {
                "type": "object",
                "properties": {},
                "required": []
            }
            
        return function_def

    def execute(self, **kwargs) -> Any:
        """Execute the tool with given arguments.
        
        Args:
            **kwargs: Tool arguments
            
        Returns:
            Tool execution result
        """
        if self.parameters:
            # Validate parameters using Pydantic
            params = self.parameters(**kwargs)
            # Convert to dict and call function
            return self.func(**params.model_dump())
        else:
            return self.func(**kwargs)

    def __call__(self, **kwargs) -> Any:
        """Make tool callable."""
        return self.execute(**kwargs)


class ToolRegistry:
    """Registry for managing available tools."""

    def __init__(self):
        """Initialize empty registry."""
        self._tools: Dict[str, Tool] = {}

    def register(self, tool: Tool) -> None:
        """Register a tool.
        
        Args:
            tool: Tool to register
            
        Raises:
            ValueError: If tool name already exists
        """
        if tool.name in self._tools:
            raise ValueError(f"Tool '{tool.name}' already registered")
        self._tools[tool.name] = tool

    def get(self, name: str) -> Optional[Tool]:
        """Get a tool by name.
        
        Args:
            name: Tool name
            
        Returns:
            Tool if found, None otherwise
        """
        return self._tools.get(name)

    def list_tools(self) -> List[str]:
        """List all registered tool names.
        
        Returns:
            List of tool names
        """
        return list(self._tools.keys())

    def get_all_tools(self) -> List[Tool]:
        """Get all registered tools.
        
        Returns:
            List of all tools
        """
        return list(self._tools.values())

    def to_openai_functions(self) -> List[Dict[str, Any]]:
        """Convert all tools to OpenAI function format.
        
        Returns:
            List of function definitions
        """
        return [tool.to_openai_function() for tool in self._tools.values()]

    def execute_tool(self, name: str, **kwargs) -> Any:
        """Execute a tool by name.
        
        Args:
            name: Tool name
            **kwargs: Tool arguments
            
        Returns:
            Tool execution result
            
        Raises:
            ValueError: If tool not found
        """
        tool = self.get(name)
        if not tool:
            raise ValueError(f"Tool '{name}' not found")
        return tool.execute(**kwargs)

    def __len__(self) -> int:
        """Get number of registered tools."""
        return len(self._tools)

    def __contains__(self, name: str) -> bool:
        """Check if tool is registered."""
        return name in self._tools


# Global registry instance
_global_registry = ToolRegistry()


def tool(
    func: Optional[Callable] = None,
    *,
    name: Optional[str] = None,
    description: Optional[str] = None,
    registry: Optional[ToolRegistry] = None,
    **metadata
) -> Union[Callable, Callable[[Callable], Callable]]:
    """Decorator to register a function as a tool.
    
    Args:
        func: Function to decorate (when used without parentheses)
        name: Tool name (defaults to function name)
        description: Tool description (defaults to function docstring)
        registry: Registry to use (defaults to global registry)
        **metadata: Additional metadata
        
    Returns:
        Decorated function or decorator
        
    Example:
        ```python
        @tool
        def search(query: str) -> str:
            '''Search the web for information.'''
            return f"Search results for: {query}"
            
        @tool(name="calculator", description="Perform calculations")
        def calc(expression: str) -> float:
            return eval(expression)
        ```
    """
    def decorator(func: Callable) -> Callable:
        # Extract tool name
        tool_name = name or func.__name__
        
        # Extract description
        tool_desc = description or inspect.getdoc(func) or f"Tool: {tool_name}"
        
        # Extract parameters from function signature
        sig = inspect.signature(func)
        params = {}
        required = []
        
        for param_name, param in sig.parameters.items():
            if param_name == "self":
                continue
                
            # Get type annotation
            param_type = param.annotation if param.annotation != inspect.Parameter.empty else Any
            
            # Get default value
            if param.default == inspect.Parameter.empty:
                required.append(param_name)
                params[param_name] = (param_type, Field(..., description=f"Parameter: {param_name}"))
            else:
                params[param_name] = (param_type, Field(default=param.default, description=f"Parameter: {param_name}"))
        
        # Create Pydantic model for parameters if any
        param_model = None
        if params:
            param_model = create_model(
                f"{tool_name}_params",
                **params
            )
        
        # Get return type
        return_type = sig.return_annotation if sig.return_annotation != inspect.Parameter.empty else Any
        
        # Create tool
        tool_instance = Tool(
            name=tool_name,
            description=tool_desc,
            func=func,
            parameters=param_model,
            returns=return_type,
            metadata=metadata
        )
        
        # Register tool
        target_registry = registry or _global_registry
        target_registry.register(tool_instance)
        
        # Add tool reference to function
        func._tool = tool_instance
        
        @wraps(func)
        def wrapper(*args, **kwargs):
            return func(*args, **kwargs)
        
        # Copy the _tool attribute to wrapper
        wrapper._tool = tool_instance
            
        return wrapper
    
    # Handle being called without parentheses (@tool)
    if func is not None:
        return decorator(func)
        
    # Handle being called with parentheses (@tool() or @tool(name="..."))
    return decorator


def get_global_registry() -> ToolRegistry:
    """Get the global tool registry.
    
    Returns:
        Global ToolRegistry instance
    """
    return _global_registry


# Built-in tools
@tool(description="Perform basic arithmetic calculations")
def calculator(expression: str) -> float:
    """Evaluate a mathematical expression.
    
    Args:
        expression: Mathematical expression to evaluate
        
    Returns:
        Result of the calculation
    """
    # Safe evaluation of mathematical expressions
    allowed_names = {
        k: v for k, v in math.__dict__.items() if not k.startswith("__")
    }
    allowed_names.update({"abs": abs, "round": round})
    
    try:
        # Remove any potentially dangerous characters
        safe_expr = expression.replace("__", "").replace("import", "").replace("exec", "")
        return eval(safe_expr, {"__builtins__": {}}, allowed_names)
    except Exception as e:
        raise ValueError(f"Invalid expression: {e}")


import math  # Import for calculator tool