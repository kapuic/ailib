"""Prompt templating system for dynamic prompt generation."""

import re
from string import Template
from typing import Any, Dict, List, Optional, Union

from .llm_client import Message, Role


class PromptTemplate:
    """Simple and powerful prompt template with variable substitution."""

    def __init__(self, template: str, role: Role = Role.USER):
        """Initialize prompt template.
        
        Args:
            template: Template string with {variable} placeholders
            role: Default role for messages created from this template
        """
        self.template = template
        self.role = role
        self._variables = self._extract_variables(template)

    def _extract_variables(self, template: str) -> List[str]:
        """Extract variable names from template.
        
        Args:
            template: Template string
            
        Returns:
            List of variable names found in template
        """
        # Find all {variable} patterns
        pattern = r'\{(\w+)\}'
        return list(set(re.findall(pattern, template)))

    def format(self, **kwargs) -> str:
        """Format template with provided variables.
        
        Args:
            **kwargs: Variable values
            
        Returns:
            Formatted string
            
        Raises:
            KeyError: If required variable is missing
        """
        # Check for missing variables
        missing = set(self._variables) - set(kwargs.keys())
        if missing:
            raise KeyError(f"Missing required variables: {missing}")
            
        # Format using simple string format
        return self.template.format(**kwargs)

    def create_message(self, **kwargs) -> Message:
        """Create a Message instance from template.
        
        Args:
            **kwargs: Variable values
            
        Returns:
            Message with formatted content
        """
        content = self.format(**kwargs)
        return Message(role=self.role, content=content)

    @property
    def variables(self) -> List[str]:
        """Get list of variables in template."""
        return self._variables.copy()

    def partial(self, **kwargs) -> "PromptTemplate":
        """Create a new template with some variables pre-filled.
        
        Args:
            **kwargs: Variables to pre-fill
            
        Returns:
            New PromptTemplate with partial substitution
        """
        # Perform partial substitution
        partial_template = self.template
        for key, value in kwargs.items():
            if key in self._variables:
                partial_template = partial_template.replace(f"{{{key}}}", str(value))
                
        return PromptTemplate(partial_template, self.role)


class Prompt:
    """Advanced prompt builder with multiple message support."""

    def __init__(self):
        """Initialize empty prompt."""
        self._messages: List[Message] = []

    def add_message(self, role: Role, content: str, **kwargs) -> "Prompt":
        """Add a message to the prompt.
        
        Args:
            role: Message role
            content: Message content
            **kwargs: Additional message attributes
            
        Returns:
            Self for chaining
        """
        message = Message(role=role, content=content, **kwargs)
        self._messages.append(message)
        return self

    def add_system(self, content: str) -> "Prompt":
        """Add a system message.
        
        Args:
            content: System message content
            
        Returns:
            Self for chaining
        """
        return self.add_message(Role.SYSTEM, content)

    def add_user(self, content: str) -> "Prompt":
        """Add a user message.
        
        Args:
            content: User message content
            
        Returns:
            Self for chaining
        """
        return self.add_message(Role.USER, content)

    def add_assistant(self, content: str) -> "Prompt":
        """Add an assistant message.
        
        Args:
            content: Assistant message content
            
        Returns:
            Self for chaining
        """
        return self.add_message(Role.ASSISTANT, content)

    def add_template(self, template: Union[str, PromptTemplate], role: Role = Role.USER, **kwargs) -> "Prompt":
        """Add a templated message.
        
        Args:
            template: Template string or PromptTemplate instance
            role: Message role (if template is string)
            **kwargs: Template variables
            
        Returns:
            Self for chaining
        """
        if isinstance(template, str):
            template = PromptTemplate(template, role)
            
        message = template.create_message(**kwargs)
        self._messages.append(message)
        return self

    def build(self) -> List[Message]:
        """Build the final list of messages.
        
        Returns:
            List of messages
        """
        return self._messages.copy()

    @classmethod
    def from_template(cls, template: str, role: Role = Role.USER, **kwargs) -> List[Message]:
        """Convenience method to create a single message from template.
        
        Args:
            template: Template string
            role: Message role
            **kwargs: Template variables
            
        Returns:
            List with single formatted message
        """
        prompt = cls()
        prompt.add_template(template, role, **kwargs)
        return prompt.build()

    def __len__(self) -> int:
        """Get number of messages."""
        return len(self._messages)

    def __iter__(self):
        """Iterate over messages."""
        return iter(self._messages)


def create_react_prompt(question: str, tools: Optional[List[str]] = None) -> List[Message]:
    """Create a ReAct (Reasoning and Acting) prompt.
    
    Args:
        question: The question to answer
        tools: List of available tool names
        
    Returns:
        List of messages for ReAct prompting
    """
    tools_str = "\n".join(f"- {tool}" for tool in (tools or []))
    
    system_prompt = f"""You are a helpful AI assistant that can use tools to answer questions.

Available tools:
{tools_str}

To use a tool, respond with:
Thought: [your reasoning about what to do]
Action: [tool name]
Action Input: [input for the tool]

After receiving the tool output, continue with:
Thought: [reflection on the result]
... (continue until you have the final answer)

When you have the final answer, respond with:
Thought: [final reasoning]
Answer: [final answer to the user]"""

    return Prompt() \
        .add_system(system_prompt) \
        .add_user(question) \
        .build()


def create_few_shot_prompt(
    instruction: str,
    examples: List[Dict[str, str]],
    query: str,
    system_message: Optional[str] = None
) -> List[Message]:
    """Create a few-shot learning prompt.
    
    Args:
        instruction: Task instruction
        examples: List of input/output examples
        query: The actual query to process
        system_message: Optional system message
        
    Returns:
        List of messages for few-shot prompting
    """
    prompt = Prompt()
    
    if system_message:
        prompt.add_system(system_message)
    else:
        prompt.add_system(f"Follow these instructions: {instruction}")
    
    # Add examples
    for example in examples:
        prompt.add_user(example["input"])
        prompt.add_assistant(example["output"])
    
    # Add the actual query
    prompt.add_user(query)
    
    return prompt.build()