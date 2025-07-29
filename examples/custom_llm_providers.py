"""Examples of implementing custom LLM providers for AILib."""

import os
from typing import Any, AsyncIterator, Dict, Iterator, List, Optional, Union
import anthropic
import google.generativeai as genai

from ailib.core import LLMClient, CompletionResponse, Message, Role


class AnthropicClient(LLMClient):
    """Anthropic Claude client implementation."""
    
    def __init__(
        self,
        model: str = "claude-3-opus-20240229",
        api_key: Optional[str] = None,
        **kwargs
    ):
        """Initialize Anthropic client.
        
        Args:
            model: Model name (e.g., 'claude-3-opus-20240229', 'claude-3-sonnet-20240229')
            api_key: Anthropic API key (defaults to ANTHROPIC_API_KEY env var)
            **kwargs: Additional client parameters
        """
        super().__init__(model, **kwargs)
        
        self.client = anthropic.Anthropic(
            api_key=api_key or os.getenv("ANTHROPIC_API_KEY")
        )
        self.async_client = anthropic.AsyncAnthropic(
            api_key=api_key or os.getenv("ANTHROPIC_API_KEY")
        )
    
    def complete(
        self,
        messages: List[Message],
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> CompletionResponse:
        """Generate completion using Anthropic API."""
        self.validate_messages(messages)
        
        # Convert messages to Anthropic format
        system_message = None
        anthropic_messages = []
        
        for msg in messages:
            if msg.role == Role.SYSTEM:
                system_message = msg.content
            else:
                anthropic_messages.append({
                    "role": "user" if msg.role == Role.USER else "assistant",
                    "content": msg.content
                })
        
        # Make API call
        response = self.client.messages.create(
            model=self.model,
            messages=anthropic_messages,
            system=system_message,
            temperature=temperature,
            max_tokens=max_tokens or 1000,
            **kwargs
        )
        
        return CompletionResponse(
            content=response.content[0].text,
            model=self.model,
            usage={
                "prompt_tokens": response.usage.input_tokens,
                "completion_tokens": response.usage.output_tokens,
                "total_tokens": response.usage.input_tokens + response.usage.output_tokens,
            },
            finish_reason=response.stop_reason,
        )
    
    async def acomplete(self, messages: List[Message], **kwargs) -> CompletionResponse:
        """Async version of complete."""
        # Similar implementation using async_client
        pass
    
    def stream(self, messages: List[Message], **kwargs) -> Iterator[str]:
        """Stream completions."""
        # Implementation using stream=True
        pass
    
    async def astream(self, messages: List[Message], **kwargs) -> AsyncIterator[str]:
        """Async stream completions."""
        pass


class GoogleGeminiClient(LLMClient):
    """Google Gemini client implementation."""
    
    def __init__(
        self,
        model: str = "gemini-pro",
        api_key: Optional[str] = None,
        **kwargs
    ):
        """Initialize Gemini client.
        
        Args:
            model: Model name (e.g., 'gemini-pro', 'gemini-pro-vision')
            api_key: Google API key (defaults to GOOGLE_API_KEY env var)
            **kwargs: Additional client parameters
        """
        super().__init__(model, **kwargs)
        
        genai.configure(api_key=api_key or os.getenv("GOOGLE_API_KEY"))
        self.model = genai.GenerativeModel(model)
    
    def complete(
        self,
        messages: List[Message],
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> CompletionResponse:
        """Generate completion using Gemini API."""
        self.validate_messages(messages)
        
        # Convert messages to Gemini format
        chat = self.model.start_chat(history=[])
        
        for msg in messages:
            if msg.role == Role.USER:
                response = chat.send_message(msg.content)
            elif msg.role == Role.ASSISTANT:
                # Add to history
                chat.history.append({
                    "role": "model",
                    "parts": [msg.content]
                })
        
        # Get the last response
        text = response.text
        
        return CompletionResponse(
            content=text,
            model=self.model._model_name,
            usage={
                "prompt_tokens": 0,  # Gemini doesn't provide token counts
                "completion_tokens": 0,
                "total_tokens": 0,
            },
            finish_reason="stop",
        )
    
    # Implement other required methods...


class OllamaClient(LLMClient):
    """Ollama (local models) client implementation."""
    
    def __init__(
        self,
        model: str = "llama2",
        base_url: str = "http://localhost:11434",
        **kwargs
    ):
        """Initialize Ollama client.
        
        Args:
            model: Model name (e.g., 'llama2', 'mistral', 'codellama')
            base_url: Ollama API base URL
            **kwargs: Additional client parameters
        """
        super().__init__(model, **kwargs)
        self.base_url = base_url
        
    def complete(
        self,
        messages: List[Message],
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> CompletionResponse:
        """Generate completion using Ollama API."""
        import requests
        
        self.validate_messages(messages)
        
        # Convert messages to Ollama format
        prompt = ""
        for msg in messages:
            if msg.role == Role.SYSTEM:
                prompt += f"System: {msg.content}\n\n"
            elif msg.role == Role.USER:
                prompt += f"User: {msg.content}\n\n"
            elif msg.role == Role.ASSISTANT:
                prompt += f"Assistant: {msg.content}\n\n"
        
        # Make API call
        response = requests.post(
            f"{self.base_url}/api/generate",
            json={
                "model": self.model,
                "prompt": prompt,
                "temperature": temperature,
                "stream": False,
            }
        )
        
        data = response.json()
        
        return CompletionResponse(
            content=data["response"],
            model=self.model,
            usage={
                "prompt_tokens": len(prompt.split()),
                "completion_tokens": len(data["response"].split()),
                "total_tokens": len(prompt.split()) + len(data["response"].split()),
            },
            finish_reason="stop",
        )
    
    # Implement other required methods...


# Usage examples:

def example_anthropic():
    """Example using Anthropic Claude."""
    from ailib import Chain, Message, Role
    
    # Initialize Claude client
    claude = AnthropicClient(model="claude-3-sonnet-20240229")
    
    # Use with chains
    chain = Chain(claude)
    chain.add_user("Explain quantum computing in simple terms")
    
    result = chain.run()
    print(result)


def example_gemini():
    """Example using Google Gemini."""
    from ailib import Agent, tool
    
    # Initialize Gemini client
    gemini = GoogleGeminiClient(model="gemini-pro")
    
    # Create agent with Gemini
    agent = Agent(llm=gemini)
    agent.run("What are the benefits of renewable energy?")


def example_ollama():
    """Example using local Ollama models."""
    from ailib import Prompt
    
    # Initialize Ollama client for local models
    ollama = OllamaClient(model="mistral")
    
    # Use for code generation
    prompt = Prompt()
    prompt.add_system("You are a code generator.")
    prompt.add_user("Write a Python function to calculate fibonacci numbers")
    
    response = ollama.complete(prompt.build())
    print(response.content)


def example_multi_provider_chain():
    """Example using multiple providers in sequence."""
    from ailib import Chain
    
    # You can even switch providers mid-chain!
    openai_client = OpenAIClient(model="gpt-3.5-turbo")
    claude_client = AnthropicClient(model="claude-3-haiku-20240307")
    
    # Start with OpenAI
    chain1 = Chain(openai_client)
    result1 = chain1.add_user("Generate a creative story premise").run()
    
    # Continue with Claude
    chain2 = Chain(claude_client)
    result2 = chain2.add_user(f"Expand this premise into a paragraph: {result1}").run()
    
    print(result2)