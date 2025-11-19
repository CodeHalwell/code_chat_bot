"""AI Provider clients with Pydantic integration."""
from typing import List, Iterator, Tuple
from abc import ABC, abstractmethod

from openai import OpenAI
from mistralai.client import MistralClient
from anthropic import Anthropic
import cohere
import tiktoken

from ..models import ChatMessage, AIProviderConfig
from ..config import ConfigManager


def approximate_token_count(text: str) -> int:
    """
    Approximate token count using word count heuristic.
    This is a rough approximation used for non-OpenAI models.
    
    Args:
        text: The text to count tokens for
        
    Returns:
        Approximate token count
    """
    return int(len(text.split()) * 1.3)


class BaseAIProvider(ABC):
    """Base class for AI providers."""
    
    def __init__(self, config: AIProviderConfig, config_manager: ConfigManager):
        self.config = config
        self.config_manager = config_manager
        self.client = self._initialize_client()
    
    @abstractmethod
    def _initialize_client(self):
        """Initialize the AI provider client."""
        pass
    
    @abstractmethod
    def generate_response(self, messages: List[ChatMessage], stream: bool = True) -> Iterator[str]:
        """Generate response from the AI model."""
        pass
    
    @abstractmethod
    def count_tokens(self, text: str) -> int:
        """Count tokens in the given text."""
        pass


class OpenAIProvider(BaseAIProvider):
    """OpenAI provider implementation."""
    
    def _initialize_client(self):
        api_key = self.config_manager.get_api_key("OpenAI")
        return OpenAI(api_key=api_key)
    
    def generate_response(self, messages: List[ChatMessage], stream: bool = True) -> Iterator[str]:
        """Generate response from OpenAI model."""
        message_dicts = [{"role": msg.role, "content": msg.content} for msg in messages]
        
        response = self.client.chat.completions.create(
            model=self.config.model,
            messages=message_dicts,
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens,
            stream=stream
        )
        
        if stream:
            for chunk in response:
                if chunk.choices[0].delta.content is not None:
                    yield chunk.choices[0].delta.content
        else:
            yield response.choices[0].message.content
    
    def count_tokens(self, text: str) -> int:
        """Count tokens using tiktoken."""
        try:
            encoding = tiktoken.encoding_for_model(self.config.model)
        except KeyError:
            encoding = tiktoken.get_encoding("cl100k_base")
        return len(encoding.encode(text))


class MistralAIProvider(BaseAIProvider):
    """MistralAI provider implementation."""
    
    def _initialize_client(self):
        api_key = self.config_manager.get_api_key("MistralAI")
        return MistralClient(api_key=api_key)
    
    def generate_response(self, messages: List[ChatMessage], stream: bool = True) -> Iterator[str]:
        """Generate response from MistralAI model."""
        message_dicts = [{"role": msg.role, "content": msg.content} for msg in messages]
        
        if stream:
            response = self.client.chat_stream(
                model=self.config.model,
                messages=message_dicts,
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens,
            )
            for chunk in response:
                if chunk.choices[0].delta.content is not None:
                    yield chunk.choices[0].delta.content
        else:
            response = self.client.chat(
                model=self.config.model,
                messages=message_dicts,
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens,
            )
            yield response.choices[0].message.content
    
    def count_tokens(self, text: str) -> int:
        """Approximate token count for Mistral models."""
        return approximate_token_count(text)


class AnthropicProvider(BaseAIProvider):
    """Anthropic provider implementation."""
    
    def _initialize_client(self):
        api_key = self.config_manager.get_api_key("Anthropic")
        return Anthropic(api_key=api_key)
    
    def generate_response(self, messages: List[ChatMessage], stream: bool = True) -> Iterator[str]:
        """Generate response from Anthropic model."""
        # Separate system message from other messages
        system_message = None
        conversation_messages = []
        
        for msg in messages:
            if msg.role == "system":
                system_message = msg.content
            else:
                conversation_messages.append({"role": msg.role, "content": msg.content})
        
        kwargs = {
            "model": self.config.model,
            "max_tokens": self.config.max_tokens,
            "temperature": self.config.temperature,
            "messages": conversation_messages,
        }
        
        if system_message:
            kwargs["system"] = system_message
        
        if stream:
            with self.client.messages.stream(**kwargs) as response:
                for chunk in response.text_stream:
                    yield chunk
        else:
            response = self.client.messages.create(**kwargs)
            yield response.content[0].text
    
    def count_tokens(self, text: str) -> int:
        """Approximate token count for Anthropic models."""
        return approximate_token_count(text)


class CohereProvider(BaseAIProvider):
    """Cohere provider implementation."""
    
    def _initialize_client(self):
        api_key = self.config_manager.get_api_key("Cohere")
        return cohere.Client(api_key=api_key)
    
    def generate_response(self, messages: List[ChatMessage], stream: bool = True) -> Iterator[str]:
        """Generate response from Cohere model."""
        # Convert messages to chat format
        chat_history = []
        message = ""
        
        for msg in messages:
            if msg.role == "user":
                message = msg.content
            elif msg.role == "assistant":
                chat_history.append({"role": "CHATBOT", "message": msg.content})
            elif msg.role == "system":
                # Add system message to chat history
                chat_history.append({"role": "SYSTEM", "message": msg.content})
        
        if stream:
            response = self.client.chat_stream(
                message=message,
                chat_history=chat_history,
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens,
            )
            for chunk in response:
                if chunk.event_type == "text-generation":
                    yield chunk.text
        else:
            response = self.client.chat(
                message=message,
                chat_history=chat_history,
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens,
            )
            yield response.text
    
    def count_tokens(self, text: str) -> int:
        """Approximate token count for Cohere models."""
        return approximate_token_count(text)


def get_provider(provider_config: AIProviderConfig, config_manager: ConfigManager) -> BaseAIProvider:
    """Factory function to get the appropriate AI provider."""
    providers = {
        "OpenAI": OpenAIProvider,
        "MistralAI": MistralAIProvider,
        "Anthropic": AnthropicProvider,
        "Cohere": CohereProvider,
    }
    
    provider_class = providers.get(provider_config.provider)
    if not provider_class:
        raise ValueError(f"Unsupported provider: {provider_config.provider}")
    
    return provider_class(provider_config, config_manager)