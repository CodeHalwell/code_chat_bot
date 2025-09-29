"""Pydantic models for data validation and configuration."""
from typing import Dict, List, Optional, Literal
from pydantic import BaseModel, Field, ConfigDict


class ModelPricing(BaseModel):
    """Model for AI model pricing information."""
    description: str
    input_price_1M_tokens: float = Field(ge=0, description="Input price per 1M tokens")
    output_price_1M_tokens: float = Field(ge=0, description="Output price per 1M tokens")


class ChatMessage(BaseModel):
    """Model for chat messages."""
    role: Literal["user", "assistant", "system"]
    content: str


class AppConfig(BaseModel):
    """Application configuration model."""
    model_config = ConfigDict(env_prefix="")
    
    openai_api_key: Optional[str] = Field(None, alias="OPENAI")
    mistral_api_key: Optional[str] = Field(None, alias="MISTRAL")
    anthropic_api_key: Optional[str] = Field(None, alias="ANTHROPIC")
    cohere_api_key: Optional[str] = Field(None, alias="COHERE")


class DocumentMetadata(BaseModel):
    """Model for document metadata."""
    file_type: Literal["pdf", "text", "csv", "web"]
    file_path: Optional[str] = None
    url: Optional[str] = None
    chunk_size: int = Field(default=500, ge=1)
    chunk_overlap: int = Field(default=75, ge=0)


class VectorSearchResult(BaseModel):
    """Model for vector database search results."""
    content: str
    score: float = Field(ge=0, le=1)


class AIProviderConfig(BaseModel):
    """Configuration for AI providers."""
    provider: Literal["OpenAI", "MistralAI", "Anthropic", "Cohere"]
    model: str
    temperature: float = Field(default=0.7, ge=0, le=2)
    max_tokens: int = Field(default=1000, ge=1)