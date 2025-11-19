"""Code Chat Bot - Multiple AI ChatBot with modern architecture."""

__version__ = "0.1.0"

from .config import ConfigManager
from .models import (
    ChatMessage, 
    AIProviderConfig, 
    DocumentMetadata,
    VectorSearchResult,
    ModelPricing,
    AppConfig
)
from .providers import get_provider
from .document_processing import DocumentProcessor, VectorStore, perform_vector_db_search

__all__ = [
    "ConfigManager",
    "ChatMessage",
    "AIProviderConfig", 
    "DocumentMetadata",
    "VectorSearchResult",
    "ModelPricing",
    "AppConfig",
    "get_provider",
    "DocumentProcessor",
    "VectorStore",
    "perform_vector_db_search",
]