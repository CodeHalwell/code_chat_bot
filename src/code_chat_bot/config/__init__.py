"""Configuration management for the chat bot."""
import os
from typing import Dict, Any
from pydantic import BaseModel
import dotenv

from ..models import AppConfig, ModelPricing

dotenv.load_dotenv()


class ConfigManager:
    """Manages application configuration and model pricing."""
    
    def __init__(self):
        self.app_config = AppConfig(
            OPENAI=os.getenv("OPENAI"),
            MISTRAL=os.getenv("MISTRAL"),
            ANTHROPIC=os.getenv("ANTHROPIC"),
            COHERE=os.getenv("COHERE"),
            GOOGLE=os.getenv("GOOGLE_API_KEY")
        )
        self.model_costs = self._load_model_costs()
    
    def _load_model_costs(self) -> Dict[str, ModelPricing]:
        """Load model pricing information."""
        return {
            # OpenAI Models
            "gpt-4o": ModelPricing(
                description="GPT-4o - Most advanced multimodal model with vision, faster and cheaper than GPT-4 Turbo.",
                input_price_1M_tokens=5.0,
                output_price_1M_tokens=15.0
            ),
            "gpt-4o-mini": ModelPricing(
                description="GPT-4o mini - Small, affordable, and intelligent model for fast, lightweight tasks.",
                input_price_1M_tokens=0.15,
                output_price_1M_tokens=0.60
            ),
            "o1-preview": ModelPricing(
                description="OpenAI o1-preview - Advanced reasoning model for complex problem-solving.",
                input_price_1M_tokens=15.0,
                output_price_1M_tokens=60.0
            ),
            "o1-mini": ModelPricing(
                description="OpenAI o1-mini - Fast reasoning model for coding and STEM tasks.",
                input_price_1M_tokens=3.0,
                output_price_1M_tokens=12.0
            ),
            "gpt-4-turbo": ModelPricing(
                description="GPT-4 Turbo with improved instruction following and JSON mode.",
                input_price_1M_tokens=10,
                output_price_1M_tokens=30
            ),
            "gpt-4": ModelPricing(
                description="Most capable GPT-4 model, great for tasks that require nuance and reasoning.",
                input_price_1M_tokens=30,
                output_price_1M_tokens=60
            ),
            "gpt-3.5-turbo": ModelPricing(
                description="Fast, inexpensive model for simple tasks.",
                input_price_1M_tokens=0.5,
                output_price_1M_tokens=1.5
            ),

            # Anthropic Models
            "claude-3-5-sonnet-20241022": ModelPricing(
                description="Claude 3.5 Sonnet - Most intelligent model, with best-in-class coding and agentic capabilities.",
                input_price_1M_tokens=3.0,
                output_price_1M_tokens=15.0
            ),
            "claude-3-opus-20240229": ModelPricing(
                description="Claude 3 Opus - Most powerful model for highly complex tasks.",
                input_price_1M_tokens=15,
                output_price_1M_tokens=75
            ),
            "claude-3-sonnet-20240229": ModelPricing(
                description="Claude 3 Sonnet - Balanced performance for enterprise workloads.",
                input_price_1M_tokens=3,
                output_price_1M_tokens=15
            ),
            "claude-3-haiku-20240307": ModelPricing(
                description="Claude 3 Haiku - Fastest and most compact model for near-instant responsiveness.",
                input_price_1M_tokens=0.25,
                output_price_1M_tokens=1.25
            ),

            # MistralAI Models
            "mistral-large-latest": ModelPricing(
                description="Mistral Large - Flagship model with advanced reasoning capabilities.",
                input_price_1M_tokens=8,
                output_price_1M_tokens=24
            ),
            "mistral-medium-latest": ModelPricing(
                description="Mistral Medium - Balanced performance and cost.",
                input_price_1M_tokens=2.7,
                output_price_1M_tokens=8.1
            ),
            "mistral-small-latest": ModelPricing(
                description="Mistral Small - Cost-effective reasoning with low latency.",
                input_price_1M_tokens=2,
                output_price_1M_tokens=6
            ),
            "open-mixtral-8x22b": ModelPricing(
                description="Mixtral 8x22B - High-performance sparse MoE model.",
                input_price_1M_tokens=2,
                output_price_1M_tokens=6
            ),
            "open-mixtral-8x7b": ModelPricing(
                description="Mixtral 8x7B - Efficient sparse MoE model.",
                input_price_1M_tokens=0.7,
                output_price_1M_tokens=0.7
            ),
            "open-mistral-7b": ModelPricing(
                description="Mistral 7B - Fast and customizable base model.",
                input_price_1M_tokens=0.25,
                output_price_1M_tokens=0.25
            ),
            "mistral-embed": ModelPricing(
                description="Mistral Embed - Advanced semantic text embeddings.",
                input_price_1M_tokens=0.1,
                output_price_1M_tokens=0.1
            ),

            # Cohere Models
            "command-r-plus": ModelPricing(
                description="Command R+ - Most powerful model for complex RAG and tool use.",
                input_price_1M_tokens=3.0,
                output_price_1M_tokens=15.0
            ),
            "command-r": ModelPricing(
                description="Command R - Optimized for RAG and long-context tasks.",
                input_price_1M_tokens=0.5,
                output_price_1M_tokens=1.5
            ),
            "command": ModelPricing(
                description="Command - Versatile model for general use cases.",
                input_price_1M_tokens=1.0,
                output_price_1M_tokens=2.0
            ),
            "command-light": ModelPricing(
                description="Command Light - Faster, lightweight version for simple tasks.",
                input_price_1M_tokens=0.3,
                output_price_1M_tokens=0.6
            ),

            # Google Gemini Models
            "gemini-1.5-pro": ModelPricing(
                description="Gemini 1.5 Pro - Most capable model with 2M token context window.",
                input_price_1M_tokens=3.5,
                output_price_1M_tokens=10.5
            ),
            "gemini-1.5-flash": ModelPricing(
                description="Gemini 1.5 Flash - Fast and efficient multimodal model.",
                input_price_1M_tokens=0.35,
                output_price_1M_tokens=1.05
            ),
            "gemini-pro": ModelPricing(
                description="Gemini Pro - Versatile model for various tasks.",
                input_price_1M_tokens=0.5,
                output_price_1M_tokens=1.5
            ),
        }
    
    def get_api_key(self, provider: str) -> str:
        """Get API key for a specific provider."""
        key_mapping = {
            "OpenAI": self.app_config.openai_api_key,
            "MistralAI": self.app_config.mistral_api_key,
            "Anthropic": self.app_config.anthropic_api_key,
            "Cohere": self.app_config.cohere_api_key,
            "Google": self.app_config.google_api_key,
        }
        return key_mapping.get(provider)
    
    def get_model_info(self, model_name: str) -> ModelPricing:
        """Get pricing information for a specific model."""
        return self.model_costs.get(model_name)
    
    def calculate_cost(self, model_name: str, input_tokens: int, output_tokens: int) -> float:
        """Calculate the cost for a model based on token usage."""
        model_info = self.get_model_info(model_name)
        if not model_info:
            return 0.0
        
        input_cost = (input_tokens / 1_000_000) * model_info.input_price_1M_tokens
        output_cost = (output_tokens / 1_000_000) * model_info.output_price_1M_tokens
        return input_cost + output_cost