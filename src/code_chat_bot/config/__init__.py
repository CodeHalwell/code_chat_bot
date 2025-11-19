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
            COHERE=os.getenv("COHERE")
        )
        self.model_costs = self._load_model_costs()
    
    def _load_model_costs(self) -> Dict[str, ModelPricing]:
        """Load model pricing information."""
        return {
            "open-mistral-7b": ModelPricing(
                description="A 7B transformer model, fast-deployed and easily customizable for various applications.",
                input_price_1M_tokens=0.25,
                output_price_1M_tokens=0.25
            ),
            "open-mixtral-8x7b": ModelPricing(
                description="A 7B sparse Mixture-of-Experts model with 12.9B active parameters from a total of 45B, designed for efficient large-scale processing.",
                input_price_1M_tokens=0.7,
                output_price_1M_tokens=0.7
            ),
            "open-mixtral-8x22b": ModelPricing(
                description="A high-performance 22B sparse Mixture-of-Experts model utilizing 39B active parameters from 141B total, suitable for complex problem solving.",
                input_price_1M_tokens=2,
                output_price_1M_tokens=6
            ),
            "mistral-small-latest": ModelPricing(
                description="Designed for cost-effective reasoning with low latency, ideal for quick response applications.",
                input_price_1M_tokens=2,
                output_price_1M_tokens=6
            ),
            "mistral-medium-latest": ModelPricing(
                description="Medium-scale model providing a balance between performance and cost, suitable for a range of applications.",
                input_price_1M_tokens=2.7,
                output_price_1M_tokens=8.1
            ),
            "mistral-large-latest": ModelPricing(
                description="The flagship model of the Mistral series, offering advanced reasoning capabilities for the most demanding tasks.",
                input_price_1M_tokens=8,
                output_price_1M_tokens=24
            ),
            "mistral-embed": ModelPricing(
                description="Advanced model for semantic extraction from text, ideal for creating meaningful text representations.",
                input_price_1M_tokens=0.1,
                output_price_1M_tokens=0.1
            ),
            "claude-3-haiku-20240307": ModelPricing(
                description="Optimized for speed and efficiency, well-suited for lightweight tasks requiring quick turnarounds.",
                input_price_1M_tokens=0.25,
                output_price_1M_tokens=1.25
            ),
            "claude-3-sonnet-20240229": ModelPricing(
                description="Designed for robust performance on demanding tasks, offering detailed and extensive responses.",
                input_price_1M_tokens=3,
                output_price_1M_tokens=15
            ),
            "claude-3-opus-20240229": ModelPricing(
                description="Most powerful model for highly complex tasks.",
                input_price_1M_tokens=15,
                output_price_1M_tokens=75
            ),
            "gpt-4": ModelPricing(
                description="Most capable GPT-4 model, great for tasks that require a lot of nuance and careful reasoning.",
                input_price_1M_tokens=30,
                output_price_1M_tokens=60
            ),
            "gpt-4-turbo-2024-04-09": ModelPricing(
                description="Latest GPT-4 Turbo with improved instruction following and JSON mode.",
                input_price_1M_tokens=10,
                output_price_1M_tokens=30
            ),
            "gpt-3.5-turbo-0125": ModelPricing(
                description="Updated GPT 3.5 Turbo model with improved instruction following.",
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