"""
Integrations Module

External service integrations (Ollama LLM).
"""

from sentiment_analyzer.integrations.ollama import (
    OllamaClient,
    OllamaConfig,
    OllamaError,
    OllamaConnectionError,
    OllamaGenerationError,
    create_ollama_client,
)
from sentiment_analyzer.integrations.prompts import PromptTemplates

__all__ = [
    "OllamaClient",
    "OllamaConfig",
    "OllamaError",
    "OllamaConnectionError",
    "OllamaGenerationError",
    "create_ollama_client",
    "PromptTemplates",
]
