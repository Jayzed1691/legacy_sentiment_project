"""
Ollama LLM Integration

Client for interacting with local Ollama models for sentiment analysis
and feedback generation.
"""

import json
import logging
import requests
from typing import Dict, List, Optional, Any
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class OllamaConfig:
    """Configuration for Ollama client."""
    base_url: str = "http://localhost:11434"
    model: str = "llama3.1"
    temperature: float = 0.7
    max_tokens: int = 2000
    timeout: int = 60


class OllamaError(Exception):
    """Base exception for Ollama errors."""
    pass


class OllamaConnectionError(OllamaError):
    """Raised when cannot connect to Ollama."""
    pass


class OllamaGenerationError(OllamaError):
    """Raised when generation fails."""
    pass


class OllamaClient:
    """
    Client for interacting with Ollama local LLM.

    Provides methods for generating text, analyzing sentiment, and creating
    structured outputs from natural language prompts.

    Example:
        >>> client = OllamaClient()
        >>> response = client.generate("Analyze the sentiment of: 'Revenue exceeded expectations'")
        >>> print(response)
    """

    def __init__(self, config: Optional[OllamaConfig] = None):
        """
        Initialize Ollama client.

        Args:
            config: OllamaConfig instance. Uses defaults if None.
        """
        self.config = config or OllamaConfig()
        self._verify_connection()

    def _verify_connection(self) -> None:
        """Verify Ollama server is accessible."""
        try:
            response = requests.get(f"{self.config.base_url}/api/tags", timeout=5)
            response.raise_for_status()
            logger.info(f"Connected to Ollama at {self.config.base_url}")

            # Check if the requested model is available
            models = response.json().get('models', [])
            model_names = [m.get('name', '') for m in models]

            if not any(self.config.model in name for name in model_names):
                logger.warning(
                    f"Model '{self.config.model}' not found. Available models: {model_names}. "
                    f"Consider running: ollama pull {self.config.model}"
                )

        except requests.exceptions.RequestException as e:
            raise OllamaConnectionError(
                f"Cannot connect to Ollama at {self.config.base_url}. "
                f"Is Ollama running? Error: {e}"
            ) from e

    def generate(self, prompt: str, system_prompt: Optional[str] = None,
                temperature: Optional[float] = None,
                max_tokens: Optional[int] = None) -> str:
        """
        Generate text from a prompt.

        Args:
            prompt: User prompt
            system_prompt: Optional system prompt to set context
            temperature: Override default temperature
            max_tokens: Override default max tokens

        Returns:
            Generated text

        Raises:
            OllamaGenerationError: If generation fails
        """
        url = f"{self.config.base_url}/api/generate"

        payload = {
            "model": self.config.model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": temperature or self.config.temperature,
                "num_predict": max_tokens or self.config.max_tokens,
            }
        }

        if system_prompt:
            payload["system"] = system_prompt

        try:
            response = requests.post(
                url,
                json=payload,
                timeout=self.config.timeout
            )
            response.raise_for_status()

            result = response.json()
            generated_text = result.get('response', '')

            logger.debug(f"Generated {len(generated_text)} characters")
            return generated_text

        except requests.exceptions.Timeout:
            raise OllamaGenerationError(
                f"Generation timed out after {self.config.timeout}s"
            )
        except requests.exceptions.RequestException as e:
            raise OllamaGenerationError(f"Generation failed: {e}") from e

    def generate_json(self, prompt: str, system_prompt: Optional[str] = None,
                     temperature: Optional[float] = None) -> Dict[str, Any]:
        """
        Generate structured JSON output.

        Automatically parses JSON from the response and handles common formatting issues.

        Args:
            prompt: User prompt (should request JSON output)
            system_prompt: Optional system prompt
            temperature: Override default temperature

        Returns:
            Parsed JSON as dictionary

        Raises:
            OllamaGenerationError: If generation fails or JSON invalid
        """
        # Add JSON formatting instruction to prompt
        enhanced_prompt = f"{prompt}\n\nIMPORTANT: Return ONLY valid JSON, no other text."

        response = self.generate(enhanced_prompt, system_prompt, temperature)

        # Try to extract JSON from response
        try:
            # Try direct parse first
            return json.loads(response)
        except json.JSONDecodeError:
            # Try to find JSON in response (sometimes LLM adds explanation text)
            import re
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                try:
                    return json.loads(json_match.group())
                except json.JSONDecodeError:
                    pass

            # Try array format
            json_match = re.search(r'\[.*\]', response, re.DOTALL)
            if json_match:
                try:
                    return json.loads(json_match.group())
                except json.JSONDecodeError:
                    pass

            raise OllamaGenerationError(
                f"Could not parse JSON from response: {response[:200]}..."
            )

    def chat(self, messages: List[Dict[str, str]],
            temperature: Optional[float] = None,
            max_tokens: Optional[int] = None) -> str:
        """
        Generate response in chat format.

        Args:
            messages: List of message dicts with 'role' and 'content'
                     Roles: 'system', 'user', 'assistant'
            temperature: Override default temperature
            max_tokens: Override default max tokens

        Returns:
            Generated response text

        Example:
            >>> messages = [
            ...     {"role": "system", "content": "You are a financial analyst"},
            ...     {"role": "user", "content": "Analyze this earnings call"}
            ... ]
            >>> response = client.chat(messages)
        """
        url = f"{self.config.base_url}/api/chat"

        payload = {
            "model": self.config.model,
            "messages": messages,
            "stream": False,
            "options": {
                "temperature": temperature or self.config.temperature,
                "num_predict": max_tokens or self.config.max_tokens,
            }
        }

        try:
            response = requests.post(
                url,
                json=payload,
                timeout=self.config.timeout
            )
            response.raise_for_status()

            result = response.json()
            message = result.get('message', {})
            content = message.get('content', '')

            logger.debug(f"Chat response: {len(content)} characters")
            return content

        except requests.exceptions.Timeout:
            raise OllamaGenerationError(
                f"Chat timed out after {self.config.timeout}s"
            )
        except requests.exceptions.RequestException as e:
            raise OllamaGenerationError(f"Chat failed: {e}") from e

    def list_models(self) -> List[str]:
        """
        List available Ollama models.

        Returns:
            List of model names
        """
        try:
            response = requests.get(f"{self.config.base_url}/api/tags", timeout=5)
            response.raise_for_status()
            models = response.json().get('models', [])
            return [m.get('name', '') for m in models]
        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to list models: {e}")
            return []

    def pull_model(self, model_name: str) -> bool:
        """
        Pull/download a model from Ollama library.

        Args:
            model_name: Name of model to pull (e.g., 'llama3.1')

        Returns:
            True if successful
        """
        url = f"{self.config.base_url}/api/pull"

        try:
            logger.info(f"Pulling model: {model_name}")
            response = requests.post(
                url,
                json={"name": model_name},
                timeout=300  # 5 minutes for download
            )
            response.raise_for_status()
            logger.info(f"Successfully pulled model: {model_name}")
            return True

        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to pull model {model_name}: {e}")
            return False


# =============================================================================
# Convenience Functions
# =============================================================================

def create_ollama_client(model: str = "llama3.1",
                        base_url: str = "http://localhost:11434") -> OllamaClient:
    """
    Convenience function to create an Ollama client.

    Args:
        model: Model name (default: llama3.1)
        base_url: Ollama server URL

    Returns:
        OllamaClient instance
    """
    config = OllamaConfig(model=model, base_url=base_url)
    return OllamaClient(config)
