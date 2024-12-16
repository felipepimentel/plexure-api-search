"""OpenRouter LLM client for API communication."""

import json
import logging
from typing import Any, Dict, Optional

import httpx

from ...config import config_instance
from ...utils.cache import DiskCache
from .base import LLMProvider

logger = logging.getLogger(__name__)

# Cache for LLM responses
llm_cache = DiskCache[Dict[str, Any]](
    namespace="llm_responses",
    ttl=config_instance.cache_ttl * 24,  # Cache LLM responses for longer
)


class OpenRouterClient(LLMProvider):
    """Client for OpenRouter LLM API integration."""

    def __init__(self, use_cache: bool = True):
        """Initialize OpenRouter client.

        Args:
            use_cache: Whether to use caching for LLM responses
        """
        self.api_key = config_instance.openrouter_api_key
        self.model = config_instance.openrouter_model
        self.max_tokens = config_instance.openrouter_max_tokens
        self.temperature = config_instance.openrouter_temperature
        self.base_url = "https://openrouter.ai/api/v1"
        self.use_cache = use_cache

        # Headers required by OpenRouter
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "HTTP-Referer": "https://github.com/plexure/plexure-api-search",  # Required by OpenRouter
            "X-Title": "Plexure API Search",  # Required by OpenRouter
            "X-Custom-Auth": "plexure-api-search",  # Recommended by OpenRouter
            "Content-Type": "application/json",
        }

        self.http_client = httpx.Client(
            headers=self.headers,
            timeout=30.0,
        )

    def call(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        cache_key: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Make a call to OpenRouter API.

        Args:
            prompt: The prompt to send to the LLM
            system_prompt: Optional system prompt to set context
            max_tokens: Optional maximum tokens to generate
            temperature: Optional temperature for response randomness
            cache_key: Optional cache key for response

        Returns:
            LLM response dictionary
        """
        try:
            # Check cache first if enabled
            if self.use_cache and cache_key:
                cached = llm_cache.get(cache_key)
                if cached is not None:
                    return cached

            # Prepare request
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": prompt})

            data = {
                "model": self.model,
                "messages": messages,
                "max_tokens": max_tokens or self.max_tokens,
                "temperature": temperature or self.temperature,
                "top_p": 1,
                "stream": False,
                "transforms": ["middle-out"],  # OpenRouter optimization
            }

            # Log request details for debugging
            logger.debug(f"Request URL: {self.base_url}/chat/completions")
            logger.debug(f"Request headers: {self.headers}")
            logger.debug(f"Request body: {json.dumps(data, indent=2)}")

            # Make request
            response = self.http_client.post(
                f"{self.base_url}/chat/completions",
                json=data,
            )

            # Log response for debugging if there's an error
            if response.status_code != 200:
                logger.error(f"OpenRouter API error response: {response.text}")
                response.raise_for_status()

            # Parse response
            result = response.json()

            # Cache response if enabled and key provided
            if self.use_cache and cache_key:
                llm_cache.set(cache_key, result)

            return result

        except Exception as e:
            logger.error(f"OpenRouter API call failed: {e}")
            return {
                "error": str(e),
                "choices": [
                    {
                        "message": {
                            "content": "API call failed"
                        }
                    }
                ]
            }
