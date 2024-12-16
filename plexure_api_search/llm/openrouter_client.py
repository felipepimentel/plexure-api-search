from typing import Dict, Any, Optional
import httpx
from plexure_api_search.llm.openrouter_client import OpenRouterClient
from plexure_api_search.utils.cache import llm_cache
from plexure_api_search.utils.logger import logger

class OpenRouterClient(OpenRouterClient):
    def _call_llm(
        self, 
        prompt: str,
        cache_key: Optional[str] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """Make a call to OpenRouter API.

        Args:
            prompt: The prompt to send to the LLM
            cache_key: Optional cache key for response
            **kwargs: Additional parameters for the API call

        Returns:
            LLM response
        """
        try:
            # Check cache first
            if cache_key:
                cached = llm_cache.get(cache_key)
                if cached is not None:
                    return cached

            # Prepare request
            data = {
                "model": self.model,
                "messages": [
                    {
                        "role": "system",
                        "content": "You are an AI assistant specialized in analyzing and documenting APIs. You provide technical, accurate, and concise responses focused on API functionality and best practices."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                "max_tokens": kwargs.get("max_tokens", self.max_tokens),
                "temperature": kwargs.get("temperature", self.temperature),
                "top_p": 1,
                "stream": False,
            }

            # Make request with updated headers
            response = self.http_client.post(
                f"{self.base_url}/chat/completions",
                json=data,
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "HTTP-Referer": "https://github.com/plexure/plexure-api-search",
                    "X-Title": "Plexure API Search",
                }
            )
            
            # Log request details for debugging
            logger.debug(f"Request URL: {self.base_url}/chat/completions")
            logger.debug(f"Request headers: {response.request.headers}")
            logger.debug(f"Request body: {data}")
            
            response.raise_for_status()
            result = response.json()

            # Cache response if key provided
            if cache_key:
                llm_cache.set(cache_key, result)

            return result

        except Exception as e:
            logger.error(f"OpenRouter API call failed: {e}")
            if isinstance(e, httpx.HTTPError):
                logger.error(f"Response content: {e.response.content if hasattr(e, 'response') else 'No response content'}")
            return {"error": str(e)} 