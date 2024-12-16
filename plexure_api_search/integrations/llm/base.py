"""Base interface for LLM providers."""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional


class LLMProvider(ABC):
    """Abstract base class for LLM providers."""

    @abstractmethod
    def call(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        cache_key: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Make a call to the LLM provider.

        Args:
            prompt: The prompt to send to the LLM
            system_prompt: Optional system prompt to set context
            max_tokens: Optional maximum tokens to generate
            temperature: Optional temperature for response randomness
            cache_key: Optional cache key for response

        Returns:
            LLM response dictionary
        """
        pass 