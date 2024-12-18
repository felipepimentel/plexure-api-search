"""Spell checking for search queries."""

import logging
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime

import spacy
from spacy.tokens import Token
from spellchecker import SpellChecker

from ..config import Config
from ..monitoring.events import Event, EventType
from ..monitoring.metrics import MetricsManager
from ..services.base import BaseService
from ..services.events import PublisherService

logger = logging.getLogger(__name__)


class SpellCheckConfig:
    """Configuration for spell checking."""

    def __init__(
        self,
        language: str = "en",
        distance: int = 2,
        min_word_length: int = 4,
        max_suggestions: int = 5,
        custom_words: Optional[List[str]] = None,
    ) -> None:
        """Initialize spell check config.

        Args:
            language: Language code
            distance: Maximum edit distance
            min_word_length: Minimum word length to check
            max_suggestions: Maximum number of suggestions
            custom_words: Custom words to add to dictionary
        """
        self.language = language
        self.distance = distance
        self.min_word_length = min_word_length
        self.max_suggestions = max_suggestions
        self.custom_words = custom_words or []


class SpellChecker(BaseService[Dict[str, Any]]):
    """Spell checking service for query enhancement."""

    def __init__(
        self,
        config: Config,
        publisher: PublisherService,
        metrics_manager: MetricsManager,
        spell_config: Optional[SpellCheckConfig] = None,
    ) -> None:
        """Initialize spell checker.

        Args:
            config: Application configuration
            publisher: Event publisher
            metrics_manager: Metrics manager
            spell_config: Spell checking configuration
        """
        super().__init__(config, publisher, metrics_manager)
        self.spell_config = spell_config or SpellCheckConfig()
        self._initialized = False
        self._spellchecker: Optional[SpellChecker] = None
        self._nlp: Optional[spacy.language.Language] = None

    async def initialize(self) -> None:
        """Initialize spell checker resources."""
        if self._initialized:
            return

        try:
            # Initialize spellchecker
            self._spellchecker = SpellChecker(
                language=self.spell_config.language,
                distance=self.spell_config.distance,
            )

            # Add custom words
            if self.spell_config.custom_words:
                self._spellchecker.word_frequency.load_words(self.spell_config.custom_words)

            # Load spaCy model
            self._nlp = spacy.load("en_core_web_sm")

            self._initialized = True
            self.publisher.publish(
                Event(
                    type=EventType.SERVICE_STARTED,
                    timestamp=datetime.now(),
                    component="spellchecker",
                    description="Spell checker initialized",
                    metadata={
                        "language": self.spell_config.language,
                        "custom_words": len(self.spell_config.custom_words),
                    },
                )
            )

        except Exception as e:
            logger.error(f"Failed to initialize spell checker: {e}")
            raise

    async def cleanup(self) -> None:
        """Clean up spell checker resources."""
        self._initialized = False
        self._spellchecker = None
        self._nlp = None

        self.publisher.publish(
            Event(
                type=EventType.SERVICE_STOPPED,
                timestamp=datetime.now(),
                component="spellchecker",
                description="Spell checker stopped",
            )
        )

    async def health_check(self) -> Dict[str, Any]:
        """Check spell checker health.

        Returns:
            Health check results
        """
        return {
            "service": "SpellChecker",
            "initialized": self._initialized,
            "spellchecker_loaded": self._spellchecker is not None,
            "nlp_loaded": self._nlp is not None,
            "language": self.spell_config.language,
        }

    async def check_query(
        self,
        query: str,
        min_word_length: Optional[int] = None,
        max_suggestions: Optional[int] = None,
    ) -> Tuple[str, Dict[str, List[str]]]:
        """Check query for spelling errors.

        Args:
            query: Search query
            min_word_length: Optional minimum word length
            max_suggestions: Optional maximum suggestions

        Returns:
            Tuple of (corrected query, suggestions by word)
        """
        if not self._initialized:
            await self.initialize()

        try:
            # Emit spell check started event
            self.publisher.publish(
                Event(
                    type=EventType.SPELL_CHECK_STARTED,
                    timestamp=datetime.now(),
                    component="spellchecker",
                    description=f"Checking query: {query}",
                )
            )

            # Process query with spaCy
            doc = self._nlp(query)

            # Get settings
            min_length = min_word_length or self.spell_config.min_word_length
            max_sugg = max_suggestions or self.spell_config.max_suggestions

            # Check each token
            corrections = {}
            corrected_tokens = []

            for token in doc:
                # Skip short words and special tokens
                if (
                    len(token.text) < min_length
                    or token.is_punct
                    or token.is_space
                    or token.is_stop
                    or token.like_num
                    or token.is_quote
                ):
                    corrected_tokens.append(token.text)
                    continue

                # Check if word is misspelled
                if not self._spellchecker.known([token.text.lower()]):
                    # Get suggestions
                    suggestions = self._spellchecker.candidates(token.text.lower())
                    suggestions = list(suggestions)[:max_sugg]

                    if suggestions:
                        # Use first suggestion as correction
                        corrections[token.text] = suggestions
                        corrected_tokens.append(suggestions[0])
                    else:
                        corrected_tokens.append(token.text)
                else:
                    corrected_tokens.append(token.text)

            # Build corrected query
            corrected_query = " ".join(corrected_tokens)

            # Emit success event
            self.publisher.publish(
                Event(
                    type=EventType.SPELL_CHECK_COMPLETED,
                    timestamp=datetime.now(),
                    component="spellchecker",
                    description="Spell check completed",
                    metadata={
                        "original_query": query,
                        "corrected_query": corrected_query,
                        "num_corrections": len(corrections),
                    },
                )
            )

            return corrected_query, corrections

        except Exception as e:
            logger.error(f"Spell check failed: {e}")
            self.publisher.publish(
                Event(
                    type=EventType.SPELL_CHECK_FAILED,
                    timestamp=datetime.now(),
                    component="spellchecker",
                    description="Spell check failed",
                    error=str(e),
                )
            )
            return query, {}

    def add_words(self, words: List[str]) -> None:
        """Add custom words to dictionary.

        Args:
            words: List of words to add
        """
        if not self._initialized or not self._spellchecker:
            return

        try:
            self._spellchecker.word_frequency.load_words(words)
            self.spell_config.custom_words.extend(words)

            self.publisher.publish(
                Event(
                    type=EventType.DICTIONARY_UPDATED,
                    timestamp=datetime.now(),
                    component="spellchecker",
                    description="Added custom words to dictionary",
                    metadata={"num_words": len(words)},
                )
            )

        except Exception as e:
            logger.error(f"Failed to add words: {e}")


# Create service instance
spell_checker = SpellChecker(
    Config(),
    PublisherService(Config(), MetricsManager()),
    MetricsManager(),
)

__all__ = ["SpellCheckConfig", "SpellChecker", "spell_checker"] 