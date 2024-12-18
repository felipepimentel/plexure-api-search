"""Query optimization for improved search performance."""

import logging
from typing import Dict, List, Any, Optional, Set
from datetime import datetime

from ..config import Config
from ..monitoring.events import Event, EventType
from ..monitoring.metrics import MetricsManager
from ..services.base import BaseService
from ..services.events import PublisherService

logger = logging.getLogger(__name__)


class OptimizationConfig:
    """Configuration for query optimization."""

    def __init__(
        self,
        min_query_length: int = 3,
        max_query_length: int = 100,
        min_term_freq: int = 2,
        max_terms: int = 10,
        min_term_length: int = 2,
        enable_stopwords: bool = True,
        enable_synonyms: bool = True,
        enable_ngrams: bool = True,
        ngram_range: tuple = (2, 3),
        min_ngram_freq: int = 2,
    ) -> None:
        """Initialize optimization config.

        Args:
            min_query_length: Minimum query length
            max_query_length: Maximum query length
            min_term_freq: Minimum term frequency
            max_terms: Maximum number of terms
            min_term_length: Minimum term length
            enable_stopwords: Whether to remove stopwords
            enable_synonyms: Whether to use synonyms
            enable_ngrams: Whether to use ngrams
            ngram_range: Range of ngram sizes
            min_ngram_freq: Minimum ngram frequency
        """
        self.min_query_length = min_query_length
        self.max_query_length = max_query_length
        self.min_term_freq = min_term_freq
        self.max_terms = max_terms
        self.min_term_length = min_term_length
        self.enable_stopwords = enable_stopwords
        self.enable_synonyms = enable_synonyms
        self.enable_ngrams = enable_ngrams
        self.ngram_range = ngram_range
        self.min_ngram_freq = min_ngram_freq


class QueryOptimizer(BaseService[Dict[str, Any]]):
    """Query optimizer for improved search performance."""

    def __init__(
        self,
        config: Config,
        publisher: PublisherService,
        metrics_manager: MetricsManager,
        optimization_config: Optional[OptimizationConfig] = None,
    ) -> None:
        """Initialize query optimizer.

        Args:
            config: Application configuration
            publisher: Event publisher
            metrics_manager: Metrics manager
            optimization_config: Optimization configuration
        """
        super().__init__(config, publisher, metrics_manager)
        self.optimization_config = optimization_config or OptimizationConfig()
        self._initialized = False
        self._stopwords: Set[str] = set()
        self._synonyms: Dict[str, List[str]] = {}
        self._term_frequencies: Dict[str, int] = {}
        self._ngram_frequencies: Dict[str, int] = {}

    async def initialize(self) -> None:
        """Initialize optimization resources."""
        if self._initialized:
            return

        try:
            # Load stopwords
            if self.optimization_config.enable_stopwords:
                await self._load_stopwords()

            # Load synonyms
            if self.optimization_config.enable_synonyms:
                await self._load_synonyms()

            self._initialized = True
            self.publisher.publish(
                Event(
                    type=EventType.SERVICE_STARTED,
                    timestamp=datetime.now(),
                    component="query_optimizer",
                    description="Query optimizer initialized",
                    metadata={
                        "stopwords_enabled": self.optimization_config.enable_stopwords,
                        "synonyms_enabled": self.optimization_config.enable_synonyms,
                        "ngrams_enabled": self.optimization_config.enable_ngrams,
                    },
                )
            )

        except Exception as e:
            logger.error(f"Failed to initialize query optimizer: {e}")
            raise

    async def cleanup(self) -> None:
        """Clean up optimization resources."""
        self._initialized = False
        self._stopwords.clear()
        self._synonyms.clear()
        self._term_frequencies.clear()
        self._ngram_frequencies.clear()

        self.publisher.publish(
            Event(
                type=EventType.SERVICE_STOPPED,
                timestamp=datetime.now(),
                component="query_optimizer",
                description="Query optimizer stopped",
            )
        )

    async def health_check(self) -> Dict[str, Any]:
        """Check optimizer health.

        Returns:
            Health check results
        """
        return {
            "service": "QueryOptimizer",
            "initialized": self._initialized,
            "stopwords_loaded": bool(self._stopwords),
            "synonyms_loaded": bool(self._synonyms),
            "term_frequencies": len(self._term_frequencies),
            "ngram_frequencies": len(self._ngram_frequencies),
            "status": "healthy" if self._initialized else "unhealthy",
        }

    async def optimize_query(
        self,
        query: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Optimize search query.

        Args:
            query: Original query
            context: Query context

        Returns:
            Optimized query
        """
        if not self._initialized:
            await self.initialize()

        try:
            # Emit optimization started event
            self.publisher.publish(
                Event(
                    type=EventType.OPTIMIZATION_STARTED,
                    timestamp=datetime.now(),
                    component="query_optimizer",
                    description=f"Starting query optimization for: {query}",
                    metadata={"context": context},
                )
            )

            # Validate query length
            if len(query) < self.optimization_config.min_query_length:
                return query
            if len(query) > self.optimization_config.max_query_length:
                query = query[:self.optimization_config.max_query_length]

            # Tokenize query
            terms = self._tokenize(query)

            # Remove stopwords
            if self.optimization_config.enable_stopwords:
                terms = [t for t in terms if t not in self._stopwords]

            # Apply minimum term length
            terms = [t for t in terms if len(t) >= self.optimization_config.min_term_length]

            # Limit number of terms
            if len(terms) > self.optimization_config.max_terms:
                terms = terms[:self.optimization_config.max_terms]

            # Add synonyms
            if self.optimization_config.enable_synonyms:
                expanded_terms = set(terms)
                for term in terms:
                    if term in self._synonyms:
                        expanded_terms.update(self._synonyms[term])
                terms = list(expanded_terms)

            # Add ngrams
            if self.optimization_config.enable_ngrams:
                ngrams = self._generate_ngrams(
                    terms,
                    self.optimization_config.ngram_range,
                )
                terms.extend(ngrams)

            # Update frequencies
            for term in terms:
                self._term_frequencies[term] = self._term_frequencies.get(term, 0) + 1

            # Remove low frequency terms
            terms = [
                t for t in terms
                if self._term_frequencies[t] >= self.optimization_config.min_term_freq
            ]

            # Reconstruct query
            optimized_query = " ".join(terms)

            # Emit success event
            self.publisher.publish(
                Event(
                    type=EventType.OPTIMIZATION_COMPLETED,
                    timestamp=datetime.now(),
                    component="query_optimizer",
                    description="Query optimization completed",
                    metadata={
                        "original_query": query,
                        "optimized_query": optimized_query,
                        "num_terms": len(terms),
                    },
                )
            )

            return optimized_query

        except Exception as e:
            logger.error(f"Query optimization failed: {e}")
            self.publisher.publish(
                Event(
                    type=EventType.OPTIMIZATION_FAILED,
                    timestamp=datetime.now(),
                    component="query_optimizer",
                    description="Query optimization failed",
                    error=str(e),
                )
            )
            return query

    def _tokenize(self, text: str) -> List[str]:
        """Tokenize text into terms.

        Args:
            text: Text to tokenize

        Returns:
            List of terms
        """
        # TODO: Implement proper tokenization
        return text.lower().split()

    def _generate_ngrams(
        self,
        terms: List[str],
        ngram_range: tuple,
    ) -> List[str]:
        """Generate ngrams from terms.

        Args:
            terms: List of terms
            ngram_range: Range of ngram sizes

        Returns:
            List of ngrams
        """
        ngrams = []
        min_n, max_n = ngram_range
        for n in range(min_n, max_n + 1):
            for i in range(len(terms) - n + 1):
                ngram = "_".join(terms[i:i + n])
                if ngram not in self._ngram_frequencies:
                    self._ngram_frequencies[ngram] = 0
                self._ngram_frequencies[ngram] += 1
                if self._ngram_frequencies[ngram] >= self.optimization_config.min_ngram_freq:
                    ngrams.append(ngram)
        return ngrams

    async def _load_stopwords(self) -> None:
        """Load stopwords."""
        # TODO: Load stopwords from file or service
        self._stopwords = {
            "a", "an", "and", "are", "as", "at", "be", "by", "for",
            "from", "has", "he", "in", "is", "it", "its", "of", "on",
            "that", "the", "to", "was", "were", "will", "with",
        }

    async def _load_synonyms(self) -> None:
        """Load synonyms."""
        # TODO: Load synonyms from file or service
        self._synonyms = {
            "quick": ["fast", "rapid", "swift"],
            "search": ["find", "query", "lookup"],
            "create": ["add", "insert", "new"],
            "update": ["modify", "change", "edit"],
            "delete": ["remove", "destroy", "erase"],
        }


# Create service instance
query_optimizer = QueryOptimizer(
    Config(),
    PublisherService(Config(), MetricsManager()),
    MetricsManager(),
)

__all__ = [
    "OptimizationConfig",
    "QueryOptimizer",
    "query_optimizer",
] 