"""Fuzzy matching for endpoint search."""

import logging
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime

import numpy as np
from rapidfuzz import fuzz, process
from rapidfuzz.utils import default_process

from ..config import Config
from ..monitoring.events import Event, EventType
from ..monitoring.metrics import MetricsManager
from ..services.base import BaseService
from ..services.events import PublisherService
from .search_models import SearchResult

logger = logging.getLogger(__name__)


class FuzzyConfig:
    """Configuration for fuzzy matching."""

    def __init__(
        self,
        min_score: float = 60.0,  # Minimum fuzzy match score (0-100)
        max_results: int = 10,  # Maximum number of results
        score_cutoff: float = 50.0,  # Score cutoff for rapidfuzz
        processor: Optional[Any] = default_process,  # String processor
        match_type: str = "WRatio",  # Type of fuzzy matching
    ) -> None:
        """Initialize fuzzy config.

        Args:
            min_score: Minimum score threshold (0-100)
            max_results: Maximum number of results
            score_cutoff: Score cutoff for rapidfuzz
            processor: String processor function
            match_type: Type of fuzzy matching
        """
        self.min_score = min_score
        self.max_results = max_results
        self.score_cutoff = score_cutoff
        self.processor = processor
        self.match_type = match_type


class FuzzyMatcher(BaseService[Dict[str, Any]]):
    """Fuzzy matching service for endpoint search."""

    def __init__(
        self,
        config: Config,
        publisher: PublisherService,
        metrics_manager: MetricsManager,
        fuzzy_config: Optional[FuzzyConfig] = None,
    ) -> None:
        """Initialize fuzzy matcher.

        Args:
            config: Application configuration
            publisher: Event publisher
            metrics_manager: Metrics manager
            fuzzy_config: Fuzzy matching configuration
        """
        super().__init__(config, publisher, metrics_manager)
        self.fuzzy_config = fuzzy_config or FuzzyConfig()
        self._initialized = False
        self._match_func = getattr(fuzz, self.fuzzy_config.match_type)

    async def initialize(self) -> None:
        """Initialize matcher resources."""
        if self._initialized:
            return

        try:
            self._initialized = True
            self.publisher.publish(
                Event(
                    type=EventType.SERVICE_STARTED,
                    timestamp=datetime.now(),
                    component="fuzzy_matcher",
                    description="Fuzzy matcher initialized",
                    metadata={
                        "match_type": self.fuzzy_config.match_type,
                        "min_score": self.fuzzy_config.min_score,
                    },
                )
            )

        except Exception as e:
            logger.error(f"Failed to initialize fuzzy matcher: {e}")
            raise

    async def cleanup(self) -> None:
        """Clean up matcher resources."""
        self._initialized = False

        self.publisher.publish(
            Event(
                type=EventType.SERVICE_STOPPED,
                timestamp=datetime.now(),
                component="fuzzy_matcher",
                description="Fuzzy matcher stopped",
            )
        )

    async def health_check(self) -> Dict[str, Any]:
        """Check matcher health.

        Returns:
            Health check results
        """
        return {
            "service": "FuzzyMatcher",
            "initialized": self._initialized,
            "match_type": self.fuzzy_config.match_type,
            "min_score": self.fuzzy_config.min_score,
        }

    async def match(
        self,
        query: str,
        endpoints: List[Dict[str, Any]],
        min_score: Optional[float] = None,
        max_results: Optional[int] = None,
    ) -> List[SearchResult]:
        """Find fuzzy matches for query in endpoints.

        Args:
            query: Search query
            endpoints: List of endpoint dictionaries
            min_score: Optional minimum score threshold
            max_results: Optional maximum number of results

        Returns:
            List of search results with fuzzy match scores
        """
        if not self._initialized:
            await self.initialize()

        if not endpoints:
            return []

        try:
            # Emit matching started event
            self.publisher.publish(
                Event(
                    type=EventType.FUZZY_MATCHING_STARTED,
                    timestamp=datetime.now(),
                    component="fuzzy_matcher",
                    description=f"Matching query against {len(endpoints)} endpoints",
                    metadata={
                        "query": query,
                        "num_endpoints": len(endpoints),
                    },
                )
            )

            # Prepare endpoint texts for matching
            endpoint_texts = []
            for endpoint in endpoints:
                # Combine method, path and description
                text = f"{endpoint['method']} {endpoint['path']}"
                if endpoint.get("description"):
                    text += f" - {endpoint['description']}"
                endpoint_texts.append(text)

            # Perform fuzzy matching
            start_time = datetime.now()
            min_score = min_score or self.fuzzy_config.min_score
            max_results = max_results or self.fuzzy_config.max_results

            # Get matches using rapidfuzz
            matches = process.extract(
                query,
                endpoint_texts,
                scorer=self._match_func,
                processor=self.fuzzy_config.processor,
                score_cutoff=self.fuzzy_config.score_cutoff,
                limit=max_results,
            )

            # Convert matches to search results
            results = []
            for text, score, idx in matches:
                if score >= min_score:
                    endpoint = endpoints[idx]
                    result = SearchResult(
                        endpoint=endpoint["path"],
                        method=endpoint["method"],
                        description=endpoint.get("description", ""),
                        score=float(score) / 100.0,  # Normalize to 0-1
                        tags=endpoint.get("tags", []),
                        parameters=endpoint.get("parameters", []),
                        responses=endpoint.get("responses", {}),
                    )
                    results.append(result)

            # Calculate metrics
            duration = (datetime.now() - start_time).total_seconds()
            avg_score = np.mean([r.score for r in results]) if results else 0.0
            max_score = max([r.score for r in results]) if results else 0.0

            # Update metrics
            self.metrics.observe(
                "fuzzy_matching_duration",
                duration,
                {"num_endpoints": len(endpoints)},
            )
            self.metrics.observe(
                "fuzzy_matching_score",
                avg_score,
                {"type": "average"},
            )
            self.metrics.observe(
                "fuzzy_matching_score",
                max_score,
                {"type": "max"},
            )

            # Emit success event
            self.publisher.publish(
                Event(
                    type=EventType.FUZZY_MATCHING_COMPLETED,
                    timestamp=datetime.now(),
                    component="fuzzy_matcher",
                    description="Fuzzy matching completed",
                    duration_ms=duration * 1000,
                    metadata={
                        "input_endpoints": len(endpoints),
                        "output_results": len(results),
                        "avg_score": avg_score,
                        "max_score": max_score,
                    },
                )
            )

            return results

        except Exception as e:
            logger.error(f"Fuzzy matching failed: {e}")
            self.publisher.publish(
                Event(
                    type=EventType.FUZZY_MATCHING_FAILED,
                    timestamp=datetime.now(),
                    component="fuzzy_matcher",
                    description="Fuzzy matching failed",
                    error=str(e),
                )
            )
            return []


# Create service instance
fuzzy_matcher = FuzzyMatcher(
    Config(),
    PublisherService(Config(), MetricsManager()),
    MetricsManager(),
)

__all__ = ["FuzzyConfig", "FuzzyMatcher", "fuzzy_matcher"] 