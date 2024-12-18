"""Result diversification for improved search quality."""

import logging
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from ..config import Config
from ..monitoring.events import Event, EventType
from ..monitoring.metrics import MetricsManager
from ..services.base import BaseService
from ..services.events import PublisherService
from .search_models import SearchResult

logger = logging.getLogger(__name__)


class DiversityConfig:
    """Configuration for result diversification."""

    def __init__(
        self,
        min_diversity_score: float = 0.3,
        max_similarity: float = 0.8,
        min_results: int = 3,
        max_results: int = 10,
        similarity_threshold: float = 0.95,
    ) -> None:
        """Initialize diversity config.

        Args:
            min_diversity_score: Minimum diversity score
            max_similarity: Maximum allowed similarity
            min_results: Minimum number of results
            max_results: Maximum number of results
            similarity_threshold: Similarity threshold
        """
        self.min_diversity_score = min_diversity_score
        self.max_similarity = max_similarity
        self.min_results = min_results
        self.max_results = max_results
        self.similarity_threshold = similarity_threshold


class DiversityManager(BaseService[Dict[str, Any]]):
    """Diversity management service for search results."""

    def __init__(
        self,
        config: Config,
        publisher: PublisherService,
        metrics_manager: MetricsManager,
        diversity_config: Optional[DiversityConfig] = None,
    ) -> None:
        """Initialize diversity manager.

        Args:
            config: Application configuration
            publisher: Event publisher
            metrics_manager: Metrics manager
            diversity_config: Diversity configuration
        """
        super().__init__(config, publisher, metrics_manager)
        self.diversity_config = diversity_config or DiversityConfig()
        self._initialized = False

    async def initialize(self) -> None:
        """Initialize diversity resources."""
        if self._initialized:
            return

        try:
            self._initialized = True
            self.publisher.publish(
                Event(
                    type=EventType.SERVICE_STARTED,
                    timestamp=datetime.now(),
                    component="diversity_manager",
                    description="Diversity manager initialized",
                    metadata={
                        "min_diversity": self.diversity_config.min_diversity_score,
                        "max_similarity": self.diversity_config.max_similarity,
                    },
                )
            )

        except Exception as e:
            logger.error(f"Failed to initialize diversity manager: {e}")
            raise

    async def cleanup(self) -> None:
        """Clean up diversity resources."""
        self._initialized = False

        self.publisher.publish(
            Event(
                type=EventType.SERVICE_STOPPED,
                timestamp=datetime.now(),
                component="diversity_manager",
                description="Diversity manager stopped",
            )
        )

    async def health_check(self) -> Dict[str, Any]:
        """Check diversity manager health.

        Returns:
            Health check results
        """
        return {
            "service": "DiversityManager",
            "initialized": self._initialized,
            "min_diversity": self.diversity_config.min_diversity_score,
            "max_similarity": self.diversity_config.max_similarity,
        }

    async def diversify_results(
        self,
        results: List[SearchResult],
        vectors: List[np.ndarray],
        min_results: Optional[int] = None,
        max_results: Optional[int] = None,
    ) -> List[SearchResult]:
        """Diversify search results based on vectors.

        Args:
            results: Search results to diversify
            vectors: Result vectors for similarity
            min_results: Optional minimum results
            max_results: Optional maximum results

        Returns:
            Diversified search results
        """
        if not self._initialized:
            await self.initialize()

        if not results or not vectors:
            return []

        try:
            # Emit diversification started event
            self.publisher.publish(
                Event(
                    type=EventType.DIVERSIFICATION_STARTED,
                    timestamp=datetime.now(),
                    component="diversity_manager",
                    description=f"Diversifying {len(results)} results",
                    metadata={
                        "num_results": len(results),
                    },
                )
            )

            # Convert vectors to numpy array
            X = np.array(vectors)

            # Get result range
            min_k = min_results or self.diversity_config.min_results
            max_k = min(
                max_results or self.diversity_config.max_results,
                len(results),
            )

            # Calculate similarity matrix
            similarity_matrix = cosine_similarity(X)

            # Initialize selected indices
            selected = []
            remaining = list(range(len(results)))

            # Select first result (highest score)
            selected.append(remaining.pop(0))

            # Select remaining results
            while len(selected) < max_k and remaining:
                # Calculate maximum similarity to selected results
                max_similarities = np.max(
                    similarity_matrix[remaining][:, selected],
                    axis=1,
                )

                # Find result with lowest maximum similarity
                best_idx = np.argmin(max_similarities)
                if max_similarities[best_idx] < self.diversity_config.max_similarity:
                    selected.append(remaining.pop(best_idx))
                else:
                    break

            # Check if we have enough results
            if len(selected) < min_k:
                self.publisher.publish(
                    Event(
                        type=EventType.DIVERSIFICATION_COMPLETED,
                        timestamp=datetime.now(),
                        component="diversity_manager",
                        description="Diversification skipped (insufficient results)",
                        metadata={
                            "selected_results": len(selected),
                            "min_results": min_k,
                        },
                    )
                )
                return results[:max_k]

            # Calculate diversity score
            if len(selected) > 1:
                diversity_score = 1.0 - np.mean(
                    similarity_matrix[selected][:, selected]
                )
            else:
                diversity_score = 1.0

            # Skip diversification if score is too low
            if diversity_score < self.diversity_config.min_diversity_score:
                self.publisher.publish(
                    Event(
                        type=EventType.DIVERSIFICATION_COMPLETED,
                        timestamp=datetime.now(),
                        component="diversity_manager",
                        description="Diversification skipped (low diversity)",
                        metadata={
                            "diversity_score": diversity_score,
                        },
                    )
                )
                return results[:max_k]

            # Get diversified results
            diversified = [results[i] for i in selected]

            # Update result scores based on diversity
            for result in diversified:
                result.score *= (1.0 + diversity_score) / 2.0

            # Emit success event
            self.publisher.publish(
                Event(
                    type=EventType.DIVERSIFICATION_COMPLETED,
                    timestamp=datetime.now(),
                    component="diversity_manager",
                    description="Diversification completed",
                    metadata={
                        "input_results": len(results),
                        "output_results": len(diversified),
                        "diversity_score": diversity_score,
                    },
                )
            )

            return diversified

        except Exception as e:
            logger.error(f"Diversification failed: {e}")
            self.publisher.publish(
                Event(
                    type=EventType.DIVERSIFICATION_FAILED,
                    timestamp=datetime.now(),
                    component="diversity_manager",
                    description="Diversification failed",
                    error=str(e),
                )
            )
            return results[:max_results or self.diversity_config.max_results]


# Create service instance
diversity_manager = DiversityManager(
    Config(),
    PublisherService(Config(), MetricsManager()),
    MetricsManager(),
)

__all__ = ["DiversityConfig", "DiversityManager", "diversity_manager"] 