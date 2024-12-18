"""Cross-encoder reranking for improved search quality."""

import logging
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime

import numpy as np
import torch
from sentence_transformers import CrossEncoder

from ..config import Config
from ..monitoring.events import Event, EventType
from ..monitoring.metrics import MetricsManager
from ..services.base import BaseService
from ..services.events import PublisherService
from ..services.models import ModelService
from .search_models import SearchResult

logger = logging.getLogger(__name__)


class RerankerConfig:
    """Configuration for reranking."""

    def __init__(
        self,
        model_name: str = "cross-encoder/stsb-roberta-base",
        batch_size: int = 32,
        min_score: float = 0.5,
        max_results: int = 10,
        use_gpu: bool = False,
    ) -> None:
        """Initialize reranker config.

        Args:
            model_name: Name of cross-encoder model
            batch_size: Batch size for inference
            min_score: Minimum score threshold
            max_results: Maximum number of results
            use_gpu: Whether to use GPU
        """
        self.model_name = model_name
        self.batch_size = batch_size
        self.min_score = min_score
        self.max_results = max_results
        self.use_gpu = use_gpu


class Reranker(BaseService[Dict[str, Any]]):
    """Cross-encoder reranking service."""

    def __init__(
        self,
        config: Config,
        publisher: PublisherService,
        metrics_manager: MetricsManager,
        model_service: ModelService,
        reranker_config: Optional[RerankerConfig] = None,
    ) -> None:
        """Initialize reranker.

        Args:
            config: Application configuration
            publisher: Event publisher
            metrics_manager: Metrics manager
            model_service: Model service
            reranker_config: Reranker configuration
        """
        super().__init__(config, publisher, metrics_manager)
        self.model_service = model_service
        self.reranker_config = reranker_config or RerankerConfig()
        self._initialized = False
        self._model: Optional[CrossEncoder] = None

    async def initialize(self) -> None:
        """Initialize reranker resources."""
        if self._initialized:
            return

        try:
            # Load cross-encoder model
            self._model = CrossEncoder(
                self.reranker_config.model_name,
                device="cuda" if self.reranker_config.use_gpu else "cpu",
            )
            self._initialized = True

            self.publisher.publish(
                Event(
                    type=EventType.SERVICE_STARTED,
                    timestamp=datetime.now(),
                    component="reranker",
                    description="Reranker initialized",
                    metadata={
                        "model": self.reranker_config.model_name,
                        "use_gpu": self.reranker_config.use_gpu,
                    },
                )
            )

        except Exception as e:
            logger.error(f"Failed to initialize reranker: {e}")
            raise

    async def cleanup(self) -> None:
        """Clean up reranker resources."""
        self._initialized = False
        self._model = None

        self.publisher.publish(
            Event(
                type=EventType.SERVICE_STOPPED,
                timestamp=datetime.now(),
                component="reranker",
                description="Reranker stopped",
            )
        )

    async def health_check(self) -> Dict[str, Any]:
        """Check reranker health.

        Returns:
            Health check results
        """
        return {
            "service": "Reranker",
            "initialized": self._initialized,
            "model_loaded": self._model is not None,
            "model_name": self.reranker_config.model_name,
            "use_gpu": self.reranker_config.use_gpu,
        }

    async def rerank(
        self,
        query: str,
        results: List[SearchResult],
        min_score: Optional[float] = None,
        max_results: Optional[int] = None,
    ) -> List[SearchResult]:
        """Rerank search results using cross-encoder.

        Args:
            query: Original search query
            results: Initial search results
            min_score: Optional minimum score threshold
            max_results: Optional maximum number of results

        Returns:
            Reranked search results
        """
        if not self._initialized:
            await self.initialize()

        if not results:
            return []

        try:
            # Emit reranking started event
            self.publisher.publish(
                Event(
                    type=EventType.RERANKING_STARTED,
                    timestamp=datetime.now(),
                    component="reranker",
                    description=f"Reranking {len(results)} results",
                    metadata={
                        "query": query,
                        "num_results": len(results),
                    },
                )
            )

            # Prepare text pairs for cross-encoder
            pairs = []
            for result in results:
                # Combine method, endpoint and description for better matching
                text = f"{result.method} {result.endpoint}"
                if result.description:
                    text += f" - {result.description}"
                pairs.append([query, text])

            # Get cross-encoder scores in batches
            start_time = datetime.now()
            scores = []
            batch_size = self.reranker_config.batch_size

            for i in range(0, len(pairs), batch_size):
                batch = pairs[i:i + batch_size]
                batch_scores = self._model.predict(
                    batch,
                    show_progress_bar=False,
                    convert_to_numpy=True,
                )
                scores.extend(batch_scores)

            # Update scores and filter results
            min_score = min_score or self.reranker_config.min_score
            max_results = max_results or self.reranker_config.max_results

            reranked = []
            for result, score in zip(results, scores):
                if score >= min_score:
                    result.score = float(score)
                    reranked.append(result)

            # Sort by score and limit results
            reranked.sort(key=lambda x: x.score, reverse=True)
            reranked = reranked[:max_results]

            # Calculate metrics
            duration = (datetime.now() - start_time).total_seconds()
            avg_score = np.mean([r.score for r in reranked]) if reranked else 0.0
            max_score = max([r.score for r in reranked]) if reranked else 0.0

            # Update metrics
            self.metrics.observe(
                "reranking_duration",
                duration,
                {"batch_size": batch_size},
            )
            self.metrics.observe(
                "reranking_score",
                avg_score,
                {"type": "average"},
            )
            self.metrics.observe(
                "reranking_score",
                max_score,
                {"type": "max"},
            )

            # Emit success event
            self.publisher.publish(
                Event(
                    type=EventType.RERANKING_COMPLETED,
                    timestamp=datetime.now(),
                    component="reranker",
                    description="Reranking completed",
                    duration_ms=duration * 1000,
                    metadata={
                        "input_results": len(results),
                        "output_results": len(reranked),
                        "avg_score": avg_score,
                        "max_score": max_score,
                    },
                )
            )

            return reranked

        except Exception as e:
            logger.error(f"Reranking failed: {e}")
            self.publisher.publish(
                Event(
                    type=EventType.RERANKING_FAILED,
                    timestamp=datetime.now(),
                    component="reranker",
                    description="Reranking failed",
                    error=str(e),
                )
            )
            return results  # Return original results on failure


# Create service instance
reranker = Reranker(
    Config(),
    PublisherService(Config(), MetricsManager()),
    MetricsManager(),
    ModelService(Config(), PublisherService(Config(), MetricsManager()), MetricsManager()),
)

__all__ = ["RerankerConfig", "Reranker", "reranker"] 