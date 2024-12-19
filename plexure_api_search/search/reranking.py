"""Cross-encoder reranking module."""

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
from sentence_transformers import CrossEncoder

from ..config import config_instance
from ..monitoring.metrics import MetricsManager

logger = logging.getLogger(__name__)

class Reranker:
    """Cross-encoder reranker."""

    def __init__(self):
        """Initialize reranker."""
        self.metrics = MetricsManager()
        self.initialized = False
        self.model = None

    def initialize(self) -> None:
        """Initialize reranker."""
        if self.initialized:
            return

        try:
            # Load model
            self.model = CrossEncoder(config_instance.models.cross_encoder)
            self.initialized = True
            logger.info("Reranker initialized")

        except Exception as e:
            logger.error(f"Failed to initialize reranker: {e}")
            raise

    def cleanup(self) -> None:
        """Clean up reranker."""
        self.model = None
        self.initialized = False
        logger.info("Reranker cleaned up")

    def rerank(
        self,
        query: str,
        results: List[Dict],
        top_k: Optional[int] = None,
    ) -> List[Dict]:
        """Rerank search results.
        
        Args:
            query: Search query
            results: Search results to rerank
            top_k: Number of results to return
            
        Returns:
            Reranked search results
        """
        if not self.initialized:
            self.initialize()

        try:
            # Start timer
            start_time = self.metrics.start_timer()

            # Create input pairs
            pairs = []
            for result in results:
                text = result.get("text", "")
                pairs.append([query, text])

            # Get cross-encoder scores
            scores = self.model.predict(
                pairs,
                show_progress_bar=False,
            )

            # Update scores
            for i, result in enumerate(results):
                result["rerank_score"] = float(scores[i])
                result["score"] = (result["score"] + result["rerank_score"]) / 2

            # Sort by score
            results.sort(key=lambda x: x["score"], reverse=True)

            # Take top results
            if top_k:
                results = results[:top_k]

            # Stop timer
            self.metrics.stop_timer(
                start_time,
                "reranking",
                {"query": query},
            )

            return results

        except Exception as e:
            logger.error(f"Failed to rerank results: {e}")
            self.metrics.increment(
                "reranking_errors",
                {"query": query},
            )
            return results

# Global instance
reranker = Reranker() 