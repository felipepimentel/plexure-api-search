"""Query intent detection module."""

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
from sentence_transformers import util

from ..config import config_instance
from ..monitoring.metrics import MetricsManager
from ..services.models import ModelService

logger = logging.getLogger(__name__)

class IntentDetector:
    """Query intent detector."""

    def __init__(self):
        """Initialize intent detector."""
        self.model_service = ModelService()
        self.metrics = MetricsManager()
        self.initialized = False

    def initialize(self) -> None:
        """Initialize intent detector."""
        if self.initialized:
            return

        try:
            # Initialize model service
            self.model_service.initialize()
            self.initialized = True
            logger.info("Intent detector initialized")

        except Exception as e:
            logger.error(f"Failed to initialize intent detector: {e}")
            raise

    def cleanup(self) -> None:
        """Clean up intent detector."""
        self.model_service.cleanup()
        self.initialized = False
        logger.info("Intent detector cleaned up")

    def detect(self, query: str) -> Dict[str, float]:
        """Detect query intent.
        
        Args:
            query: Search query
            
        Returns:
            Dictionary of intent types and scores
        """
        if not self.initialized:
            self.initialize()

        try:
            # Start timer
            start_time = self.metrics.start_timer()

            # Encode query
            query_embedding = self.model_service.encode(query)

            # Calculate similarities with intent types
            intent_scores = {}
            for intent_type, intent_embedding in self._get_intent_embeddings().items():
                similarity = util.pytorch_cos_sim(
                    query_embedding,
                    intent_embedding,
                )[0][0].item()
                intent_scores[intent_type] = similarity

            # Stop timer
            self.metrics.stop_timer(
                start_time,
                "intent_detection",
                {"query": query},
            )

            return intent_scores

        except Exception as e:
            logger.error(f"Failed to detect intent: {e}")
            self.metrics.increment(
                "intent_errors",
                {"query": query},
            )
            raise

    def _get_intent_embeddings(self) -> Dict[str, np.ndarray]:
        """Get intent embeddings.
        
        Returns:
            Dictionary of intent types and embeddings
        """
        # Define intent types and examples
        intent_types = {
            "search": [
                "find endpoints",
                "search for APIs",
                "look for endpoints",
                "show me APIs",
            ],
            "filter": [
                "filter endpoints",
                "show only",
                "exclude",
                "include",
            ],
            "compare": [
                "compare endpoints",
                "difference between",
                "which is better",
                "pros and cons",
            ],
            "explain": [
                "explain endpoint",
                "how does it work",
                "what does it do",
                "tell me about",
            ],
            "example": [
                "show example",
                "give me sample",
                "usage example",
                "how to use",
            ],
        }

        # Encode intent examples
        intent_embeddings = {}
        for intent_type, examples in intent_types.items():
            embeddings = self.model_service.encode(examples)
            intent_embeddings[intent_type] = np.mean(embeddings, axis=0)

        return intent_embeddings

# Global instance
intent_detector = IntentDetector() 