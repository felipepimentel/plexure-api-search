"""API search with vector embeddings."""

import logging
from typing import Dict, List, Optional, Any

import numpy as np
from dependency_injector.wiring import inject
from dependency_injector.providers import Provider

from ..config import config_instance
from ..monitoring.metrics import MetricsManager
from ..services.models import model_service
from ..services.vector_store import vector_store

logger = logging.getLogger(__name__)

class APISearcher:
    """API searcher."""

    def __init__(self):
        """Initialize searcher."""
        self.metrics = MetricsManager()
        self.initialized = False

    def initialize(self) -> None:
        """Initialize searcher."""
        if self.initialized:
            return

        try:
            # Initialize services
            model_service.initialize()
            vector_store.initialize()
            self.initialized = True
            logger.info("Searcher initialized")

        except Exception as e:
            logger.error(f"Failed to initialize searcher: {e}")
            raise

    def cleanup(self) -> None:
        """Clean up searcher."""
        model_service.cleanup()
        vector_store.cleanup()
        self.initialized = False
        logger.info("Searcher cleaned up")

    def search(
        self,
        query: str,
        top_k: int = 10,
    ) -> List[Dict[str, Any]]:
        """Search for API endpoints.
        
        Args:
            query: Search query
            top_k: Number of results to return
            
        Returns:
            List of search results with scores
        """
        if not self.initialized:
            self.initialize()

        try:
            # Start timer
            start_time = self.metrics.start_timer()

            # Generate query embedding
            query_embedding = model_service.encode(query)

            # Search vectors
            distances, indices = vector_store.search_vectors(query_embedding, top_k)

            # Create results
            results = []
            for i, (distance, index) in enumerate(zip(distances, indices)):
                if index >= 0:  # Skip invalid indices
                    results.append({
                        "score": float(1.0 - distance),  # Convert distance to similarity
                        "index": int(index),
                    })

            # Stop timer
            self.metrics.stop_timer(start_time, "search")
            self.metrics.increment(
                "searches_performed",
                {"results": len(results)},
            )

            return results

        except Exception as e:
            logger.error(f"Failed to search: {e}")
            self.metrics.increment("search_errors")
            raise

# Global instance
api_searcher = APISearcher() 