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
from ..indexing import api_indexer

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
        min_score: float = 0.3,
        expand_query: bool = True,
        rerank_results: bool = True,
    ) -> List[Dict[str, Any]]:
        """Search for API endpoints.
        
        Args:
            query: Search query
            top_k: Number of results to return
            min_score: Minimum similarity score
            expand_query: Whether to expand query
            rerank_results: Whether to rerank results
            
        Returns:
            List of search results with scores and metadata
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
                    score = float(1.0 - distance)  # Convert distance to similarity
                    if score >= min_score:
                        # Get endpoint metadata
                        metadata = api_indexer.endpoint_metadata.get(int(index), {})
                        result = {
                            "score": score,
                            "method": metadata.get("method", ""),
                            "endpoint": metadata.get("path", ""),
                            "description": metadata.get("description", ""),
                            "summary": metadata.get("summary", ""),
                            "parameters": metadata.get("parameters", []),
                            "responses": metadata.get("responses", {}),
                            "tags": metadata.get("tags", []),
                        }
                        results.append(result)

            # Stop timer and update metrics
            self.metrics.stop_timer(start_time, "search", {"operation": "search"})
            self.metrics.increment(
                "searches_performed",
                {"status": "success", "count": len(results)},
            )

            return results

        except Exception as e:
            logger.error(f"Failed to search: {e}")
            self.metrics.increment("search_errors")
            raise

# Global instance
api_searcher = APISearcher() 