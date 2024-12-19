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
        min_score: float = 0.1,
        expand_query: bool = True,
        rerank_results: bool = True,
    ) -> List[Dict[str, Any]]:
        """Search for API endpoints.
        
        Args:
            query: Search query
            top_k: Number of results to return
            min_score: Minimum similarity score (0.0 to 1.0)
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

            # Log search parameters
            logger.debug(f"Searching with query: {query}")
            logger.debug(f"Parameters: top_k={top_k}, min_score={min_score}")

            # Clean query
            query = query.strip()
            if not query:
                logger.warning("Empty query")
                return []

            # Generate query embedding
            query_embedding = model_service.encode(query)
            logger.debug(f"Generated query embedding shape: {query_embedding.shape}")

            # Search vectors
            distances, indices = vector_store.search_vectors(query_embedding, top_k)
            logger.debug(f"Raw search results: {len(indices)} indices, {len(distances)} distances")
            if len(indices) > 0:
                logger.debug(f"Distance range: {np.min(distances)} to {np.max(distances)}")

            # Create results
            results = []
            for i, (distance, index) in enumerate(zip(distances, indices)):
                if index >= 0:  # Skip invalid indices
                    score = float(1.0 - distance)  # Convert distance to similarity
                    logger.debug(f"Result {i}: index={index}, distance={distance}, score={score}")
                    if score >= min_score:
                        # Get endpoint metadata
                        metadata = api_indexer.endpoint_metadata.get(int(index), {})
                        if not metadata:
                            logger.warning(f"No metadata found for index {index}")
                            continue

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
                        logger.debug(f"Added result: {result['method']} {result['endpoint']}")

            # Sort results by score
            results.sort(key=lambda x: x["score"], reverse=True)

            # Log results summary
            logger.debug(f"Found {len(results)} results above threshold")
            if results:
                logger.debug(f"Top score: {results[0]['score']}")
                logger.debug(f"Score range: {results[-1]['score']} to {results[0]['score']}")

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