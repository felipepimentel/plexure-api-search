"""
Search Engine Core for Plexure API Search

This module provides the core search functionality for the Plexure API Search system.
It implements semantic search over API endpoints using vector embeddings and similarity
metrics to find relevant endpoints based on natural language queries.

Key Features:
- Semantic search using vector embeddings
- Query preprocessing and expansion
- Result ranking and scoring
- Metadata enrichment
- Caching integration
- Performance optimization
- Error handling
- Monitoring

The Searcher class provides:
- Natural language query processing
- Vector similarity search
- Result ranking and filtering
- Metadata association
- Cache management
- Performance tracking
- Error recovery

Search Pipeline:
1. Query Processing:
   - Query cleaning
   - Query expansion
   - Query vectorization

2. Search Execution:
   - Vector similarity search
   - Result ranking
   - Metadata enrichment
   - Score normalization

3. Result Processing:
   - Result filtering
   - Score thresholding
   - Metadata formatting
   - Cache management

Example Usage:
    from plexure_api_search.search import Searcher

    # Initialize searcher
    searcher = Searcher()

    # Perform search
    results = searcher.search(
        query="find authentication endpoints",
        limit=10,
        min_score=0.5
    )

    # Process results
    for result in results:
        print(f"Endpoint: {result.endpoint}")
        print(f"Score: {result.score}")
        print(f"Metadata: {result.metadata}")

Performance Features:
- Query caching
- Batch processing
- Result reranking
- Score optimization
- Resource management
"""

import logging
from typing import Dict, List

from ..monitoring.metrics import MetricsManager
from ..services.models import model_service
from ..services.vector_store import vector_store

logger = logging.getLogger(__name__)


class APISearcher:
    """API endpoint searcher."""

    def __init__(self):
        """Initialize searcher."""
        self.metrics = MetricsManager()
        self.initialized = False

    def initialize(self) -> None:
        """Initialize searcher."""
        if self.initialized:
            return

        try:
            # Initialize dependencies
            model_service.initialize()
            vector_store.initialize()
            self.initialized = True
            logger.info("Searcher initialized")

        except Exception as e:
            logger.error(f"Failed to initialize searcher: {e}")
            raise

    def cleanup(self) -> None:
        """Clean up searcher."""
        if self.initialized:
            vector_store.cleanup()
            model_service.cleanup()
            self.initialized = False
            logger.info("Searcher cleaned up")

    def search(self, query: str, top_k: int = 10) -> List[Dict]:
        """Search for API endpoints.

        Args:
            query: Search query
            top_k: Number of results to return

        Returns:
            List of search results with scores and metadata
        """
        if not self.initialized:
            self.initialize()

        try:
            # Get query embedding
            query_vector = model_service.get_embeddings(query)

            # Search vector store
            distances, indices, metadata = vector_store.search_vectors(
                query_vector, k=top_k
            )

            # Process results
            results = []
            for i, (distance, index) in enumerate(zip(distances[0], indices[0])):
                # Get metadata for this result
                endpoint_metadata = metadata[i]
                if endpoint_metadata is None:
                    logger.warning(f"No metadata found for index {index}")
                    continue

                # Convert distance to similarity score (0-1)
                score = float(1.0 - distance)

                # Format result
                result = {
                    "score": score,
                    "method": endpoint_metadata.get("method", ""),
                    "path": endpoint_metadata.get("path", ""),
                    "description": endpoint_metadata.get("description", ""),
                    "summary": endpoint_metadata.get("summary", ""),
                    "parameters": endpoint_metadata.get("parameters", []),
                    "responses": endpoint_metadata.get("responses", {}),
                    "tags": endpoint_metadata.get("tags", []),
                }
                results.append(result)

            # Sort by score descending
            results.sort(key=lambda x: x["score"], reverse=True)

            # Update metrics
            self.metrics.increment_counter(
                "searches_performed", labels={"query_type": "semantic"}
            )

            return results

        except Exception as e:
            logger.error(f"Search failed: {e}")
            self.metrics.increment_counter("search_errors")
            return []


# Global instance
searcher = APISearcher()
