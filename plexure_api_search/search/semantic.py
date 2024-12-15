"""Semantic search implementation using vector embeddings."""

import logging
from typing import Any, Dict, List, Optional

from ..embeddings.transformer import get_embeddings
from ..storage.vector_stores.pinecone import vector_store
from .base import BaseSearch

logger = logging.getLogger(__name__)


class SemanticSearch(BaseSearch):
    """Semantic search implementation using vector embeddings."""

    def __init__(self):
        """Initialize semantic search with vector store."""
        self.vector_store = vector_store

    def prepare_query(self, query: str) -> List[float]:
        """Convert query to vector embedding.

        Args:
            query: Raw query string

        Returns:
            Query vector embedding
        """
        try:
            # Get query embedding
            query_vector = get_embeddings(query)
            return query_vector
        except Exception as e:
            logger.error(f"Error preparing query: {e}")
            raise

    def search(
        self,
        query: str,
        top_k: int = 10,
        filters: Optional[Dict] = None,
        **kwargs: Any,
    ) -> List[Dict]:
        """Search for semantically similar endpoints.

        Args:
            query: Search query string
            top_k: Number of results to return
            filters: Optional filters to apply
            **kwargs: Additional search parameters

        Returns:
            List of search results with scores and metadata
        """
        try:
            # Prepare query vector
            query_vector = self.prepare_query(query)

            # Perform vector search
            raw_results = self.vector_store.search(
                query_vector=query_vector,
                top_k=top_k,
                filters=filters,
                include_metadata=True,
            )

            # Process results
            results = self.process_results(raw_results.get("matches", []))

            logger.info(f"Found {len(results)} results for query: {query}")
            return results

        except Exception as e:
            logger.error(f"Search failed: {e}")
            return []

    def process_results(self, results: List[Dict]) -> List[Dict]:
        """Process and format search results.

        Args:
            results: Raw search results from vector store

        Returns:
            Processed and formatted results with normalized scores
        """
        processed_results = []

        try:
            for match in results:
                # Get score and metadata
                score = (1 + match.get("score", 0)) / 2  # Normalize to 0-1
                metadata = match.get("metadata", {})

                # Create result entry
                result = {
                    "score": score,
                    "api_name": metadata.get("api_name", "N/A"),
                    "api_version": metadata.get("api_version", "N/A"),
                    "method": metadata.get("method", "N/A"),
                    "path": metadata.get("path", "N/A"),
                    "description": metadata.get("description", ""),
                    "summary": metadata.get("summary", ""),
                    "parameters": metadata.get("parameters", []),
                    "responses": metadata.get("responses", []),
                    "tags": metadata.get("tags", []),
                    "requires_auth": metadata.get("requires_auth", False),
                    "deprecated": metadata.get("deprecated", False),
                }

                processed_results.append(result)

        except Exception as e:
            logger.error(f"Error processing results: {e}")

        return processed_results


# Global search instance
semantic_search = SemanticSearch()

__all__ = ["semantic_search"] 