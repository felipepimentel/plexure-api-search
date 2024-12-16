"""Endpoint storage and retrieval management."""

import logging
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
from pydantic import BaseModel

from ..config import config_instance
from ..embedding.embeddings import ModernEmbeddings
from ..utils.cache import DiskCache

logger = logging.getLogger(__name__)


class EndpointData(BaseModel):
    """Endpoint data model."""

    id: str
    method: str
    path: str
    description: str
    api_name: str
    api_version: str
    parameters: List[Any]
    responses: List[Any]
    tags: List[str]
    requires_auth: bool
    deprecated: bool
    metadata: Dict[str, Any]
    vector: Optional[List[float]] = None
    sparse_vector: Optional[List[float]] = None


class EndpointStore:
    """Manages endpoint storage and retrieval."""

    def __init__(self):
        """Initialize endpoint store."""
        self.embeddings = ModernEmbeddings()
        self.endpoints: Dict[str, EndpointData] = {}
        self.vector_matrix: Optional[np.ndarray] = None
        self.id_mapping: Dict[str, int] = {}
        self.reverse_mapping: Dict[int, str] = {}
        
        # Cache for endpoint data
        self.cache = DiskCache[Dict[str, Any]](
            namespace="endpoint_store",
            ttl=config_instance.cache_ttl * 24,  # Cache for longer
        )

    def add_endpoint(self, endpoint: Dict[str, Any]) -> None:
        """Add endpoint to store.

        Args:
            endpoint: Endpoint data
        """
        try:
            # Create endpoint model
            endpoint_data = EndpointData(**endpoint)
            
            # Generate vector if not provided
            if not endpoint_data.vector:
                text = self._prepare_endpoint_text(endpoint_data)
                vector = self.embeddings.get_embeddings(text)
                endpoint_data.vector = vector.tolist()
                
            # Add to storage
            self.endpoints[endpoint_data.id] = endpoint_data
            
            # Update vector matrix
            self._update_vector_matrix()
            
            # Cache endpoint data
            self.cache.set(endpoint_data.id, endpoint_data.dict())
            
            logger.info(f"Added endpoint {endpoint_data.method} {endpoint_data.path}")

        except Exception as e:
            logger.error(f"Failed to add endpoint: {e}")

    def get_endpoint(self, endpoint_id: str) -> Optional[EndpointData]:
        """Get endpoint by ID.

        Args:
            endpoint_id: Endpoint ID

        Returns:
            Endpoint data if found
        """
        try:
            # Check cache first
            cached = self.cache.get(endpoint_id)
            if cached:
                return EndpointData(**cached)
                
            # Check in-memory storage
            return self.endpoints.get(endpoint_id)

        except Exception as e:
            logger.error(f"Failed to get endpoint {endpoint_id}: {e}")
            return None

    def search_endpoints(
        self,
        query: str,
        top_k: int = 10,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[Tuple[EndpointData, float]]:
        """Search endpoints using vector similarity.

        Args:
            query: Search query
            top_k: Number of results to return
            filters: Optional filters to apply

        Returns:
            List of (endpoint, score) tuples
        """
        try:
            if not self.vector_matrix:
                return []
                
            # Get query vector
            query_vector = self.embeddings.get_embeddings(query)
            
            # Calculate similarities
            similarities = np.dot(self.vector_matrix, query_vector) / (
                np.linalg.norm(self.vector_matrix, axis=1) * np.linalg.norm(query_vector)
            )
            
            # Get top k indices
            top_indices = np.argsort(similarities)[-top_k:][::-1]
            
            # Get results
            results = []
            for idx in top_indices:
                endpoint_id = self.reverse_mapping[idx]
                endpoint = self.get_endpoint(endpoint_id)
                if endpoint:
                    score = float(similarities[idx])
                    
                    # Apply filters if provided
                    if filters and not self._apply_filters(endpoint, filters):
                        continue
                        
                    results.append((endpoint, score))
                    
            return results

        except Exception as e:
            logger.error(f"Failed to search endpoints: {e}")
            return []

    def _prepare_endpoint_text(self, endpoint: EndpointData) -> str:
        """Prepare endpoint text for embedding.

        Args:
            endpoint: Endpoint data

        Returns:
            Formatted endpoint text
        """
        try:
            parts = []
            
            # Add method and path
            parts.append(f"{endpoint.method} {endpoint.path}")
            
            # Add description
            if endpoint.description:
                parts.append(endpoint.description)
                
            # Add parameters
            if endpoint.parameters:
                param_text = "Parameters: " + ", ".join(str(p) for p in endpoint.parameters)
                parts.append(param_text)
                
            # Add responses
            if endpoint.responses:
                response_text = "Responses: " + ", ".join(str(r) for r in endpoint.responses)
                parts.append(response_text)
                
            # Add tags
            if endpoint.tags:
                parts.append("Tags: " + ", ".join(endpoint.tags))
                
            return " | ".join(parts)

        except Exception as e:
            logger.error(f"Failed to prepare endpoint text: {e}")
            return ""

    def _update_vector_matrix(self) -> None:
        """Update vector matrix for efficient similarity search."""
        try:
            # Create new mappings
            self.id_mapping = {
                endpoint_id: i
                for i, endpoint_id in enumerate(self.endpoints.keys())
            }
            self.reverse_mapping = {
                i: endpoint_id
                for endpoint_id, i in self.id_mapping.items()
            }
            
            # Create vector matrix
            vectors = []
            for endpoint_id in self.endpoints:
                endpoint = self.endpoints[endpoint_id]
                if endpoint.vector:
                    vectors.append(endpoint.vector)
                    
            if vectors:
                self.vector_matrix = np.array(vectors)
                logger.info(f"Updated vector matrix with shape {self.vector_matrix.shape}")
            else:
                self.vector_matrix = None
                logger.warning("No vectors available for matrix")

        except Exception as e:
            logger.error(f"Failed to update vector matrix: {e}")
            self.vector_matrix = None

    def _apply_filters(self, endpoint: EndpointData, filters: Dict[str, Any]) -> bool:
        """Apply filters to endpoint.

        Args:
            endpoint: Endpoint data
            filters: Filters to apply

        Returns:
            True if endpoint passes filters
        """
        try:
            for field, value in filters.items():
                if field == "method" and value.upper() != endpoint.method.upper():
                    return False
                    
                elif field == "api_name" and value != endpoint.api_name:
                    return False
                    
                elif field == "api_version" and value != endpoint.api_version:
                    return False
                    
                elif field == "requires_auth" and value != endpoint.requires_auth:
                    return False
                    
                elif field == "deprecated" and value != endpoint.deprecated:
                    return False
                    
                elif field == "tags" and not any(tag in endpoint.tags for tag in value):
                    return False
                    
            return True

        except Exception as e:
            logger.error(f"Failed to apply filters: {e}")
            return False


# Global endpoint store instance
endpoint_store = EndpointStore()

__all__ = ["endpoint_store", "EndpointData", "EndpointStore"] 