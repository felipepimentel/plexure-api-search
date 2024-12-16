"""Multi-vector search strategy for complex endpoints."""

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from ...embedding.embeddings import ModernEmbeddings
from .base import BaseSearchStrategy, SearchConfig, SearchResult, StrategyFactory

logger = logging.getLogger(__name__)


@dataclass
class EndpointVector:
    """Vector representation of endpoint component."""

    component: str
    vector: np.ndarray
    weight: float


class MultiVectorStrategy(BaseSearchStrategy):
    """Multi-vector search strategy."""

    def __init__(self, config: Optional[SearchConfig] = None):
        """Initialize multi-vector strategy.

        Args:
            config: Optional strategy configuration
        """
        super().__init__(config)
        self.embeddings = ModernEmbeddings()

    def _create_endpoint_vectors(
        self,
        endpoint: Dict[str, Any],
    ) -> List[EndpointVector]:
        """Create multiple vectors for endpoint components.

        Args:
            endpoint: Endpoint data

        Returns:
            List of component vectors
        """
        vectors = []
        
        try:
            # Path vector (highest weight)
            path_text = f"{endpoint['method']} {endpoint['path']}"
            path_vector = self.embeddings.get_embeddings(path_text)
            vectors.append(EndpointVector(
                component="path",
                vector=path_vector,
                weight=0.4,
            ))
            
            # Description vector
            if endpoint.get("description"):
                desc_vector = self.embeddings.get_embeddings(endpoint["description"])
                vectors.append(EndpointVector(
                    component="description",
                    vector=desc_vector,
                    weight=0.3,
                ))
                
            # Parameters vector
            params = endpoint.get("parameters", [])
            if params:
                param_text = " ".join(str(p) for p in params)
                param_vector = self.embeddings.get_embeddings(param_text)
                vectors.append(EndpointVector(
                    component="parameters",
                    vector=param_vector,
                    weight=0.15,
                ))
                
            # Response vector
            responses = endpoint.get("responses", [])
            if responses:
                response_text = " ".join(str(r) for r in responses)
                response_vector = self.embeddings.get_embeddings(response_text)
                vectors.append(EndpointVector(
                    component="responses",
                    vector=response_vector,
                    weight=0.15,
                ))
                
            return vectors

        except Exception as e:
            logger.error(f"Error creating endpoint vectors: {e}")
            return []

    def _calculate_similarity(
        self,
        query_vector: np.ndarray,
        endpoint_vectors: List[EndpointVector],
    ) -> float:
        """Calculate weighted similarity score.

        Args:
            query_vector: Query embedding
            endpoint_vectors: List of endpoint component vectors

        Returns:
            Weighted similarity score
        """
        try:
            similarities = []
            weights = []
            
            for vec in endpoint_vectors:
                # Calculate cosine similarity
                similarity = np.dot(query_vector, vec.vector) / (
                    np.linalg.norm(query_vector) * np.linalg.norm(vec.vector)
                )
                similarities.append(similarity)
                weights.append(vec.weight)
                
            # Return weighted average
            return float(np.average(similarities, weights=weights))

        except Exception as e:
            logger.error(f"Error calculating similarity: {e}")
            return 0.0

    def _get_component_matches(
        self,
        query_vector: np.ndarray,
        endpoint_vectors: List[EndpointVector],
        threshold: float = 0.5,
    ) -> Dict[str, float]:
        """Get matching scores for each component.

        Args:
            query_vector: Query embedding
            endpoint_vectors: List of endpoint component vectors
            threshold: Minimum similarity threshold

        Returns:
            Dictionary of component matches
        """
        try:
            matches = {}
            
            for vec in endpoint_vectors:
                similarity = np.dot(query_vector, vec.vector) / (
                    np.linalg.norm(query_vector) * np.linalg.norm(vec.vector)
                )
                if similarity >= threshold:
                    matches[vec.component] = float(similarity)
                    
            return matches

        except Exception as e:
            logger.error(f"Error getting component matches: {e}")
            return {}

    def search(
        self,
        query: str,
        top_k: int = 10,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[SearchResult]:
        """Execute multi-vector search strategy.

        Args:
            query: Search query
            top_k: Number of results to return
            filters: Optional filters to apply

        Returns:
            Search results
        """
        try:
            # Get query embedding
            query_vector = self.embeddings.get_embeddings(query)
            
            # Get endpoints from vector store
            endpoints = []  # TODO: Implement retrieval from vector store
            
            # Calculate similarities using multiple vectors
            scored_results = []
            
            for endpoint in endpoints:
                # Create vectors for endpoint components
                endpoint_vectors = self._create_endpoint_vectors(endpoint)
                if not endpoint_vectors:
                    continue
                    
                # Calculate overall similarity
                similarity = self._calculate_similarity(query_vector, endpoint_vectors)
                
                # Get component matches
                matches = self._get_component_matches(query_vector, endpoint_vectors)
                
                # Create result with metadata
                scored_results.append({
                    **endpoint,
                    "score": similarity,
                    "component_matches": matches,
                })
                
            # Sort by score
            scored_results.sort(key=lambda x: x["score"], reverse=True)
            
            # Convert to SearchResult objects
            search_results = []
            for result in scored_results[:top_k]:
                search_results.append(
                    SearchResult(
                        id=result["id"],
                        score=float(result["score"]),
                        method=result["method"],
                        path=result["path"],
                        description=result["description"],
                        api_name=result["api_name"],
                        api_version=result["api_version"],
                        metadata={
                            **result.get("metadata", {}),
                            "component_matches": result["component_matches"],
                        },
                        strategy="multi_vector",
                    )
                )
                
            return search_results

        except Exception as e:
            logger.error(f"Multi-vector search failed: {e}")
            return [] 