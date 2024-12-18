"""Advanced API search with triple vector embeddings and contextual boosting."""

import logging
import traceback
import time
from typing import Any, Dict, List, Optional
from datetime import datetime

import numpy as np

from ..config import config_instance
from ..embedding.embeddings import EmbeddingManager
from ..integrations import pinecone_instance
from ..utils.cache import DiskCache
from .boosting import ContextualBooster, BusinessContext, BusinessValue
from .quality import QualityMetrics
from .search_models import SearchResult
from .understanding import ZeroShotUnderstanding
from ..monitoring.metrics_manager import metrics_manager
from ..monitoring.events import Event, EventType, publisher
from .vectorizer import TripleVectorizer, Triple

logger = logging.getLogger(__name__)

search_cache = DiskCache[Dict[str, Any]](
    namespace="search",
    ttl=config_instance.cache_ttl,  # 1 hour
)


class BusinessInsight:
    """Business insight for search results."""

    def __init__(
        self,
        title: str,
        description: str,
        impact_score: float,
        recommendations: List[str],
        metrics: Dict[str, float],
        category: str,
    ):
        """Initialize business insight.

        Args:
            title: Insight title
            description: Detailed description
            impact_score: Business impact score (0-1)
            recommendations: List of actionable recommendations
            metrics: Related business metrics
            category: Insight category
        """
        self.title = title
        self.description = description
        self.impact_score = impact_score
        self.recommendations = recommendations
        self.metrics = metrics
        self.category = category
        self.timestamp = datetime.now()


class APISearcher:
    """Advanced API search engine with multiple strategies."""

    def __init__(self, top_k: int = 10, use_cache: bool = True):
        """Initialize searcher.

        Args:
            top_k: Number of results to return (default: 10)
            use_cache: Whether to use caching (default: True)
        """
        self.client = pinecone_instance
        self.embedding_manager = EmbeddingManager()
        self.vectorizer = TripleVectorizer(self.embedding_manager)
        self.booster = ContextualBooster()
        self.understanding = ZeroShotUnderstanding()
        self.metrics = QualityMetrics()
        self.top_k = top_k
        self.use_cache = use_cache
        self.metrics = metrics_manager
        self.quality_metrics = QualityMetrics()

    def vectorize_query(self, query: str) -> np.ndarray:
        """Vectorize a search query.

        Args:
            query: Search query

        Returns:
            Query vector
        """
        # Emit vectorization started event
        publisher.emit(Event(
            type=EventType.EMBEDDING_STARTED,
            timestamp=datetime.now(),
            component="vectorizer",
            description=f"Vectorizing query: {query}"
        ))
        
        try:
            # Create a Triple object for the query
            query_triple = Triple(
                endpoint=query,
                method="*",
                description=query
            )
            
            # Use the vectorizer to get query embedding
            vector = self.vectorizer.vectorize(query_triple)
            
            # Emit success event
            publisher.emit(Event(
                type=EventType.EMBEDDING_COMPLETED,
                timestamp=datetime.now(),
                component="vectorizer",
                description="Query vectorization completed",
                success=True
            ))
            
            return vector
            
        except Exception as e:
            # Emit failure event
            publisher.emit(Event(
                type=EventType.EMBEDDING_FAILED,
                timestamp=datetime.now(),
                component="vectorizer",
                description="Query vectorization failed",
                error=str(e),
                success=False
            ))
            raise

    def _base_search(
        self,
        query: str,
        filters: Optional[Dict] = None,
        include_metadata: bool = True,
        use_cache: bool = True,
        enhance_results: bool = True,
    ) -> List[Dict[str, Any]]:
        """Enhanced search with caching and reranking."""
        start_time = time.time()
        
        # Emit search started event
        publisher.emit(Event(
            type=EventType.SEARCH_STARTED,
            timestamp=datetime.now(),
            component="search",
            description=f"Starting search for query: {query}",
            metadata={"filters": filters}
        ))
        
        try:
            # Check cache first
            if use_cache:
                publisher.emit(Event(
                    type=EventType.CACHE_UPDATE,
                    timestamp=datetime.now(),
                    component="search",
                    description="Checking search cache"
                ))
                
                cached_results = search_cache.get(query)
                if cached_results:
                    publisher.emit(Event(
                        type=EventType.CACHE_HIT,
                        timestamp=datetime.now(),
                        component="search",
                        description="Using cached search results",
                        metadata={"results_count": len(cached_results)}
                    ))
                    return cached_results

            publisher.emit(Event(
                type=EventType.CACHE_MISS,
                timestamp=datetime.now(),
                component="search",
                description="Cache miss, performing vector search"
            ))

            # Vector search
            publisher.emit(Event(
                type=EventType.SEARCH_QUERY_PROCESSED,
                timestamp=datetime.now(),
                component="search",
                description="Processing search query"
            ))
            
            query_vector = self.vectorize_query(query)
            
            # Use FAISS for local search
            results = []
            try:
                # Get embeddings for all endpoints
                publisher.emit(Event(
                    type=EventType.SEARCH_STARTED,
                    timestamp=datetime.now(),
                    component="faiss",
                    description="Loading endpoints for search"
                ))
                
                endpoints = self._get_endpoints()
                endpoint_vectors = []
                
                # Vectorize endpoints
                publisher.emit(Event(
                    type=EventType.EMBEDDING_STARTED,
                    timestamp=datetime.now(),
                    component="vectorizer",
                    description="Vectorizing endpoints"
                ))
                
                for endpoint in endpoints:
                    triple = Triple(
                        endpoint=endpoint["path"],
                        method=endpoint["method"],
                        description=endpoint.get("description", "")
                    )
                    vector = self.vectorizer.vectorize(triple)
                    endpoint_vectors.append((vector, endpoint))
                
                publisher.emit(Event(
                    type=EventType.EMBEDDING_COMPLETED,
                    timestamp=datetime.now(),
                    component="vectorizer",
                    description=f"Vectorized {len(endpoints)} endpoints"
                ))
                
                # Compute similarities
                publisher.emit(Event(
                    type=EventType.SEARCH_QUERY_PROCESSED,
                    timestamp=datetime.now(),
                    component="search",
                    description="Computing similarities"
                ))
                
                similarities = []
                for vector, endpoint in endpoint_vectors:
                    similarity = np.dot(query_vector, vector) / (
                        np.linalg.norm(query_vector) * np.linalg.norm(vector)
                    )
                    similarities.append((similarity, endpoint))
                
                # Sort by similarity
                similarities.sort(key=lambda x: x[0], reverse=True)
                
                # Convert to results format
                results = []
                for similarity, endpoint in similarities[:self.top_k]:
                    processed_result = {
                        "score": float(similarity),
                        "method": endpoint.get("method", ""),
                        "endpoint": endpoint.get("path", ""),
                        "description": endpoint.get("description", ""),
                        "summary": endpoint.get("summary", ""),
                        "tags": endpoint.get("tags", []),
                        "parameters": endpoint.get("parameters", []),
                        "responses": endpoint.get("responses", {}),
                        "id": endpoint.get("id", "")
                    }
                    results.append(processed_result)
                
                publisher.emit(Event(
                    type=EventType.SEARCH_RESULTS_FOUND,
                    timestamp=datetime.now(),
                    component="search",
                    description=f"Found {len(results)} initial results",
                    metadata={
                        "result_count": len(results),
                        "top_score": max(r["score"] for r in results) if results else 0
                    }
                ))
                
            except Exception as e:
                logger.error(f"Search failed: {e}")
                publisher.emit(Event(
                    type=EventType.SEARCH_FAILED,
                    timestamp=datetime.now(),
                    component="search",
                    description="Vector search failed",
                    error=str(e),
                    success=False
                ))
                results = []

            # Enhance results if needed
            if enhance_results and results:
                publisher.emit(Event(
                    type=EventType.SEARCH_STARTED,
                    timestamp=datetime.now(),
                    component="enhancer",
                    description="Enhancing search results"
                ))
                
                results = self._enhance_results(query, results)
                
                publisher.emit(Event(
                    type=EventType.SEARCH_COMPLETED,
                    timestamp=datetime.now(),
                    component="enhancer",
                    description="Results enhancement completed",
                    metadata={"enhanced_count": len(results)}
                ))

            # Cache results
            if use_cache:
                publisher.emit(Event(
                    type=EventType.CACHE_UPDATE,
                    timestamp=datetime.now(),
                    component="search",
                    description="Caching search results"
                ))
                search_cache.set(query, results)

            # Record search time
            search_time = time.time() - start_time
            self.metrics.record_search_latency(search_time)

            # Emit search completed event
            publisher.emit(Event(
                type=EventType.SEARCH_COMPLETED,
                timestamp=datetime.now(),
                component="search",
                description="Search completed successfully",
                duration_ms=search_time * 1000,
                metadata={
                    "results_count": len(results),
                    "search_time": search_time,
                    "query": query
                }
            ))

            return results

        except Exception as e:
            # Log the full traceback for debugging
            logger.error(f"Search failed: {e}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            
            # Emit search failed event
            publisher.emit(Event(
                type=EventType.SEARCH_FAILED,
                timestamp=datetime.now(),
                component="search",
                description="Search failed",
                error=str(e),
                success=False,
                metadata={"query": query}
            ))
            
            return []

    def _get_endpoints(self) -> List[Dict[str, Any]]:
        """Get all endpoints from the API directory.
        
        Returns:
            List of endpoint dictionaries
        """
        import os
        import yaml
        from pathlib import Path
        
        endpoints = []
        api_dir = Path(config_instance.api_dir)
        
        # Ensure API directory exists
        if not api_dir.exists():
            logger.error(f"API directory not found: {api_dir}")
            return []
            
        logger.info(f"Loading endpoints from {api_dir}")
        
        # Walk through all YAML files in the API directory
        for root, _, files in os.walk(api_dir):
            for file in files:
                if file.endswith(('.yaml', '.yml')):
                    try:
                        file_path = Path(root) / file
                        logger.info(f"Processing file: {file_path}")
                        
                        with open(file_path, 'r') as f:
                            spec = yaml.safe_load(f)
                            
                        # Extract paths and methods from OpenAPI spec
                        if spec and isinstance(spec, dict) and 'paths' in spec:
                            for path, methods in spec['paths'].items():
                                if not isinstance(methods, dict):
                                    continue
                                    
                                for method, details in methods.items():
                                    if not isinstance(details, dict):
                                        continue
                                        
                                    if method.lower() not in ['get', 'post', 'put', 'delete', 'patch']:
                                        continue
                                        
                                    # Create endpoint dictionary with required fields
                                    endpoint = {
                                        'id': f"{method.upper()}_{path}",
                                        'path': path,
                                        'method': method.upper(),
                                        'description': str(details.get('description', '')),
                                        'summary': str(details.get('summary', '')),
                                        'tags': list(details.get('tags', [])),
                                        'parameters': list(details.get('parameters', [])),
                                        'responses': dict(details.get('responses', {})),
                                        'file': str(file_path.relative_to(api_dir))
                                    }
                                    
                                    # Log endpoint found
                                    logger.debug(f"Found endpoint: {endpoint['method']} {endpoint['path']}")
                                    endpoints.append(endpoint)
                                    
                    except Exception as e:
                        logger.error(f"Failed to process {file}: {e}")
                        continue
        
        logger.info(f"Found {len(endpoints)} endpoints")
        return endpoints

    def _process_results(self, results: List[Dict[str, Any]]) -> List[SearchResult]:
        """Process raw search results.

        Args:
            results: Raw search results

        Returns:
            List of processed search results
        """
        processed = []
        for result in results:
            try:
                metadata = result.get("metadata", {})
                
                # Create SearchResult object
                search_result = SearchResult(
                    endpoint=metadata.get("path", ""),
                    method=metadata.get("method", ""),
                    description=metadata.get("description", ""),
                    score=float(result.get("score", 0.0)),
                    tags=metadata.get("tags", []),
                    parameters=metadata.get("parameters", []),
                    responses=metadata.get("responses", {}),
                )
                processed.append(search_result)
            except Exception as e:
                logger.error(f"Failed to process result: {e}")
                continue
        return processed

    def _rerank_results(
        self, query: str, results: List[SearchResult]
    ) -> List[SearchResult]:
        """Rerank results using cross-encoder or bi-encoder.

        Args:
            query: Original search query
            results: List of search results to rerank

        Returns:
            Reranked search results
        """
        try:
            # Try cross-encoder first
            if hasattr(self.embedding_manager, "cross_encoder") and self.embedding_manager.cross_encoder is not None:
                try:
                    # Get cross-encoder scores
                    pairs = [(query, result.description) for result in results]
                    scores = self.embedding_manager.cross_encoder.predict(pairs)

                    # Update scores and sort
                    for result, score in zip(results, scores):
                        result.score = float(score)

                    return sorted(results, key=lambda x: x.score, reverse=True)[:self.top_k]
                except Exception as e:
                    logger.warning(f"Cross-encoder reranking failed, falling back to bi-encoder: {e}")

            # Fall back to bi-encoder similarity
            query_embedding = self.embedding_manager.get_embeddings(query)
            for result in results:
                result_embedding = self.embedding_manager.get_embeddings(result.description)
                similarity = np.dot(query_embedding, result_embedding) / (
                    np.linalg.norm(query_embedding) * np.linalg.norm(result_embedding)
                )
                result.score = float((similarity + 1) / 2)  # Normalize to 0-1

            return sorted(results, key=lambda x: x.score, reverse=True)[:self.top_k]

        except Exception as e:
            logger.error(f"Reranking failed: {e}")
            return results[:self.top_k]  # Return original results if reranking fails

    def search(
        self,
        query: str,
        filters: Optional[Dict] = None,
        include_metadata: bool = True,
        enhance_results: bool = True,
    ) -> List[Dict[str, Any]]:
        """Search for API endpoints.

        Args:
            query: Search query
            filters: Optional filters to apply
            include_metadata: Whether to include metadata
            enhance_results: Whether to enhance results with LLM

        Returns:
            List of search results with scores and metadata
        """
        return self._base_search(
            query=query,
            filters=filters,
            include_metadata=include_metadata,
            use_cache=self.use_cache,
            enhance_results=enhance_results,
        )

    def _enhance_results(self, query: str, results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Enhance search results with additional information.
        
        Args:
            query: Original search query
            results: List of search results
            
        Returns:
            Enhanced results
        """
        try:
            # Get query categories
            publisher.emit(Event(
                type=EventType.SEARCH_STARTED,
                timestamp=datetime.now(),
                component="understanding",
                description="Analyzing query categories"
            ))
            
            query_categories = self.understanding.get_categories(query)
            
            publisher.emit(Event(
                type=EventType.SEARCH_COMPLETED,
                timestamp=datetime.now(),
                component="understanding",
                description=f"Found categories: {', '.join(query_categories)}",
                metadata={"categories": query_categories}
            ))
            
            # Enhance each result
            enhanced_results = []
            for result in results:
                try:
                    # Add category matches
                    endpoint_text = f"{result.get('method', '')} {result.get('endpoint', '')} {result.get('description', '')}"
                    endpoint_categories = self.understanding.get_categories(endpoint_text)
                    
                    # Calculate category overlap
                    category_overlap = set(query_categories) & set(endpoint_categories)
                    category_score = len(category_overlap) / max(len(query_categories), 1)
                    
                    # Adjust score based on category match
                    final_score = (result.get("score", 0) * 0.7 + category_score * 0.3)
                    
                    # Create enhanced result
                    enhanced_result = {
                        **result,
                        "score": float(final_score),
                        "categories": endpoint_categories,
                        "matching_categories": list(category_overlap)
                    }
                    
                    enhanced_results.append(enhanced_result)
                    
                except Exception as e:
                    logger.error(f"Failed to enhance result: {e}")
                    enhanced_results.append(result)
            
            # Sort by final score
            enhanced_results.sort(key=lambda x: x["score"], reverse=True)
            
            publisher.emit(Event(
                type=EventType.SEARCH_COMPLETED,
                timestamp=datetime.now(),
                component="enhancer",
                description="Results enhancement completed",
                metadata={
                    "enhanced_count": len(enhanced_results),
                    "categories_found": len(query_categories)
                }
            ))
            
            return enhanced_results
            
        except Exception as e:
            logger.error(f"Failed to enhance results: {e}")
            publisher.emit(Event(
                type=EventType.SEARCH_FAILED,
                timestamp=datetime.now(),
                component="enhancer",
                description="Results enhancement failed",
                error=str(e),
                success=False
            ))
            return results
