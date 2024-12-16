"""Specialized caching system for search scores."""

import hashlib
import json
import logging
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import numpy as np

from ..config import config_instance
from ..monitoring.cache_metrics import cache_metrics_manager
from ..utils.cache import DiskCache
from ..utils.compression import CompressedCache, CompressionMethod
from .storage import EndpointData

logger = logging.getLogger(__name__)


@dataclass
class ScoreData:
    """Score data with metadata."""

    endpoint_id: str
    score: float
    strategy: str
    components: Dict[str, float]
    timestamp: float


class ScoreCache:
    """Cache for search scores."""

    def __init__(self, strategy_name: str):
        """Initialize score cache.

        Args:
            strategy_name: Name of the search strategy
        """
        self.strategy_name = strategy_name
        base_cache = DiskCache[Dict[str, Any]](
            namespace=f"scores_{strategy_name}",
            ttl=config_instance.semantic_cache_ttl,
        )
        self.cache = CompressedCache(
            cache=base_cache,
            method=CompressionMethod.GZIP,
            compression_level=6,
        )

    def _generate_key(
        self,
        query: str,
        filters: Optional[Dict[str, Any]] = None,
        extra_context: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Generate cache key.

        Args:
            query: Search query
            filters: Optional filters
            extra_context: Optional additional context

        Returns:
            Cache key
        """
        try:
            # Create key components
            key_parts = [
                ("query", query.lower().strip()),
                ("filters", json.dumps(filters, sort_keys=True) if filters else ""),
                ("context", json.dumps(extra_context, sort_keys=True) if extra_context else ""),
                ("strategy", self.strategy_name),
            ]
            
            # Create key string
            key_str = "|".join(f"{k}:{v}" for k, v in key_parts)
            
            # Hash key
            return hashlib.sha256(key_str.encode()).hexdigest()

        except Exception as e:
            logger.error(f"Failed to generate cache key: {e}")
            return query

    def get_scores(
        self,
        query: str,
        filters: Optional[Dict[str, Any]] = None,
        extra_context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, ScoreData]:
        """Get cached scores.

        Args:
            query: Search query
            filters: Optional filters
            extra_context: Optional additional context

        Returns:
            Dictionary of endpoint ID to score data
        """
        try:
            start_time = time.time()
            
            # Generate cache key
            key = self._generate_key(query, filters, extra_context)
            
            # Get from cache
            cached = self.cache.get(key)
            latency = time.time() - start_time
            
            if not cached:
                # Track cache miss
                cache_metrics_manager.track_cache_miss(
                    strategy=self.strategy_name,
                    latency=latency,
                    cache_type="disk",
                )
                return {}
                
            # Track cache hit
            cache_metrics_manager.track_cache_hit(
                strategy=self.strategy_name,
                latency=latency,
                cache_type="disk",
            )
                
            # Convert to ScoreData objects
            scores = {}
            for endpoint_id, data in cached.items():
                scores[endpoint_id] = ScoreData(**data)
                
            # Update cache stats
            compression_stats = self.cache.get_compression_stats()
            if key in compression_stats["by_key"]:
                stats = compression_stats["by_key"][key]
                cache_metrics_manager.update_cache_stats(
                    strategy=self.strategy_name,
                    num_items=len(scores),
                    memory_bytes=stats["compressed_size"],
                )
                
            return scores

        except Exception as e:
            logger.error(f"Failed to get cached scores: {e}")
            return {}

    def set_scores(
        self,
        query: str,
        scores: Dict[str, ScoreData],
        filters: Optional[Dict[str, Any]] = None,
        extra_context: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Cache scores.

        Args:
            query: Search query
            scores: Score data by endpoint ID
            filters: Optional filters
            extra_context: Optional additional context
        """
        try:
            start_time = time.time()
            
            # Generate cache key
            key = self._generate_key(query, filters, extra_context)
            
            # Convert to dictionaries
            cache_data = {
                endpoint_id: {
                    "endpoint_id": score.endpoint_id,
                    "score": score.score,
                    "strategy": score.strategy,
                    "components": score.components,
                    "timestamp": score.timestamp,
                }
                for endpoint_id, score in scores.items()
            }
            
            # Save to cache
            self.cache.set(key, cache_data)
            
            # Track latency
            latency = time.time() - start_time
            cache_metrics_manager.cache_latency.labels(
                operation="set",
                strategy=self.strategy_name,
            ).observe(latency)
            
            # Update cache stats
            compression_stats = self.cache.get_compression_stats()
            if key in compression_stats["by_key"]:
                stats = compression_stats["by_key"][key]
                cache_metrics_manager.update_cache_stats(
                    strategy=self.strategy_name,
                    num_items=len(scores),
                    memory_bytes=stats["compressed_size"],
                )

        except Exception as e:
            logger.error(f"Failed to cache scores: {e}")

    def invalidate(
        self,
        endpoint_ids: Optional[Set[str]] = None,
    ) -> None:
        """Invalidate cache entries.

        Args:
            endpoint_ids: Optional set of endpoint IDs to invalidate
        """
        try:
            if endpoint_ids:
                # Get all cache keys
                all_keys = self.cache.cache.get_all_keys()
                
                # Track number of invalidations
                num_invalidated = 0
                
                # Check each key
                for key in all_keys:
                    cached = self.cache.get(key)
                    if not cached:
                        continue
                        
                    # Check if any endpoint ID matches
                    if any(endpoint_id in cached for endpoint_id in endpoint_ids):
                        self.cache.delete(key)
                        num_invalidated += 1
                        
                # Track invalidations
                if num_invalidated > 0:
                    cache_metrics_manager.track_cache_invalidation(
                        strategy=self.strategy_name,
                        num_items=num_invalidated,
                    )
            else:
                # Clear entire cache
                self.cache.clear()
                
                # Track full invalidation
                cache_metrics_manager.track_cache_invalidation(
                    strategy=self.strategy_name,
                    num_items=1000,  # Use large number to indicate full clear
                )

        except Exception as e:
            logger.error(f"Failed to invalidate cache: {e}")


class SemanticScoreCache(ScoreCache):
    """Score cache with semantic similarity matching."""

    def __init__(
        self,
        strategy_name: str,
        similarity_threshold: float = 0.8,
    ):
        """Initialize semantic score cache.

        Args:
            strategy_name: Name of the search strategy
            similarity_threshold: Minimum similarity for cache hit
        """
        super().__init__(strategy_name)
        self.similarity_threshold = similarity_threshold
        self.query_vectors: Dict[str, np.ndarray] = {}

    def _get_query_vector(self, query: str) -> Optional[np.ndarray]:
        """Get query vector.

        Args:
            query: Search query

        Returns:
            Query vector if available
        """
        try:
            start_time = time.time()
            
            # Check if vector exists
            if query in self.query_vectors:
                # Track vector cache hit
                latency = time.time() - start_time
                cache_metrics_manager.track_cache_hit(
                    strategy=f"{self.strategy_name}_vector",
                    latency=latency,
                    cache_type="memory",
                )
                return self.query_vectors[query]
                
            # Generate vector
            from ..embedding.embeddings import embeddings
            vector = embeddings.get_embeddings(query)
            
            # Track vector cache miss
            latency = time.time() - start_time
            cache_metrics_manager.track_cache_miss(
                strategy=f"{self.strategy_name}_vector",
                latency=latency,
                cache_type="memory",
            )
            
            # Cache vector
            self.query_vectors[query] = vector
            
            # Update vector cache stats
            cache_metrics_manager.update_cache_stats(
                strategy=f"{self.strategy_name}_vector",
                num_items=len(self.query_vectors),
                memory_bytes=sum(v.nbytes for v in self.query_vectors.values()),
            )
            
            return vector

        except Exception as e:
            logger.error(f"Failed to get query vector: {e}")
            return None

    def _find_similar_query(
        self,
        query: str,
        filters: Optional[Dict[str, Any]] = None,
    ) -> Optional[str]:
        """Find semantically similar cached query.

        Args:
            query: Search query
            filters: Optional filters

        Returns:
            Similar cached query if found
        """
        try:
            start_time = time.time()
            
            # Get query vector
            query_vector = self._get_query_vector(query)
            if query_vector is None:
                return None
                
            # Get all cache keys
            all_keys = self.cache.cache.get_all_keys()
            
            # Find similar queries
            max_similarity = 0.0
            similar_query = None
            
            for key in all_keys:
                # Extract query from key
                cached_query = key.split("|")[0].split(":")[1]
                
                # Get vector
                cached_vector = self._get_query_vector(cached_query)
                if cached_vector is None:
                    continue
                    
                # Calculate similarity
                similarity = np.dot(query_vector, cached_vector) / (
                    np.linalg.norm(query_vector) * np.linalg.norm(cached_vector)
                )
                
                # Update if more similar
                if similarity > max_similarity and similarity >= self.similarity_threshold:
                    max_similarity = similarity
                    similar_query = cached_query
            
            # Track semantic search latency
            latency = time.time() - start_time
            cache_metrics_manager.cache_latency.labels(
                operation="semantic_search",
                strategy=self.strategy_name,
            ).observe(latency)
            
            return similar_query

        except Exception as e:
            logger.error(f"Failed to find similar query: {e}")
            return None

    def get_scores(
        self,
        query: str,
        filters: Optional[Dict[str, Any]] = None,
        extra_context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, ScoreData]:
        """Get cached scores with semantic matching.

        Args:
            query: Search query
            filters: Optional filters
            extra_context: Optional additional context

        Returns:
            Dictionary of endpoint ID to score data
        """
        try:
            start_time = time.time()
            
            # Try exact match first
            scores = super().get_scores(query, filters, extra_context)
            if scores:
                return scores
                
            # Try semantic match
            similar_query = self._find_similar_query(query, filters)
            if similar_query:
                scores = super().get_scores(similar_query, filters, extra_context)
                if scores:
                    # Track semantic hit
                    latency = time.time() - start_time
                    cache_metrics_manager.track_cache_hit(
                        strategy=self.strategy_name,
                        latency=latency,
                        semantic=True,
                        cache_type="disk",
                    )
                    return scores
                
            # Track miss
            latency = time.time() - start_time
            cache_metrics_manager.track_cache_miss(
                strategy=self.strategy_name,
                latency=latency,
                cache_type="disk",
            )
            return {}

        except Exception as e:
            logger.error(f"Failed to get semantic cached scores: {e}")
            return {} 