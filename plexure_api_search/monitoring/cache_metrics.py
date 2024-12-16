"""Cache monitoring and metrics system."""

import logging
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Set

from prometheus_client import Counter, Gauge, Histogram

from ..config import config_instance
from ..utils.cache import DiskCache

logger = logging.getLogger(__name__)


@dataclass
class CacheMetrics:
    """Cache performance metrics."""

    hits: int = 0
    misses: int = 0
    semantic_hits: int = 0
    total_latency: float = 0.0
    cached_items: int = 0
    memory_usage: float = 0.0
    invalidations: int = 0
    evictions: int = 0

    @property
    def hit_rate(self) -> float:
        """Calculate cache hit rate."""
        total = self.hits + self.misses
        return self.hits / total if total > 0 else 0.0

    @property
    def semantic_hit_rate(self) -> float:
        """Calculate semantic cache hit rate."""
        total = self.hits + self.misses
        return self.semantic_hits / total if total > 0 else 0.0

    @property
    def avg_latency(self) -> float:
        """Calculate average cache latency."""
        total = self.hits + self.misses
        return self.total_latency / total if total > 0 else 0.0


class CacheMetricsManager:
    """Manager for cache performance metrics."""

    def __init__(self):
        """Initialize cache metrics manager."""
        self.metrics: Dict[str, CacheMetrics] = {}
        self.cache = DiskCache[Dict[str, Any]](
            namespace="cache_metrics",
            ttl=config_instance.cache_ttl * 24,  # Cache metrics for longer
        )

        # Initialize Prometheus metrics
        if config_instance.metrics_backend == "prometheus":
            # Cache hits/misses
            self.cache_hits = Counter(
                "cache_hits_total",
                "Total number of cache hits",
                ["cache_type", "strategy"],
            )
            self.cache_misses = Counter(
                "cache_misses_total",
                "Total number of cache misses",
                ["cache_type", "strategy"],
            )
            self.semantic_hits = Counter(
                "semantic_cache_hits_total",
                "Total number of semantic cache hits",
                ["strategy"],
            )

            # Cache performance
            self.cache_latency = Histogram(
                "cache_operation_latency_seconds",
                "Cache operation latency in seconds",
                ["operation", "strategy"],
                buckets=[0.001, 0.005, 0.01, 0.025, 0.05, 0.1],
            )
            self.cached_items = Gauge(
                "cache_items_total",
                "Total number of cached items",
                ["strategy"],
            )
            self.cache_memory = Gauge(
                "cache_memory_bytes",
                "Cache memory usage in bytes",
                ["strategy"],
            )

            # Cache operations
            self.cache_invalidations = Counter(
                "cache_invalidations_total",
                "Total number of cache invalidations",
                ["strategy"],
            )
            self.cache_evictions = Counter(
                "cache_evictions_total",
                "Total number of cache evictions",
                ["strategy"],
            )

    def get_metrics(self, strategy: str) -> CacheMetrics:
        """Get metrics for strategy.

        Args:
            strategy: Strategy name

        Returns:
            Cache metrics
        """
        if strategy not in self.metrics:
            self.metrics[strategy] = CacheMetrics()
        return self.metrics[strategy]

    def track_cache_hit(
        self,
        strategy: str,
        latency: float,
        semantic: bool = False,
        cache_type: str = "disk",
    ) -> None:
        """Track cache hit.

        Args:
            strategy: Strategy name
            latency: Operation latency
            semantic: Whether it was a semantic hit
            cache_type: Type of cache
        """
        try:
            metrics = self.get_metrics(strategy)
            metrics.hits += 1
            metrics.total_latency += latency

            if semantic:
                metrics.semantic_hits += 1

            # Update Prometheus metrics
            if config_instance.metrics_backend == "prometheus":
                self.cache_hits.labels(
                    cache_type=cache_type,
                    strategy=strategy,
                ).inc()
                self.cache_latency.labels(
                    operation="hit",
                    strategy=strategy,
                ).observe(latency)
                if semantic:
                    self.semantic_hits.labels(strategy=strategy).inc()

        except Exception as e:
            logger.error(f"Failed to track cache hit: {e}")

    def track_cache_miss(
        self,
        strategy: str,
        latency: float,
        cache_type: str = "disk",
    ) -> None:
        """Track cache miss.

        Args:
            strategy: Strategy name
            latency: Operation latency
            cache_type: Type of cache
        """
        try:
            metrics = self.get_metrics(strategy)
            metrics.misses += 1
            metrics.total_latency += latency

            # Update Prometheus metrics
            if config_instance.metrics_backend == "prometheus":
                self.cache_misses.labels(
                    cache_type=cache_type,
                    strategy=strategy,
                ).inc()
                self.cache_latency.labels(
                    operation="miss",
                    strategy=strategy,
                ).observe(latency)

        except Exception as e:
            logger.error(f"Failed to track cache miss: {e}")

    def track_cache_invalidation(
        self,
        strategy: str,
        num_items: int = 1,
    ) -> None:
        """Track cache invalidation.

        Args:
            strategy: Strategy name
            num_items: Number of items invalidated
        """
        try:
            metrics = self.get_metrics(strategy)
            metrics.invalidations += num_items

            # Update Prometheus metrics
            if config_instance.metrics_backend == "prometheus":
                self.cache_invalidations.labels(strategy=strategy).inc(num_items)

        except Exception as e:
            logger.error(f"Failed to track cache invalidation: {e}")

    def track_cache_eviction(
        self,
        strategy: str,
        num_items: int = 1,
    ) -> None:
        """Track cache eviction.

        Args:
            strategy: Strategy name
            num_items: Number of items evicted
        """
        try:
            metrics = self.get_metrics(strategy)
            metrics.evictions += num_items

            # Update Prometheus metrics
            if config_instance.metrics_backend == "prometheus":
                self.cache_evictions.labels(strategy=strategy).inc(num_items)

        except Exception as e:
            logger.error(f"Failed to track cache eviction: {e}")

    def update_cache_stats(
        self,
        strategy: str,
        num_items: int,
        memory_bytes: float,
    ) -> None:
        """Update cache statistics.

        Args:
            strategy: Strategy name
            num_items: Number of cached items
            memory_bytes: Memory usage in bytes
        """
        try:
            metrics = self.get_metrics(strategy)
            metrics.cached_items = num_items
            metrics.memory_usage = memory_bytes

            # Update Prometheus metrics
            if config_instance.metrics_backend == "prometheus":
                self.cached_items.labels(strategy=strategy).set(num_items)
                self.cache_memory.labels(strategy=strategy).set(memory_bytes)

        except Exception as e:
            logger.error(f"Failed to update cache stats: {e}")

    def get_cache_stats(self, strategy: str) -> Dict[str, Any]:
        """Get cache statistics.

        Args:
            strategy: Strategy name

        Returns:
            Dictionary of cache statistics
        """
        try:
            metrics = self.get_metrics(strategy)
            return {
                "hit_rate": metrics.hit_rate,
                "semantic_hit_rate": metrics.semantic_hit_rate,
                "avg_latency": metrics.avg_latency,
                "hits": metrics.hits,
                "misses": metrics.misses,
                "semantic_hits": metrics.semantic_hits,
                "cached_items": metrics.cached_items,
                "memory_usage": metrics.memory_usage,
                "invalidations": metrics.invalidations,
                "evictions": metrics.evictions,
            }

        except Exception as e:
            logger.error(f"Failed to get cache stats: {e}")
            return {}

    def save_metrics(self) -> None:
        """Save metrics to persistent storage."""
        try:
            # Convert metrics to dictionaries
            metrics_data = {
                strategy: {
                    "hits": m.hits,
                    "misses": m.misses,
                    "semantic_hits": m.semantic_hits,
                    "total_latency": m.total_latency,
                    "cached_items": m.cached_items,
                    "memory_usage": m.memory_usage,
                    "invalidations": m.invalidations,
                    "evictions": m.evictions,
                }
                for strategy, m in self.metrics.items()
            }

            # Save to cache
            self.cache.set("metrics", metrics_data)

        except Exception as e:
            logger.error(f"Failed to save metrics: {e}")

    def load_metrics(self) -> None:
        """Load metrics from persistent storage."""
        try:
            # Load from cache
            metrics_data = self.cache.get("metrics")
            if not metrics_data:
                return

            # Convert to CacheMetrics objects
            self.metrics = {
                strategy: CacheMetrics(**data)
                for strategy, data in metrics_data.items()
            }

        except Exception as e:
            logger.error(f"Failed to load metrics: {e}")


# Global metrics manager instance
cache_metrics_manager = CacheMetricsManager()

__all__ = ["cache_metrics_manager", "CacheMetrics"] 