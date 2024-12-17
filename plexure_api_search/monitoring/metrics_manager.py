"""Monitoring and metrics management."""

import logging
from dataclasses import dataclass, field
from typing import Dict, Any, Optional
from datetime import datetime

logger = logging.getLogger(__name__)


@dataclass
class MetricsManager:
    """Manager for collecting and reporting metrics."""

    metrics: Dict[str, Any] = field(default_factory=dict)
    start_time: Optional[datetime] = None

    def __post_init__(self):
        """Initialize metrics manager."""
        self.start_time = datetime.now()
        self.metrics = {
            "search_latency": [],
            "index_latency": [],
            "cache_hits": 0,
            "cache_misses": 0,
            "total_searches": 0,
            "total_indexing": 0,
            "errors": [],
        }

    def record_search_latency(self, latency_ms: float):
        """Record search latency.

        Args:
            latency_ms: Latency in milliseconds
        """
        self.metrics["search_latency"].append(latency_ms)
        self.metrics["total_searches"] += 1

    def record_index_latency(self, latency_ms: float):
        """Record indexing latency.

        Args:
            latency_ms: Latency in milliseconds
        """
        self.metrics["index_latency"].append(latency_ms)
        self.metrics["total_indexing"] += 1

    def record_cache_hit(self):
        """Record cache hit."""
        self.metrics["cache_hits"] += 1

    def record_cache_miss(self):
        """Record cache miss."""
        self.metrics["cache_misses"] += 1

    def record_error(self, error: Exception, context: Optional[Dict[str, Any]] = None):
        """Record error with context.

        Args:
            error: Exception that occurred
            context: Optional context about the error
        """
        self.metrics["errors"].append({
            "error": str(error),
            "type": type(error).__name__,
            "timestamp": datetime.now().isoformat(),
            "context": context or {},
        })

    def get_metrics(self) -> Dict[str, Any]:
        """Get current metrics.

        Returns:
            Dictionary of metrics
        """
        uptime = (datetime.now() - self.start_time).total_seconds()
        
        # Calculate averages
        avg_search_latency = (
            sum(self.metrics["search_latency"]) / len(self.metrics["search_latency"])
            if self.metrics["search_latency"]
            else 0
        )
        avg_index_latency = (
            sum(self.metrics["index_latency"]) / len(self.metrics["index_latency"])
            if self.metrics["index_latency"]
            else 0
        )
        
        return {
            "uptime_seconds": uptime,
            "total_searches": self.metrics["total_searches"],
            "total_indexing": self.metrics["total_indexing"],
            "avg_search_latency_ms": avg_search_latency,
            "avg_index_latency_ms": avg_index_latency,
            "cache_hits": self.metrics["cache_hits"],
            "cache_misses": self.metrics["cache_misses"],
            "error_count": len(self.metrics["errors"]),
            "recent_errors": self.metrics["errors"][-5:],  # Last 5 errors
        }

    def reset(self):
        """Reset all metrics."""
        self.__post_init__()


# Global metrics manager instance
metrics_manager = MetricsManager() 