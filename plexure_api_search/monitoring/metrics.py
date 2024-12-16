"""Advanced metrics and monitoring system."""

import json
import logging
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
from prometheus_client import Counter, Gauge, Histogram, start_http_server

from ..config import config_instance
from ..utils.cache import DiskCache

logger = logging.getLogger(__name__)

# Cache for metrics
metrics_cache = DiskCache[Dict[str, Any]](
    namespace="metrics",
    ttl=config_instance.cache_ttl * 24,  # Cache metrics for longer
)


@dataclass
class SearchMetrics:
    """Search quality metrics."""

    query: str
    timestamp: datetime
    latency: float
    num_results: int
    top_score: float
    mean_score: float
    user_feedback: Optional[float] = None
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format."""
        return {
            "query": self.query,
            "timestamp": self.timestamp.isoformat(),
            "latency": self.latency,
            "num_results": self.num_results,
            "top_score": self.top_score,
            "mean_score": self.mean_score,
            "user_feedback": self.user_feedback,
            "error": self.error,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SearchMetrics":
        """Create from dictionary format."""
        data["timestamp"] = datetime.fromisoformat(data["timestamp"])
        return cls(**data)


class MetricsManager:
    """Advanced metrics and monitoring manager."""

    def __init__(self):
        """Initialize metrics manager."""
        self.metrics_dir = config_instance.metrics_dir
        self.enable_telemetry = config_instance.enable_telemetry
        
        # Create metrics directory
        self.metrics_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize Prometheus metrics if enabled
        if self.enable_telemetry and config_instance.metrics_backend == "prometheus":
            # Search metrics
            self.search_latency = Histogram(
                "search_latency_seconds",
                "Search request latency in seconds",
                buckets=[0.1, 0.25, 0.5, 1.0, 2.5, 5.0],
            )
            self.search_errors = Counter(
                "search_errors_total",
                "Total number of search errors",
                ["error_type"],
            )
            self.search_requests = Counter(
                "search_requests_total",
                "Total number of search requests",
            )
            self.results_returned = Histogram(
                "search_results_count",
                "Number of results returned per search",
                buckets=[0, 1, 5, 10, 20, 50],
            )
            
            # Cache metrics
            self.cache_hits = Counter(
                "cache_hits_total",
                "Total number of cache hits",
                ["cache_type"],
            )
            self.cache_misses = Counter(
                "cache_misses_total",
                "Total number of cache misses",
                ["cache_type"],
            )
            
            # Quality metrics
            self.search_quality = Gauge(
                "search_quality_score",
                "Search quality score (0-1)",
            )
            self.user_satisfaction = Gauge(
                "user_satisfaction_score",
                "User satisfaction score (0-1)",
            )
            
            # Start Prometheus server
            start_http_server(8000)
            logger.info("Started Prometheus metrics server on port 8000")

    def track_search(
        self,
        query: str,
        results: List[Dict[str, Any]],
        latency: float,
        error: Optional[str] = None,
    ) -> None:
        """Track search metrics.

        Args:
            query: Search query
            results: Search results
            latency: Search latency in seconds
            error: Optional error message
        """
        try:
            # Calculate basic metrics
            num_results = len(results)
            scores = [r.get("score", 0.0) for r in results]
            top_score = max(scores) if scores else 0.0
            mean_score = np.mean(scores) if scores else 0.0

            # Create metrics object
            metrics = SearchMetrics(
                query=query,
                timestamp=datetime.now(),
                latency=latency,
                num_results=num_results,
                top_score=top_score,
                mean_score=mean_score,
                error=error,
            )

            # Save to file
            self._save_metrics(metrics)

            # Update Prometheus metrics if enabled
            if self.enable_telemetry and config_instance.metrics_backend == "prometheus":
                self.search_latency.observe(latency)
                self.search_requests.inc()
                self.results_returned.observe(num_results)
                
                if error:
                    self.search_errors.labels(error_type=type(error).__name__).inc()

                # Update quality metrics
                quality_score = self._calculate_quality_score(metrics)
                self.search_quality.set(quality_score)

        except Exception as e:
            logger.error(f"Failed to track search metrics: {e}")

    def track_cache(self, cache_type: str, hit: bool) -> None:
        """Track cache metrics.

        Args:
            cache_type: Type of cache
            hit: Whether it was a cache hit
        """
        try:
            if self.enable_telemetry and config_instance.metrics_backend == "prometheus":
                if hit:
                    self.cache_hits.labels(cache_type=cache_type).inc()
                else:
                    self.cache_misses.labels(cache_type=cache_type).inc()

        except Exception as e:
            logger.error(f"Failed to track cache metrics: {e}")

    def track_user_feedback(self, query: str, feedback: float) -> None:
        """Track user feedback.

        Args:
            query: Search query
            feedback: User feedback score (0-1)
        """
        try:
            # Update metrics file with feedback
            metrics_file = self.metrics_dir / f"{self._get_metrics_key(query)}.json"
            if metrics_file.exists():
                metrics = SearchMetrics.from_dict(json.loads(metrics_file.read_text()))
                metrics.user_feedback = feedback
                self._save_metrics(metrics)

            # Update Prometheus metrics
            if self.enable_telemetry and config_instance.metrics_backend == "prometheus":
                self.user_satisfaction.set(feedback)

        except Exception as e:
            logger.error(f"Failed to track user feedback: {e}")

    def get_metrics(
        self,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
    ) -> List[SearchMetrics]:
        """Get metrics within time range.

        Args:
            start_time: Optional start time
            end_time: Optional end time

        Returns:
            List of metrics objects
        """
        try:
            metrics = []
            for file in self.metrics_dir.glob("*.json"):
                try:
                    data = json.loads(file.read_text())
                    metric = SearchMetrics.from_dict(data)
                    
                    # Filter by time range
                    if start_time and metric.timestamp < start_time:
                        continue
                    if end_time and metric.timestamp > end_time:
                        continue
                        
                    metrics.append(metric)
                except Exception as e:
                    logger.error(f"Failed to load metrics from {file}: {e}")
                    continue

            return sorted(metrics, key=lambda x: x.timestamp)

        except Exception as e:
            logger.error(f"Failed to get metrics: {e}")
            return []

    def calculate_statistics(
        self,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
    ) -> Dict[str, Any]:
        """Calculate search statistics.

        Args:
            start_time: Optional start time
            end_time: Optional end time

        Returns:
            Dictionary of statistics
        """
        try:
            metrics = self.get_metrics(start_time, end_time)
            
            if not metrics:
                return {}

            # Calculate statistics
            latencies = [m.latency for m in metrics]
            scores = [m.top_score for m in metrics]
            result_counts = [m.num_results for m in metrics]
            feedback = [m.user_feedback for m in metrics if m.user_feedback is not None]
            
            stats = {
                "total_searches": len(metrics),
                "total_errors": len([m for m in metrics if m.error]),
                "avg_latency": np.mean(latencies),
                "p95_latency": np.percentile(latencies, 95),
                "avg_results": np.mean(result_counts),
                "avg_top_score": np.mean(scores),
                "user_satisfaction": np.mean(feedback) if feedback else None,
            }

            return stats

        except Exception as e:
            logger.error(f"Failed to calculate statistics: {e}")
            return {}

    def _save_metrics(self, metrics: SearchMetrics) -> None:
        """Save metrics to file.

        Args:
            metrics: Metrics object to save
        """
        try:
            metrics_file = self.metrics_dir / f"{self._get_metrics_key(metrics.query)}.json"
            metrics_file.write_text(json.dumps(metrics.to_dict(), indent=2))

        except Exception as e:
            logger.error(f"Failed to save metrics: {e}")

    def _get_metrics_key(self, query: str) -> str:
        """Generate metrics file key.

        Args:
            query: Search query

        Returns:
            Metrics key
        """
        timestamp = int(time.time())
        return f"{timestamp}_{hash(query)}"

    def _calculate_quality_score(self, metrics: SearchMetrics) -> float:
        """Calculate search quality score.

        Args:
            metrics: Search metrics

        Returns:
            Quality score between 0 and 1
        """
        try:
            # Combine multiple factors
            factors = [
                metrics.top_score,
                1.0 if metrics.num_results > 0 else 0.0,
                1.0 if metrics.latency < 1.0 else 0.5,
            ]
            
            if metrics.user_feedback is not None:
                factors.append(metrics.user_feedback)
                
            return float(np.mean(factors))

        except Exception as e:
            logger.error(f"Failed to calculate quality score: {e}")
            return 0.0


# Global metrics manager instance
metrics_manager = MetricsManager()

__all__ = ["metrics_manager", "SearchMetrics"] 