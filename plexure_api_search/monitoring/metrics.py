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

    _instance = None
    _initialized = False

    def __new__(cls):
        """Create or return singleton instance."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        """Initialize metrics manager."""
        if self._initialized:
            return

        try:
            # Initialize metrics
            self.search_latency = Histogram(
                "search_latency_seconds",
                "Search request latency in seconds",
                ["endpoint"],
            )

            self.index_latency = Histogram(
                "index_latency_seconds",
                "Index operation latency in seconds",
                ["operation"],
            )

            self.search_requests = Counter(
                "search_requests_total",
                "Total number of search requests",
                ["endpoint", "status"],
            )

            self.index_operations = Counter(
                "index_operations_total",
                "Total number of index operations",
                ["operation", "status"],
            )

            self.cache_hits = Counter(
                "cache_hits_total",
                "Total number of cache hits",
                ["cache"],
            )

            self.cache_misses = Counter(
                "cache_misses_total",
                "Total number of cache misses",
                ["cache"],
            )

            self.model_load_time = Histogram(
                "model_load_seconds",
                "Model loading time in seconds",
                ["model"],
            )

            self.model_inference_time = Histogram(
                "model_inference_seconds",
                "Model inference time in seconds",
                ["model"],
            )

            self.vector_store_operations = Counter(
                "vector_store_operations_total",
                "Total number of vector store operations",
                ["operation", "status"],
            )

            self.vector_store_latency = Histogram(
                "vector_store_latency_seconds",
                "Vector store operation latency in seconds",
                ["operation"],
            )

            self.api_files = Gauge(
                "api_files_total",
                "Total number of API files indexed",
            )

            self.api_endpoints = Gauge(
                "api_endpoints_total",
                "Total number of API endpoints indexed",
            )

            self.memory_usage = Gauge(
                "memory_usage_bytes",
                "Memory usage in bytes",
            )

            self.cpu_usage = Gauge(
                "cpu_usage_percent",
                "CPU usage percentage",
            )

            # Start metrics server if enabled
            if config_instance.enable_telemetry:
                try:
                    start_http_server(8000)
                    logger.info("Started Prometheus metrics server on port 8000")
                except OSError as e:
                    if e.errno == 98:  # Address already in use
                        logger.warning("Metrics server port 8000 already in use, continuing without metrics server")
                    else:
                        logger.error(f"Failed to start metrics server: {e}")

            self._initialized = True

        except ValueError as e:
            # Ignore duplicate registration errors
            if "Duplicated timeseries" not in str(e):
                raise
            logger.debug("Metrics already registered")

        except Exception as e:
            logger.error(f"Failed to initialize metrics: {e}")
            raise

    def start_timer(self) -> float:
        """Start a timer.
        
        Returns:
            Start time in seconds
        """
        return time.time()

    def stop_timer(
        self,
        start_time: float,
        metric: str,
        labels: Optional[Dict[str, str]] = None,
    ) -> None:
        """Stop a timer and record duration.
        
        Args:
            start_time: Start time from start_timer()
            metric: Metric name
            labels: Metric labels
        """
        duration = time.time() - start_time
        labels = labels or {}

        if metric == "search":
            self.search_latency.labels(**labels).observe(duration)
        elif metric == "index":
            self.index_latency.labels(**labels).observe(duration)
        elif metric == "model_load":
            self.model_load_time.labels(**labels).observe(duration)
        elif metric == "model_inference":
            self.model_inference_time.labels(**labels).observe(duration)
        elif metric == "vector_store":
            self.vector_store_latency.labels(**labels).observe(duration)

    def increment(
        self,
        metric: str,
        labels: Optional[Dict[str, str]] = None,
    ) -> None:
        """Increment a counter metric.
        
        Args:
            metric: Metric name
            labels: Metric labels
        """
        labels = labels or {}

        if metric == "search_requests":
            self.search_requests.labels(**labels).inc()
        elif metric == "index_operations":
            self.index_operations.labels(**labels).inc()
        elif metric == "cache_hits":
            self.cache_hits.labels(**labels).inc()
        elif metric == "cache_misses":
            self.cache_misses.labels(**labels).inc()
        elif metric == "vector_store_operations":
            self.vector_store_operations.labels(**labels).inc()

    def set_gauge(
        self,
        metric: str,
        value: float,
        labels: Optional[Dict[str, str]] = None,
    ) -> None:
        """Set a gauge metric value.
        
        Args:
            metric: Metric name
            value: Metric value
            labels: Metric labels
        """
        labels = labels or {}

        if metric == "api_files":
            self.api_files.set(value)
        elif metric == "api_endpoints":
            self.api_endpoints.set(value)
        elif metric == "memory_usage":
            self.memory_usage.set(value)
        elif metric == "cpu_usage":
            self.cpu_usage.set(value)

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
            if config_instance.enable_telemetry:
                self.search_latency.labels(endpoint=query).observe(latency)
                self.search_requests.labels(endpoint=query, status="success").inc()
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
            if config_instance.enable_telemetry:
                if hit:
                    self.cache_hits.labels(cache=cache_type).inc()
                else:
                    self.cache_misses.labels(cache=cache_type).inc()

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
            if config_instance.enable_telemetry:
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