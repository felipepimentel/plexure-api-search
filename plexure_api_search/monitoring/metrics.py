"""Metrics management."""

import logging
from typing import Dict, List, Optional

from prometheus_client import Counter, Gauge, Histogram, start_http_server

logger = logging.getLogger(__name__)

class MetricsManager:
    """Metrics manager."""

    _instance = None
    _server_started = False

    def __new__(cls):
        """Create singleton instance."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        """Initialize metrics manager."""
        if not hasattr(self, "initialized"):
            self.port = 5555
            self.initialized = False
            self.counters: Dict[str, Counter] = {}
            self.gauges: Dict[str, Gauge] = {}
            self.histograms: Dict[str, Histogram] = {}

    def initialize(self) -> None:
        """Initialize metrics manager."""
        if self.initialized:
            return

        try:
            # Start metrics server if not already started
            if not self._server_started:
                try:
                    start_http_server(self.port)
                    self._server_started = True
                    logger.info(f"Started Prometheus metrics server on port {self.port}")
                except OSError as e:
                    if e.errno == 98:  # Address already in use
                        self._server_started = True
                        logger.info(f"Metrics server already running on port {self.port}")
                    else:
                        raise

            # Initialize metrics
            self._initialize_metrics()
            self.initialized = True

        except Exception as e:
            logger.error(f"Failed to initialize metrics manager: {e}")
            raise

    def _initialize_metrics(self) -> None:
        """Initialize metrics."""
        # Counters
        self.counters["embeddings_generated"] = Counter(
            "embeddings_generated_total",
            "Total number of embeddings generated",
        )
        self.counters["embedding_errors"] = Counter(
            "embedding_errors_total",
            "Total number of embedding errors",
        )
        self.counters["searches_performed"] = Counter(
            "searches_performed_total",
            "Total number of searches performed",
            ["query_type"],
        )
        self.counters["search_errors"] = Counter(
            "search_errors_total",
            "Total number of search errors",
        )
        self.counters["contract_errors"] = Counter(
            "contract_errors_total",
            "Total number of contract errors",
        )

        # Gauges
        self.gauges["index_size"] = Gauge(
            "index_size",
            "Number of vectors in index",
        )
        self.gauges["metadata_size"] = Gauge(
            "metadata_size",
            "Number of metadata entries",
        )

        # Histograms
        self.histograms["search_latency"] = Histogram(
            "search_latency_seconds",
            "Search latency in seconds",
        )
        self.histograms["embedding_latency"] = Histogram(
            "embedding_latency_seconds",
            "Embedding latency in seconds",
        )

    def increment_counter(self, name: str, value: int = 1, labels: Optional[Dict] = None) -> None:
        """Increment counter.
        
        Args:
            name: Counter name
            value: Value to increment by
            labels: Counter labels
        """
        if not self.initialized:
            self.initialize()

        try:
            counter = self.counters.get(name)
            if not counter:
                logger.error(f"Counter {name} not found")
                return

            if labels:
                counter.labels(**labels).inc(value)
            else:
                counter.inc(value)

        except Exception as e:
            logger.error(f"Failed to increment counter {name}: {e}")

    def set_gauge(self, name: str, value: float) -> None:
        """Set gauge value.
        
        Args:
            name: Gauge name
            value: Value to set
        """
        if not self.initialized:
            self.initialize()

        try:
            gauge = self.gauges.get(name)
            if not gauge:
                logger.error(f"Gauge {name} not found")
                return

            gauge.set(value)

        except Exception as e:
            logger.error(f"Failed to set gauge {name}: {e}")

    def observe_value(self, name: str, value: float) -> None:
        """Observe histogram value.
        
        Args:
            name: Histogram name
            value: Value to observe
        """
        if not self.initialized:
            self.initialize()

        try:
            histogram = self.histograms.get(name)
            if not histogram:
                logger.error(f"Histogram {name} not found")
                return

            histogram.observe(value)

        except Exception as e:
            logger.error(f"Failed to observe value for {name}: {e}")

# Global instance
metrics_manager = MetricsManager() 