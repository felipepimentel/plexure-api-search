"""Query analytics for monitoring search patterns."""

import logging
import asyncio
from typing import Dict, List, Any, Optional, Set, Union, Counter
from datetime import datetime, timedelta
from dataclasses import dataclass
import json
import os
from pathlib import Path
import numpy as np
from collections import defaultdict

from ..config import Config
from .events import Event, EventType
from .metrics import MetricsManager
from ..services.base import BaseService
from ..services.events import PublisherService

logger = logging.getLogger(__name__)


@dataclass
class QueryStats:
    """Statistics for a search query."""

    query: str  # Original query
    count: int  # Number of times queried
    avg_latency: float  # Average latency in seconds
    avg_results: float  # Average number of results
    success_rate: float  # Success rate (0-1)
    last_seen: datetime  # Last time query was seen
    metadata: Dict[str, Any]  # Additional metadata


@dataclass
class TimeWindow:
    """Time window for analytics."""

    start: datetime  # Window start time
    end: datetime  # Window end time
    total_queries: int  # Total number of queries
    unique_queries: int  # Number of unique queries
    avg_latency: float  # Average latency
    success_rate: float  # Success rate
    top_queries: List[QueryStats]  # Top queries by count


class AnalyticsConfig:
    """Configuration for query analytics."""

    def __init__(
        self,
        window_size: int = 3600,  # 1 hour
        max_windows: int = 24,  # Keep 24 hours of data
        min_query_count: int = 5,  # Minimum queries for analysis
        top_k: int = 10,  # Number of top queries to track
        export_interval: float = 3600.0,  # Export every hour
        export_dir: str = "analytics",  # Directory to store exports
        enable_pattern_analysis: bool = True,
        enable_trend_analysis: bool = True,
        enable_correlation_analysis: bool = True,
    ) -> None:
        """Initialize analytics config.

        Args:
            window_size: Time window size in seconds
            max_windows: Maximum number of windows to keep
            min_query_count: Minimum query count for analysis
            top_k: Number of top queries to track
            export_interval: Export interval in seconds
            export_dir: Directory to store exports
            enable_pattern_analysis: Whether to analyze query patterns
            enable_trend_analysis: Whether to analyze query trends
            enable_correlation_analysis: Whether to analyze query correlations
        """
        self.window_size = window_size
        self.max_windows = max_windows
        self.min_query_count = min_query_count
        self.top_k = top_k
        self.export_interval = export_interval
        self.export_dir = export_dir
        self.enable_pattern_analysis = enable_pattern_analysis
        self.enable_trend_analysis = enable_trend_analysis
        self.enable_correlation_analysis = enable_correlation_analysis


class QueryAnalytics(BaseService[Dict[str, Any]]):
    """Query analytics implementation."""

    def __init__(
        self,
        config: Config,
        publisher: PublisherService,
        metrics_manager: MetricsManager,
        analytics_config: Optional[AnalyticsConfig] = None,
    ) -> None:
        """Initialize query analytics.

        Args:
            config: Application configuration
            publisher: Event publisher
            metrics_manager: Metrics manager
            analytics_config: Analytics configuration
        """
        super().__init__(config, publisher, metrics_manager)
        self.analytics_config = analytics_config or AnalyticsConfig()
        self._initialized = False
        self._current_window: Optional[TimeWindow] = None
        self._windows: List[TimeWindow] = []
        self._query_stats: Dict[str, QueryStats] = {}
        self._patterns: Dict[str, List[str]] = defaultdict(list)
        self._trends: Dict[str, List[float]] = defaultdict(list)
        self._correlations: Dict[str, Dict[str, float]] = defaultdict(dict)
        self._export_task: Optional[asyncio.Task] = None

        # Create export directory
        os.makedirs(self.analytics_config.export_dir, exist_ok=True)

    async def initialize(self) -> None:
        """Initialize analytics resources."""
        if self._initialized:
            return

        try:
            # Start new window
            self._start_window()

            # Start export task
            self._export_task = asyncio.create_task(self._export_loop())

            self._initialized = True
            self.publisher.publish(
                Event(
                    type=EventType.SERVICE_STARTED,
                    timestamp=datetime.now(),
                    component="query_analytics",
                    description="Query analytics initialized",
                    metadata={
                        "window_size": self.analytics_config.window_size,
                        "max_windows": self.analytics_config.max_windows,
                        "pattern_analysis": self.analytics_config.enable_pattern_analysis,
                        "trend_analysis": self.analytics_config.enable_trend_analysis,
                        "correlation_analysis": self.analytics_config.enable_correlation_analysis,
                    },
                )
            )

        except Exception as e:
            logger.error(f"Failed to initialize query analytics: {e}")
            raise

    async def cleanup(self) -> None:
        """Clean up analytics resources."""
        if self._export_task:
            self._export_task.cancel()
            try:
                await self._export_task
            except asyncio.CancelledError:
                pass

        self._initialized = False
        self._current_window = None
        self._windows.clear()
        self._query_stats.clear()
        self._patterns.clear()
        self._trends.clear()
        self._correlations.clear()

        self.publisher.publish(
            Event(
                type=EventType.SERVICE_STOPPED,
                timestamp=datetime.now(),
                component="query_analytics",
                description="Query analytics stopped",
            )
        )

    async def health_check(self) -> Dict[str, Any]:
        """Check analytics health.

        Returns:
            Health check results
        """
        return {
            "service": "QueryAnalytics",
            "initialized": self._initialized,
            "num_windows": len(self._windows),
            "num_queries": len(self._query_stats),
            "current_window": bool(self._current_window),
            "status": "healthy" if self._initialized else "unhealthy",
        }

    async def record_query(
        self,
        query: str,
        latency: float,
        num_results: int,
        success: bool,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Record search query.

        Args:
            query: Search query
            latency: Query latency in seconds
            num_results: Number of results
            success: Whether query was successful
            metadata: Additional metadata
        """
        if not self._initialized:
            return

        try:
            # Check window
            now = datetime.now()
            if not self._current_window or now >= self._current_window.end:
                self._start_window()

            # Update query stats
            if query not in self._query_stats:
                self._query_stats[query] = QueryStats(
                    query=query,
                    count=0,
                    avg_latency=0.0,
                    avg_results=0.0,
                    success_rate=0.0,
                    last_seen=now,
                    metadata=metadata or {},
                )

            stats = self._query_stats[query]
            stats.count += 1
            stats.avg_latency = (
                (stats.avg_latency * (stats.count - 1) + latency)
                / stats.count
            )
            stats.avg_results = (
                (stats.avg_results * (stats.count - 1) + num_results)
                / stats.count
            )
            stats.success_rate = (
                (stats.success_rate * (stats.count - 1) + float(success))
                / stats.count
            )
            stats.last_seen = now
            if metadata:
                stats.metadata.update(metadata)

            # Update window stats
            if self._current_window:
                self._current_window.total_queries += 1
                self._current_window.unique_queries = len(self._query_stats)
                self._current_window.avg_latency = (
                    (self._current_window.avg_latency * (self._current_window.total_queries - 1) + latency)
                    / self._current_window.total_queries
                )
                self._current_window.success_rate = (
                    (self._current_window.success_rate * (self._current_window.total_queries - 1) + float(success))
                    / self._current_window.total_queries
                )

            # Update patterns
            if self.analytics_config.enable_pattern_analysis:
                self._update_patterns(query)

            # Update trends
            if self.analytics_config.enable_trend_analysis:
                self._update_trends(query, latency, num_results, success)

            # Update correlations
            if self.analytics_config.enable_correlation_analysis:
                self._update_correlations(query)

            # Emit event
            self.publisher.publish(
                Event(
                    type=EventType.QUERY_RECORDED,
                    timestamp=now,
                    component="query_analytics",
                    description=f"Query recorded: {query}",
                    metadata={
                        "query": query,
                        "latency": latency,
                        "num_results": num_results,
                        "success": success,
                        "metadata": metadata,
                    },
                )
            )

        except Exception as e:
            logger.error(f"Failed to record query: {e}")

    def _start_window(self) -> None:
        """Start new time window."""
        now = datetime.now()
        window = TimeWindow(
            start=now,
            end=now + timedelta(seconds=self.analytics_config.window_size),
            total_queries=0,
            unique_queries=0,
            avg_latency=0.0,
            success_rate=0.0,
            top_queries=[],
        )

        # Update top queries
        sorted_queries = sorted(
            self._query_stats.values(),
            key=lambda x: x.count,
            reverse=True,
        )
        window.top_queries = sorted_queries[:self.analytics_config.top_k]

        # Add window
        self._current_window = window
        self._windows.append(window)

        # Remove old windows
        if len(self._windows) > self.analytics_config.max_windows:
            self._windows.pop(0)

    def _update_patterns(self, query: str) -> None:
        """Update query patterns.

        Args:
            query: Search query
        """
        # Extract terms
        terms = query.lower().split()

        # Update patterns
        for i in range(len(terms)):
            for j in range(i + 1, len(terms) + 1):
                pattern = " ".join(terms[i:j])
                self._patterns[pattern].append(query)

        # Remove old patterns
        for pattern in list(self._patterns.keys()):
            if len(self._patterns[pattern]) < self.analytics_config.min_query_count:
                del self._patterns[pattern]

    def _update_trends(
        self,
        query: str,
        latency: float,
        num_results: int,
        success: bool,
    ) -> None:
        """Update query trends.

        Args:
            query: Search query
            latency: Query latency
            num_results: Number of results
            success: Whether query was successful
        """
        # Update trends
        self._trends[f"{query}_latency"].append(latency)
        self._trends[f"{query}_results"].append(num_results)
        self._trends[f"{query}_success"].append(float(success))

        # Remove old trends
        max_points = self.analytics_config.max_windows
        for key in self._trends:
            if len(self._trends[key]) > max_points:
                self._trends[key] = self._trends[key][-max_points:]

    def _update_correlations(self, query: str) -> None:
        """Update query correlations.

        Args:
            query: Search query
        """
        # Calculate correlations with other queries
        for other_query in self._query_stats:
            if query != other_query:
                correlation = self._calculate_correlation(query, other_query)
                self._correlations[query][other_query] = correlation
                self._correlations[other_query][query] = correlation

    def _calculate_correlation(
        self,
        query1: str,
        query2: str,
    ) -> float:
        """Calculate correlation between queries.

        Args:
            query1: First query
            query2: Second query

        Returns:
            Correlation coefficient
        """
        # Get query trends
        trend1 = self._trends.get(f"{query1}_latency", [])
        trend2 = self._trends.get(f"{query2}_latency", [])

        # Calculate correlation if enough data
        if len(trend1) >= self.analytics_config.min_query_count and len(trend2) >= self.analytics_config.min_query_count:
            return float(np.corrcoef(trend1, trend2)[0, 1])

        return 0.0

    async def _export_loop(self) -> None:
        """Background task for exporting analytics."""
        while True:
            try:
                # Sleep for export interval
                await asyncio.sleep(self.analytics_config.export_interval)

                # Export analytics
                await self._export_analytics()

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Export loop failed: {e}")

    async def _export_analytics(self) -> None:
        """Export analytics data."""
        try:
            # Create export data
            data = {
                "timestamp": datetime.now().isoformat(),
                "windows": [
                    {
                        "start": window.start.isoformat(),
                        "end": window.end.isoformat(),
                        "total_queries": window.total_queries,
                        "unique_queries": window.unique_queries,
                        "avg_latency": window.avg_latency,
                        "success_rate": window.success_rate,
                        "top_queries": [
                            {
                                "query": stats.query,
                                "count": stats.count,
                                "avg_latency": stats.avg_latency,
                                "avg_results": stats.avg_results,
                                "success_rate": stats.success_rate,
                                "last_seen": stats.last_seen.isoformat(),
                                "metadata": stats.metadata,
                            }
                            for stats in window.top_queries
                        ],
                    }
                    for window in self._windows
                ],
                "patterns": {
                    pattern: queries[:self.analytics_config.top_k]
                    for pattern, queries in self._patterns.items()
                },
                "trends": {
                    key: values[-self.analytics_config.max_windows:]
                    for key, values in self._trends.items()
                },
                "correlations": {
                    query: {
                        other: corr
                        for other, corr in correlations.items()
                        if abs(corr) >= 0.5  # Only export strong correlations
                    }
                    for query, correlations in self._correlations.items()
                },
            }

            # Save to file
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"analytics_{timestamp}.json"
            filepath = os.path.join(self.analytics_config.export_dir, filename)

            with open(filepath, "w") as f:
                json.dump(data, f, indent=2)

            # Emit event
            self.publisher.publish(
                Event(
                    type=EventType.ANALYTICS_EXPORTED,
                    timestamp=datetime.now(),
                    component="query_analytics",
                    description="Analytics exported",
                    metadata={
                        "filename": filename,
                        "num_windows": len(self._windows),
                        "num_patterns": len(self._patterns),
                        "num_trends": len(self._trends),
                        "num_correlations": len(self._correlations),
                    },
                )
            )

        except Exception as e:
            logger.error(f"Failed to export analytics: {e}")


# Create analytics instance
query_analytics = QueryAnalytics(
    Config(),
    PublisherService(Config(), MetricsManager()),
    MetricsManager(),
)

__all__ = [
    "QueryStats",
    "TimeWindow",
    "AnalyticsConfig",
    "QueryAnalytics",
    "query_analytics",
] 