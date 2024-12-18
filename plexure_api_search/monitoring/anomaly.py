"""Anomaly detection for monitoring system behavior."""

import logging
import asyncio
from typing import Dict, List, Any, Optional, Set, Union, Tuple
from datetime import datetime, timedelta
import numpy as np
from dataclasses import dataclass
from collections import defaultdict
from enum import Enum

from ..config import Config
from .events import Event, EventType
from .metrics import MetricsManager
from ..services.base import BaseService
from ..services.events import PublisherService

logger = logging.getLogger(__name__)


class AnomalyType(Enum):
    """Types of anomalies."""

    SPIKE = "spike"  # Sudden increase
    DROP = "drop"  # Sudden decrease
    TREND = "trend"  # Gradual change
    PATTERN = "pattern"  # Pattern change
    OUTLIER = "outlier"  # Statistical outlier


@dataclass
class AnomalyRule:
    """Rule for anomaly detection."""

    metric: str  # Metric name
    window: int  # Time window in seconds
    threshold: float  # Detection threshold
    type: AnomalyType  # Anomaly type
    severity: str = "warning"  # Anomaly severity
    enabled: bool = True  # Whether rule is enabled


@dataclass
class AnomalyEvent:
    """Detected anomaly event."""

    rule: AnomalyRule  # Applied rule
    value: float  # Anomalous value
    expected: float  # Expected value
    deviation: float  # Deviation from expected
    timestamp: datetime  # Event timestamp
    metadata: Dict[str, Any]  # Additional metadata


class AnomalyConfig:
    """Configuration for anomaly detection."""

    def __init__(
        self,
        default_window: int = 300,  # 5 minutes
        min_data_points: int = 30,
        confidence_level: float = 0.95,
        enable_spike_detection: bool = True,
        enable_drop_detection: bool = True,
        enable_trend_detection: bool = True,
        enable_pattern_detection: bool = True,
        enable_outlier_detection: bool = True,
        cleanup_interval: float = 3600.0,  # 1 hour
    ) -> None:
        """Initialize anomaly config.

        Args:
            default_window: Default time window in seconds
            min_data_points: Minimum data points required
            confidence_level: Statistical confidence level
            enable_spike_detection: Whether to detect spikes
            enable_drop_detection: Whether to detect drops
            enable_trend_detection: Whether to detect trends
            enable_pattern_detection: Whether to detect pattern changes
            enable_outlier_detection: Whether to detect outliers
            cleanup_interval: Data cleanup interval in seconds
        """
        self.default_window = default_window
        self.min_data_points = min_data_points
        self.confidence_level = confidence_level
        self.enable_spike_detection = enable_spike_detection
        self.enable_drop_detection = enable_drop_detection
        self.enable_trend_detection = enable_trend_detection
        self.enable_pattern_detection = enable_pattern_detection
        self.enable_outlier_detection = enable_outlier_detection
        self.cleanup_interval = cleanup_interval


class AnomalyDetector(BaseService[Dict[str, Any]]):
    """Anomaly detector implementation."""

    def __init__(
        self,
        config: Config,
        publisher: PublisherService,
        metrics_manager: MetricsManager,
        anomaly_config: Optional[AnomalyConfig] = None,
    ) -> None:
        """Initialize anomaly detector.

        Args:
            config: Application configuration
            publisher: Event publisher
            metrics_manager: Metrics manager
            anomaly_config: Anomaly detection configuration
        """
        super().__init__(config, publisher, metrics_manager)
        self.anomaly_config = anomaly_config or AnomalyConfig()
        self._initialized = False
        self._rules: Dict[str, AnomalyRule] = {}
        self._data: Dict[str, List[Tuple[datetime, float]]] = defaultdict(list)
        self._cleanup_task: Optional[asyncio.Task] = None

        # Register default rules
        self._register_default_rules()

    def _register_default_rules(self) -> None:
        """Register default anomaly rules."""
        # Search performance rules
        self.add_rule(
            AnomalyRule(
                metric="search_latency",
                window=300,  # 5 minutes
                threshold=2.0,  # 2x standard deviation
                type=AnomalyType.SPIKE,
                severity="warning",
            )
        )
        self.add_rule(
            AnomalyRule(
                metric="search_errors",
                window=300,  # 5 minutes
                threshold=0.1,  # 10% error rate
                type=AnomalyType.SPIKE,
                severity="error",
            )
        )

        # Index performance rules
        self.add_rule(
            AnomalyRule(
                metric="index_latency",
                window=300,  # 5 minutes
                threshold=2.0,  # 2x standard deviation
                type=AnomalyType.SPIKE,
                severity="warning",
            )
        )
        self.add_rule(
            AnomalyRule(
                metric="index_errors",
                window=300,  # 5 minutes
                threshold=0.1,  # 10% error rate
                type=AnomalyType.SPIKE,
                severity="error",
            )
        )

        # Resource usage rules
        self.add_rule(
            AnomalyRule(
                metric="memory_usage",
                window=300,  # 5 minutes
                threshold=0.9,  # 90% usage
                type=AnomalyType.SPIKE,
                severity="warning",
            )
        )
        self.add_rule(
            AnomalyRule(
                metric="cpu_usage",
                window=300,  # 5 minutes
                threshold=0.9,  # 90% usage
                type=AnomalyType.SPIKE,
                severity="warning",
            )
        )

        # Quality metrics rules
        self.add_rule(
            AnomalyRule(
                metric="search_quality",
                window=3600,  # 1 hour
                threshold=0.2,  # 20% drop
                type=AnomalyType.DROP,
                severity="warning",
            )
        )
        self.add_rule(
            AnomalyRule(
                metric="relevance_score",
                window=3600,  # 1 hour
                threshold=0.2,  # 20% drop
                type=AnomalyType.DROP,
                severity="warning",
            )
        )

    async def initialize(self) -> None:
        """Initialize anomaly detector resources."""
        if self._initialized:
            return

        try:
            # Start cleanup task
            self._cleanup_task = asyncio.create_task(self._cleanup_loop())

            self._initialized = True
            self.publisher.publish(
                Event(
                    type=EventType.SERVICE_STARTED,
                    timestamp=datetime.now(),
                    component="anomaly_detector",
                    description="Anomaly detector initialized",
                    metadata={
                        "num_rules": len(self._rules),
                        "window": self.anomaly_config.default_window,
                        "confidence": self.anomaly_config.confidence_level,
                    },
                )
            )

        except Exception as e:
            logger.error(f"Failed to initialize anomaly detector: {e}")
            raise

    async def cleanup(self) -> None:
        """Clean up anomaly detector resources."""
        if self._cleanup_task:
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass

        self._initialized = False
        self._rules.clear()
        self._data.clear()

        self.publisher.publish(
            Event(
                type=EventType.SERVICE_STOPPED,
                timestamp=datetime.now(),
                component="anomaly_detector",
                description="Anomaly detector stopped",
            )
        )

    async def health_check(self) -> Dict[str, Any]:
        """Check anomaly detector health.

        Returns:
            Health check results
        """
        return {
            "service": "AnomalyDetector",
            "initialized": self._initialized,
            "num_rules": len(self._rules),
            "num_metrics": len(self._data),
            "status": "healthy" if self._initialized else "unhealthy",
        }

    def add_rule(self, rule: AnomalyRule) -> None:
        """Add anomaly rule.

        Args:
            rule: Anomaly rule
        """
        self._rules[rule.metric] = rule

    async def process_metric(
        self,
        metric: str,
        value: float,
        timestamp: Optional[datetime] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Optional[AnomalyEvent]:
        """Process metric value for anomalies.

        Args:
            metric: Metric name
            value: Metric value
            timestamp: Optional timestamp
            metadata: Optional metadata

        Returns:
            Anomaly event if detected
        """
        if not self._initialized:
            return None

        try:
            # Get rule for metric
            rule = self._rules.get(metric)
            if not rule or not rule.enabled:
                return None

            # Add data point
            timestamp = timestamp or datetime.now()
            self._data[metric].append((timestamp, value))

            # Check for anomaly
            anomaly = await self._check_anomaly(rule, value, timestamp, metadata or {})
            if anomaly:
                # Emit event
                self.publisher.publish(
                    Event(
                        type=EventType.ANOMALY_DETECTED,
                        timestamp=datetime.now(),
                        component="anomaly_detector",
                        description=f"Anomaly detected for {metric}",
                        metadata={
                            "metric": metric,
                            "value": value,
                            "expected": anomaly.expected,
                            "deviation": anomaly.deviation,
                            "type": anomaly.rule.type.value,
                            "severity": anomaly.rule.severity,
                        },
                    )
                )

            return anomaly

        except Exception as e:
            logger.error(f"Metric processing failed: {e}")
            return None

    async def _check_anomaly(
        self,
        rule: AnomalyRule,
        value: float,
        timestamp: datetime,
        metadata: Dict[str, Any],
    ) -> Optional[AnomalyEvent]:
        """Check for anomaly in metric value.

        Args:
            rule: Anomaly rule
            value: Current value
            timestamp: Current timestamp
            metadata: Additional metadata

        Returns:
            Anomaly event if detected
        """
        # Get historical data
        window_start = timestamp - timedelta(seconds=rule.window)
        history = [
            v for t, v in self._data[rule.metric]
            if window_start <= t <= timestamp
        ]

        # Check minimum data points
        if len(history) < self.anomaly_config.min_data_points:
            return None

        # Convert to numpy array
        data = np.array(history)

        # Check anomaly type
        if rule.type == AnomalyType.SPIKE:
            return self._check_spike(rule, data, value, timestamp, metadata)
        elif rule.type == AnomalyType.DROP:
            return self._check_drop(rule, data, value, timestamp, metadata)
        elif rule.type == AnomalyType.TREND:
            return self._check_trend(rule, data, value, timestamp, metadata)
        elif rule.type == AnomalyType.PATTERN:
            return self._check_pattern(rule, data, value, timestamp, metadata)
        elif rule.type == AnomalyType.OUTLIER:
            return self._check_outlier(rule, data, value, timestamp, metadata)

        return None

    def _check_spike(
        self,
        rule: AnomalyRule,
        history: np.ndarray,
        value: float,
        timestamp: datetime,
        metadata: Dict[str, Any],
    ) -> Optional[AnomalyEvent]:
        """Check for spike anomaly.

        Args:
            rule: Anomaly rule
            history: Historical data
            value: Current value
            timestamp: Current timestamp
            metadata: Additional metadata

        Returns:
            Anomaly event if detected
        """
        mean = np.mean(history)
        std = np.std(history)
        z_score = (value - mean) / std if std > 0 else 0

        if z_score > rule.threshold:
            return AnomalyEvent(
                rule=rule,
                value=value,
                expected=mean,
                deviation=z_score,
                timestamp=timestamp,
                metadata=metadata,
            )

        return None

    def _check_drop(
        self,
        rule: AnomalyRule,
        history: np.ndarray,
        value: float,
        timestamp: datetime,
        metadata: Dict[str, Any],
    ) -> Optional[AnomalyEvent]:
        """Check for drop anomaly.

        Args:
            rule: Anomaly rule
            history: Historical data
            value: Current value
            timestamp: Current timestamp
            metadata: Additional metadata

        Returns:
            Anomaly event if detected
        """
        mean = np.mean(history)
        std = np.std(history)
        z_score = (mean - value) / std if std > 0 else 0

        if z_score > rule.threshold:
            return AnomalyEvent(
                rule=rule,
                value=value,
                expected=mean,
                deviation=z_score,
                timestamp=timestamp,
                metadata=metadata,
            )

        return None

    def _check_trend(
        self,
        rule: AnomalyRule,
        history: np.ndarray,
        value: float,
        timestamp: datetime,
        metadata: Dict[str, Any],
    ) -> Optional[AnomalyEvent]:
        """Check for trend anomaly.

        Args:
            rule: Anomaly rule
            history: Historical data
            value: Current value
            timestamp: Current timestamp
            metadata: Additional metadata

        Returns:
            Anomaly event if detected
        """
        # Calculate trend using linear regression
        x = np.arange(len(history))
        slope, _ = np.polyfit(x, history, 1)
        trend = slope * len(history)

        if abs(trend) > rule.threshold:
            return AnomalyEvent(
                rule=rule,
                value=value,
                expected=history[-1],
                deviation=trend,
                timestamp=timestamp,
                metadata=metadata,
            )

        return None

    def _check_pattern(
        self,
        rule: AnomalyRule,
        history: np.ndarray,
        value: float,
        timestamp: datetime,
        metadata: Dict[str, Any],
    ) -> Optional[AnomalyEvent]:
        """Check for pattern anomaly.

        Args:
            rule: Anomaly rule
            history: Historical data
            value: Current value
            timestamp: Current timestamp
            metadata: Additional metadata

        Returns:
            Anomaly event if detected
        """
        # TODO: Implement pattern detection
        return None

    def _check_outlier(
        self,
        rule: AnomalyRule,
        history: np.ndarray,
        value: float,
        timestamp: datetime,
        metadata: Dict[str, Any],
    ) -> Optional[AnomalyEvent]:
        """Check for outlier anomaly.

        Args:
            rule: Anomaly rule
            history: Historical data
            value: Current value
            timestamp: Current timestamp
            metadata: Additional metadata

        Returns:
            Anomaly event if detected
        """
        q1 = np.percentile(history, 25)
        q3 = np.percentile(history, 75)
        iqr = q3 - q1
        lower = q1 - rule.threshold * iqr
        upper = q3 + rule.threshold * iqr

        if value < lower or value > upper:
            return AnomalyEvent(
                rule=rule,
                value=value,
                expected=(q1 + q3) / 2,
                deviation=min(abs(value - lower), abs(value - upper)),
                timestamp=timestamp,
                metadata=metadata,
            )

        return None

    async def _cleanup_loop(self) -> None:
        """Background task for data cleanup."""
        while True:
            try:
                # Sleep for cleanup interval
                await asyncio.sleep(self.anomaly_config.cleanup_interval)

                # Remove old data points
                now = datetime.now()
                max_window = max(rule.window for rule in self._rules.values())
                cutoff = now - timedelta(seconds=max_window)

                for metric in self._data:
                    self._data[metric] = [
                        (t, v) for t, v in self._data[metric]
                        if t >= cutoff
                    ]

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Data cleanup failed: {e}")


# Create detector instance
anomaly_detector = AnomalyDetector(
    Config(),
    PublisherService(Config(), MetricsManager()),
    MetricsManager(),
)

__all__ = [
    "AnomalyType",
    "AnomalyRule",
    "AnomalyEvent",
    "AnomalyConfig",
    "AnomalyDetector",
    "anomaly_detector",
] 