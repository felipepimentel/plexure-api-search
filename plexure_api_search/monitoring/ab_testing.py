"""A/B testing framework for evaluating search improvements."""

import logging
import asyncio
from typing import Dict, List, Any, Optional, Set, Union, Callable
from datetime import datetime, timedelta
from dataclasses import dataclass
import json
import os
from pathlib import Path
import numpy as np
from collections import defaultdict
import random
import hashlib

from ..config import Config
from .events import Event, EventType
from .metrics import MetricsManager
from ..services.base import BaseService
from ..services.events import PublisherService

logger = logging.getLogger(__name__)


@dataclass
class Variant:
    """Test variant configuration."""

    name: str  # Variant name
    weight: float  # Traffic allocation weight
    config: Dict[str, Any]  # Variant configuration
    description: str  # Variant description
    metadata: Dict[str, Any]  # Additional metadata


@dataclass
class Experiment:
    """A/B test experiment configuration."""

    name: str  # Experiment name
    description: str  # Experiment description
    variants: List[Variant]  # Test variants
    metrics: List[str]  # Metrics to track
    start_time: datetime  # Experiment start time
    end_time: datetime  # Experiment end time
    enabled: bool = True  # Whether experiment is enabled
    metadata: Dict[str, Any] = None  # Additional metadata


@dataclass
class ExperimentResult:
    """A/B test experiment result."""

    experiment: Experiment  # Experiment configuration
    variant: Variant  # Test variant
    user_id: str  # User identifier
    metrics: Dict[str, float]  # Metric values
    timestamp: datetime  # Result timestamp
    metadata: Dict[str, Any]  # Additional metadata


class ABConfig:
    """Configuration for A/B testing."""

    def __init__(
        self,
        min_sample_size: int = 1000,  # Minimum sample size per variant
        confidence_level: float = 0.95,  # Statistical confidence level
        export_interval: float = 3600.0,  # Export every hour
        export_dir: str = "experiments",  # Directory to store exports
        enable_significance_testing: bool = True,
        enable_sequential_testing: bool = True,
        enable_multi_armed_bandit: bool = True,
        enable_feature_flags: bool = True,
    ) -> None:
        """Initialize A/B testing config.

        Args:
            min_sample_size: Minimum sample size per variant
            confidence_level: Statistical confidence level
            export_interval: Export interval in seconds
            export_dir: Directory to store exports
            enable_significance_testing: Whether to perform significance testing
            enable_sequential_testing: Whether to perform sequential testing
            enable_multi_armed_bandit: Whether to use multi-armed bandit
            enable_feature_flags: Whether to use feature flags
        """
        self.min_sample_size = min_sample_size
        self.confidence_level = confidence_level
        self.export_interval = export_interval
        self.export_dir = export_dir
        self.enable_significance_testing = enable_significance_testing
        self.enable_sequential_testing = enable_sequential_testing
        self.enable_multi_armed_bandit = enable_multi_armed_bandit
        self.enable_feature_flags = enable_feature_flags


class ABTesting(BaseService[Dict[str, Any]]):
    """A/B testing implementation."""

    def __init__(
        self,
        config: Config,
        publisher: PublisherService,
        metrics_manager: MetricsManager,
        ab_config: Optional[ABConfig] = None,
    ) -> None:
        """Initialize A/B testing.

        Args:
            config: Application configuration
            publisher: Event publisher
            metrics_manager: Metrics manager
            ab_config: A/B testing configuration
        """
        super().__init__(config, publisher, metrics_manager)
        self.ab_config = ab_config or ABConfig()
        self._initialized = False
        self._experiments: Dict[str, Experiment] = {}
        self._results: Dict[str, List[ExperimentResult]] = defaultdict(list)
        self._assignments: Dict[str, Dict[str, str]] = defaultdict(dict)
        self._export_task: Optional[asyncio.Task] = None

        # Create export directory
        os.makedirs(self.ab_config.export_dir, exist_ok=True)

    async def initialize(self) -> None:
        """Initialize A/B testing resources."""
        if self._initialized:
            return

        try:
            # Start export task
            self._export_task = asyncio.create_task(self._export_loop())

            self._initialized = True
            self.publisher.publish(
                Event(
                    type=EventType.SERVICE_STARTED,
                    timestamp=datetime.now(),
                    component="ab_testing",
                    description="A/B testing initialized",
                    metadata={
                        "min_sample_size": self.ab_config.min_sample_size,
                        "confidence_level": self.ab_config.confidence_level,
                        "significance_testing": self.ab_config.enable_significance_testing,
                        "sequential_testing": self.ab_config.enable_sequential_testing,
                        "multi_armed_bandit": self.ab_config.enable_multi_armed_bandit,
                        "feature_flags": self.ab_config.enable_feature_flags,
                    },
                )
            )

        except Exception as e:
            logger.error(f"Failed to initialize A/B testing: {e}")
            raise

    async def cleanup(self) -> None:
        """Clean up A/B testing resources."""
        if self._export_task:
            self._export_task.cancel()
            try:
                await self._export_task
            except asyncio.CancelledError:
                pass

        self._initialized = False
        self._experiments.clear()
        self._results.clear()
        self._assignments.clear()

        self.publisher.publish(
            Event(
                type=EventType.SERVICE_STOPPED,
                timestamp=datetime.now(),
                component="ab_testing",
                description="A/B testing stopped",
            )
        )

    async def health_check(self) -> Dict[str, Any]:
        """Check A/B testing health.

        Returns:
            Health check results
        """
        return {
            "service": "ABTesting",
            "initialized": self._initialized,
            "num_experiments": len(self._experiments),
            "num_results": sum(len(results) for results in self._results.values()),
            "num_assignments": sum(len(assignments) for assignments in self._assignments.values()),
            "status": "healthy" if self._initialized else "unhealthy",
        }

    def create_experiment(
        self,
        name: str,
        description: str,
        variants: List[Variant],
        metrics: List[str],
        duration: timedelta,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Experiment:
        """Create new experiment.

        Args:
            name: Experiment name
            description: Experiment description
            variants: Test variants
            metrics: Metrics to track
            duration: Experiment duration
            metadata: Additional metadata

        Returns:
            Created experiment
        """
        if not self._initialized:
            raise RuntimeError("A/B testing not initialized")

        # Validate variants
        total_weight = sum(variant.weight for variant in variants)
        if not np.isclose(total_weight, 1.0):
            raise ValueError("Variant weights must sum to 1.0")

        # Create experiment
        experiment = Experiment(
            name=name,
            description=description,
            variants=variants,
            metrics=metrics,
            start_time=datetime.now(),
            end_time=datetime.now() + duration,
            metadata=metadata or {},
        )

        # Register experiment
        self._experiments[name] = experiment

        # Emit event
        self.publisher.publish(
            Event(
                type=EventType.EXPERIMENT_CREATED,
                timestamp=datetime.now(),
                component="ab_testing",
                description=f"Experiment created: {name}",
                metadata={
                    "name": name,
                    "description": description,
                    "variants": [v.name for v in variants],
                    "metrics": metrics,
                    "duration": str(duration),
                    "metadata": metadata,
                },
            )
        )

        return experiment

    def get_variant(
        self,
        experiment_name: str,
        user_id: str,
    ) -> Optional[Variant]:
        """Get variant assignment for user.

        Args:
            experiment_name: Experiment name
            user_id: User identifier

        Returns:
            Assigned variant
        """
        if not self._initialized:
            return None

        try:
            # Get experiment
            experiment = self._experiments.get(experiment_name)
            if not experiment or not experiment.enabled:
                return None

            # Check if experiment is active
            now = datetime.now()
            if now < experiment.start_time or now > experiment.end_time:
                return None

            # Get existing assignment
            if experiment_name in self._assignments[user_id]:
                variant_name = self._assignments[user_id][experiment_name]
                return next(
                    (v for v in experiment.variants if v.name == variant_name),
                    None,
                )

            # Assign variant
            if self.ab_config.enable_multi_armed_bandit:
                variant = self._assign_bandit(experiment, user_id)
            else:
                variant = self._assign_random(experiment, user_id)

            # Store assignment
            self._assignments[user_id][experiment_name] = variant.name

            # Emit event
            self.publisher.publish(
                Event(
                    type=EventType.VARIANT_ASSIGNED,
                    timestamp=datetime.now(),
                    component="ab_testing",
                    description=f"Variant assigned: {variant.name}",
                    metadata={
                        "experiment": experiment_name,
                        "user_id": user_id,
                        "variant": variant.name,
                    },
                )
            )

            return variant

        except Exception as e:
            logger.error(f"Failed to get variant: {e}")
            return None

    def record_metrics(
        self,
        experiment_name: str,
        user_id: str,
        metrics: Dict[str, float],
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Record experiment metrics.

        Args:
            experiment_name: Experiment name
            user_id: User identifier
            metrics: Metric values
            metadata: Additional metadata
        """
        if not self._initialized:
            return

        try:
            # Get experiment and variant
            experiment = self._experiments.get(experiment_name)
            if not experiment or not experiment.enabled:
                return

            variant_name = self._assignments[user_id].get(experiment_name)
            if not variant_name:
                return

            variant = next(
                (v for v in experiment.variants if v.name == variant_name),
                None,
            )
            if not variant:
                return

            # Create result
            result = ExperimentResult(
                experiment=experiment,
                variant=variant,
                user_id=user_id,
                metrics=metrics,
                timestamp=datetime.now(),
                metadata=metadata or {},
            )

            # Store result
            self._results[experiment_name].append(result)

            # Emit event
            self.publisher.publish(
                Event(
                    type=EventType.METRICS_RECORDED,
                    timestamp=datetime.now(),
                    component="ab_testing",
                    description=f"Metrics recorded: {experiment_name}",
                    metadata={
                        "experiment": experiment_name,
                        "user_id": user_id,
                        "variant": variant_name,
                        "metrics": metrics,
                        "metadata": metadata,
                    },
                )
            )

            # Check significance
            if self.ab_config.enable_significance_testing:
                self._check_significance(experiment_name)

        except Exception as e:
            logger.error(f"Failed to record metrics: {e}")

    def _assign_random(
        self,
        experiment: Experiment,
        user_id: str,
    ) -> Variant:
        """Assign variant randomly.

        Args:
            experiment: Experiment configuration
            user_id: User identifier

        Returns:
            Assigned variant
        """
        # Use hash for consistent assignment
        hash_input = f"{experiment.name}:{user_id}"
        hash_value = int(hashlib.md5(hash_input.encode()).hexdigest(), 16)
        random.seed(hash_value)

        # Select variant based on weights
        weights = [variant.weight for variant in experiment.variants]
        return random.choices(experiment.variants, weights=weights)[0]

    def _assign_bandit(
        self,
        experiment: Experiment,
        user_id: str,
    ) -> Variant:
        """Assign variant using multi-armed bandit.

        Args:
            experiment: Experiment configuration
            user_id: User identifier

        Returns:
            Assigned variant
        """
        # Get results for experiment
        results = self._results.get(experiment.name, [])
        if not results:
            return self._assign_random(experiment, user_id)

        # Calculate success rates
        success_rates = {}
        for variant in experiment.variants:
            variant_results = [
                r for r in results
                if r.variant.name == variant.name
            ]
            if not variant_results:
                success_rates[variant.name] = 0.0
                continue

            # Use first metric as success metric
            metric = experiment.metrics[0]
            values = [r.metrics[metric] for r in variant_results]
            success_rates[variant.name] = np.mean(values)

        # Select variant using Thompson sampling
        weights = [success_rates[v.name] for v in experiment.variants]
        return random.choices(experiment.variants, weights=weights)[0]

    def _check_significance(self, experiment_name: str) -> None:
        """Check statistical significance.

        Args:
            experiment_name: Experiment name
        """
        # Get experiment and results
        experiment = self._experiments[experiment_name]
        results = self._results[experiment_name]

        # Check sample size
        variant_results = defaultdict(list)
        for result in results:
            variant_results[result.variant.name].append(result)

        if any(
            len(results) < self.ab_config.min_sample_size
            for results in variant_results.values()
        ):
            return

        # Calculate statistics
        stats = {}
        for metric in experiment.metrics:
            metric_stats = {}
            for variant_name, variant_results in variant_results.items():
                values = [r.metrics[metric] for r in variant_results]
                metric_stats[variant_name] = {
                    "mean": np.mean(values),
                    "std": np.std(values),
                    "count": len(values),
                }
            stats[metric] = metric_stats

        # Perform t-tests
        control = experiment.variants[0].name
        for metric in experiment.metrics:
            metric_stats = stats[metric]
            control_stats = metric_stats[control]
            for variant_name, variant_stats in metric_stats.items():
                if variant_name == control:
                    continue

                # Calculate t-statistic
                n1 = control_stats["count"]
                n2 = variant_stats["count"]
                m1 = control_stats["mean"]
                m2 = variant_stats["mean"]
                v1 = control_stats["std"] ** 2
                v2 = variant_stats["std"] ** 2

                t = (m1 - m2) / np.sqrt(v1/n1 + v2/n2)
                p = 2 * (1 - stats.t.cdf(abs(t), df=n1+n2-2))

                # Check significance
                if p < (1 - self.ab_config.confidence_level):
                    self.publisher.publish(
                        Event(
                            type=EventType.SIGNIFICANCE_DETECTED,
                            timestamp=datetime.now(),
                            component="ab_testing",
                            description=f"Significance detected: {experiment_name}",
                            metadata={
                                "experiment": experiment_name,
                                "metric": metric,
                                "control": control,
                                "variant": variant_name,
                                "control_mean": m1,
                                "variant_mean": m2,
                                "t_statistic": t,
                                "p_value": p,
                            },
                        )
                    )

    async def _export_loop(self) -> None:
        """Background task for exporting results."""
        while True:
            try:
                # Sleep for export interval
                await asyncio.sleep(self.ab_config.export_interval)

                # Export results
                await self._export_results()

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Export loop failed: {e}")

    async def _export_results(self) -> None:
        """Export experiment results."""
        try:
            # Create export data
            data = {
                "timestamp": datetime.now().isoformat(),
                "experiments": [
                    {
                        "name": experiment.name,
                        "description": experiment.description,
                        "start_time": experiment.start_time.isoformat(),
                        "end_time": experiment.end_time.isoformat(),
                        "variants": [
                            {
                                "name": variant.name,
                                "weight": variant.weight,
                                "description": variant.description,
                                "metadata": variant.metadata,
                            }
                            for variant in experiment.variants
                        ],
                        "metrics": experiment.metrics,
                        "metadata": experiment.metadata,
                        "results": [
                            {
                                "variant": result.variant.name,
                                "user_id": result.user_id,
                                "metrics": result.metrics,
                                "timestamp": result.timestamp.isoformat(),
                                "metadata": result.metadata,
                            }
                            for result in self._results[experiment.name]
                        ],
                    }
                    for experiment in self._experiments.values()
                ],
            }

            # Save to file
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"experiments_{timestamp}.json"
            filepath = os.path.join(self.ab_config.export_dir, filename)

            with open(filepath, "w") as f:
                json.dump(data, f, indent=2)

            # Emit event
            self.publisher.publish(
                Event(
                    type=EventType.RESULTS_EXPORTED,
                    timestamp=datetime.now(),
                    component="ab_testing",
                    description="Results exported",
                    metadata={
                        "filename": filename,
                        "num_experiments": len(self._experiments),
                        "num_results": sum(len(results) for results in self._results.values()),
                    },
                )
            )

        except Exception as e:
            logger.error(f"Failed to export results: {e}")


# Create testing instance
ab_testing = ABTesting(
    Config(),
    PublisherService(Config(), MetricsManager()),
    MetricsManager(),
)

__all__ = [
    "Variant",
    "Experiment",
    "ExperimentResult",
    "ABConfig",
    "ABTesting",
    "ab_testing",
] 