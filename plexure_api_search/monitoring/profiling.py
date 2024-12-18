"""Performance profiling for monitoring system performance."""

import logging
import asyncio
from typing import Dict, List, Any, Optional, Set, Union, Callable
from datetime import datetime, timedelta
from dataclasses import dataclass
import time
import cProfile
import pstats
import io
import tracemalloc
import functools
import json
import os
from pathlib import Path

from ..config import Config
from .events import Event, EventType
from .metrics import MetricsManager
from ..services.base import BaseService
from ..services.events import PublisherService

logger = logging.getLogger(__name__)


@dataclass
class ProfileConfig:
    """Configuration for performance profiling."""

    enable_cpu_profiling: bool = True
    enable_memory_profiling: bool = True
    enable_io_profiling: bool = True
    enable_network_profiling: bool = True
    profile_interval: float = 60.0  # Profile every minute
    snapshot_interval: float = 300.0  # Take snapshot every 5 minutes
    retention_days: int = 7  # Keep profiles for 7 days
    profile_dir: str = "profiles"  # Directory to store profiles
    min_duration: float = 0.1  # Minimum duration to profile (seconds)


@dataclass
class ProfileSnapshot:
    """Performance profile snapshot."""

    timestamp: datetime
    duration: float
    cpu_stats: Dict[str, Any]
    memory_stats: Dict[str, Any]
    io_stats: Dict[str, Any]
    network_stats: Dict[str, Any]
    metadata: Dict[str, Any]


class Profiler(BaseService[Dict[str, Any]]):
    """Performance profiler implementation."""

    def __init__(
        self,
        config: Config,
        publisher: PublisherService,
        metrics_manager: MetricsManager,
        profile_config: Optional[ProfileConfig] = None,
    ) -> None:
        """Initialize profiler.

        Args:
            config: Application configuration
            publisher: Event publisher
            metrics_manager: Metrics manager
            profile_config: Profile configuration
        """
        super().__init__(config, publisher, metrics_manager)
        self.profile_config = profile_config or ProfileConfig()
        self._initialized = False
        self._profiler = cProfile.Profile()
        self._snapshots: List[ProfileSnapshot] = []
        self._current_snapshot: Optional[ProfileSnapshot] = None
        self._profile_task: Optional[asyncio.Task] = None
        self._snapshot_task: Optional[asyncio.Task] = None
        self._cleanup_task: Optional[asyncio.Task] = None

        # Create profile directory
        os.makedirs(self.profile_config.profile_dir, exist_ok=True)

        # Start memory tracking
        if self.profile_config.enable_memory_profiling:
            tracemalloc.start()

    async def initialize(self) -> None:
        """Initialize profiler resources."""
        if self._initialized:
            return

        try:
            # Start background tasks
            self._profile_task = asyncio.create_task(self._profile_loop())
            self._snapshot_task = asyncio.create_task(self._snapshot_loop())
            self._cleanup_task = asyncio.create_task(self._cleanup_loop())

            self._initialized = True
            self.publisher.publish(
                Event(
                    type=EventType.SERVICE_STARTED,
                    timestamp=datetime.now(),
                    component="profiler",
                    description="Profiler initialized",
                    metadata={
                        "cpu_profiling": self.profile_config.enable_cpu_profiling,
                        "memory_profiling": self.profile_config.enable_memory_profiling,
                        "io_profiling": self.profile_config.enable_io_profiling,
                        "network_profiling": self.profile_config.enable_network_profiling,
                    },
                )
            )

        except Exception as e:
            logger.error(f"Failed to initialize profiler: {e}")
            raise

    async def cleanup(self) -> None:
        """Clean up profiler resources."""
        # Stop background tasks
        if self._profile_task:
            self._profile_task.cancel()
            try:
                await self._profile_task
            except asyncio.CancelledError:
                pass

        if self._snapshot_task:
            self._snapshot_task.cancel()
            try:
                await self._snapshot_task
            except asyncio.CancelledError:
                pass

        if self._cleanup_task:
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass

        # Stop profiling
        if self._profiler:
            self._profiler.disable()

        # Stop memory tracking
        if self.profile_config.enable_memory_profiling:
            tracemalloc.stop()

        self._initialized = False
        self._snapshots.clear()
        self._current_snapshot = None

        self.publisher.publish(
            Event(
                type=EventType.SERVICE_STOPPED,
                timestamp=datetime.now(),
                component="profiler",
                description="Profiler stopped",
            )
        )

    async def health_check(self) -> Dict[str, Any]:
        """Check profiler health.

        Returns:
            Health check results
        """
        return {
            "service": "Profiler",
            "initialized": self._initialized,
            "num_snapshots": len(self._snapshots),
            "current_snapshot": bool(self._current_snapshot),
            "status": "healthy" if self._initialized else "unhealthy",
        }

    def profile(self, func: Callable) -> Callable:
        """Decorator for profiling functions.

        Args:
            func: Function to profile

        Returns:
            Wrapped function
        """
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            if not self._initialized:
                return await func(*args, **kwargs)

            # Start profiling
            start_time = time.time()
            self._profiler.enable()

            try:
                # Call function
                result = await func(*args, **kwargs)

                # Stop profiling
                self._profiler.disable()
                duration = time.time() - start_time

                # Record profile if duration exceeds threshold
                if duration >= self.profile_config.min_duration:
                    await self._record_profile(func.__name__, duration)

                return result

            except Exception as e:
                # Stop profiling on error
                self._profiler.disable()
                raise

        return wrapper

    async def _record_profile(
        self,
        name: str,
        duration: float,
    ) -> None:
        """Record performance profile.

        Args:
            name: Profile name
            duration: Profile duration
        """
        try:
            # Get CPU stats
            cpu_stats = {}
            if self.profile_config.enable_cpu_profiling:
                stats = pstats.Stats(self._profiler)
                stream = io.StringIO()
                stats.stream = stream
                stats.sort_stats("cumulative")
                stats.print_stats()
                cpu_stats = {
                    "stats": stream.getvalue(),
                    "total_calls": stats.total_calls,
                    "total_time": stats.total_tt,
                }

            # Get memory stats
            memory_stats = {}
            if self.profile_config.enable_memory_profiling:
                snapshot = tracemalloc.take_snapshot()
                top_stats = snapshot.statistics("lineno")
                memory_stats = {
                    "current": tracemalloc.get_traced_memory()[0],
                    "peak": tracemalloc.get_traced_memory()[1],
                    "top_stats": [
                        {
                            "file": stat.traceback[0].filename,
                            "line": stat.traceback[0].lineno,
                            "size": stat.size,
                            "count": stat.count,
                        }
                        for stat in top_stats[:10]
                    ],
                }

            # Get IO stats
            io_stats = {}
            if self.profile_config.enable_io_profiling:
                # TODO: Implement IO profiling
                pass

            # Get network stats
            network_stats = {}
            if self.profile_config.enable_network_profiling:
                # TODO: Implement network profiling
                pass

            # Create snapshot
            snapshot = ProfileSnapshot(
                timestamp=datetime.now(),
                duration=duration,
                cpu_stats=cpu_stats,
                memory_stats=memory_stats,
                io_stats=io_stats,
                network_stats=network_stats,
                metadata={
                    "name": name,
                    "pid": os.getpid(),
                },
            )

            # Save snapshot
            self._current_snapshot = snapshot
            self._snapshots.append(snapshot)

            # Save to file
            await self._save_snapshot(snapshot)

            # Emit event
            self.publisher.publish(
                Event(
                    type=EventType.PROFILE_RECORDED,
                    timestamp=datetime.now(),
                    component="profiler",
                    description=f"Profile recorded: {name}",
                    metadata={
                        "name": name,
                        "duration": duration,
                        "cpu_stats": bool(cpu_stats),
                        "memory_stats": bool(memory_stats),
                        "io_stats": bool(io_stats),
                        "network_stats": bool(network_stats),
                    },
                )
            )

        except Exception as e:
            logger.error(f"Failed to record profile: {e}")

    async def _save_snapshot(self, snapshot: ProfileSnapshot) -> None:
        """Save profile snapshot to file.

        Args:
            snapshot: Profile snapshot
        """
        try:
            # Create filename
            timestamp = snapshot.timestamp.strftime("%Y%m%d_%H%M%S")
            filename = f"profile_{timestamp}.json"
            filepath = os.path.join(self.profile_config.profile_dir, filename)

            # Convert to JSON
            data = {
                "timestamp": snapshot.timestamp.isoformat(),
                "duration": snapshot.duration,
                "cpu_stats": snapshot.cpu_stats,
                "memory_stats": snapshot.memory_stats,
                "io_stats": snapshot.io_stats,
                "network_stats": snapshot.network_stats,
                "metadata": snapshot.metadata,
            }

            # Save to file
            with open(filepath, "w") as f:
                json.dump(data, f, indent=2)

        except Exception as e:
            logger.error(f"Failed to save snapshot: {e}")

    async def _profile_loop(self) -> None:
        """Background task for continuous profiling."""
        while True:
            try:
                # Sleep for profile interval
                await asyncio.sleep(self.profile_config.profile_interval)

                # Record profile
                await self._record_profile(
                    name="system",
                    duration=self.profile_config.profile_interval,
                )

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Profile loop failed: {e}")

    async def _snapshot_loop(self) -> None:
        """Background task for taking snapshots."""
        while True:
            try:
                # Sleep for snapshot interval
                await asyncio.sleep(self.profile_config.snapshot_interval)

                # Take snapshot
                if self.profile_config.enable_memory_profiling:
                    snapshot = tracemalloc.take_snapshot()
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    filename = f"memory_{timestamp}.snapshot"
                    filepath = os.path.join(self.profile_config.profile_dir, filename)
                    snapshot.dump(filepath)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Snapshot loop failed: {e}")

    async def _cleanup_loop(self) -> None:
        """Background task for cleaning up old profiles."""
        while True:
            try:
                # Sleep for one day
                await asyncio.sleep(86400)

                # Get cutoff time
                cutoff = datetime.now() - timedelta(days=self.profile_config.retention_days)

                # Remove old snapshots
                self._snapshots = [
                    snapshot for snapshot in self._snapshots
                    if snapshot.timestamp >= cutoff
                ]

                # Remove old files
                for path in Path(self.profile_config.profile_dir).glob("*"):
                    if path.stat().st_mtime < cutoff.timestamp():
                        path.unlink()

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Cleanup loop failed: {e}")


# Create profiler instance
profiler = Profiler(
    Config(),
    PublisherService(Config(), MetricsManager()),
    MetricsManager(),
)

__all__ = [
    "ProfileConfig",
    "ProfileSnapshot",
    "Profiler",
    "profiler",
] 