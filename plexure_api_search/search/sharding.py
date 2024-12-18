"""Sharding capabilities for large indices."""

import logging
from typing import Dict, List, Any, Optional, Set
from datetime import datetime
import hashlib
import asyncio
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from sklearn.preprocessing import MinMaxScaler

from ..config import Config
from ..monitoring.events import Event, EventType
from ..monitoring.metrics import MetricsManager
from ..services.base import BaseService
from ..services.events import PublisherService
from .search_models import SearchResult
from .searcher import APISearcher

logger = logging.getLogger(__name__)


class ShardConfig:
    """Configuration for index shard."""

    def __init__(
        self,
        shard_id: str,
        host: str = "localhost",
        port: int = 8000,
        weight: float = 1.0,
        max_concurrent: int = 10,
        timeout: float = 5.0,
        retry_count: int = 3,
        retry_delay: float = 1.0,
        health_interval: float = 60.0,
        size_limit: int = 1000000,  # Maximum number of vectors per shard
        replication_factor: int = 2,  # Number of replicas per shard
    ) -> None:
        """Initialize shard config.

        Args:
            shard_id: Shard identifier
            host: Shard hostname
            port: Shard port
            weight: Shard weight for load balancing
            max_concurrent: Maximum concurrent requests
            timeout: Request timeout in seconds
            retry_count: Number of retries
            retry_delay: Delay between retries in seconds
            health_interval: Health check interval in seconds
            size_limit: Maximum number of vectors per shard
            replication_factor: Number of replicas per shard
        """
        self.shard_id = shard_id
        self.host = host
        self.port = port
        self.weight = weight
        self.max_concurrent = max_concurrent
        self.timeout = timeout
        self.retry_count = retry_count
        self.retry_delay = retry_delay
        self.health_interval = health_interval
        self.size_limit = size_limit
        self.replication_factor = replication_factor


class IndexShard:
    """Shard for distributed index."""

    def __init__(
        self,
        config: ShardConfig,
        searcher: APISearcher,
    ) -> None:
        """Initialize index shard.

        Args:
            config: Shard configuration
            searcher: API searcher instance
        """
        self.config = config
        self.searcher = searcher
        self._semaphore = asyncio.Semaphore(config.max_concurrent)
        self._executor = ThreadPoolExecutor(max_workers=config.max_concurrent)
        self._healthy = True
        self._last_health_check = datetime.now()
        self._size = 0

    async def search(
        self,
        query: str,
        **kwargs: Any,
    ) -> List[SearchResult]:
        """Perform search on shard.

        Args:
            query: Search query
            **kwargs: Additional search parameters

        Returns:
            Search results
        """
        async with self._semaphore:
            try:
                # Execute search in thread pool
                loop = asyncio.get_event_loop()
                results = await loop.run_in_executor(
                    self._executor,
                    lambda: self.searcher.search(query, **kwargs),
                )
                return results

            except Exception as e:
                logger.error(f"Shard {self.config.shard_id} search failed: {e}")
                return []

    async def health_check(self) -> bool:
        """Check shard health.

        Returns:
            True if healthy, False otherwise
        """
        try:
            # Check if health check is needed
            now = datetime.now()
            if (now - self._last_health_check).total_seconds() < self.config.health_interval:
                return self._healthy

            # Perform health check
            loop = asyncio.get_event_loop()
            health = await loop.run_in_executor(
                self._executor,
                lambda: self.searcher.health_check(),
            )

            self._healthy = health.get("status") == "healthy"
            self._last_health_check = now
            return self._healthy

        except Exception as e:
            logger.error(f"Shard {self.config.shard_id} health check failed: {e}")
            self._healthy = False
            return False

    async def add_vectors(
        self,
        vectors: List[np.ndarray],
        metadata: List[Dict[str, Any]],
    ) -> bool:
        """Add vectors to shard.

        Args:
            vectors: List of vectors to add
            metadata: List of metadata for vectors

        Returns:
            True if successful, False otherwise
        """
        try:
            # Check size limit
            if self._size + len(vectors) > self.config.size_limit:
                logger.error(f"Shard {self.config.shard_id} size limit exceeded")
                return False

            # Add vectors to index
            loop = asyncio.get_event_loop()
            success = await loop.run_in_executor(
                self._executor,
                lambda: self.searcher.add_vectors(vectors, metadata),
            )

            if success:
                self._size += len(vectors)

            return success

        except Exception as e:
            logger.error(f"Shard {self.config.shard_id} add vectors failed: {e}")
            return False


class ShardingConfig:
    """Configuration for index sharding."""

    def __init__(
        self,
        min_shards: int = 2,
        max_shards: int = 10,
        min_healthy: float = 0.5,
        strategy: str = "consistent_hashing",
        timeout: float = 10.0,
        retry_count: int = 3,
        retry_delay: float = 1.0,
        health_interval: float = 60.0,
        rebalance_threshold: float = 0.2,  # Trigger rebalancing when imbalance > 20%
        rebalance_interval: float = 3600.0,  # Rebalance every hour
    ) -> None:
        """Initialize sharding config.

        Args:
            min_shards: Minimum number of shards
            max_shards: Maximum number of shards
            min_healthy: Minimum healthy shard ratio
            strategy: Sharding strategy
            timeout: Request timeout in seconds
            retry_count: Number of retries
            retry_delay: Delay between retries in seconds
            health_interval: Health check interval in seconds
            rebalance_threshold: Threshold for triggering rebalancing
            rebalance_interval: Interval between rebalancing
        """
        self.min_shards = min_shards
        self.max_shards = max_shards
        self.min_healthy = min_healthy
        self.strategy = strategy
        self.timeout = timeout
        self.retry_count = retry_count
        self.retry_delay = retry_delay
        self.health_interval = health_interval
        self.rebalance_threshold = rebalance_threshold
        self.rebalance_interval = rebalance_interval


class ShardingManager(BaseService[Dict[str, Any]]):
    """Sharding manager for distributed index."""

    def __init__(
        self,
        config: Config,
        publisher: PublisherService,
        metrics_manager: MetricsManager,
        sharding_config: Optional[ShardingConfig] = None,
    ) -> None:
        """Initialize sharding manager.

        Args:
            config: Application configuration
            publisher: Event publisher
            metrics_manager: Metrics manager
            sharding_config: Sharding configuration
        """
        super().__init__(config, publisher, metrics_manager)
        self.sharding_config = sharding_config or ShardingConfig()
        self._initialized = False
        self._shards: Dict[str, IndexShard] = {}
        self._hash_ring: List[str] = []  # Consistent hashing ring
        self._last_rebalance = datetime.now()

    async def initialize(self) -> None:
        """Initialize sharding resources."""
        if self._initialized:
            return

        try:
            # Initialize shards
            for i in range(self.sharding_config.min_shards):
                shard_id = f"shard_{i}"
                shard_config = ShardConfig(
                    shard_id=shard_id,
                    host="localhost",
                    port=8000 + i,
                    weight=1.0,
                    max_concurrent=10,
                    timeout=self.sharding_config.timeout,
                    retry_count=self.sharding_config.retry_count,
                    retry_delay=self.sharding_config.retry_delay,
                    health_interval=self.sharding_config.health_interval,
                )
                self._shards[shard_id] = IndexShard(
                    config=shard_config,
                    searcher=APISearcher(),
                )

            # Initialize hash ring
            self._rebuild_hash_ring()

            self._initialized = True
            self.publisher.publish(
                Event(
                    type=EventType.SERVICE_STARTED,
                    timestamp=datetime.now(),
                    component="sharding_manager",
                    description="Sharding manager initialized",
                    metadata={
                        "num_shards": len(self._shards),
                        "strategy": self.sharding_config.strategy,
                    },
                )
            )

        except Exception as e:
            logger.error(f"Failed to initialize sharding manager: {e}")
            raise

    async def cleanup(self) -> None:
        """Clean up sharding resources."""
        self._initialized = False
        self._shards.clear()
        self._hash_ring.clear()

        self.publisher.publish(
            Event(
                type=EventType.SERVICE_STOPPED,
                timestamp=datetime.now(),
                component="sharding_manager",
                description="Sharding manager stopped",
            )
        )

    async def health_check(self) -> Dict[str, Any]:
        """Check sharding manager health.

        Returns:
            Health check results
        """
        # Check shard health
        healthy_shards = 0
        for shard in self._shards.values():
            if await shard.health_check():
                healthy_shards += 1

        # Calculate health ratio
        health_ratio = healthy_shards / len(self._shards) if self._shards else 0.0
        is_healthy = health_ratio >= self.sharding_config.min_healthy

        return {
            "service": "ShardingManager",
            "initialized": self._initialized,
            "num_shards": len(self._shards),
            "healthy_shards": healthy_shards,
            "health_ratio": health_ratio,
            "status": "healthy" if is_healthy else "unhealthy",
        }

    async def search(
        self,
        query: str,
        **kwargs: Any,
    ) -> List[SearchResult]:
        """Perform sharded search.

        Args:
            query: Search query
            **kwargs: Additional search parameters

        Returns:
            Combined search results
        """
        if not self._initialized:
            await self.initialize()

        try:
            # Get healthy shards
            healthy_shards = [
                shard for shard in self._shards.values()
                if await shard.health_check()
            ]

            if not healthy_shards:
                logger.error("No healthy shards available")
                return []

            # Emit search started event
            self.publisher.publish(
                Event(
                    type=EventType.SEARCH_STARTED,
                    timestamp=datetime.now(),
                    component="sharding_manager",
                    description=f"Starting sharded search for query: {query}",
                    metadata={
                        "num_shards": len(healthy_shards),
                        "strategy": self.sharding_config.strategy,
                    },
                )
            )

            # Select shards based on strategy
            if self.sharding_config.strategy == "consistent_hashing":
                # Use consistent hashing to select shard
                shard_id = self._get_shard_for_key(query)
                selected_shards = [self._shards[shard_id]] if shard_id in self._shards else []
            else:
                # Use all healthy shards
                selected_shards = healthy_shards

            # Perform sharded search
            tasks = []
            for shard in selected_shards:
                task = asyncio.create_task(shard.search(query, **kwargs))
                tasks.append(task)

            # Wait for results with timeout
            results = []
            done, pending = await asyncio.wait(
                tasks,
                timeout=self.sharding_config.timeout,
            )

            # Cancel pending tasks
            for task in pending:
                task.cancel()

            # Combine results
            for task in done:
                try:
                    shard_results = await task
                    results.extend(shard_results)
                except Exception as e:
                    logger.error(f"Shard search failed: {e}")

            # Sort by score
            results.sort(key=lambda x: x.score, reverse=True)

            # Check if rebalancing is needed
            await self._check_rebalancing()

            # Emit success event
            self.publisher.publish(
                Event(
                    type=EventType.SEARCH_COMPLETED,
                    timestamp=datetime.now(),
                    component="sharding_manager",
                    description="Sharded search completed",
                    metadata={
                        "num_results": len(results),
                        "num_shards": len(selected_shards),
                    },
                )
            )

            return results

        except Exception as e:
            logger.error(f"Sharded search failed: {e}")
            self.publisher.publish(
                Event(
                    type=EventType.SEARCH_FAILED,
                    timestamp=datetime.now(),
                    component="sharding_manager",
                    description="Sharded search failed",
                    error=str(e),
                )
            )
            return []

    async def add_vectors(
        self,
        vectors: List[np.ndarray],
        metadata: List[Dict[str, Any]],
    ) -> bool:
        """Add vectors to sharded index.

        Args:
            vectors: List of vectors to add
            metadata: List of metadata for vectors

        Returns:
            True if successful, False otherwise
        """
        if not self._initialized:
            await self.initialize()

        try:
            # Group vectors by shard
            shard_vectors: Dict[str, List[int]] = {}
            for i, meta in enumerate(metadata):
                shard_id = self._get_shard_for_key(str(meta))
                if shard_id not in shard_vectors:
                    shard_vectors[shard_id] = []
                shard_vectors[shard_id].append(i)

            # Add vectors to shards
            tasks = []
            for shard_id, indices in shard_vectors.items():
                if shard_id not in self._shards:
                    continue

                shard = self._shards[shard_id]
                shard_vectors_subset = [vectors[i] for i in indices]
                shard_metadata_subset = [metadata[i] for i in indices]
                task = asyncio.create_task(
                    shard.add_vectors(shard_vectors_subset, shard_metadata_subset)
                )
                tasks.append(task)

            # Wait for all tasks to complete
            results = await asyncio.gather(*tasks, return_exceptions=True)

            # Check results
            success = all(isinstance(r, bool) and r for r in results)

            # Check if rebalancing is needed
            if success:
                await self._check_rebalancing()

            return success

        except Exception as e:
            logger.error(f"Add vectors failed: {e}")
            return False

    def _get_shard_for_key(self, key: str) -> str:
        """Get shard ID for key using consistent hashing.

        Args:
            key: Key to hash

        Returns:
            Shard ID
        """
        if not self._hash_ring:
            return next(iter(self._shards))

        # Calculate hash
        hash_value = int(hashlib.md5(key.encode()).hexdigest(), 16)

        # Find shard in ring
        for shard_id in self._hash_ring:
            shard_hash = int(hashlib.md5(shard_id.encode()).hexdigest(), 16)
            if hash_value <= shard_hash:
                return shard_id

        # Wrap around to first shard
        return self._hash_ring[0]

    def _rebuild_hash_ring(self) -> None:
        """Rebuild consistent hashing ring."""
        self._hash_ring = sorted(
            self._shards.keys(),
            key=lambda x: int(hashlib.md5(x.encode()).hexdigest(), 16),
        )

    async def _check_rebalancing(self) -> None:
        """Check if rebalancing is needed."""
        # Check if enough time has passed
        now = datetime.now()
        if (now - self._last_rebalance).total_seconds() < self.sharding_config.rebalance_interval:
            return

        # Calculate shard sizes
        sizes = [shard._size for shard in self._shards.values()]
        if not sizes:
            return

        # Calculate imbalance
        avg_size = sum(sizes) / len(sizes)
        max_imbalance = max(abs(size - avg_size) / avg_size for size in sizes)

        # Check if rebalancing is needed
        if max_imbalance > self.sharding_config.rebalance_threshold:
            await self._rebalance_shards()

        self._last_rebalance = now

    async def _rebalance_shards(self) -> None:
        """Rebalance shards."""
        # TODO: Implement shard rebalancing
        # This would involve:
        # 1. Calculating optimal distribution
        # 2. Moving vectors between shards
        # 3. Updating hash ring
        pass


# Create service instance
sharding_manager = ShardingManager(
    Config(),
    PublisherService(Config(), MetricsManager()),
    MetricsManager(),
)

__all__ = [
    "ShardConfig",
    "IndexShard",
    "ShardingConfig",
    "ShardingManager",
    "sharding_manager",
] 