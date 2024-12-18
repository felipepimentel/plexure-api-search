"""Distributed search capabilities for improved performance."""

import logging
from typing import Dict, List, Any, Optional, Set
from datetime import datetime
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


class NodeConfig:
    """Configuration for search node."""

    def __init__(
        self,
        node_id: str,
        host: str = "localhost",
        port: int = 8000,
        weight: float = 1.0,
        max_concurrent: int = 10,
        timeout: float = 5.0,
        retry_count: int = 3,
        retry_delay: float = 1.0,
        health_interval: float = 60.0,
    ) -> None:
        """Initialize node config.

        Args:
            node_id: Node identifier
            host: Node hostname
            port: Node port
            weight: Node weight for load balancing
            max_concurrent: Maximum concurrent requests
            timeout: Request timeout in seconds
            retry_count: Number of retries
            retry_delay: Delay between retries in seconds
            health_interval: Health check interval in seconds
        """
        self.node_id = node_id
        self.host = host
        self.port = port
        self.weight = weight
        self.max_concurrent = max_concurrent
        self.timeout = timeout
        self.retry_count = retry_count
        self.retry_delay = retry_delay
        self.health_interval = health_interval


class SearchNode:
    """Distributed search node."""

    def __init__(
        self,
        config: NodeConfig,
        searcher: APISearcher,
    ) -> None:
        """Initialize search node.

        Args:
            config: Node configuration
            searcher: API searcher instance
        """
        self.config = config
        self.searcher = searcher
        self._semaphore = asyncio.Semaphore(config.max_concurrent)
        self._executor = ThreadPoolExecutor(max_workers=config.max_concurrent)
        self._healthy = True
        self._last_health_check = datetime.now()

    async def search(
        self,
        query: str,
        **kwargs: Any,
    ) -> List[SearchResult]:
        """Perform search on node.

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
                logger.error(f"Node {self.config.node_id} search failed: {e}")
                return []

    async def health_check(self) -> bool:
        """Check node health.

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
            logger.error(f"Node {self.config.node_id} health check failed: {e}")
            self._healthy = False
            return False


class DistributedConfig:
    """Configuration for distributed search."""

    def __init__(
        self,
        min_nodes: int = 2,
        max_nodes: int = 10,
        min_healthy: float = 0.5,
        strategy: str = "round_robin",
        timeout: float = 10.0,
        retry_count: int = 3,
        retry_delay: float = 1.0,
        health_interval: float = 60.0,
    ) -> None:
        """Initialize distributed config.

        Args:
            min_nodes: Minimum number of nodes
            max_nodes: Maximum number of nodes
            min_healthy: Minimum healthy node ratio
            strategy: Load balancing strategy
            timeout: Request timeout in seconds
            retry_count: Number of retries
            retry_delay: Delay between retries in seconds
            health_interval: Health check interval in seconds
        """
        self.min_nodes = min_nodes
        self.max_nodes = max_nodes
        self.min_healthy = min_healthy
        self.strategy = strategy
        self.timeout = timeout
        self.retry_count = retry_count
        self.retry_delay = retry_delay
        self.health_interval = health_interval


class DistributedManager(BaseService[Dict[str, Any]]):
    """Distributed search manager."""

    def __init__(
        self,
        config: Config,
        publisher: PublisherService,
        metrics_manager: MetricsManager,
        distributed_config: Optional[DistributedConfig] = None,
    ) -> None:
        """Initialize distributed manager.

        Args:
            config: Application configuration
            publisher: Event publisher
            metrics_manager: Metrics manager
            distributed_config: Distributed configuration
        """
        super().__init__(config, publisher, metrics_manager)
        self.distributed_config = distributed_config or DistributedConfig()
        self._initialized = False
        self._nodes: Dict[str, SearchNode] = {}
        self._current_node = 0

    async def initialize(self) -> None:
        """Initialize distributed resources."""
        if self._initialized:
            return

        try:
            # Initialize nodes
            for i in range(self.distributed_config.min_nodes):
                node_id = f"node_{i}"
                node_config = NodeConfig(
                    node_id=node_id,
                    host="localhost",
                    port=8000 + i,
                    weight=1.0,
                    max_concurrent=10,
                    timeout=self.distributed_config.timeout,
                    retry_count=self.distributed_config.retry_count,
                    retry_delay=self.distributed_config.retry_delay,
                    health_interval=self.distributed_config.health_interval,
                )
                self._nodes[node_id] = SearchNode(
                    config=node_config,
                    searcher=APISearcher(),
                )

            self._initialized = True
            self.publisher.publish(
                Event(
                    type=EventType.SERVICE_STARTED,
                    timestamp=datetime.now(),
                    component="distributed_manager",
                    description="Distributed manager initialized",
                    metadata={
                        "num_nodes": len(self._nodes),
                        "strategy": self.distributed_config.strategy,
                    },
                )
            )

        except Exception as e:
            logger.error(f"Failed to initialize distributed manager: {e}")
            raise

    async def cleanup(self) -> None:
        """Clean up distributed resources."""
        self._initialized = False
        self._nodes.clear()
        self._current_node = 0

        self.publisher.publish(
            Event(
                type=EventType.SERVICE_STOPPED,
                timestamp=datetime.now(),
                component="distributed_manager",
                description="Distributed manager stopped",
            )
        )

    async def health_check(self) -> Dict[str, Any]:
        """Check distributed manager health.

        Returns:
            Health check results
        """
        # Check node health
        healthy_nodes = 0
        for node in self._nodes.values():
            if await node.health_check():
                healthy_nodes += 1

        # Calculate health ratio
        health_ratio = healthy_nodes / len(self._nodes) if self._nodes else 0.0
        is_healthy = health_ratio >= self.distributed_config.min_healthy

        return {
            "service": "DistributedManager",
            "initialized": self._initialized,
            "num_nodes": len(self._nodes),
            "healthy_nodes": healthy_nodes,
            "health_ratio": health_ratio,
            "status": "healthy" if is_healthy else "unhealthy",
        }

    async def search(
        self,
        query: str,
        **kwargs: Any,
    ) -> List[SearchResult]:
        """Perform distributed search.

        Args:
            query: Search query
            **kwargs: Additional search parameters

        Returns:
            Combined search results
        """
        if not self._initialized:
            await self.initialize()

        try:
            # Get healthy nodes
            healthy_nodes = [
                node for node in self._nodes.values()
                if await node.health_check()
            ]

            if not healthy_nodes:
                logger.error("No healthy nodes available")
                return []

            # Emit search started event
            self.publisher.publish(
                Event(
                    type=EventType.SEARCH_STARTED,
                    timestamp=datetime.now(),
                    component="distributed_manager",
                    description=f"Starting distributed search for query: {query}",
                    metadata={
                        "num_nodes": len(healthy_nodes),
                        "strategy": self.distributed_config.strategy,
                    },
                )
            )

            # Select nodes based on strategy
            if self.distributed_config.strategy == "round_robin":
                # Use round-robin selection
                node = healthy_nodes[self._current_node % len(healthy_nodes)]
                self._current_node = (self._current_node + 1) % len(healthy_nodes)
                selected_nodes = [node]
            else:
                # Use all healthy nodes
                selected_nodes = healthy_nodes

            # Perform distributed search
            tasks = []
            for node in selected_nodes:
                task = asyncio.create_task(node.search(query, **kwargs))
                tasks.append(task)

            # Wait for results with timeout
            results = []
            done, pending = await asyncio.wait(
                tasks,
                timeout=self.distributed_config.timeout,
            )

            # Cancel pending tasks
            for task in pending:
                task.cancel()

            # Combine results
            for task in done:
                try:
                    node_results = await task
                    results.extend(node_results)
                except Exception as e:
                    logger.error(f"Node search failed: {e}")

            # Sort by score
            results.sort(key=lambda x: x.score, reverse=True)

            # Emit success event
            self.publisher.publish(
                Event(
                    type=EventType.SEARCH_COMPLETED,
                    timestamp=datetime.now(),
                    component="distributed_manager",
                    description="Distributed search completed",
                    metadata={
                        "num_results": len(results),
                        "num_nodes": len(selected_nodes),
                    },
                )
            )

            return results

        except Exception as e:
            logger.error(f"Distributed search failed: {e}")
            self.publisher.publish(
                Event(
                    type=EventType.SEARCH_FAILED,
                    timestamp=datetime.now(),
                    component="distributed_manager",
                    description="Distributed search failed",
                    error=str(e),
                )
            )
            return []


# Create service instance
distributed_manager = DistributedManager(
    Config(),
    PublisherService(Config(), MetricsManager()),
    MetricsManager(),
)

__all__ = [
    "NodeConfig",
    "SearchNode",
    "DistributedConfig",
    "DistributedManager",
    "distributed_manager",
] 