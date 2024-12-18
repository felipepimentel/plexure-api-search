"""Pinecone connection pool implementation."""

import asyncio
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional, Set

import pinecone
from pinecone import Index

from ..config import Config
from ..monitoring.events import Event, EventType
from ..monitoring.metrics import MetricsManager
from ..services.base import BaseService, ServiceException
from ..services.circuit_breaker import CircuitBreaker, CircuitBreakerService
from ..services.events import PublisherService

logger = logging.getLogger(__name__)


class PineconeConnection:
    """Pinecone connection wrapper."""

    def __init__(
        self,
        index: Index,
        created_at: datetime,
        last_used: datetime,
    ) -> None:
        """Initialize connection.

        Args:
            index: Pinecone index
            created_at: Creation timestamp
            last_used: Last usage timestamp
        """
        self.index = index
        self.created_at = created_at
        self.last_used = last_used
        self.in_use = False


class PineconePool(BaseService[Dict[str, Any]]):
    """Pinecone connection pool implementation."""

    def __init__(
        self,
        config: Config,
        publisher: PublisherService,
        metrics_manager: MetricsManager,
        circuit_breaker: CircuitBreakerService,
        min_size: int = 2,
        max_size: int = 10,
        max_idle_time: int = 300,  # 5 minutes
    ) -> None:
        """Initialize connection pool.

        Args:
            config: Application configuration
            publisher: Event publisher
            metrics_manager: Metrics manager
            circuit_breaker: Circuit breaker service
            min_size: Minimum pool size
            max_size: Maximum pool size
            max_idle_time: Maximum idle time in seconds
        """
        super().__init__(config, publisher, metrics_manager)
        self.min_size = min_size
        self.max_size = max_size
        self.max_idle_time = max_idle_time
        self.circuit_breaker = circuit_breaker.get_circuit("pinecone_pool")

        self.connections: List[PineconeConnection] = []
        self.lock = asyncio.Lock()
        self._cleanup_task: Optional[asyncio.Task] = None
        self._initialized = False

    async def initialize(self) -> None:
        """Initialize pool resources."""
        if self._initialized:
            return

        try:
            # Initialize Pinecone
            pinecone.init(
                api_key=self.config.pinecone_api_key,
                environment=self.config.pinecone_environment,
            )

            # Create initial connections
            for _ in range(self.min_size):
                await self._create_connection()

            # Start cleanup task
            self._cleanup_task = asyncio.create_task(self._cleanup_loop())
            self._initialized = True

            self.publisher.publish(
                Event(
                    type=EventType.SERVICE_STARTED,
                    timestamp=datetime.now(),
                    component="pinecone_pool",
                    description="Pinecone connection pool initialized",
                    metadata={
                        "min_size": self.min_size,
                        "max_size": self.max_size,
                    },
                )
            )

        except Exception as e:
            logger.error(f"Failed to initialize Pinecone pool: {e}")
            raise ServiceException(
                message="Failed to initialize Pinecone pool",
                service_name="PineconePool",
                error_code="INIT_FAILED",
                details={"error": str(e)},
            )

    async def cleanup(self) -> None:
        """Clean up pool resources."""
        if self._cleanup_task:
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass

        # Close all connections
        async with self.lock:
            for conn in self.connections:
                if not conn.in_use:
                    await self._close_connection(conn)
            self.connections.clear()

        self._initialized = False
        self.publisher.publish(
            Event(
                type=EventType.SERVICE_STOPPED,
                timestamp=datetime.now(),
                component="pinecone_pool",
                description="Pinecone connection pool stopped",
            )
        )

    async def health_check(self) -> Dict[str, Any]:
        """Check pool health.

        Returns:
            Health check results
        """
        return {
            "service": "PineconePool",
            "initialized": self._initialized,
            "total_connections": len(self.connections),
            "active_connections": sum(1 for c in self.connections if c.in_use),
        }

    async def acquire(self) -> PineconeConnection:
        """Acquire a connection from the pool.

        Returns:
            Pinecone connection

        Raises:
            ServiceException: If no connection available
        """
        if not self._initialized:
            await self.initialize()

        try:
            async with self.lock:
                # Try to find available connection
                for conn in self.connections:
                    if not conn.in_use:
                        conn.in_use = True
                        conn.last_used = datetime.now()
                        return conn

                # Create new connection if possible
                if len(self.connections) < self.max_size:
                    conn = await self._create_connection()
                    conn.in_use = True
                    return conn

                # No connections available
                raise ServiceException(
                    message="No connections available",
                    service_name="PineconePool",
                    error_code="NO_CONNECTIONS",
                )

        except Exception as e:
            logger.error(f"Failed to acquire connection: {e}")
            self.metrics.increment("connection_errors", 1)
            raise ServiceException(
                message="Failed to acquire connection",
                service_name="PineconePool",
                error_code="ACQUIRE_FAILED",
                details={"error": str(e)},
            )

    async def release(self, conn: PineconeConnection) -> None:
        """Release a connection back to the pool.

        Args:
            conn: Connection to release
        """
        try:
            async with self.lock:
                if conn in self.connections:
                    conn.in_use = False
                    conn.last_used = datetime.now()

        except Exception as e:
            logger.error(f"Failed to release connection: {e}")
            self.metrics.increment("connection_errors", 1)

    async def _create_connection(self) -> PineconeConnection:
        """Create a new connection.

        Returns:
            New connection

        Raises:
            ServiceException: If connection creation fails
        """
        try:
            # Create connection with circuit breaker
            async def create() -> PineconeConnection:
                index = pinecone.Index(self.config.pinecone_index_name)
                now = datetime.now()
                return PineconeConnection(
                    index=index,
                    created_at=now,
                    last_used=now,
                )

            conn = await self.circuit_breaker.call(create)
            self.connections.append(conn)
            self.metrics.increment("connections_created", 1)
            return conn

        except Exception as e:
            logger.error(f"Failed to create connection: {e}")
            self.metrics.increment("connection_errors", 1)
            raise ServiceException(
                message="Failed to create connection",
                service_name="PineconePool",
                error_code="CREATE_FAILED",
                details={"error": str(e)},
            )

    async def _close_connection(self, conn: PineconeConnection) -> None:
        """Close a connection.

        Args:
            conn: Connection to close
        """
        try:
            if conn in self.connections:
                self.connections.remove(conn)
                self.metrics.increment("connections_closed", 1)

        except Exception as e:
            logger.error(f"Failed to close connection: {e}")
            self.metrics.increment("connection_errors", 1)

    async def _cleanup_loop(self) -> None:
        """Cleanup idle connections."""
        while True:
            try:
                async with self.lock:
                    now = datetime.now()
                    to_close = []

                    # Find idle connections
                    for conn in self.connections:
                        if not conn.in_use:
                            idle_time = (now - conn.last_used).total_seconds()
                            if idle_time > self.max_idle_time:
                                to_close.append(conn)

                    # Close idle connections if above min_size
                    if len(self.connections) - len(to_close) >= self.min_size:
                        for conn in to_close:
                            await self._close_connection(conn)

                    # Update metrics
                    self.metrics.observe(
                        "total_connections",
                        len(self.connections),
                    )
                    self.metrics.observe(
                        "active_connections",
                        sum(1 for c in self.connections if c.in_use),
                    )

                await asyncio.sleep(60)  # Check every minute

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Connection cleanup failed: {e}")
                await asyncio.sleep(60)  # Retry after error


# Create service instance
pinecone_pool = PineconePool(
    Config(),
    PublisherService(Config(), MetricsManager()),
    MetricsManager(),
    CircuitBreakerService(Config(), PublisherService(Config(), MetricsManager()), MetricsManager()),
)

__all__ = ["PineconeConnection", "PineconePool", "pinecone_pool"] 