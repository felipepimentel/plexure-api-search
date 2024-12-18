"""Vector store service implementation."""

import logging
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
from dependency_injector.wiring import inject, provider

from ..config import Config
from ..monitoring.events import Event, EventType
from ..monitoring.metrics import MetricsManager
from .base import BaseService, ServiceException
from .events import PublisherService

logger = logging.getLogger(__name__)


class VectorStoreTransaction:
    """Vector store transaction context manager."""

    def __init__(self, store: "BaseVectorStore") -> None:
        """Initialize transaction.

        Args:
            store: Vector store instance
        """
        self.store = store
        self.changes: List[Dict[str, Any]] = []
        self.committed = False

    def __enter__(self) -> "VectorStoreTransaction":
        """Enter transaction context."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Exit transaction context."""
        if exc_type is None and not self.committed:
            self.commit()
        elif exc_type is not None:
            self.rollback()

    def add(self, operation: Dict[str, Any]) -> None:
        """Add operation to transaction.

        Args:
            operation: Operation details
        """
        self.changes.append(operation)

    def commit(self) -> None:
        """Commit transaction changes."""
        if not self.committed:
            self.store._commit_transaction(self)
            self.committed = True

    def rollback(self) -> None:
        """Rollback transaction changes."""
        if not self.committed:
            self.store._rollback_transaction(self)
            self.changes = []


class BaseVectorStore(ABC):
    """Base vector store interface."""

    @abstractmethod
    def begin_transaction(self) -> VectorStoreTransaction:
        """Begin a new transaction.

        Returns:
            Transaction context manager
        """
        pass

    @abstractmethod
    def upsert(
        self,
        vectors: List[np.ndarray],
        metadata: List[Dict[str, Any]],
        ids: Optional[List[str]] = None,
    ) -> None:
        """Upsert vectors to store.

        Args:
            vectors: Vector data
            metadata: Vector metadata
            ids: Optional vector IDs
        """
        pass

    @abstractmethod
    def search(
        self,
        query_vector: np.ndarray,
        top_k: int = 10,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """Search for similar vectors.

        Args:
            query_vector: Query vector
            top_k: Number of results
            filters: Optional filters

        Returns:
            Search results
        """
        pass

    @abstractmethod
    def delete(self, ids: List[str]) -> None:
        """Delete vectors by ID.

        Args:
            ids: Vector IDs to delete
        """
        pass

    @abstractmethod
    def _commit_transaction(self, transaction: VectorStoreTransaction) -> None:
        """Commit transaction changes.

        Args:
            transaction: Transaction to commit
        """
        pass

    @abstractmethod
    def _rollback_transaction(self, transaction: VectorStoreTransaction) -> None:
        """Rollback transaction changes.

        Args:
            transaction: Transaction to rollback
        """
        pass


class VectorStoreService(BaseService[Dict[str, Any]]):
    """Vector store service implementation."""

    def __init__(
        self,
        config: Config,
        publisher: PublisherService,
        metrics_manager: MetricsManager,
        store: BaseVectorStore,
    ) -> None:
        """Initialize vector store service.

        Args:
            config: Application configuration
            publisher: Event publisher
            metrics_manager: Metrics manager
            store: Vector store implementation
        """
        super().__init__(config, publisher, metrics_manager)
        self.store = store
        self._initialized = False

    async def initialize(self) -> None:
        """Initialize service resources."""
        if self._initialized:
            return

        try:
            # Initialize store
            self._initialized = True
            self.publisher.publish(
                Event(
                    type=EventType.SERVICE_STARTED,
                    timestamp=datetime.now(),
                    component="vector_store",
                    description="Vector store service initialized",
                )
            )

        except Exception as e:
            logger.error(f"Failed to initialize vector store: {e}")
            raise ServiceException(
                message="Failed to initialize vector store",
                service_name="VectorStore",
                error_code="INIT_FAILED",
                details={"error": str(e)},
            )

    async def cleanup(self) -> None:
        """Clean up service resources."""
        self._initialized = False
        self.publisher.publish(
            Event(
                type=EventType.SERVICE_STOPPED,
                timestamp=datetime.now(),
                component="vector_store",
                description="Vector store service stopped",
            )
        )

    async def health_check(self) -> Dict[str, Any]:
        """Check service health.

        Returns:
            Health check results
        """
        return {
            "service": "VectorStore",
            "initialized": self._initialized,
            "store_type": self.store.__class__.__name__,
        }

    def begin_transaction(self) -> VectorStoreTransaction:
        """Begin a new transaction.

        Returns:
            Transaction context manager
        """
        return self.store.begin_transaction()

    async def upsert(
        self,
        vectors: List[np.ndarray],
        metadata: List[Dict[str, Any]],
        ids: Optional[List[str]] = None,
    ) -> None:
        """Upsert vectors to store.

        Args:
            vectors: Vector data
            metadata: Vector metadata
            ids: Optional vector IDs
        """
        if not self._initialized:
            await self.initialize()

        try:
            with self.store.begin_transaction() as transaction:
                self.store.upsert(vectors, metadata, ids)
                self.metrics.increment(
                    "vectors_upserted",
                    len(vectors),
                )
                self.publisher.publish(
                    Event(
                        type=EventType.VECTORS_UPSERTED,
                        timestamp=datetime.now(),
                        component="vector_store",
                        description=f"Upserted {len(vectors)} vectors",
                        metadata={"count": len(vectors)},
                    )
                )

        except Exception as e:
            logger.error(f"Failed to upsert vectors: {e}")
            self.metrics.increment("vector_errors", 1)
            raise ServiceException(
                message="Failed to upsert vectors",
                service_name="VectorStore",
                error_code="UPSERT_FAILED",
                details={"error": str(e)},
            )

    async def search(
        self,
        query_vector: np.ndarray,
        top_k: int = 10,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """Search for similar vectors.

        Args:
            query_vector: Query vector
            top_k: Number of results
            filters: Optional filters

        Returns:
            Search results
        """
        if not self._initialized:
            await self.initialize()

        try:
            results = self.store.search(query_vector, top_k, filters)
            self.metrics.increment("searches_performed", 1)
            self.metrics.observe(
                "search_results",
                len(results),
            )
            return results

        except Exception as e:
            logger.error(f"Failed to search vectors: {e}")
            self.metrics.increment("search_errors", 1)
            raise ServiceException(
                message="Failed to search vectors",
                service_name="VectorStore",
                error_code="SEARCH_FAILED",
                details={"error": str(e)},
            )

    async def delete(self, ids: List[str]) -> None:
        """Delete vectors by ID.

        Args:
            ids: Vector IDs to delete
        """
        if not self._initialized:
            await self.initialize()

        try:
            with self.store.begin_transaction() as transaction:
                self.store.delete(ids)
                self.metrics.increment(
                    "vectors_deleted",
                    len(ids),
                )
                self.publisher.publish(
                    Event(
                        type=EventType.VECTORS_DELETED,
                        timestamp=datetime.now(),
                        component="vector_store",
                        description=f"Deleted {len(ids)} vectors",
                        metadata={"count": len(ids)},
                    )
                )

        except Exception as e:
            logger.error(f"Failed to delete vectors: {e}")
            self.metrics.increment("vector_errors", 1)
            raise ServiceException(
                message="Failed to delete vectors",
                service_name="VectorStore",
                error_code="DELETE_FAILED",
                details={"error": str(e)},
            )


__all__ = [
    "BaseVectorStore",
    "VectorStoreService",
    "VectorStoreTransaction",
] 