"""Result caching for improved search performance."""

import logging
from typing import Dict, List, Any, Optional, Set
from datetime import datetime
import hashlib
import json
import asyncio
from collections import OrderedDict

from ..config import Config
from ..monitoring.events import Event, EventType
from ..monitoring.metrics import MetricsManager
from ..services.base import BaseService
from ..services.events import PublisherService

logger = logging.getLogger(__name__)


class CacheConfig:
    """Configuration for result caching."""

    def __init__(
        self,
        max_size: int = 10000,
        ttl: int = 3600,  # 1 hour
        min_hits: int = 2,
        max_age: int = 86400,  # 24 hours
        enable_query_cache: bool = True,
        enable_vector_cache: bool = True,
        enable_result_cache: bool = True,
        cache_dir: str = "cache",
        persist_cache: bool = True,
        compression: bool = True,
    ) -> None:
        """Initialize cache config.

        Args:
            max_size: Maximum number of cache entries
            ttl: Cache time-to-live in seconds
            min_hits: Minimum hits for caching
            max_age: Maximum age of cache entries in seconds
            enable_query_cache: Whether to cache query results
            enable_vector_cache: Whether to cache vector embeddings
            enable_result_cache: Whether to cache search results
            cache_dir: Directory for persistent cache
            persist_cache: Whether to persist cache to disk
            compression: Whether to compress cache entries
        """
        self.max_size = max_size
        self.ttl = ttl
        self.min_hits = min_hits
        self.max_age = max_age
        self.enable_query_cache = enable_query_cache
        self.enable_vector_cache = enable_vector_cache
        self.enable_result_cache = enable_result_cache
        self.cache_dir = cache_dir
        self.persist_cache = persist_cache
        self.compression = compression


class CacheEntry:
    """Cache entry with metadata."""

    def __init__(
        self,
        key: str,
        value: Any,
        timestamp: datetime,
        ttl: int,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Initialize cache entry.

        Args:
            key: Cache key
            value: Cached value
            timestamp: Entry timestamp
            ttl: Time-to-live in seconds
            metadata: Additional metadata
        """
        self.key = key
        self.value = value
        self.timestamp = timestamp
        self.ttl = ttl
        self.metadata = metadata or {}
        self.hits = 0
        self.last_access = timestamp

    def is_expired(self) -> bool:
        """Check if entry is expired.

        Returns:
            True if expired, False otherwise
        """
        now = datetime.now()
        age = (now - self.timestamp).total_seconds()
        return age > self.ttl

    def update_stats(self) -> None:
        """Update entry statistics."""
        self.hits += 1
        self.last_access = datetime.now()


class ResultCache(BaseService[Dict[str, Any]]):
    """Cache manager for search results."""

    def __init__(
        self,
        config: Config,
        publisher: PublisherService,
        metrics_manager: MetricsManager,
        cache_config: Optional[CacheConfig] = None,
    ) -> None:
        """Initialize cache manager.

        Args:
            config: Application configuration
            publisher: Event publisher
            metrics_manager: Metrics manager
            cache_config: Cache configuration
        """
        super().__init__(config, publisher, metrics_manager)
        self.cache_config = cache_config or CacheConfig()
        self._initialized = False
        self._query_cache: OrderedDict[str, CacheEntry] = OrderedDict()
        self._vector_cache: OrderedDict[str, CacheEntry] = OrderedDict()
        self._result_cache: OrderedDict[str, CacheEntry] = OrderedDict()
        self._cleanup_task: Optional[asyncio.Task] = None

    async def initialize(self) -> None:
        """Initialize cache resources."""
        if self._initialized:
            return

        try:
            # Load persistent cache if enabled
            if self.cache_config.persist_cache:
                await self._load_cache()

            # Start cleanup task
            self._cleanup_task = asyncio.create_task(self._cleanup_loop())

            self._initialized = True
            self.publisher.publish(
                Event(
                    type=EventType.SERVICE_STARTED,
                    timestamp=datetime.now(),
                    component="result_cache",
                    description="Result cache initialized",
                    metadata={
                        "query_cache_enabled": self.cache_config.enable_query_cache,
                        "vector_cache_enabled": self.cache_config.enable_vector_cache,
                        "result_cache_enabled": self.cache_config.enable_result_cache,
                    },
                )
            )

        except Exception as e:
            logger.error(f"Failed to initialize result cache: {e}")
            raise

    async def cleanup(self) -> None:
        """Clean up cache resources."""
        if self._cleanup_task:
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass

        # Save persistent cache if enabled
        if self.cache_config.persist_cache:
            await self._save_cache()

        self._initialized = False
        self._query_cache.clear()
        self._vector_cache.clear()
        self._result_cache.clear()

        self.publisher.publish(
            Event(
                type=EventType.SERVICE_STOPPED,
                timestamp=datetime.now(),
                component="result_cache",
                description="Result cache stopped",
            )
        )

    async def health_check(self) -> Dict[str, Any]:
        """Check cache health.

        Returns:
            Health check results
        """
        return {
            "service": "ResultCache",
            "initialized": self._initialized,
            "query_cache_size": len(self._query_cache),
            "vector_cache_size": len(self._vector_cache),
            "result_cache_size": len(self._result_cache),
            "status": "healthy" if self._initialized else "unhealthy",
        }

    async def get_query_cache(
        self,
        query: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> Optional[Any]:
        """Get cached query results.

        Args:
            query: Search query
            context: Query context

        Returns:
            Cached results if found, None otherwise
        """
        if not self._initialized or not self.cache_config.enable_query_cache:
            return None

        try:
            # Generate cache key
            key = self._generate_key(query, context)

            # Get cache entry
            entry = self._query_cache.get(key)
            if not entry or entry.is_expired():
                return None

            # Update stats
            entry.update_stats()
            self._query_cache.move_to_end(key)

            return entry.value

        except Exception as e:
            logger.error(f"Query cache lookup failed: {e}")
            return None

    async def set_query_cache(
        self,
        query: str,
        value: Any,
        context: Optional[Dict[str, Any]] = None,
        ttl: Optional[int] = None,
    ) -> None:
        """Set query cache entry.

        Args:
            query: Search query
            value: Results to cache
            context: Query context
            ttl: Optional TTL override
        """
        if not self._initialized or not self.cache_config.enable_query_cache:
            return

        try:
            # Generate cache key
            key = self._generate_key(query, context)

            # Create cache entry
            entry = CacheEntry(
                key=key,
                value=value,
                timestamp=datetime.now(),
                ttl=ttl or self.cache_config.ttl,
                metadata={"query": query, "context": context},
            )

            # Add to cache
            self._query_cache[key] = entry
            self._query_cache.move_to_end(key)

            # Enforce size limit
            while len(self._query_cache) > self.cache_config.max_size:
                self._query_cache.popitem(last=False)

        except Exception as e:
            logger.error(f"Query cache update failed: {e}")

    async def get_vector_cache(
        self,
        text: str,
        model_name: str,
    ) -> Optional[Any]:
        """Get cached vector embedding.

        Args:
            text: Text to embed
            model_name: Model name

        Returns:
            Cached vector if found, None otherwise
        """
        if not self._initialized or not self.cache_config.enable_vector_cache:
            return None

        try:
            # Generate cache key
            key = self._generate_key(text, {"model": model_name})

            # Get cache entry
            entry = self._vector_cache.get(key)
            if not entry or entry.is_expired():
                return None

            # Update stats
            entry.update_stats()
            self._vector_cache.move_to_end(key)

            return entry.value

        except Exception as e:
            logger.error(f"Vector cache lookup failed: {e}")
            return None

    async def set_vector_cache(
        self,
        text: str,
        vector: Any,
        model_name: str,
        ttl: Optional[int] = None,
    ) -> None:
        """Set vector cache entry.

        Args:
            text: Text to embed
            vector: Vector embedding
            model_name: Model name
            ttl: Optional TTL override
        """
        if not self._initialized or not self.cache_config.enable_vector_cache:
            return

        try:
            # Generate cache key
            key = self._generate_key(text, {"model": model_name})

            # Create cache entry
            entry = CacheEntry(
                key=key,
                value=vector,
                timestamp=datetime.now(),
                ttl=ttl or self.cache_config.ttl,
                metadata={"text": text, "model": model_name},
            )

            # Add to cache
            self._vector_cache[key] = entry
            self._vector_cache.move_to_end(key)

            # Enforce size limit
            while len(self._vector_cache) > self.cache_config.max_size:
                self._vector_cache.popitem(last=False)

        except Exception as e:
            logger.error(f"Vector cache update failed: {e}")

    async def get_result_cache(
        self,
        query: str,
        filters: Optional[Dict[str, Any]] = None,
    ) -> Optional[Any]:
        """Get cached search results.

        Args:
            query: Search query
            filters: Search filters

        Returns:
            Cached results if found, None otherwise
        """
        if not self._initialized or not self.cache_config.enable_result_cache:
            return None

        try:
            # Generate cache key
            key = self._generate_key(query, {"filters": filters})

            # Get cache entry
            entry = self._result_cache.get(key)
            if not entry or entry.is_expired():
                return None

            # Update stats
            entry.update_stats()
            self._result_cache.move_to_end(key)

            return entry.value

        except Exception as e:
            logger.error(f"Result cache lookup failed: {e}")
            return None

    async def set_result_cache(
        self,
        query: str,
        results: Any,
        filters: Optional[Dict[str, Any]] = None,
        ttl: Optional[int] = None,
    ) -> None:
        """Set result cache entry.

        Args:
            query: Search query
            results: Search results
            filters: Search filters
            ttl: Optional TTL override
        """
        if not self._initialized or not self.cache_config.enable_result_cache:
            return

        try:
            # Generate cache key
            key = self._generate_key(query, {"filters": filters})

            # Create cache entry
            entry = CacheEntry(
                key=key,
                value=results,
                timestamp=datetime.now(),
                ttl=ttl or self.cache_config.ttl,
                metadata={"query": query, "filters": filters},
            )

            # Add to cache
            self._result_cache[key] = entry
            self._result_cache.move_to_end(key)

            # Enforce size limit
            while len(self._result_cache) > self.cache_config.max_size:
                self._result_cache.popitem(last=False)

        except Exception as e:
            logger.error(f"Result cache update failed: {e}")

    def _generate_key(
        self,
        value: Any,
        context: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Generate cache key.

        Args:
            value: Value to generate key for
            context: Additional context

        Returns:
            Cache key
        """
        # Convert value and context to string
        key_parts = [str(value)]
        if context:
            key_parts.append(json.dumps(context, sort_keys=True))

        # Generate hash
        key = "_".join(key_parts)
        return hashlib.md5(key.encode()).hexdigest()

    async def _cleanup_loop(self) -> None:
        """Background task for cache cleanup."""
        while True:
            try:
                # Sleep for cleanup interval
                await asyncio.sleep(60)  # Check every minute

                # Remove expired entries
                now = datetime.now()
                for cache in [self._query_cache, self._vector_cache, self._result_cache]:
                    expired = []
                    for key, entry in cache.items():
                        if entry.is_expired():
                            expired.append(key)
                    for key in expired:
                        del cache[key]

                # Save persistent cache if enabled
                if self.cache_config.persist_cache:
                    await self._save_cache()

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Cache cleanup failed: {e}")

    async def _load_cache(self) -> None:
        """Load persistent cache from disk."""
        # TODO: Implement cache persistence
        pass

    async def _save_cache(self) -> None:
        """Save persistent cache to disk."""
        # TODO: Implement cache persistence
        pass


# Create service instance
result_cache = ResultCache(
    Config(),
    PublisherService(Config(), MetricsManager()),
    MetricsManager(),
)

__all__ = [
    "CacheConfig",
    "CacheEntry",
    "ResultCache",
    "result_cache",
] 