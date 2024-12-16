"""Hierarchical caching system combining memory, disk and Redis."""

import logging
from typing import Any, Dict, Optional, TypeVar

from ..config import config_instance
from .cache import BaseCache, DiskCache, RedisCache, RedisUnavailable

logger = logging.getLogger(__name__)

T = TypeVar("T")

class HierarchicalCache(BaseCache[T]):
    """Hierarchical cache implementation combining memory, disk and Redis caches."""

    def __init__(
        self,
        namespace: str,
        memory_ttl: Optional[int] = None,
        disk_ttl: Optional[int] = None,
        redis_ttl: Optional[int] = None,
    ):
        """Initialize hierarchical cache.

        Args:
            namespace: Cache namespace
            memory_ttl: Optional memory cache TTL (defaults to config)
            disk_ttl: Optional disk cache TTL (defaults to config)
            redis_ttl: Optional Redis cache TTL (defaults to config)
        """
        self.namespace = namespace
        
        # Initialize caches based on configuration
        self.caches: Dict[str, BaseCache[T]] = {}
        
        if config_instance.memory_cache_enabled:
            from .memory_cache import MemoryCache
            self.caches["memory"] = MemoryCache(
                namespace=namespace,
                ttl=memory_ttl or config_instance.memory_cache_ttl,
                max_size=config_instance.memory_cache_max_size,
            )
            
        if config_instance.disk_cache_enabled:
            self.caches["disk"] = DiskCache(
                namespace=namespace,
                ttl=disk_ttl or config_instance.disk_cache_ttl,
            )
            
        if config_instance.redis_enabled:
            try:
                self.caches["redis"] = RedisCache(
                    namespace=namespace,
                    ttl=redis_ttl or config_instance.redis_cache_ttl,
                )
            except RedisUnavailable:
                logger.warning(
                    "Redis cache was enabled but Redis package is not installed. "
                    "Continuing without Redis cache. "
                    "Install Redis with: pip install redis"
                )
            except Exception as e:
                logger.warning(f"Failed to initialize Redis cache: {e}")

        if not self.caches:
            logger.warning(
                "No caches were enabled. Defaulting to disk cache."
            )
            self.caches["disk"] = DiskCache(
                namespace=namespace,
                ttl=disk_ttl or config_instance.disk_cache_ttl,
            )

    def get(self, key: str) -> Optional[T]:
        """Get value from cache hierarchy.

        Checks caches in order: memory -> disk -> redis
        On cache miss in faster tier, populates it from slower tier

        Args:
            key: Cache key

        Returns:
            Cached value or None if not found
        """
        # Check each cache tier
        for cache_name, cache in self.caches.items():
            value = cache.get(key)
            if value is not None:
                # Found in current tier, populate faster tiers
                self._populate_faster_tiers(cache_name, key, value)
                return value
                
        return None

    def set(self, key: str, value: T, ttl: Optional[int] = None) -> None:
        """Set value in all cache tiers.

        Args:
            key: Cache key
            value: Value to cache
            ttl: Optional TTL in seconds (overrides defaults)
        """
        for cache in self.caches.values():
            cache.set(key, value, ttl)

    def delete(self, key: str) -> None:
        """Delete value from all cache tiers.

        Args:
            key: Cache key
        """
        for cache in self.caches.values():
            cache.delete(key)

    def clear(self) -> None:
        """Clear all cache tiers."""
        for cache in self.caches.values():
            cache.clear()

    def _populate_faster_tiers(self, found_tier: str, key: str, value: T) -> None:
        """Populate faster cache tiers after a hit in a slower tier.

        Args:
            found_tier: Name of tier where value was found
            key: Cache key
            value: Value to populate
        """
        # Order of tiers from fastest to slowest
        tier_order = ["memory", "disk", "redis"]
        
        # Find tiers faster than where value was found
        found_index = tier_order.index(found_tier)
        faster_tiers = tier_order[:found_index]
        
        # Populate faster tiers
        for tier in faster_tiers:
            if tier in self.caches:
                self.caches[tier].set(key, value) 