"""Advanced caching system with disk and Redis support."""

import json
import logging
import os
import pickle
import time
from pathlib import Path
from typing import Any, Dict, Generic, Optional, TypeVar, Union

from ..config import config_instance

logger = logging.getLogger(__name__)

T = TypeVar("T")

class RedisUnavailable(Exception):
    """Raised when Redis is not available but was requested."""
    pass

# Conditional Redis import
try:
    import redis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False
    redis = None  # type: ignore


class BaseCache(Generic[T]):
    """Base cache interface."""

    def get(self, key: str) -> Optional[T]:
        """Get value from cache.

        Args:
            key: Cache key

        Returns:
            Cached value or None if not found
        """
        raise NotImplementedError

    def set(self, key: str, value: T, ttl: Optional[int] = None) -> None:
        """Set value in cache.

        Args:
            key: Cache key
            value: Value to cache
            ttl: Optional TTL in seconds
        """
        raise NotImplementedError

    def delete(self, key: str) -> None:
        """Delete value from cache.

        Args:
            key: Cache key
        """
        raise NotImplementedError

    def clear(self) -> None:
        """Clear all cached values."""
        raise NotImplementedError


class DiskCache(BaseCache[T]):
    """Disk-based cache implementation."""

    def __init__(
        self,
        namespace: str,
        ttl: int = 3600,
        cache_dir: Optional[Path] = None,
    ):
        """Initialize disk cache.

        Args:
            namespace: Cache namespace
            ttl: Cache TTL in seconds
            cache_dir: Optional cache directory
        """
        self.namespace = namespace
        self.ttl = ttl
        self.cache_dir = cache_dir or config_instance.cache_dir / namespace
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _get_path(self, key: str) -> Path:
        """Get cache file path.

        Args:
            key: Cache key

        Returns:
            Cache file path
        """
        # Use hash to avoid file system issues with long keys
        hashed_key = str(hash(key))
        return self.cache_dir / f"{hashed_key}.cache"

    def _is_valid(self, path: Path) -> bool:
        """Check if cache entry is still valid.

        Args:
            path: Cache file path

        Returns:
            True if valid, False otherwise
        """
        try:
            if not path.exists():
                return False
            
            # Check TTL
            mtime = path.stat().st_mtime
            age = time.time() - mtime
            return age < self.ttl

        except Exception as e:
            logger.error(f"Error checking cache validity: {e}")
            return False

    def get(self, key: str) -> Optional[T]:
        """Get value from cache.

        Args:
            key: Cache key

        Returns:
            Cached value or None if not found
        """
        try:
            path = self._get_path(key)
            
            # Check validity
            if not self._is_valid(path):
                if path.exists():
                    path.unlink()
                return None

            # Load value
            with path.open("rb") as f:
                return pickle.load(f)

        except Exception as e:
            logger.error(f"Error reading from cache: {e}")
            return None

    def set(self, key: str, value: T, ttl: Optional[int] = None) -> None:
        """Set value in cache.

        Args:
            key: Cache key
            value: Value to cache
            ttl: Optional TTL in seconds (overrides default)
        """
        try:
            path = self._get_path(key)
            
            # Save value
            with path.open("wb") as f:
                pickle.dump(value, f)

            # Update modification time for TTL
            os.utime(path, None)

        except Exception as e:
            logger.error(f"Error writing to cache: {e}")

    def delete(self, key: str) -> None:
        """Delete value from cache.

        Args:
            key: Cache key
        """
        try:
            path = self._get_path(key)
            if path.exists():
                path.unlink()

        except Exception as e:
            logger.error(f"Error deleting from cache: {e}")

    def clear(self) -> None:
        """Clear all cached values."""
        try:
            for path in self.cache_dir.glob("*.cache"):
                path.unlink()

        except Exception as e:
            logger.error(f"Error clearing cache: {e}")


class RedisCache(BaseCache[T]):
    """Redis-based cache implementation."""

    def __init__(
        self,
        namespace: str,
        ttl: int = 3600,
        redis_url: Optional[str] = None,
    ):
        """Initialize Redis cache.

        Args:
            namespace: Cache namespace
            ttl: Cache TTL in seconds
            redis_url: Optional Redis URL (defaults to config)

        Raises:
            RedisUnavailable: If Redis package is not installed
        """
        if not REDIS_AVAILABLE:
            raise RedisUnavailable(
                "Redis package is not installed. Please install it with: pip install redis"
            )

        self.namespace = namespace
        self.ttl = ttl
        
        # Connect to Redis
        redis_url = redis_url or config_instance.redis_url
        if not redis_url:
            raise ValueError("Redis URL is required")
            
        self.redis = redis.from_url(redis_url)
        logger.info(f"Connected to Redis at {redis_url}")

    def _get_key(self, key: str) -> str:
        """Get namespaced cache key.

        Args:
            key: Cache key

        Returns:
            Namespaced key
        """
        return f"{self.namespace}:{key}"

    def get(self, key: str) -> Optional[T]:
        """Get value from cache.

        Args:
            key: Cache key

        Returns:
            Cached value or None if not found
        """
        try:
            value = self.redis.get(self._get_key(key))
            if value is None:
                return None
                
            return pickle.loads(value)

        except Exception as e:
            logger.error(f"Error reading from Redis: {e}")
            return None

    def set(self, key: str, value: T, ttl: Optional[int] = None) -> None:
        """Set value in cache.

        Args:
            key: Cache key
            value: Value to cache
            ttl: Optional TTL in seconds (overrides default)
        """
        try:
            # Serialize value
            serialized = pickle.dumps(value)
            
            # Set in Redis with TTL
            self.redis.setex(
                self._get_key(key),
                ttl or self.ttl,
                serialized,
            )

        except Exception as e:
            logger.error(f"Error writing to Redis: {e}")

    def delete(self, key: str) -> None:
        """Delete value from cache.

        Args:
            key: Cache key
        """
        try:
            self.redis.delete(self._get_key(key))

        except Exception as e:
            logger.error(f"Error deleting from Redis: {e}")

    def clear(self) -> None:
        """Clear all cached values in namespace."""
        try:
            # Get all keys in namespace
            pattern = f"{self.namespace}:*"
            keys = self.redis.keys(pattern)
            
            # Delete all keys
            if keys:
                self.redis.delete(*keys)

        except Exception as e:
            logger.error(f"Error clearing Redis cache: {e}")


def get_cache(
    namespace: str,
    ttl: int = 3600,
    backend: Optional[str] = None,
) -> BaseCache[T]:
    """Get cache instance based on configuration.

    Args:
        namespace: Cache namespace
        ttl: Cache TTL in seconds
        backend: Optional backend override

    Returns:
        Cache instance
    """
    backend = backend or config_instance.cache_backend
    
    if backend == "redis" and config_instance.redis_url:
        return RedisCache(
            namespace=namespace,
            ttl=ttl,
            redis_url=config_instance.redis_url,
        )
    else:
        return DiskCache(
            namespace=namespace,
            ttl=ttl,
        )


__all__ = ["BaseCache", "DiskCache", "RedisCache", "get_cache"]
