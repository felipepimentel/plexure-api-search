"""Distributed cache implementation using Redis."""

import json
import logging
import pickle
import time
from typing import Any, Dict, List, Optional, Set, TypeVar, Union

import redis
from redis.cluster import RedisCluster
from redis.exceptions import ConnectionError, RedisError

from ..config import config_instance
from .compression import CompressedCache, CompressionMethod

logger = logging.getLogger(__name__)

T = TypeVar("T")


class RedisCache:
    """Redis-based distributed cache."""

    def __init__(
        self,
        namespace: str,
        ttl: Optional[int] = None,
        cluster_mode: bool = False,
        compression_enabled: bool = True,
    ):
        """Initialize Redis cache.

        Args:
            namespace: Cache namespace
            ttl: Optional TTL in seconds
            cluster_mode: Whether to use Redis cluster
            compression_enabled: Whether to enable compression
        """
        self.namespace = namespace
        self.ttl = ttl or config_instance.cache_ttl
        self.cluster_mode = cluster_mode
        self.compression_enabled = compression_enabled

        # Initialize Redis client
        try:
            if cluster_mode:
                self.client = RedisCluster(
                    host=config_instance.redis_host,
                    port=config_instance.redis_port,
                    password=config_instance.redis_password,
                    decode_responses=False,
                    skip_full_coverage_check=True,
                )
            else:
                self.client = redis.Redis(
                    host=config_instance.redis_host,
                    port=config_instance.redis_port,
                    password=config_instance.redis_password,
                    decode_responses=False,
                    db=0,
                )

            # Test connection
            self.client.ping()
            logger.info(f"Connected to Redis {'cluster' if cluster_mode else 'server'}")

            # Initialize compression if enabled
            if compression_enabled:
                self.cache = CompressedCache(
                    cache=self,
                    method=CompressionMethod.GZIP,
                    compression_level=6,
                )

        except (ConnectionError, RedisError) as e:
            logger.error(f"Failed to connect to Redis: {e}")
            self.client = None

    def _format_key(self, key: str) -> str:
        """Format cache key with namespace.

        Args:
            key: Original key

        Returns:
            Formatted key
        """
        return f"{self.namespace}:{key}"

    def get(self, key: str) -> Optional[Any]:
        """Get value from cache.

        Args:
            key: Cache key

        Returns:
            Cached value if found
        """
        try:
            if not self.client:
                return None

            # Get from Redis
            value = self.client.get(self._format_key(key))
            if not value:
                return None

            # Deserialize value
            try:
                return pickle.loads(value)
            except Exception:
                return None

        except Exception as e:
            logger.error(f"Failed to get from Redis: {e}")
            return None

    def set(
        self,
        key: str,
        value: Any,
        ttl: Optional[int] = None,
    ) -> bool:
        """Set value in cache.

        Args:
            key: Cache key
            value: Value to cache
            ttl: Optional TTL in seconds

        Returns:
            Whether the operation succeeded
        """
        try:
            if not self.client:
                return False

            # Serialize value
            serialized = pickle.dumps(value)

            # Set in Redis
            formatted_key = self._format_key(key)
            if ttl is not None:
                return bool(self.client.setex(formatted_key, ttl, serialized))
            else:
                return bool(self.client.set(formatted_key, serialized, ex=self.ttl))

        except Exception as e:
            logger.error(f"Failed to set in Redis: {e}")
            return False

    def delete(self, key: str) -> bool:
        """Delete value from cache.

        Args:
            key: Cache key

        Returns:
            Whether the operation succeeded
        """
        try:
            if not self.client:
                return False

            return bool(self.client.delete(self._format_key(key)))

        except Exception as e:
            logger.error(f"Failed to delete from Redis: {e}")
            return False

    def clear(self) -> bool:
        """Clear all values in namespace.

        Returns:
            Whether the operation succeeded
        """
        try:
            if not self.client:
                return False

            # Get all keys in namespace
            pattern = f"{self.namespace}:*"
            keys = self.client.keys(pattern)
            if not keys:
                return True

            # Delete all keys
            return bool(self.client.delete(*keys))

        except Exception as e:
            logger.error(f"Failed to clear Redis namespace: {e}")
            return False

    def get_all_keys(self) -> List[str]:
        """Get all keys in namespace.

        Returns:
            List of cache keys
        """
        try:
            if not self.client:
                return []

            # Get all keys in namespace
            pattern = f"{self.namespace}:*"
            keys = self.client.keys(pattern)
            if not keys:
                return []

            # Remove namespace prefix
            prefix_len = len(self.namespace) + 1
            return [key[prefix_len:].decode() for key in keys]

        except Exception as e:
            logger.error(f"Failed to get Redis keys: {e}")
            return []

    def exists(self, key: str) -> bool:
        """Check if key exists.

        Args:
            key: Cache key

        Returns:
            Whether the key exists
        """
        try:
            if not self.client:
                return False

            return bool(self.client.exists(self._format_key(key)))

        except Exception as e:
            logger.error(f"Failed to check Redis key: {e}")
            return False

    def ttl(self, key: str) -> int:
        """Get TTL for key.

        Args:
            key: Cache key

        Returns:
            TTL in seconds, -1 if no TTL, -2 if key doesn't exist
        """
        try:
            if not self.client:
                return -2

            return self.client.ttl(self._format_key(key))

        except Exception as e:
            logger.error(f"Failed to get Redis TTL: {e}")
            return -2

    def incr(self, key: str) -> Optional[int]:
        """Increment counter.

        Args:
            key: Counter key

        Returns:
            New counter value
        """
        try:
            if not self.client:
                return None

            return self.client.incr(self._format_key(key))

        except Exception as e:
            logger.error(f"Failed to increment Redis counter: {e}")
            return None

    def decr(self, key: str) -> Optional[int]:
        """Decrement counter.

        Args:
            key: Counter key

        Returns:
            New counter value
        """
        try:
            if not self.client:
                return None

            return self.client.decr(self._format_key(key))

        except Exception as e:
            logger.error(f"Failed to decrement Redis counter: {e}")
            return None

    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics.

        Returns:
            Dictionary of cache statistics
        """
        try:
            if not self.client:
                return {}

            # Get Redis info
            info = self.client.info()

            # Get namespace stats
            pattern = f"{self.namespace}:*"
            keys = self.client.keys(pattern)
            num_keys = len(keys)

            # Calculate memory usage
            memory_usage = 0
            for key in keys:
                memory_usage += self.client.memory_usage(key) or 0

            return {
                "total_keys": num_keys,
                "memory_bytes": memory_usage,
                "hits": info.get("keyspace_hits", 0),
                "misses": info.get("keyspace_misses", 0),
                "evicted_keys": info.get("evicted_keys", 0),
                "expired_keys": info.get("expired_keys", 0),
                "connected_clients": info.get("connected_clients", 0),
                "used_memory": info.get("used_memory", 0),
                "used_memory_peak": info.get("used_memory_peak", 0),
                "total_connections_received": info.get("total_connections_received", 0),
                "total_commands_processed": info.get("total_commands_processed", 0),
            }

        except Exception as e:
            logger.error(f"Failed to get Redis stats: {e}")
            return {}

    def health_check(self) -> bool:
        """Check if cache is healthy.

        Returns:
            Whether the cache is healthy
        """
        try:
            if not self.client:
                return False

            # Try to ping Redis
            return bool(self.client.ping())

        except Exception as e:
            logger.error(f"Redis health check failed: {e}")
            return False


class DistributedCache(RedisCache):
    """Distributed cache with additional features."""

    def __init__(
        self,
        namespace: str,
        ttl: Optional[int] = None,
        cluster_mode: bool = False,
        compression_enabled: bool = True,
        max_retries: int = 3,
        retry_delay: float = 0.1,
    ):
        """Initialize distributed cache.

        Args:
            namespace: Cache namespace
            ttl: Optional TTL in seconds
            cluster_mode: Whether to use Redis cluster
            compression_enabled: Whether to enable compression
            max_retries: Maximum number of retries
            retry_delay: Delay between retries in seconds
        """
        super().__init__(namespace, ttl, cluster_mode, compression_enabled)
        self.max_retries = max_retries
        self.retry_delay = retry_delay

    def _retry_operation(self, operation: Any) -> Any:
        """Retry operation with exponential backoff.

        Args:
            operation: Operation to retry

        Returns:
            Operation result
        """
        for attempt in range(self.max_retries):
            try:
                return operation()
            except (ConnectionError, RedisError) as e:
                if attempt == self.max_retries - 1:
                    raise e
                time.sleep(self.retry_delay * (2 ** attempt))

    def get(self, key: str) -> Optional[Any]:
        """Get value from cache with retry.

        Args:
            key: Cache key

        Returns:
            Cached value if found
        """
        return self._retry_operation(lambda: super().get(key))

    def set(
        self,
        key: str,
        value: Any,
        ttl: Optional[int] = None,
    ) -> bool:
        """Set value in cache with retry.

        Args:
            key: Cache key
            value: Value to cache
            ttl: Optional TTL in seconds

        Returns:
            Whether the operation succeeded
        """
        return self._retry_operation(lambda: super().set(key, value, ttl))

    def delete(self, key: str) -> bool:
        """Delete value from cache with retry.

        Args:
            key: Cache key

        Returns:
            Whether the operation succeeded
        """
        return self._retry_operation(lambda: super().delete(key))

    def clear(self) -> bool:
        """Clear all values in namespace with retry.

        Returns:
            Whether the operation succeeded
        """
        return self._retry_operation(lambda: super().clear())

    def get_all_keys(self) -> List[str]:
        """Get all keys in namespace with retry.

        Returns:
            List of cache keys
        """
        return self._retry_operation(lambda: super().get_all_keys())

    def exists(self, key: str) -> bool:
        """Check if key exists with retry.

        Args:
            key: Cache key

        Returns:
            Whether the key exists
        """
        return self._retry_operation(lambda: super().exists(key))

    def ttl(self, key: str) -> int:
        """Get TTL for key with retry.

        Args:
            key: Cache key

        Returns:
            TTL in seconds, -1 if no TTL, -2 if key doesn't exist
        """
        return self._retry_operation(lambda: super().ttl(key))

    def incr(self, key: str) -> Optional[int]:
        """Increment counter with retry.

        Args:
            key: Counter key

        Returns:
            New counter value
        """
        return self._retry_operation(lambda: super().incr(key))

    def decr(self, key: str) -> Optional[int]:
        """Decrement counter with retry.

        Args:
            key: Counter key

        Returns:
            New counter value
        """
        return self._retry_operation(lambda: super().decr(key))

    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics with retry.

        Returns:
            Dictionary of cache statistics
        """
        return self._retry_operation(lambda: super().get_stats())

    def health_check(self) -> bool:
        """Check if cache is healthy with retry.

        Returns:
            Whether the cache is healthy
        """
        return self._retry_operation(lambda: super().health_check()) 