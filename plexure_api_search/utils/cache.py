"""Generic disk-based cache implementation."""

import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Generic, Optional, TypeVar

import numpy as np
from diskcache import Cache

from ..config import config_instance

logger = logging.getLogger(__name__)

T = TypeVar("T")


class DiskCache(Generic[T]):
    """Generic disk-based cache implementation."""

    def __init__(
        self,
        namespace: str,
        ttl: int,
        cache_dir: Optional[str] = None,
    ):
        """Initialize cache.

        Args:
            namespace: Cache namespace (e.g. 'embeddings', 'search')
            ttl: Time to live in seconds
            cache_dir: Optional custom cache directory. If not provided, uses config_instance.cache_dir
        """
        self.namespace = namespace
        self.ttl = ttl

        base_dir = cache_dir or config_instance.cache_dir
        self.cache_dir = Path(f"{base_dir}/{namespace}")
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # Initialize disk cache
        self.cache = Cache(str(self.cache_dir))

    def get(self, key: str) -> Optional[T]:
        """Get cached value.

        Args:
            key: Cache key

        Returns:
            Cached value if found and valid, None otherwise
        """
        cache_key = self._get_cache_key(key)
        cached = self.cache.get(cache_key)

        if cached is None:
            return None

        # Check if expired
        if (
            datetime.fromisoformat(cached["timestamp"]) + timedelta(seconds=self.ttl)
            < datetime.now()
        ):
            self.cache.delete(cache_key)
            return None

        return self._deserialize_value(cached["value"])

    def set(self, key: str, value: T) -> None:
        """Store value in cache.

        Args:
            key: Cache key
            value: Value to cache
        """
        cache_key = self._get_cache_key(key)
        self.cache.set(
            cache_key,
            {
                "timestamp": datetime.now().isoformat(),
                "value": self._serialize_value(value),
            },
        )

    def delete(self, key: str) -> None:
        """Delete a value from cache.

        Args:
            key: Cache key
        """
        cache_key = self._get_cache_key(key)
        self.cache.delete(cache_key)

    def clear(self) -> None:
        """Clear all cached values."""
        self.cache.clear()

    def _get_cache_key(self, key: str) -> str:
        """Generate cache key.

        Args:
            key: Original key

        Returns:
            Namespaced cache key
        """
        return f"{self.namespace}:{key.lower().strip()}"

    def _serialize_value(self, value: T) -> Any:
        """Serialize value for storage.

        Handles numpy arrays and other special types.

        Args:
            value: Value to serialize

        Returns:
            Serialized value
        """
        if isinstance(value, np.ndarray):
            return {
                "__type__": "ndarray",
                "data": value.tolist(),
                "dtype": str(value.dtype),
            }
        elif isinstance(value, dict):
            return {key: self._serialize_value(val) for key, val in value.items()}
        elif isinstance(value, list):
            return [self._serialize_value(item) for item in value]
        return value

    def _deserialize_value(self, value: Any) -> T:
        """Deserialize value from storage.

        Handles numpy arrays and other special types.

        Args:
            value: Value to deserialize

        Returns:
            Deserialized value
        """
        if isinstance(value, dict):
            if value.get("__type__") == "ndarray":
                return np.array(value["data"], dtype=value["dtype"])
            return {key: self._deserialize_value(val) for key, val in value.items()}
        elif isinstance(value, list):
            return [self._deserialize_value(item) for item in value]
        return value
