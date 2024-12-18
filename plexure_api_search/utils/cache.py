"""Advanced caching system with disk-based storage."""

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
        cache_dir: Optional[str] = None,
    ):
        """Initialize disk cache.

        Args:
            namespace: Cache namespace
            ttl: Cache TTL in seconds
            cache_dir: Optional cache directory (defaults to config)
        """
        self.namespace = namespace
        self.ttl = ttl
        
        # Set up cache directory
        cache_dir = cache_dir or getattr(config_instance, "cache_dir", None)
        if not cache_dir:
            cache_dir = os.path.join(os.path.expanduser("~"), ".cache", "plexure_api_search")
            
        self.cache_dir = os.path.join(cache_dir, namespace)
        os.makedirs(self.cache_dir, exist_ok=True)
        logger.info(f"Using disk cache at {self.cache_dir}")

    def _get_path(self, key: str) -> str:
        """Get cache file path.

        Args:
            key: Cache key

        Returns:
            Cache file path
        """
        # Use hash of key to avoid filesystem issues
        hashed_key = str(hash(key))
        return os.path.join(self.cache_dir, f"{hashed_key}.cache")

    def _is_expired(self, path: str) -> bool:
        """Check if cache entry is expired.

        Args:
            path: Cache file path

        Returns:
            Whether the entry is expired
        """
        try:
            mtime = os.path.getmtime(path)
            age = time.time() - mtime
            return age > self.ttl
        except OSError:
            return True

    def get(self, key: str) -> Optional[T]:
        """Get value from cache.

        Args:
            key: Cache key

        Returns:
            Cached value if found and not expired
        """
        try:
            path = self._get_path(key)
            
            # Check if file exists and not expired
            if not os.path.exists(path) or self._is_expired(path):
                return None
                
            # Read and deserialize value
            with open(path, "rb") as f:
                return pickle.load(f)
                
        except Exception as e:
            logger.error(f"Error reading from disk cache: {e}")
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
            
            # Serialize and write value
            with open(path, "wb") as f:
                pickle.dump(value, f)
                
            # Update modification time for TTL
            os.utime(path, None)
            
        except Exception as e:
            logger.error(f"Error writing to disk cache: {e}")

    def delete(self, key: str) -> None:
        """Delete value from cache.

        Args:
            key: Cache key
        """
        try:
            path = self._get_path(key)
            if os.path.exists(path):
                os.remove(path)
                
        except Exception as e:
            logger.error(f"Error deleting from disk cache: {e}")

    def clear(self) -> None:
        """Clear all cached values in namespace."""
        try:
            for filename in os.listdir(self.cache_dir):
                path = os.path.join(self.cache_dir, filename)
                try:
                    if os.path.isfile(path):
                        os.remove(path)
                except Exception as e:
                    logger.error(f"Error removing cache file {path}: {e}")
                    
        except Exception as e:
            logger.error(f"Error clearing disk cache: {e}")

    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics.

        Returns:
            Dictionary of cache statistics
        """
        try:
            # Count files and calculate size
            total_files = 0
            total_size = 0
            expired_files = 0
            
            for filename in os.listdir(self.cache_dir):
                path = os.path.join(self.cache_dir, filename)
                if os.path.isfile(path):
                    total_files += 1
                    total_size += os.path.getsize(path)
                    if self._is_expired(path):
                        expired_files += 1
            
            return {
                "total_files": total_files,
                "total_size_bytes": total_size,
                "expired_files": expired_files,
                "cache_dir": self.cache_dir,
                "namespace": self.namespace,
                "ttl": self.ttl,
            }
            
        except Exception as e:
            logger.error(f"Error getting disk cache stats: {e}")
            return {}


# Default cache instance
default_cache = DiskCache("default")

__all__ = ["BaseCache", "DiskCache", "default_cache"]
