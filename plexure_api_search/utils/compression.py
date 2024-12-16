"""Cache compression utilities."""

import gzip
import json
import logging
import lzma
import pickle
import zlib
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, Optional, Union

import numpy as np

logger = logging.getLogger(__name__)


class CompressionMethod(Enum):
    """Available compression methods."""

    NONE = "none"
    GZIP = "gzip"
    LZMA = "lzma"
    ZLIB = "zlib"


@dataclass
class CompressionStats:
    """Compression statistics."""

    original_size: int
    compressed_size: int
    method: CompressionMethod
    compression_time: float
    decompression_time: float

    @property
    def compression_ratio(self) -> float:
        """Calculate compression ratio."""
        return self.compressed_size / self.original_size if self.original_size > 0 else 1.0

    @property
    def space_saved(self) -> float:
        """Calculate space saved percentage."""
        return 1.0 - self.compression_ratio


class CacheCompressor:
    """Handles cache data compression."""

    def __init__(
        self,
        method: CompressionMethod = CompressionMethod.GZIP,
        compression_level: int = 6,
    ):
        """Initialize compressor.

        Args:
            method: Compression method to use
            compression_level: Compression level (1-9, higher = better compression but slower)
        """
        self.method = method
        self.compression_level = compression_level
        self.stats: Dict[str, CompressionStats] = {}

    def compress(
        self,
        data: Any,
        key: Optional[str] = None,
    ) -> bytes:
        """Compress data.

        Args:
            data: Data to compress
            key: Optional key for tracking stats

        Returns:
            Compressed data
        """
        try:
            import time
            start_time = time.time()

            # Serialize data
            if isinstance(data, (dict, list)):
                serialized = json.dumps(data).encode()
            elif isinstance(data, np.ndarray):
                serialized = pickle.dumps(data)
            else:
                serialized = pickle.dumps(data)

            original_size = len(serialized)
            compressed: bytes

            # Apply compression
            if self.method == CompressionMethod.GZIP:
                compressed = gzip.compress(serialized, compresslevel=self.compression_level)
            elif self.method == CompressionMethod.LZMA:
                compressed = lzma.compress(serialized, preset=self.compression_level)
            elif self.method == CompressionMethod.ZLIB:
                compressed = zlib.compress(serialized, level=self.compression_level)
            else:
                compressed = serialized

            compression_time = time.time() - start_time

            # Track stats if key provided
            if key:
                self.stats[key] = CompressionStats(
                    original_size=original_size,
                    compressed_size=len(compressed),
                    method=self.method,
                    compression_time=compression_time,
                    decompression_time=0.0,  # Will be updated on decompress
                )

            return compressed

        except Exception as e:
            logger.error(f"Compression failed: {e}")
            return pickle.dumps(data)

    def decompress(
        self,
        data: bytes,
        key: Optional[str] = None,
    ) -> Any:
        """Decompress data.

        Args:
            data: Compressed data
            key: Optional key for tracking stats

        Returns:
            Decompressed data
        """
        try:
            import time
            start_time = time.time()

            # Apply decompression
            if self.method == CompressionMethod.GZIP:
                decompressed = gzip.decompress(data)
            elif self.method == CompressionMethod.LZMA:
                decompressed = lzma.decompress(data)
            elif self.method == CompressionMethod.ZLIB:
                decompressed = zlib.decompress(data)
            else:
                decompressed = data

            # Try JSON first, fallback to pickle
            try:
                result = json.loads(decompressed.decode())
            except (json.JSONDecodeError, UnicodeDecodeError):
                result = pickle.loads(decompressed)

            decompression_time = time.time() - start_time

            # Update stats if key exists
            if key and key in self.stats:
                self.stats[key].decompression_time = decompression_time

            return result

        except Exception as e:
            logger.error(f"Decompression failed: {e}")
            try:
                return pickle.loads(data)
            except Exception:
                return None

    def get_stats(self, key: str) -> Optional[CompressionStats]:
        """Get compression stats for key.

        Args:
            key: Stats key

        Returns:
            Compression stats if available
        """
        return self.stats.get(key)

    def clear_stats(self) -> None:
        """Clear compression stats."""
        self.stats.clear()

    def get_method_stats(self) -> Dict[str, Dict[str, float]]:
        """Get aggregated stats by compression method.

        Returns:
            Dictionary of method stats
        """
        method_stats: Dict[str, Dict[str, float]] = {}

        for stats in self.stats.values():
            method = stats.method.value
            if method not in method_stats:
                method_stats[method] = {
                    "total_original_size": 0,
                    "total_compressed_size": 0,
                    "avg_compression_time": 0.0,
                    "avg_decompression_time": 0.0,
                    "count": 0,
                }

            method_stats[method]["total_original_size"] += stats.original_size
            method_stats[method]["total_compressed_size"] += stats.compressed_size
            method_stats[method]["avg_compression_time"] += stats.compression_time
            method_stats[method]["avg_decompression_time"] += stats.decompression_time
            method_stats[method]["count"] += 1

        # Calculate averages
        for stats in method_stats.values():
            count = stats["count"]
            if count > 0:
                stats["avg_compression_time"] /= count
                stats["avg_decompression_time"] /= count
                stats["compression_ratio"] = stats["total_compressed_size"] / stats["total_original_size"]
                stats["space_saved"] = 1.0 - stats["compression_ratio"]

        return method_stats


class CompressedCache:
    """Cache wrapper with compression support."""

    def __init__(
        self,
        cache: Any,
        method: CompressionMethod = CompressionMethod.GZIP,
        compression_level: int = 6,
    ):
        """Initialize compressed cache.

        Args:
            cache: Base cache instance
            method: Compression method
            compression_level: Compression level
        """
        self.cache = cache
        self.compressor = CacheCompressor(method, compression_level)

    def get(self, key: str) -> Any:
        """Get value from cache.

        Args:
            key: Cache key

        Returns:
            Cached value if found
        """
        try:
            compressed = self.cache.get(key)
            if compressed is None:
                return None

            return self.compressor.decompress(compressed, key)

        except Exception as e:
            logger.error(f"Failed to get from compressed cache: {e}")
            return None

    def set(
        self,
        key: str,
        value: Any,
        ttl: Optional[int] = None,
    ) -> None:
        """Set value in cache.

        Args:
            key: Cache key
            value: Value to cache
            ttl: Optional TTL in seconds
        """
        try:
            compressed = self.compressor.compress(value, key)
            self.cache.set(key, compressed, ttl)

        except Exception as e:
            logger.error(f"Failed to set in compressed cache: {e}")

    def delete(self, key: str) -> None:
        """Delete value from cache.

        Args:
            key: Cache key
        """
        try:
            self.cache.delete(key)
            if key in self.compressor.stats:
                del self.compressor.stats[key]

        except Exception as e:
            logger.error(f"Failed to delete from compressed cache: {e}")

    def clear(self) -> None:
        """Clear all cached values."""
        try:
            self.cache.clear()
            self.compressor.clear_stats()

        except Exception as e:
            logger.error(f"Failed to clear compressed cache: {e}")

    def get_compression_stats(self) -> Dict[str, Any]:
        """Get compression statistics.

        Returns:
            Dictionary of compression stats
        """
        return {
            "by_key": {
                key: stats.__dict__
                for key, stats in self.compressor.stats.items()
            },
            "by_method": self.compressor.get_method_stats(),
        } 