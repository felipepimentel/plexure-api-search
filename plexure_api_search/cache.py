"""Caching utilities for search operations."""

import hashlib
import json
from datetime import datetime, timedelta
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Optional, Union

import numpy as np
from diskcache import Cache

from .config import CACHE_DIR


class SearchCache:
    """Cache for search results."""

    def __init__(
        self,
        cache_dir: str = f"{CACHE_DIR}/search",
        ttl: int = 3600,  # 1 hour
    ):
        """Initialize search cache.

        Args:
            cache_dir: Directory to store cache.
            ttl: Time to live in seconds.
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.ttl = ttl

        # Initialize disk cache
        self.cache = Cache(str(self.cache_dir))

    def get_search_results(
        self, query: str
    ) -> Optional[List[Dict[str, Union[str, float]]]]:
        """Get cached search results.

        Args:
            query: Search query.

        Returns:
            Cached results if found and valid, None otherwise.
        """
        key = self._get_cache_key(query)
        cached = self.cache.get(key)

        if cached is None:
            return None

        # Check if expired
        if (
            datetime.fromisoformat(cached["timestamp"]) + timedelta(seconds=self.ttl)
            < datetime.now()
        ):
            self.cache.delete(key)
            return None

        return cached["results"]

    def store_search_results(
        self, query: str, results: List[Dict[str, Union[str, float]]]
    ) -> None:
        """Store search results in cache.

        Args:
            query: Search query.
            results: Search results to cache.
        """
        key = self._get_cache_key(query)
        self.cache.set(
            key, {"timestamp": datetime.now().isoformat(), "results": results}
        )

    def _get_cache_key(self, query: str) -> str:
        """Generate cache key for query.

        Args:
            query: Search query.

        Returns:
            Cache key.
        """
        return f"search:{query.lower().strip()}"

    def clear(self) -> None:
        """Clear all cached results."""
        self.cache.clear()


class EmbeddingCache:
    """Cache for embeddings."""

    def __init__(
        self,
        cache_dir: str = f"{CACHE_DIR}/embeddings",
        ttl: int = 86400,  # 24 hours
    ):
        """Initialize embedding cache.

        Args:
            cache_dir: Directory to store cache.
            ttl: Time to live in seconds.
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.ttl = ttl

        # Initialize disk cache
        self.cache = Cache(str(self.cache_dir))

    def get_embedding(self, text: str) -> Optional[np.ndarray]:
        """Get cached embedding.

        Args:
            text: Text to get embedding for.

        Returns:
            Cached embedding if found and valid, None otherwise.
        """
        key = self._get_cache_key(text)
        cached = self.cache.get(key)

        if cached is None:
            return None

        # Check if expired
        if (
            datetime.fromisoformat(cached["timestamp"]) + timedelta(seconds=self.ttl)
            < datetime.now()
        ):
            self.cache.delete(key)
            return None

        return np.array(cached["embedding"])

    def store_embedding(self, text: str, embedding: np.ndarray) -> None:
        """Store embedding in cache.

        Args:
            text: Text the embedding is for.
            embedding: Embedding to cache.
        """
        key = self._get_cache_key(text)
        self.cache.set(
            key,
            {"timestamp": datetime.now().isoformat(), "embedding": embedding.tolist()},
        )

    def _get_cache_key(self, text: str) -> str:
        """Generate cache key for text.

        Args:
            text: Text to generate key for.

        Returns:
            Cache key.
        """
        return f"embedding:{text.lower().strip()}"

    def clear(self) -> None:
        """Clear all cached embeddings."""
        self.cache.clear()


class SmartCache:
    def __init__(self, max_size: int = 1000):
        self.max_size = max_size

    @lru_cache(maxsize=1000)
    def get_cached_search(self, query_hash: str) -> List[Dict]:
        """Get cached search results."""
        pass

    def cache_search(self, query: str, results: List[Dict]) -> None:
        """Cache search results with query fingerprint."""
        query_hash = self._generate_query_hash(query)
        self.get_cached_search.cache_clear()

    def _generate_query_hash(self, query: str) -> str:
        """Generate stable hash for query."""
        return hashlib.sha256(json.dumps(query, sort_keys=True).encode()).hexdigest()
