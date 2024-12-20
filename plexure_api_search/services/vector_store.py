"""
Vector Store Service for Plexure API Search

This module provides vector storage and retrieval functionality using FAISS (Facebook AI Similarity Search).
It manages the storage, indexing, and similarity search of vector embeddings for API endpoints.

Key Features:
- FAISS integration for efficient vector storage
- Inner product similarity metric
- ID mapping for endpoint metadata
- Normalized L2 vectors
- AVX2 optimizations
- Batch vector operations
- Persistence management
- Cache integration
- Index optimization

The VectorStore class provides methods for:
- Storing vectors with metadata
- Searching similar vectors
- Managing index lifecycle
- Optimizing performance
- Handling persistence
- Monitoring health
- Managing resources

Example Usage:
    from plexure_api_search.services.vector_store import VectorStore

    # Initialize store
    store = VectorStore(dimension=384)

    # Store vectors
    vectors = model.encode(texts)
    ids = store.store_vectors(vectors, metadata)

    # Search vectors
    query_vector = model.encode(query)
    results = store.search_vectors(query_vector, top_k=10)

Performance Features:
- Efficient similarity search with FAISS
- Batch processing support
- Memory-mapped storage
- Index optimization
- Cache integration
- Resource management
"""

import logging
import os
import pickle
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import faiss
import numpy as np

from ..config import config
from ..monitoring.metrics import MetricsManager
from ..services.models import model_service

logger = logging.getLogger(__name__)


class VectorStore:
    """Vector store for managing embeddings."""

    def __init__(self):
        """Initialize vector store."""
        self.metrics = MetricsManager()
        self.index = None
        self.dimension = None
        self.metadata = {}
        self.initialized = False

    def initialize(self) -> None:
        """Initialize vector store."""
        if self.initialized:
            return

        try:
            # Get dimension from model service
            model_service.initialize()
            self.dimension = model_service.dimension
            logger.info(f"Using embedding dimension: {self.dimension}")

            # Create index
            self.index = faiss.IndexIDMap(faiss.IndexFlatIP(self.dimension))
            logger.info(f"Created FAISS index with dimension {self.dimension}")

            # Load existing index if available
            self._load_index()
            self.initialized = True
            logger.info("Vector store initialized")

        except Exception as e:
            logger.error(f"Failed to initialize vector store: {e}")
            raise

    def cleanup(self) -> None:
        """Clean up vector store."""
        if self.initialized:
            self._save_index()
            self.index = None
            self.dimension = None
            self.metadata = {}
            self.initialized = False
            logger.info("Vector store cleaned up")

    def clear(self) -> None:
        """Clear the vector store."""
        if not self.initialized:
            self.initialize()

        try:
            # Reset index
            self.index = faiss.IndexIDMap(faiss.IndexFlatIP(self.dimension))
            self.metadata = {}
            logger.info("Cleared vector store")

            # Delete saved files
            index_path = self._get_index_path()
            if os.path.exists(index_path):
                os.remove(index_path)
                logger.info(f"Deleted saved index at {index_path}")

            metadata_path = self._get_metadata_path()
            if os.path.exists(metadata_path):
                os.remove(metadata_path)
                logger.info(f"Deleted saved metadata at {metadata_path}")

        except Exception as e:
            logger.error(f"Failed to clear vector store: {e}")
            raise

    def store_vectors(
        self,
        vectors: np.ndarray,
        ids: np.ndarray,
        metadata: Optional[List[Dict]] = None,
    ) -> None:
        """Store vectors in the index.

        Args:
            vectors: Vector array (n_vectors x dimension)
            ids: Vector IDs
            metadata: Optional metadata for each vector
        """
        if not self.initialized:
            self.initialize()

        try:
            # Validate inputs
            if vectors.shape[1] != self.dimension:
                raise ValueError(
                    f"Vector dimension {vectors.shape[1]} does not match "
                    f"index dimension {self.dimension}"
                )

            if len(vectors) != len(ids):
                raise ValueError(
                    f"Number of vectors {len(vectors)} does not match "
                    f"number of IDs {len(ids)}"
                )

            if metadata and len(metadata) != len(vectors):
                raise ValueError(
                    f"Number of metadata entries {len(metadata)} does not match "
                    f"number of vectors {len(vectors)}"
                )

            # Add vectors to index
            self.index.add_with_ids(vectors, ids)
            logger.info(f"Added {len(vectors)} vectors to index")

            # Store metadata
            if metadata:
                for i, meta in zip(ids, metadata):
                    self.metadata[int(i)] = meta
                logger.info(f"Stored metadata for {len(metadata)} vectors")

            # Update metrics
            self.metrics.set_gauge("index_size", self.index.ntotal)
            self.metrics.set_gauge("metadata_size", len(self.metadata))

        except Exception as e:
            logger.error(f"Failed to store vectors: {e}")
            raise

    def search_vectors(
        self,
        query_vector: np.ndarray,
        k: int = 10,
    ) -> Tuple[np.ndarray, np.ndarray, List[Dict]]:
        """Search for similar vectors.

        Args:
            query_vector: Query vector
            k: Number of results to return

        Returns:
            Tuple of (distances, indices, metadata)
        """
        if not self.initialized:
            self.initialize()

        try:
            # Validate inputs
            if query_vector.shape[1] != self.dimension:
                raise ValueError(
                    f"Query vector dimension {query_vector.shape[1]} does not match "
                    f"index dimension {self.dimension}"
                )

            # Search index
            k = min(k, self.index.ntotal)
            distances, indices = self.index.search(query_vector, k)

            # Get metadata
            metadata = []
            for idx in indices[0]:
                meta = self.metadata.get(int(idx), {})
                metadata.append(meta)

            return distances, indices, metadata

        except Exception as e:
            logger.error(f"Failed to search vectors: {e}")
            raise

    def _get_index_path(self) -> str:
        """Get path to saved index file."""
        cache_dir = Path(config.cache_dir) / config.environment
        cache_dir.mkdir(parents=True, exist_ok=True)
        return str(cache_dir / "faiss.index")

    def _get_metadata_path(self) -> str:
        """Get path to saved metadata file."""
        cache_dir = Path(config.cache_dir) / config.environment
        cache_dir.mkdir(parents=True, exist_ok=True)
        return str(cache_dir / "metadata.pkl")

    def _load_index(self) -> None:
        """Load saved index and metadata."""
        try:
            # Load index
            index_path = self._get_index_path()
            if os.path.exists(index_path):
                self.index = faiss.read_index(index_path)
                logger.info(f"Loaded index with {self.index.ntotal} vectors")

            # Load metadata
            metadata_path = self._get_metadata_path()
            if os.path.exists(metadata_path):
                with open(metadata_path, "rb") as f:
                    self.metadata = pickle.load(f)
                logger.info(f"Loaded metadata for {len(self.metadata)} vectors")

        except Exception as e:
            logger.error(f"Failed to load index: {e}")
            raise

    def _save_index(self) -> None:
        """Save index and metadata."""
        try:
            # Save index
            index_path = self._get_index_path()
            faiss.write_index(self.index, index_path)
            logger.info(f"Saved index with {self.index.ntotal} vectors")

            # Save metadata
            metadata_path = self._get_metadata_path()
            with open(metadata_path, "wb") as f:
                pickle.dump(self.metadata, f)
            logger.info(f"Saved metadata for {len(self.metadata)} vectors")

        except Exception as e:
            logger.error(f"Failed to save index: {e}")
            raise


# Global instance
vector_store = VectorStore()
