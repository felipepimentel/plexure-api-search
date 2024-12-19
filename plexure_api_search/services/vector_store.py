"""Vector store service."""

import logging
import os
from pathlib import Path
from typing import List, Optional, Tuple

import faiss
import numpy as np
from dependency_injector.wiring import inject

from ..config import config_instance
from ..monitoring.metrics import MetricsManager
from .models import model_service

logger = logging.getLogger(__name__)

class VectorStore:
    """Vector store for embeddings."""

    def __init__(self):
        """Initialize vector store."""
        self.metrics = MetricsManager()
        self.index = None
        self.dimension = None
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
            self.index = faiss.IndexFlatIP(self.dimension)
            logger.info(f"Created FAISS index with dimension {self.dimension}")

            # Load existing index if available
            index_path = self._get_index_path()
            if index_path.exists():
                logger.info(f"Loading existing index from {index_path}")
                self.index = faiss.read_index(str(index_path))
                logger.info(f"Loaded index with {self.index.ntotal} vectors")

            self.initialized = True
            logger.info("Vector store initialized")

        except Exception as e:
            logger.error(f"Failed to initialize vector store: {e}")
            raise

    def cleanup(self) -> None:
        """Clean up vector store."""
        if self.index is not None:
            # Save index
            index_path = self._get_index_path()
            logger.info(f"Saving index with {self.index.ntotal} vectors to {index_path}")
            faiss.write_index(self.index, str(index_path))

            self.index = None
            self.initialized = False
            logger.info("Vector store cleaned up")

    def store_vectors(self, vectors: np.ndarray, ids: Optional[np.ndarray] = None) -> None:
        """Store vectors in index.
        
        Args:
            vectors: Vectors to store (n_vectors x dimension)
            ids: Optional vector IDs (n_vectors)
        """
        if not self.initialized:
            self.initialize()

        try:
            # Validate inputs
            if vectors.ndim != 2:
                raise ValueError(f"Expected 2D array, got shape {vectors.shape}")
            if vectors.shape[1] != self.dimension:
                raise ValueError(
                    f"Expected vectors with dimension {self.dimension}, got {vectors.shape[1]}"
                )
            if ids is not None and len(ids) != len(vectors):
                raise ValueError(
                    f"Number of IDs ({len(ids)}) does not match number of vectors ({len(vectors)})"
                )

            # Log vector info
            logger.debug(f"Adding {len(vectors)} vectors to index")
            logger.debug(f"Vector shape: {vectors.shape}")
            if ids is not None:
                logger.debug(f"ID array shape: {ids.shape}")
                logger.debug(f"ID array type: {ids.dtype}")
                logger.debug(f"Sample IDs: {ids[:5] if len(ids) > 5 else ids}")

            # Add vectors
            if ids is not None:
                self.index.add_with_ids(vectors, ids)
            else:
                self.index.add(vectors)

            logger.info(f"Added {len(vectors)} vectors to index")
            logger.debug(f"Index now contains {self.index.ntotal} vectors")

            # Update metrics
            self.metrics.increment(
                "vectors_stored",
                {"count": len(vectors), "total": self.index.ntotal},
            )

        except Exception as e:
            logger.error(f"Failed to store vectors: {e}")
            self.metrics.increment("vector_store_errors")
            raise

    def search_vectors(
        self, query_vector: np.ndarray, k: int = 10
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Search for similar vectors.
        
        Args:
            query_vector: Query vector
            k: Number of results to return
            
        Returns:
            Tuple of (distances, indices)
        """
        if not self.initialized:
            self.initialize()

        try:
            # Validate input
            if query_vector.ndim != 1:
                raise ValueError(f"Expected 1D array, got shape {query_vector.shape}")
            if len(query_vector) != self.dimension:
                raise ValueError(
                    f"Expected vector with dimension {self.dimension}, got {len(query_vector)}"
                )

            # Reshape query vector
            query_vector = query_vector.reshape(1, -1)

            # Search
            distances, indices = self.index.search(query_vector, k)
            
            # Get first row since we only have one query vector
            distances = distances[0]
            indices = indices[0]

            # Log results
            logger.debug(f"Found {len(indices)} results")
            if len(indices) > 0:
                logger.debug(f"Distance range: {np.min(distances)} to {np.max(distances)}")

            return distances, indices

        except Exception as e:
            logger.error(f"Failed to search vectors: {e}")
            self.metrics.increment("vector_search_errors")
            raise

    def clear(self) -> None:
        """Clear the vector store."""
        if not self.initialized:
            self.initialize()

        try:
            # Reset index
            self.index = faiss.IndexFlatIP(self.dimension)
            logger.info("Cleared vector store")

            # Delete saved index
            index_path = self._get_index_path()
            if index_path.exists():
                index_path.unlink()
                logger.info(f"Deleted saved index at {index_path}")

            self.metrics.increment("vector_store_cleared")

        except Exception as e:
            logger.error(f"Failed to clear vector store: {e}")
            self.metrics.increment("vector_store_errors")
            raise

    def _get_index_path(self) -> Path:
        """Get path to saved index file."""
        cache_dir = Path(config_instance.cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)
        return cache_dir / "faiss.index"

# Global instance
vector_store = VectorStore() 