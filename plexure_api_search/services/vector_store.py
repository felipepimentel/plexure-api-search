"""Vector store service for managing vectors."""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import faiss
from dependency_injector.wiring import inject
from dependency_injector.providers import Provider

from ..config import config_instance
from ..monitoring.metrics import MetricsManager
from .base import BaseService

logger = logging.getLogger(__name__)

class VectorStore(BaseService):
    """Vector store service for managing vectors."""

    def __init__(self):
        """Initialize vector store."""
        super().__init__()
        self.index = None
        self.metrics = MetricsManager()
        self.initialized = False
        self.dimension = config_instance.vectors.dimension
        self.next_id = 0
        self.vectors = []

    def initialize(self) -> None:
        """Initialize vector store."""
        if self.initialized:
            return

        try:
            # Create empty list for vectors
            self.vectors = []
            
            self.initialized = True
            logger.info("Vector store initialized")

        except Exception as e:
            logger.error(f"Failed to initialize vector store: {e}")
            raise

    def cleanup(self) -> None:
        """Clean up vector store."""
        self.vectors = []
        self.initialized = False
        logger.info("Vector store cleaned up")

    def health_check(self) -> Dict[str, bool]:
        """Check service health.
        
        Returns:
            Health check results
        """
        return {
            "initialized": self.initialized,
            "vectors_ready": len(self.vectors) > 0,
        }

    def store_vectors(
        self,
        vectors: Union[np.ndarray, List[np.ndarray]],
        ids: Optional[List[int]] = None,
    ) -> None:
        """Store vectors in index.
        
        Args:
            vectors: Vectors to store
            ids: Optional vector IDs
        """
        if not self.initialized:
            self.initialize()

        try:
            # Convert to numpy array if needed
            if isinstance(vectors, list):
                vectors = np.array(vectors)

            # Ensure 2D array
            if len(vectors.shape) == 1:
                vectors = np.array([vectors])

            # Convert to float32
            vectors = vectors.astype(np.float32)

            # Log vector info
            logger.debug(f"Input vector shape: {vectors.shape}")
            logger.debug(f"Input vector type: {vectors.dtype}")
            logger.debug(f"Input vector sample: {vectors[0][:5]}")
            logger.debug(f"Input vector min: {np.min(vectors)}")
            logger.debug(f"Input vector max: {np.max(vectors)}")
            logger.debug(f"Input vector mean: {np.mean(vectors)}")

            # Validate dimensions
            if vectors.shape[1] != self.dimension:
                raise ValueError(
                    f"Vector dimension mismatch: {vectors.shape[1]} != {self.dimension}"
                )

            # Normalize vectors
            faiss.normalize_L2(vectors)

            # Add vectors
            start_time = self.metrics.start_timer()
            self.index.add(vectors)

            # Log index info
            logger.debug(f"Index size after add: {self.index.ntotal}")

            self.metrics.stop_timer(start_time, "vector_store")
            self.metrics.increment(
                "vectors_stored",
                {"count": len(vectors)},
            )

        except Exception as e:
            logger.error(f"Failed to store vectors: {e}")
            self.metrics.increment("vector_store_errors")
            raise

    def search_vectors(
        self,
        query_vector: Union[np.ndarray, List[float]],
        k: int = 10,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Search for similar vectors.
        
        Args:
            query_vector: Query vector
            k: Number of results
            
        Returns:
            Distances and indices of similar vectors
        """
        if not self.initialized:
            self.initialize()

        try:
            # Convert to numpy array if needed
            if isinstance(query_vector, list):
                query_vector = np.array(query_vector)

            # Ensure 2D array
            if len(query_vector.shape) == 1:
                query_vector = np.array([query_vector])

            # Convert to float32
            query_vector = query_vector.astype(np.float32)

            # Log query vector shape
            logger.debug(f"Query vector shape: {query_vector.shape}")

            # Validate dimensions
            if query_vector.shape[1] != self.dimension:
                raise ValueError(
                    f"Vector dimension mismatch: {query_vector.shape[1]} != {self.dimension}"
                )

            # Search
            start_time = self.metrics.start_timer()
            distances, indices = self.index.search(query_vector, k)
            self.metrics.stop_timer(start_time, "vector_search")

            return distances[0], indices[0]

        except Exception as e:
            logger.error(f"Failed to search vectors: {e}")
            self.metrics.increment("vector_search_errors")
            raise

# Global instance
vector_store = VectorStore() 