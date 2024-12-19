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
        try:
            # Initialize FAISS index with ID support
            base_index = faiss.IndexFlatIP(self.dimension)
            self.index = faiss.IndexIDMap(base_index)
            
            # Log initialization
            logger.info(f"Vector store initialized with dimension {self.dimension}")
            logger.debug(f"Using index type: {type(self.index).__name__}")
            logger.debug(f"Using base index type: {type(base_index).__name__}")
            
            self.initialized = True
            
        except Exception as e:
            logger.error(f"Failed to initialize vector store: {e}")
            self.metrics.increment("vector_store_errors")
            raise

    def cleanup(self) -> None:
        """Clean up vector store."""
        if self.initialized:
            # Reset index
            self.index = None
            self.initialized = False
            self.next_id = 0
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
        vectors: Union[List[List[float]], np.ndarray],
        ids: Optional[Union[List[int], np.ndarray]] = None,
    ) -> None:
        """Store vectors in the index.
        
        Args:
            vectors: List of vectors or numpy array
            ids: Optional list of IDs or numpy array
        """
        if not self.initialized:
            self.initialize()

        try:
            # Convert vectors to numpy array if needed
            if isinstance(vectors, list):
                vectors = np.array(vectors)

            # Ensure vectors are 2D array
            if len(vectors.shape) == 1:
                vectors = vectors.reshape(1, -1)

            # Convert to float32
            vectors = vectors.astype(np.float32)

            # Validate dimensions
            if vectors.shape[1] != self.dimension:
                raise ValueError(f"Vector dimension {vectors.shape[1]} does not match expected dimension {self.dimension}")

            # Normalize vectors
            faiss.normalize_L2(vectors)

            # Log vector info
            logger.debug(f"Adding {len(vectors)} vectors to index")
            logger.debug(f"Vector shape: {vectors.shape}")
            logger.debug(f"Vector type: {vectors.dtype}")

            # Handle IDs
            if ids is not None:
                # Convert IDs to numpy array if needed
                if isinstance(ids, list):
                    ids = np.array(ids)
                # Ensure IDs are 1D array
                ids = ids.flatten()
                # Convert to int64
                ids = ids.astype(np.int64)
                # Log ID info
                logger.debug(f"Number of IDs: {len(ids)}")
                logger.debug(f"ID array shape: {ids.shape}")
                logger.debug(f"ID array type: {ids.dtype}")
                logger.debug(f"Sample IDs: {ids[:5] if len(ids) > 5 else ids}")
                # Validate ID count
                if len(ids) != len(vectors):
                    raise ValueError(f"Number of IDs ({len(ids)}) does not match number of vectors ({len(vectors)})")
                # Add vectors with IDs
                self.index.add_with_ids(vectors, ids)
            else:
                # Add vectors without IDs
                self.index.add(vectors)

            # Update metrics
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
        query_vector: Union[List[float], np.ndarray],
        top_k: int = 10,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Search for similar vectors.
        
        Args:
            query_vector: Query vector
            top_k: Number of results to return
            
        Returns:
            Tuple of (distances, indices)
        """
        if not self.initialized:
            self.initialize()

        try:
            # Convert to numpy array if needed
            if isinstance(query_vector, list):
                query_vector = np.array(query_vector)

            # Ensure 2D array
            if len(query_vector.shape) == 1:
                query_vector = query_vector.reshape(1, -1)

            # Convert to float32
            query_vector = query_vector.astype(np.float32)

            # Validate dimensions
            if query_vector.shape[1] != self.dimension:
                raise ValueError(f"Query vector dimension {query_vector.shape[1]} does not match expected dimension {self.dimension}")

            # Normalize query vector
            faiss.normalize_L2(query_vector)

            # Log query info
            logger.debug(f"Searching with query vector shape: {query_vector.shape}")
            logger.debug(f"Query vector type: {query_vector.dtype}")

            # Search index
            distances, indices = self.index.search(query_vector, top_k)

            # Log results
            logger.debug(f"Found {len(indices[0])} results")
            logger.debug(f"Distance range: {np.min(distances)} to {np.max(distances)}")

            # Update metrics
            self.metrics.increment(
                "vectors_searched",
                {"count": 1},
            )

            return distances[0], indices[0]

        except Exception as e:
            logger.error(f"Failed to search vectors: {e}")
            self.metrics.increment("vector_store_errors")
            raise

# Global instance
vector_store = VectorStore() 