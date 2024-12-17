"""FAISS preprocessing for vector optimization."""

import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import faiss

from ..config import config_instance

logger = logging.getLogger(__name__)


class FAISSPreprocessor:
    """Preprocesses vectors using FAISS for optimization."""

    def __init__(self):
        """Initialize FAISS preprocessor."""
        self.dimension = config_instance.vector_dimension
        self.index = faiss.IndexFlatL2(self.dimension)

    def preprocess_vectors(
        self, vectors: List[Dict], batch_size: int = 100
    ) -> Tuple[List[Dict], Dict[str, Any]]:
        """Preprocess vectors using FAISS.

        Args:
            vectors: List of vector entries to preprocess
            batch_size: Size of batches for processing

        Returns:
            Tuple of (preprocessed vectors, preprocessing metadata)
        """
        try:
            if not vectors:
                logger.warning("No vectors to preprocess")
                return [], {"preprocessed": 0}

            # Extract vector data
            vector_data = []
            for entry in vectors:
                if "values" not in entry or not isinstance(entry["values"], (list, np.ndarray)):
                    logger.warning(f"Invalid vector entry: {entry}")
                    continue
                    
                # Convert to numpy array and validate
                vector = np.array(entry["values"], dtype=np.float32)
                if vector.size != self.dimension:
                    logger.warning(f"Invalid vector dimension: {vector.size} != {self.dimension}")
                    continue
                    
                vector_data.append(vector)

            if not vector_data:
                logger.warning("No valid vectors found")
                return [], {"preprocessed": 0}

            # Convert to numpy array
            vector_array = np.array(vector_data, dtype=np.float32)

            # Add to FAISS index
            self.index.add(vector_array)

            # Update original vectors with preprocessed data
            preprocessed = []
            for i, entry in enumerate(vectors):
                if i < len(vector_data):  # Only process valid vectors
                    entry_copy = entry.copy()
                    entry_copy["values"] = vector_data[i].tolist()
                    preprocessed.append(entry_copy)

            return preprocessed, {
                "preprocessed": len(preprocessed),
                "dimension": self.dimension,
            }

        except Exception as e:
            logger.error(f"Failed to preprocess vectors: {e}")
            return [], {"preprocessed": 0, "error": str(e)}

    def optimize_query(self, query_vector: List[float]) -> List[float]:
        """Optimize query vector using FAISS.

        Args:
            query_vector: Query vector to optimize

        Returns:
            Optimized query vector
        """
        try:
            # Convert to numpy array
            query = np.array(query_vector, dtype=np.float32).reshape(1, -1)
            if query.shape[1] != self.dimension:
                logger.warning(f"Invalid query dimension: {query.shape[1]} != {self.dimension}")
                # Pad or truncate to match dimension
                if query.shape[1] < self.dimension:
                    padding = np.zeros((1, self.dimension - query.shape[1]), dtype=np.float32)
                    query = np.concatenate([query, padding], axis=1)
                else:
                    query = query[:, :self.dimension]

            # Search in index to find nearest neighbors
            D, I = self.index.search(query, 1)

            # If no neighbors found, return original vector
            if len(I) == 0 or len(I[0]) == 0:
                return query_vector

            # Get optimized vector
            optimized = self.index.reconstruct(int(I[0][0]))
            return optimized.tolist()

        except Exception as e:
            logger.error(f"Failed to optimize query vector: {e}")
            return query_vector

    def clear_local_index(self) -> None:
        """Clear the local FAISS index."""
        try:
            self.index = faiss.IndexFlatL2(self.dimension)
            logger.info("Cleared local FAISS index")
        except Exception as e:
            logger.error(f"Failed to clear local index: {e}")