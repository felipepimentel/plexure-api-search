"""FAISS vector preprocessing and optimization layer."""

import logging
import os
from typing import Any, Dict, List, Optional, Tuple

import faiss
import numpy as np

from ..config import config_instance
from ..utils.cache import DiskCache

logger = logging.getLogger(__name__)

# Cache for temporary local storage
local_cache = DiskCache[Dict[str, Any]](
    namespace="faiss_local",
    ttl=config_instance.cache_ttl,  # 1 hour
)

class FAISSPreprocessor:
    """Local vector preprocessing and optimization using FAISS."""

    def __init__(
        self,
        dimension: int = config_instance.vector_dimension,
        index_type: str = "IVFFlat",
        nlist: int = 10,
        index_path: Optional[str] = None,
    ):
        """Initialize FAISS preprocessor.

        Args:
            dimension: Vector dimension
            index_type: FAISS index type (IVFFlat, Flat, etc.)
            nlist: Number of clusters for IVF indices
            index_path: Path to load/save index
        """
        self.dimension = dimension
        self.index_type = index_type
        self.nlist = min(nlist, dimension)
        self.index_path = index_path or os.path.join(
            config_instance.cache_dir, "faiss_preprocessing.bin"
        )
        
        # Initialize index
        self._initialize_index()
        
        # Load existing index if available
        if os.path.exists(self.index_path):
            self._load_index()

    def _initialize_index(self):
        """Initialize FAISS index for preprocessing."""
        try:
            base_index = faiss.IndexFlatL2(self.dimension)
            self.index = faiss.IndexIDMap(base_index)
            self.is_trained = False
            logger.info("Initialized FAISS preprocessing index")
        except Exception as e:
            logger.error(f"Failed to initialize FAISS index: {e}")
            raise

    def _load_index(self):
        """Load preprocessing index from disk."""
        try:
            self.index = faiss.read_index(self.index_path)
            logger.info(f"Loaded FAISS preprocessing index from {self.index_path}")
        except Exception as e:
            logger.error(f"Failed to load FAISS index: {e}")
            self._initialize_index()

    def _save_index(self):
        """Save preprocessing index to disk."""
        try:
            faiss.write_index(self.index, self.index_path)
            logger.info(f"Saved FAISS preprocessing index to {self.index_path}")
        except Exception as e:
            logger.error(f"Failed to save FAISS index: {e}")

    def preprocess_vectors(
        self, vectors: List[Dict], batch_size: int = 100
    ) -> Tuple[List[Dict], Dict[str, Any]]:
        """Preprocess and optimize vectors before storage.

        Args:
            vectors: List of vector entries to process
            batch_size: Size of batches for processing

        Returns:
            Tuple of (optimized vectors, preprocessing metadata)
        """
        try:
            # Convert vectors to numpy array for processing
            vector_data = []
            vector_ids = []
            metadata_map = {}
            
            for vector in vectors:
                vector_id = vector["id"]
                vector_values = np.array(vector["values"], dtype=np.float32)
                metadata = vector.get("metadata", {})
                
                vector_data.append(vector_values)
                vector_ids.append(vector_id)
                metadata_map[vector_id] = metadata

            if not vector_data:
                return [], {}

            # Convert to numpy array
            vector_array = np.array(vector_data)

            # Perform dimensionality reduction if needed
            if vector_array.shape[1] > self.dimension:
                logger.info(f"Reducing dimension from {vector_array.shape[1]} to {self.dimension}")
                pca = faiss.PCAMatrix(vector_array.shape[1], self.dimension)
                pca.train(vector_array)
                vector_array = pca.apply_py(vector_array)

            # Normalize vectors
            faiss.normalize_L2(vector_array)

            # Train index if needed
            if not self.is_trained and len(vector_array) > 1:
                self.index.train(vector_array)
                self.is_trained = True

            # Add to local index for future reference
            self.index.add_with_ids(vector_array, np.array([hash(id) % (2**63) for id in vector_ids]))
            self._save_index()

            # Prepare optimized vectors for storage
            optimized_vectors = []
            for i, (vec_id, vec) in enumerate(zip(vector_ids, vector_array)):
                optimized_vectors.append({
                    "id": vec_id,
                    "values": vec.tolist(),
                    "metadata": metadata_map[vec_id]
                })

            # Store preprocessing metadata
            preprocessing_meta = {
                "original_dim": len(vectors[0]["values"]),
                "reduced_dim": self.dimension,
                "normalized": True,
                "num_vectors": len(vectors)
            }

            return optimized_vectors, preprocessing_meta

        except Exception as e:
            logger.error(f"Vector preprocessing failed: {e}")
            return vectors, {}  # Return original vectors if preprocessing fails

    def optimize_query(self, query_vector: List[float]) -> List[float]:
        """Optimize query vector to match preprocessed vectors.

        Args:
            query_vector: Original query vector

        Returns:
            Optimized query vector
        """
        try:
            # Convert to numpy array
            query_array = np.array([query_vector], dtype=np.float32)

            # Apply same preprocessing as vectors
            if query_array.shape[1] > self.dimension:
                pca = faiss.PCAMatrix(query_array.shape[1], self.dimension)
                query_array = pca.apply_py(query_array)

            # Normalize
            faiss.normalize_L2(query_array)

            return query_array[0].tolist()

        except Exception as e:
            logger.error(f"Query optimization failed: {e}")
            return query_vector  # Return original query if optimization fails

    def clear_local_index(self) -> bool:
        """Clear local preprocessing index.

        Returns:
            True if successful, False otherwise
        """
        try:
            self._initialize_index()
            local_cache.clear()
            self._save_index()
            return True
        except Exception as e:
            logger.error(f"Failed to clear local index: {e}")
            return False 