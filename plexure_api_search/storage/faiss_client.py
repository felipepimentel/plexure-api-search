"""FAISS vector store client for high-performance similarity search."""

import logging
import os
from typing import Any, Dict, List, Optional

import faiss
import numpy as np

from ..config import config_instance
from ..utils.cache import DiskCache

logger = logging.getLogger(__name__)

# Cache for metadata since FAISS only stores vectors
metadata_cache = DiskCache[Dict[str, Any]](
    namespace="faiss_metadata",
    ttl=config_instance.cache_ttl,  # 1 hour
)

# Minimum number of vectors needed for IVF index
MIN_VECTORS_FOR_IVF = 100

class FAISSClient:
    """High-performance vector similarity search using FAISS."""

    def __init__(
        self,
        dimension: int = config_instance.vector_dimension,
        index_type: str = "IVFFlat",  # IVFFlat for better speed/accuracy trade-off
        nlist: int = 10,  # Reduced number of clusters for IVF
        index_path: Optional[str] = None,
    ):
        """Initialize FAISS client.

        Args:
            dimension: Vector dimension
            index_type: FAISS index type (IVFFlat, Flat, etc.)
            nlist: Number of clusters for IVF indices
            index_path: Path to load/save index
        """
        self.dimension = dimension
        self.index_type = index_type
        self.nlist = min(nlist, dimension)  # Ensure nlist isn't larger than dimension
        self.index_path = index_path or os.path.join(
            config_instance.cache_dir, "faiss_index.bin"
        )
        
        # Initialize index
        self._initialize_index()
        
        # Load existing index if available
        if os.path.exists(self.index_path):
            self._load_index()

    def _initialize_index(self):
        """Initialize FAISS index based on configuration."""
        try:
            # Always start with a flat index for small datasets
            base_index = faiss.IndexFlatL2(self.dimension)
            # Wrap with IDMap to support custom ids
            self.index = faiss.IndexIDMap(base_index)
            self.needs_training = False
            self.is_flat = True
            
            logger.info(f"Initialized FAISS flat index with ID mapping")
            
        except Exception as e:
            logger.error(f"Failed to initialize FAISS index: {e}")
            raise

    def _maybe_convert_to_ivf(self, vectors: np.ndarray):
        """Convert to IVF index if we have enough vectors."""
        if (
            self.index_type == "IVFFlat" 
            and self.is_flat 
            and len(vectors) >= MIN_VECTORS_FOR_IVF
        ):
            try:
                # Create IVF index
                nlist = min(self.nlist, len(vectors) // 10)  # Use 10% of vectors as clusters
                quantizer = faiss.IndexFlatL2(self.dimension)
                base_index = faiss.IndexIVFFlat(
                    quantizer, self.dimension, nlist, faiss.METRIC_L2
                )
                
                # Train on existing vectors
                base_index.train(vectors)
                
                # Wrap with IDMap
                new_index = faiss.IndexIDMap(base_index)
                
                # Add existing vectors if any
                if self.index.ntotal > 0:
                    # Get existing vectors and ids
                    existing_ids = []
                    existing_vectors = []
                    for i in range(self.index.ntotal):
                        vector = faiss.vector_to_array(self.index.index.get_vector(i))
                        id = self.index.id_map[i]
                        existing_vectors.append(vector)
                        existing_ids.append(id)
                    
                    # Add to new index
                    if existing_vectors:
                        new_index.add_with_ids(
                            np.array(existing_vectors), 
                            np.array(existing_ids)
                        )
                
                # Replace index
                self.index = new_index
                self.is_flat = False
                logger.info(f"Converted to IVF index with {nlist} clusters")
                
            except Exception as e:
                logger.error(f"Failed to convert to IVF index: {e}")
                # Continue with flat index if conversion fails
                pass

    def _load_index(self):
        """Load index from disk."""
        try:
            self.index = faiss.read_index(self.index_path)
            # Check if base index is flat
            base_index = (
                self.index.index if isinstance(self.index, faiss.IndexIDMap)
                else self.index
            )
            self.is_flat = isinstance(base_index, faiss.IndexFlat)
            logger.info(f"Loaded FAISS index from {self.index_path}")
        except Exception as e:
            logger.error(f"Failed to load FAISS index: {e}")
            # Initialize new index if load fails
            self._initialize_index()

    def _save_index(self):
        """Save index to disk."""
        try:
            faiss.write_index(self.index, self.index_path)
            logger.info(f"Saved FAISS index to {self.index_path}")
        except Exception as e:
            logger.error(f"Failed to save FAISS index: {e}")

    def _verify_connection(self) -> Dict[str, Any]:
        """Verify index is ready and return stats.

        Returns:
            Dictionary with index statistics
        """
        try:
            return {
                "total_vector_count": self.index.ntotal,
                "dimension": self.dimension,
                "index_type": "Flat" if self.is_flat else "IVFFlat",
            }
        except Exception as e:
            logger.error(f"Failed to verify FAISS index: {e}")
            raise

    def upsert_vectors(self, vectors: List[Dict], batch_size: int = 100) -> int:
        """Upsert vectors in batches.

        Args:
            vectors: List of vector entries to upsert
            batch_size: Size of batches for upserting

        Returns:
            Number of vectors successfully upserted
        """
        total_upserted = 0
        current_batch = []
        current_ids = []
        current_metadata = {}

        try:
            for vector in vectors:
                vector_id = vector["id"]
                vector_values = np.array(vector["values"], dtype=np.float32)
                
                # Store metadata separately
                metadata_cache.set(vector_id, vector.get("metadata", {}))
                
                current_batch.append(vector_values)
                current_ids.append(hash(vector_id) % (2**63 - 1))  # Convert string ID to int64
                current_metadata[vector_id] = vector.get("metadata", {})

                if len(current_batch) >= batch_size:
                    # Convert batch to numpy array
                    batch_array = np.array(current_batch)
                    
                    # Check if we should convert to IVF
                    if total_upserted == 0:
                        self._maybe_convert_to_ivf(batch_array)
                    
                    # Add vectors to index
                    if not self.is_flat and not self.index.index.is_trained:
                        self.index.index.train(batch_array)
                    
                    self.index.add_with_ids(batch_array, np.array(current_ids))
                    total_upserted += len(current_batch)
                    
                    # Save index periodically
                    if total_upserted % 1000 == 0:
                        self._save_index()
                    
                    logger.info(
                        f"Upserted batch of {len(current_batch)} vectors. Total: {total_upserted}"
                    )
                    
                    current_batch = []
                    current_ids = []

            # Handle remaining vectors
            if current_batch:
                batch_array = np.array(current_batch)
                
                # Check if we should convert to IVF
                if total_upserted == 0:
                    self._maybe_convert_to_ivf(batch_array)
                
                # Add vectors to index
                if not self.is_flat and not self.index.index.is_trained:
                    self.index.index.train(batch_array)
                
                self.index.add_with_ids(batch_array, np.array(current_ids))
                total_upserted += len(current_batch)
                self._save_index()
                
                logger.info(
                    f"Upserted final batch of {len(current_batch)} vectors. Total: {total_upserted}"
                )

            return total_upserted

        except Exception as e:
            logger.error(f"Failed to upsert vectors: {e}")
            return total_upserted

    def search_vectors(
        self,
        query_vector: List[float],
        top_k: int = 10,
        filters: Optional[Dict] = None,
        include_metadata: bool = True,
    ) -> Dict[str, Any]:
        """Search for similar vectors.

        Args:
            query_vector: Query vector
            top_k: Number of results to return
            filters: Optional filters to apply (not supported in basic FAISS)
            include_metadata: Whether to include metadata

        Returns:
            Search results with scores and metadata
        """
        try:
            # Convert query to numpy array
            query_array = np.array([query_vector], dtype=np.float32)
            
            # Perform search
            distances, indices = self.index.search(query_array, top_k)
            
            # Process results
            matches = []
            for i, (distance, idx) in enumerate(zip(distances[0], indices[0])):
                if idx == -1:  # FAISS returns -1 for empty slots
                    continue
                    
                # Convert score to similarity (FAISS returns L2 distance)
                score = 1 / (1 + distance)
                
                # Get metadata if requested
                metadata = {}
                if include_metadata:
                    vector_id = str(idx)  # Convert back to string
                    metadata = metadata_cache.get(vector_id) or {}
                
                matches.append({
                    "id": str(idx),
                    "score": float(score),
                    "metadata": metadata,
                })
            
            return {"matches": matches}

        except Exception as e:
            logger.error(f"Search failed: {e}")
            return {"matches": []}

    def delete_all(self) -> bool:
        """Delete all vectors from the index.

        Returns:
            True if successful, False otherwise
        """
        try:
            # Reset index
            self._initialize_index()
            
            # Clear metadata cache
            metadata_cache.clear()
            
            # Save empty index
            self._save_index()
            
            return True
        except Exception as e:
            logger.error(f"Failed to delete all vectors: {e}")
            return False 