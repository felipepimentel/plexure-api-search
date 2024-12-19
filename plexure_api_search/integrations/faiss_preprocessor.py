"""FAISS preprocessor for vector operations."""

import logging
from typing import List, Optional, Tuple, Dict, Any, Union

import faiss
import numpy as np

from ..config import config_instance

logger = logging.getLogger(__name__)


class FAISSPreprocessor:
    """FAISS preprocessor for vector operations."""

    def __init__(self):
        """Initialize FAISS preprocessor."""
        self.dimension = None  # Will be set during initialization
        self.index = None
        self.initialized = False

    def initialize(self) -> None:
        """Initialize FAISS index."""
        if self.initialized:
            return

        try:
            # Get dimension from config or wait for first vector
            self.dimension = config_instance.vectors.dimension
            if self.dimension:
                # Create index with configured dimension
                self.index = faiss.IndexFlatL2(self.dimension)
                self.initialized = True
                logger.info("FAISS preprocessor initialized")
            else:
                logger.info("FAISS preprocessor waiting for first vector to determine dimension")

        except Exception as e:
            logger.error(f"Failed to initialize FAISS preprocessor: {e}")
            raise

    def cleanup(self) -> None:
        """Clean up FAISS preprocessor."""
        if self.index:
            self.index = None
        self.initialized = False
        logger.info("FAISS preprocessor cleaned up")

    def add(self, vectors: List[np.ndarray]) -> None:
        """Add vectors to index.
        
        Args:
            vectors: List of vectors to add
        """
        if not vectors:
            return

        try:
            # Initialize index if needed
            if not self.initialized:
                # Get dimension from first vector
                first_vector = vectors[0]
                self.dimension = first_vector.shape[0]
                self.index = faiss.IndexFlatL2(self.dimension)
                self.initialized = True
                logger.info(f"FAISS preprocessor initialized with dimension {self.dimension}")

            # Convert to numpy array
            vectors_array = np.array(vectors).astype('float32')

            # Verify dimensions
            if vectors_array.shape[1] != self.dimension:
                raise ValueError(f"Vector dimension mismatch: expected {self.dimension}, got {vectors_array.shape[1]}")

            # Add to index
            self.index.add(vectors_array)
            logger.debug(f"Added {len(vectors)} vectors to index")

        except Exception as e:
            logger.error(f"Failed to add vectors: {e}")
            raise

    def search(
        self,
        query_vector: np.ndarray,
        k: int = 10,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Search for similar vectors.
        
        Args:
            query_vector: Query vector
            k: Number of results to return
            
        Returns:
            Tuple of (distances, indices)
        """
        if not self.initialized:
            raise ValueError("FAISS preprocessor not initialized")

        try:
            # Verify dimension
            if query_vector.shape[0] != self.dimension:
                raise ValueError(f"Query vector dimension mismatch: expected {self.dimension}, got {query_vector.shape[0]}")

            # Convert to numpy array
            query_array = np.array([query_vector]).astype('float32')

            # Search index
            distances, indices = self.index.search(query_array, k)
            return distances[0], indices[0]

        except Exception as e:
            logger.error(f"Failed to search vectors: {e}")
            raise

    def preprocess_vectors(
        self,
        vectors: List[Dict[str, np.ndarray]],
    ) -> np.ndarray:
        """Preprocess vectors for FAISS.
        
        Args:
            vectors: List of vector dictionaries with semantic, lexical, and contextual embeddings
            
        Returns:
            Preprocessed vectors as numpy array
        """
        try:
            if not vectors:
                return np.array([])

            # Get dimension from first vector if not initialized
            if not self.dimension:
                first_vector = vectors[0]
                single_dim = first_vector["semantic"].shape[0]
                self.dimension = single_dim * 3  # Combined dimension for all three embeddings

            # Convert list of dictionaries to numpy array
            preprocessed = []
            for vector_dict in vectors:
                # Verify dimensions
                semantic_dim = vector_dict["semantic"].shape[0]
                lexical_dim = vector_dict["lexical"].shape[0]
                contextual_dim = vector_dict["contextual"].shape[0]

                if semantic_dim != lexical_dim or semantic_dim != contextual_dim:
                    raise ValueError(f"Inconsistent embedding dimensions: semantic={semantic_dim}, lexical={lexical_dim}, contextual={contextual_dim}")

                # Concatenate semantic, lexical and contextual vectors
                combined = np.concatenate([
                    vector_dict["semantic"],
                    vector_dict["lexical"],
                    vector_dict["contextual"]
                ])

                # Verify final dimension
                if combined.shape[0] != self.dimension:
                    raise ValueError(f"Combined vector dimension mismatch: expected {self.dimension}, got {combined.shape[0]}")

                preprocessed.append(combined)
                
            return np.array(preprocessed).astype('float32')

        except Exception as e:
            logger.error(f"Failed to preprocess vectors: {e}")
            raise

    def postprocess_vectors(
        self,
        vectors: np.ndarray,
    ) -> List[Dict[str, np.ndarray]]:
        """Postprocess vectors from FAISS.
        
        Args:
            vectors: Preprocessed vectors from FAISS
            
        Returns:
            List of vector dictionaries
        """
        try:
            if not vectors.size:
                return []

            # Calculate single embedding dimension
            single_dim = self.dimension // 3

            # Split concatenated vectors back into components
            postprocessed = []
            for vector in vectors:
                semantic = vector[:single_dim]
                lexical = vector[single_dim:single_dim*2]
                contextual = vector[single_dim*2:single_dim*3]
                postprocessed.append({
                    "semantic": semantic,
                    "lexical": lexical,
                    "contextual": contextual
                })
            return postprocessed

        except Exception as e:
            logger.error(f"Failed to postprocess vectors: {e}")
            raise

# Global instance
preprocessor = FAISSPreprocessor()