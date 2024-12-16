"""Vector storage management with FAISS preprocessing and Pinecone storage."""

import logging
from typing import Any, Dict, List, Optional

from .faiss_preprocessor import FAISSPreprocessor
from .pinecone_client import PineconeClient

logger = logging.getLogger(__name__)


class VectorManager:
    """Manages vector preprocessing and storage."""

    def __init__(self):
        """Initialize vector manager."""
        self.preprocessor = FAISSPreprocessor()
        self.storage = PineconeClient()

    def upsert_vectors(self, vectors: List[Dict], batch_size: int = 100) -> int:
        """Preprocess and store vectors.

        Args:
            vectors: List of vector entries to store
            batch_size: Size of batches for processing

        Returns:
            Number of vectors successfully stored
        """
        try:
            # Preprocess vectors using FAISS
            optimized_vectors, preprocessing_meta = (
                self.preprocessor.preprocess_vectors(vectors, batch_size)
            )

            if not optimized_vectors:
                logger.warning("No vectors to store after preprocessing")
                return 0

            # Store preprocessed vectors in Pinecone
            stored_count = self.storage.upsert_vectors(optimized_vectors, batch_size)

            logger.info(
                f"Stored {stored_count} vectors with preprocessing: {preprocessing_meta}"
            )
            return stored_count

        except Exception as e:
            logger.error(f"Failed to store vectors: {e}")
            return 0

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
            filters: Optional filters to apply
            include_metadata: Whether to include metadata

        Returns:
            Search results with scores and metadata
        """
        try:
            # Optimize query vector
            optimized_query = self.preprocessor.optimize_query(query_vector)

            # Search in Pinecone
            results = self.storage.search_vectors(
                optimized_query,
                top_k=top_k,
                filters=filters,
                include_metadata=include_metadata,
            )

            return results

        except Exception as e:
            logger.error(f"Search failed: {e}")
            return {"matches": []}

    def delete_all(self) -> bool:
        """Delete all vectors and clear preprocessing index.

        Returns:
            True if successful, False otherwise
        """
        try:
            # Clear local preprocessing index
            self.preprocessor.clear_local_index()

            # Delete all vectors from storage
            return self.storage.delete_all()
        except Exception as e:
            logger.error(f"Failed to delete all vectors: {e}")
            return False
