"""Storage management for API vectors and metadata."""

import logging
from typing import Dict, List, Optional

from ..config.settings import settings
from ..storage.vector_stores.pinecone import vector_store
from ..utils.file import load_data, save_data
from .processor import processor
from .vectorizer import vectorizer

logger = logging.getLogger(__name__)


class APIStorage:
    """Storage manager for API vectors and metadata."""

    def __init__(self):
        """Initialize storage manager."""
        self.vector_store = vector_store

    def store_vectors(
        self,
        vectors: Optional[List[Dict]] = None,
        directory: Optional[str] = None,
        batch_size: int = 10,
        save_backup: bool = True,
    ) -> int:
        """Store vectors in vector database.

        Args:
            vectors: List of vectors to store (optional)
            directory: Directory to process API files from (optional)
            batch_size: Size of batches for storage
            save_backup: Whether to save backup of vectors

        Returns:
            Number of vectors stored
        """
        try:
            # Get vectors if not provided
            if vectors is None:
                vectors = vectorizer.vectorize_endpoints(directory=directory)

            # Save backup if requested
            if save_backup and vectors:
                backup_path = settings.paths.data_dir / "vectors_backup.json"
                if save_data({"vectors": vectors}, str(backup_path)):
                    logger.info(f"Saved vectors backup to {backup_path}")

            # Store vectors
            stored = self.vector_store.upsert(vectors, batch_size=batch_size)
            logger.info(f"Stored {stored} vectors in database")

            return stored

        except Exception as e:
            logger.error(f"Error storing vectors: {e}")
            return 0

    def load_backup(self, backup_path: Optional[str] = None) -> List[Dict]:
        """Load vectors from backup file.

        Args:
            backup_path: Path to backup file (optional)

        Returns:
            List of vector entries
        """
        try:
            if backup_path is None:
                backup_path = str(settings.paths.data_dir / "vectors_backup.json")

            data = load_data(backup_path)
            if data and "vectors" in data:
                vectors = data["vectors"]
                logger.info(f"Loaded {len(vectors)} vectors from backup")
                return vectors

            return []

        except Exception as e:
            logger.error(f"Error loading backup: {e}")
            return []

    def restore_from_backup(
        self,
        backup_path: Optional[str] = None,
        batch_size: int = 10,
    ) -> int:
        """Restore vectors from backup file.

        Args:
            backup_path: Path to backup file (optional)
            batch_size: Size of batches for storage

        Returns:
            Number of vectors restored
        """
        try:
            # Load vectors from backup
            vectors = self.load_backup(backup_path)
            if not vectors:
                return 0

            # Clear existing vectors
            if not self.vector_store.delete_all():
                logger.error("Failed to clear existing vectors")
                return 0

            # Store vectors from backup
            stored = self.vector_store.upsert(vectors, batch_size=batch_size)
            logger.info(f"Restored {stored} vectors from backup")

            return stored

        except Exception as e:
            logger.error(f"Error restoring from backup: {e}")
            return 0

    def get_stats(self) -> Dict:
        """Get storage statistics.

        Returns:
            Dictionary with storage statistics
        """
        try:
            vector_count = self.vector_store.get_vector_count()
            return {
                "vector_count": vector_count,
                "dimension": settings.model.embedding_dimension,
            }
        except Exception as e:
            logger.error(f"Error getting stats: {e}")
            return {}


# Global storage instance
storage = APIStorage()

__all__ = ["storage", "APIStorage"] 