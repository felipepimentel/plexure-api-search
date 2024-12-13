"""Centralized Pinecone client for vector operations."""

import logging
import time
from typing import Any, Dict, List, Optional

from pinecone import Pinecone, ServerlessSpec

logger = logging.getLogger(__name__)


class PineconeClient:
    """Centralized client for Pinecone operations."""

    def __init__(
        self,
        api_key: str,
        environment: str,
        index_name: str,
        dimension: int = 384,
        metric: str = "dotproduct",
        cloud: str = "aws",
        region: str = "us-east-1",
    ):
        """Initialize Pinecone client.

        Args:
            api_key: Pinecone API key
            environment: Pinecone environment
            index_name: Name of the index to use
            dimension: Vector dimension (default: 384)
            metric: Distance metric (default: dotproduct)
            cloud: Cloud provider (default: aws)
            region: Cloud region (default: us-east-1)
        """
        if not api_key:
            raise ValueError("Pinecone API key is required")
        if not index_name:
            raise ValueError("Index name is required")
        if not environment:
            raise ValueError("Environment is required")

        self.index_name = index_name
        self.dimension = dimension
        self.metric = metric

        # Initialize Pinecone
        try:
            pc = Pinecone(api_key=api_key)

            # Check if index exists
            existing_indexes = [index.name for index in pc.list_indexes()]

            if index_name not in existing_indexes:
                logger.info(f"Creating new index: {index_name}")
                pc.create_index(
                    name=index_name,
                    dimension=dimension,
                    metric=metric,
                    spec=ServerlessSpec(cloud=cloud, region=region),
                )
                time.sleep(2)  # Wait for index creation

            # Get index instance
            self.index = pc.Index(index_name)

            # Verify connection and get stats
            self._verify_connection()

        except Exception as e:
            raise RuntimeError(f"Failed to initialize Pinecone: {str(e)}")

    def _verify_connection(self) -> Dict:
        """Verify connection and get index stats."""
        try:
            stats = self.index.describe_index_stats()
            logger.info("\nIndex stats:")
            logger.info(f"Total vectors: {stats.get('total_vector_count', 0)}")
            logger.info(f"Dimension: {stats.get('dimension', self.dimension)}")
            return stats
        except Exception as e:
            logger.error(f"Could not get index stats: {e}")
            return {}

    def upsert_vectors(self, vectors: List[Dict], batch_size: int = 10) -> int:
        """Upsert vectors in batches.

        Args:
            vectors: List of vector entries to upsert
            batch_size: Size of batches for upserting

        Returns:
            Number of vectors successfully upserted
        """
        total_upserted = 0
        current_batch = []

        for vector in vectors:
            current_batch.append(vector)

            if len(current_batch) >= batch_size:
                try:
                    self.index.upsert(vectors=current_batch)
                    total_upserted += len(current_batch)
                    logger.info(
                        f"Upserted batch of {len(current_batch)} vectors. Total: {total_upserted}"
                    )
                except Exception as e:
                    logger.error(f"Batch upsert failed: {e}")
                    # Try one by one
                    for v in current_batch:
                        try:
                            self.index.upsert(vectors=[v])
                            total_upserted += 1
                        except Exception as e:
                            logger.error(f"Single vector upsert failed: {e}")
                current_batch = []

        # Upsert remaining vectors
        if current_batch:
            try:
                self.index.upsert(vectors=current_batch)
                total_upserted += len(current_batch)
                logger.info(
                    f"Upserted final batch of {len(current_batch)} vectors. Total: {total_upserted}"
                )
            except Exception as e:
                logger.error(f"Final batch upsert failed: {e}")
                # Try one by one
                for v in current_batch:
                    try:
                        self.index.upsert(vectors=[v])
                        total_upserted += 1
                    except Exception as e:
                        logger.error(f"Single vector upsert failed: {e}")

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
            filters: Optional filters to apply
            include_metadata: Whether to include metadata

        Returns:
            Search results from Pinecone
        """
        print("include_metadata", include_metadata)
        try:
            results = self.index.query(
                vector=query_vector,
                top_k=top_k,
                include_metadata=include_metadata,
                filter=filters,
            )
            return results
        except Exception as e:
            logger.error(f"Search failed: {e}")
            raise

    def delete_all(self) -> bool:
        """Delete all vectors from the index.

        Returns:
            True if successful, False otherwise
        """
        try:
            self.index.delete(delete_all=True)
            time.sleep(2)  # Wait for deletion
            stats = self._verify_connection()
            return stats.get("total_vector_count", 0) == 0
        except Exception as e:
            logger.error(f"Failed to delete all vectors: {e}")
            return False

    def get_vector_count(self) -> int:
        """Get total number of vectors in the index.

        Returns:
            Total vector count
        """
        try:
            stats = self._verify_connection()
            return stats.get("total_vector_count", 0)
        except Exception as e:
            logger.error(f"Failed to get vector count: {e}")
            return 0
