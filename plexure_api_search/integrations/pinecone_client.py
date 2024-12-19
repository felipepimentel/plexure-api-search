"""Pinecone client for vector storage."""

import logging
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pinecone

from ..config import config_instance

logger = logging.getLogger(__name__)

class PineconeClient:
    """Pinecone client for vector storage."""

    def __init__(self):
        """Initialize Pinecone client."""
        self.initialized = False
        self.index = None

    def initialize(self) -> None:
        """Initialize Pinecone client."""
        if self.initialized:
            return

        try:
            # Initialize Pinecone
            pinecone.init(
                api_key=config_instance.pinecone.api_key,
                environment=config_instance.pinecone.environment,
            )

            # Get or create index
            if config_instance.pinecone.index_name not in pinecone.list_indexes():
                pinecone.create_index(
                    name=config_instance.pinecone.index_name,
                    dimension=config_instance.vectors.dimension,
                    metric=config_instance.vectors.metric,
                )

            # Connect to index
            self.index = pinecone.Index(config_instance.pinecone.index_name)
            self.initialized = True
            logger.info("Pinecone client initialized")

        except Exception as e:
            logger.error(f"Failed to initialize Pinecone client: {e}")
            raise

    def cleanup(self) -> None:
        """Clean up Pinecone client."""
        if self.index:
            self.index = None
        self.initialized = False
        logger.info("Pinecone client cleaned up")

    def upsert(
        self,
        vectors: List[np.ndarray],
        metadata: List[Dict],
        namespace: Optional[str] = None,
    ) -> None:
        """Upsert vectors to Pinecone.
        
        Args:
            vectors: List of vectors to upsert
            metadata: List of metadata for each vector
            namespace: Optional namespace
        """
        if not self.initialized:
            self.initialize()

        try:
            # Convert vectors to list
            vectors_list = [v.tolist() for v in vectors]

            # Create vector IDs
            ids = [str(i) for i in range(len(vectors))]

            # Upsert vectors
            self.index.upsert(
                vectors=list(zip(ids, vectors_list, metadata)),
                namespace=namespace,
            )
            logger.debug(f"Upserted {len(vectors)} vectors")

        except Exception as e:
            logger.error(f"Failed to upsert vectors: {e}")
            raise

    def search(
        self,
        query_vector: np.ndarray,
        top_k: int = 10,
        namespace: Optional[str] = None,
    ) -> List[Tuple[str, float, Dict]]:
        """Search for similar vectors in Pinecone.
        
        Args:
            query_vector: Query vector
            top_k: Number of results to return
            namespace: Optional namespace
            
        Returns:
            List of (id, score, metadata) tuples
        """
        if not self.initialized:
            self.initialize()

        try:
            # Convert vector to list
            vector_list = query_vector.tolist()

            # Search index
            results = self.index.query(
                vector=vector_list,
                top_k=top_k,
                namespace=namespace,
                include_metadata=True,
            )

            # Extract matches
            matches = []
            for match in results.matches:
                matches.append(
                    (match.id, match.score, match.metadata)
                )

            return matches

        except Exception as e:
            logger.error(f"Failed to search vectors: {e}")
            raise

    def delete(
        self,
        ids: Optional[List[str]] = None,
        namespace: Optional[str] = None,
    ) -> None:
        """Delete vectors from Pinecone.
        
        Args:
            ids: List of vector IDs to delete
            namespace: Optional namespace
        """
        if not self.initialized:
            self.initialize()

        try:
            # Delete vectors
            self.index.delete(
                ids=ids,
                namespace=namespace,
            )
            logger.debug(f"Deleted {len(ids) if ids else 'all'} vectors")

        except Exception as e:
            logger.error(f"Failed to delete vectors: {e}")
            raise

# Global instance
pinecone_instance = PineconeClient()
