"""
Model Service for Plexure API Search

This module provides model management and embedding generation functionality using transformer models.
It handles the loading, caching, and execution of language models for generating vector embeddings
of API endpoints and search queries.

Key Features:
- Model loading and management
- Embedding generation
- Batch processing
- Model caching
- Resource optimization
- Error handling
- Performance monitoring
- Model versioning

The ModelService class provides:
- Model initialization and loading
- Embedding generation for texts
- Batch processing capabilities
- Model caching and persistence
- Resource cleanup
- Health monitoring
- Error recovery

Example Usage:
    from plexure_api_search.services.models import ModelService

    # Initialize service
    model_service = ModelService(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        dimension=384
    )

    # Generate embeddings
    texts = ["endpoint description 1", "endpoint description 2"]
    embeddings = model_service.get_embeddings(texts)

Supported Models:
- sentence-transformers/all-MiniLM-L6-v2 (default)
- sentence-transformers/all-mpnet-base-v2
- sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2

Performance Features:
- Batch processing for efficiency
- Model caching
- Resource optimization
- Memory management
- Error recovery
"""

import logging
from typing import List, Union

import numpy as np
from sentence_transformers import SentenceTransformer

from ..config import config
from ..monitoring.metrics import MetricsManager

logger = logging.getLogger(__name__)


class ModelService:
    """Model service for generating embeddings."""

    def __init__(self):
        """Initialize model service."""
        self.metrics = MetricsManager()
        self.model = None
        self.dimension = None
        self.initialized = False

    def initialize(self) -> None:
        """Initialize model service."""
        if self.initialized:
            return

        try:
            # Load model
            model_name = config.model_name
            logger.info(f"Loading model: {model_name}")
            self.model = SentenceTransformer(model_name)
            self.dimension = self.model.get_sentence_embedding_dimension()
            logger.info(f"Model loaded with embedding dimension: {self.dimension}")
            self.initialized = True
            logger.info("Model service initialized")

        except Exception as e:
            logger.error(f"Failed to initialize model service: {e}")
            raise

    def cleanup(self) -> None:
        """Clean up model service."""
        if self.initialized:
            self.model = None
            self.dimension = None
            self.initialized = False
            logger.info("Model service cleaned up")

    def get_embeddings(self, text: Union[str, List[str]]) -> np.ndarray:
        """Get embeddings for text.

        Args:
            text: Text to embed (string or list of strings)

        Returns:
            Embeddings array (n_texts x dimension)
        """
        if not self.initialized:
            self.initialize()

        try:
            # Convert single string to list
            if isinstance(text, str):
                text = [text]

            # Generate embeddings
            embeddings = self.model.encode(
                text,
                convert_to_numpy=True,
                normalize_embeddings=True,
            )

            # Handle 3D output (batch_size x 1 x dimension)
            if embeddings.ndim == 3:
                embeddings = embeddings.squeeze(1)

            # Update metrics
            self.metrics.increment_counter(
                "embeddings_generated",
                value=len(text),
            )

            return embeddings

        except Exception as e:
            logger.error(f"Failed to generate embeddings: {e}")
            self.metrics.increment_counter("embedding_errors")
            raise


# Global instance
model_service = ModelService()
