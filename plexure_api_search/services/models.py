"""Model service for text embeddings."""

import logging
from typing import List, Union

import numpy as np
from sentence_transformers import SentenceTransformer
from dependency_injector.wiring import inject

from ..config import config_instance
from ..monitoring.metrics import MetricsManager

logger = logging.getLogger(__name__)

class ModelService:
    """Model service for text embeddings."""

    def __init__(self):
        """Initialize model service."""
        self.metrics = MetricsManager()
        self.model = None
        self.dimension = 768  # Default dimension for all-MiniLM-L6-v2
        self.initialized = False

    def initialize(self) -> None:
        """Initialize model service."""
        if self.initialized:
            return

        try:
            # Load model
            model_name = config_instance.model_name
            logger.info(f"Loading model: {model_name}")
            self.model = SentenceTransformer(model_name)
            
            # Get embedding dimension
            self.dimension = self.model.get_sentence_embedding_dimension()
            logger.info(f"Model loaded with embedding dimension: {self.dimension}")

            self.initialized = True
            logger.info("Model service initialized")

        except Exception as e:
            logger.error(f"Failed to initialize model service: {e}")
            raise

    def cleanup(self) -> None:
        """Clean up model service."""
        if self.model is not None:
            self.model = None
            self.initialized = False
            logger.info("Model service cleaned up")

    def encode(
        self,
        texts: Union[str, List[str]],
        batch_size: int = 32,
        normalize: bool = True,
    ) -> np.ndarray:
        """Encode texts to embeddings.
        
        Args:
            texts: Text or list of texts to encode
            batch_size: Batch size for encoding
            normalize: Whether to L2 normalize embeddings
            
        Returns:
            Array of embeddings (n_texts x dimension)
        """
        if not self.initialized:
            self.initialize()

        try:
            # Convert single text to list
            if isinstance(texts, str):
                texts = [texts]

            # Log encoding info
            logger.debug(f"Encoding {len(texts)} texts")
            logger.debug(f"Sample text: {texts[0][:100]}...")

            # Encode texts
            embeddings = self.model.encode(
                texts,
                batch_size=batch_size,
                normalize_embeddings=normalize,
                show_progress_bar=False,
            )

            # Convert to numpy array
            embeddings = np.array(embeddings)

            # Log embedding info
            logger.debug(f"Generated embeddings shape: {embeddings.shape}")
            logger.debug(f"Embeddings type: {embeddings.dtype}")
            if len(embeddings) > 0:
                logger.debug(f"First embedding norm: {np.linalg.norm(embeddings[0])}")

            # Update metrics
            self.metrics.increment(
                "texts_encoded",
                {"count": len(texts), "batch_size": batch_size},
            )

            return embeddings

        except Exception as e:
            logger.error(f"Failed to encode texts: {e}")
            self.metrics.increment("encoding_errors")
            raise

# Global instance
model_service = ModelService() 