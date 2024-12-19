"""Model service for managing embeddings and models."""

import logging
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
from sentence_transformers import SentenceTransformer
from dependency_injector.wiring import inject
from dependency_injector.providers import Provider

from ..config import config_instance
from ..monitoring.metrics import MetricsManager
from .base import BaseService

logger = logging.getLogger(__name__)

class ModelService(BaseService):
    """Model service for managing embeddings and models."""

    def __init__(self):
        """Initialize model service."""
        super().__init__()
        self.models: Dict[str, SentenceTransformer] = {}
        self.metrics = MetricsManager()
        self.initialized = False
        self.embedding_dim = config_instance.vectors.dimension

    def initialize(self) -> None:
        """Initialize model service."""
        if self.initialized:
            return

        try:
            # Load models
            self.models["bi_encoder"] = self._load_model(config_instance.bi_encoder_model)
            self.models["bi_encoder_fallback"] = self._load_model(config_instance.bi_encoder_fallback)
            self.models["multilingual"] = self._load_model(config_instance.multilingual_model)

            self.initialized = True
            logger.info("Model service initialized")

        except Exception as e:
            logger.error(f"Failed to initialize model service: {e}")
            raise

    def cleanup(self) -> None:
        """Clean up model service."""
        self.models.clear()
        self.initialized = False
        logger.info("Model service cleaned up")

    def health_check(self) -> Dict[str, bool]:
        """Check service health.
        
        Returns:
            Health check results
        """
        return {
            "initialized": self.initialized,
            "models_loaded": len(self.models) > 0,
        }

    def encode(
        self,
        texts: Union[str, List[str]],
        model_name: str = "bi_encoder",
        normalize: bool = True,
    ) -> np.ndarray:
        """Encode texts using specified model.
        
        Args:
            texts: Text or list of texts to encode
            model_name: Name of model to use
            normalize: Whether to normalize vectors
            
        Returns:
            Text embeddings
        """
        if not self.initialized:
            self.initialize()

        try:
            # Get model
            model = self.models.get(model_name)
            if model is None:
                logger.warning(f"Model {model_name} not found, using fallback")
                model = self.models["bi_encoder_fallback"]

            # Convert single text to list
            if isinstance(texts, str):
                texts = [texts]

            # Clean and validate texts
            cleaned_texts = []
            for text in texts:
                if not isinstance(text, str):
                    text = str(text)
                if not text.strip():
                    text = "empty"
                cleaned_texts.append(text)

            logger.debug(f"Number of texts to encode: {len(cleaned_texts)}")
            logger.debug(f"Sample text: {cleaned_texts[0] if cleaned_texts else 'No texts'}")

            # Encode texts
            start_time = self.metrics.start_timer()
            embeddings = model.encode(
                cleaned_texts,
                normalize_embeddings=False,
                show_progress_bar=False,
                convert_to_numpy=True,
            )

            # Convert to numpy array if needed
            if not isinstance(embeddings, np.ndarray):
                embeddings = np.array(embeddings)

            # Ensure 2D array
            if len(embeddings.shape) == 1:
                embeddings = np.array([embeddings])

            logger.debug(f"Raw embeddings shape: {embeddings.shape}")
            logger.debug(f"Raw embeddings type: {embeddings.dtype}")

            # Resize if needed
            if embeddings.shape[1] != self.embedding_dim:
                logger.warning(f"Resizing embeddings from {embeddings.shape[1]} to {self.embedding_dim}")
                resized = np.zeros((len(embeddings), self.embedding_dim), dtype=np.float32)
                min_dim = min(embeddings.shape[1], self.embedding_dim)
                resized[:, :min_dim] = embeddings[:, :min_dim]
                embeddings = resized

            logger.debug(f"Resized embeddings shape: {embeddings.shape}")
            logger.debug(f"Resized embeddings type: {embeddings.dtype}")

            # Ensure float32
            embeddings = embeddings.astype(np.float32)

            # Normalize if requested
            if normalize:
                norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
                norms[norms == 0] = 1
                embeddings = embeddings / norms

            # Ensure contiguous array
            embeddings = np.ascontiguousarray(embeddings)

            logger.debug(f"Final embeddings shape: {embeddings.shape}")
            logger.debug(f"Final embeddings type: {embeddings.dtype}")
            logger.debug(f"Final embeddings sample: {embeddings[0][:5] if len(embeddings) > 0 else 'No embeddings'}")
            logger.debug(f"Final embeddings min: {np.min(embeddings)}")
            logger.debug(f"Final embeddings max: {np.max(embeddings)}")
            logger.debug(f"Final embeddings mean: {np.mean(embeddings)}")
            logger.debug(f"Final embeddings is contiguous: {embeddings.flags['C_CONTIGUOUS']}")

            self.metrics.stop_timer(
                start_time,
                "model_encode",
                {"model": model_name},
            )

            return embeddings

        except Exception as e:
            logger.error(f"Failed to encode texts: {e}")
            self.metrics.increment("model_errors", {"model": model_name})
            raise

    def _load_model(self, model_name: str) -> SentenceTransformer:
        """Load model from disk or download.
        
        Args:
            model_name: Name of model to load
            
        Returns:
            Loaded model
        """
        try:
            # Load model
            start_time = self.metrics.start_timer()
            model = SentenceTransformer(model_name)
            self.metrics.stop_timer(
                start_time,
                "model_load",
                {"model": model_name},
            )

            return model

        except Exception as e:
            logger.error(f"Failed to load model {model_name}: {e}")
            self.metrics.increment("model_errors", {"model": model_name})
            raise

# Global instance
model_service = ModelService() 