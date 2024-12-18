"""Model service implementation."""

import hashlib
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Type, Union

import torch
from dependency_injector.wiring import inject, provider
from sentence_transformers import SentenceTransformer

from ..config import Config
from ..monitoring.events import Event, EventType
from ..monitoring.metrics import MetricsManager
from .base import BaseService, ServiceException
from .events import PublisherService

logger = logging.getLogger(__name__)


class ModelVersion:
    """Model version information."""

    def __init__(
        self,
        model_id: str,
        version: str,
        path: str,
        config: Dict[str, Any],
        created_at: datetime,
    ) -> None:
        """Initialize model version.

        Args:
            model_id: Model identifier
            version: Version string
            path: Model path
            config: Model configuration
            created_at: Creation timestamp
        """
        self.model_id = model_id
        self.version = version
        self.path = path
        self.config = config
        self.created_at = created_at
        self.hash = self._compute_hash()

    def _compute_hash(self) -> str:
        """Compute version hash.

        Returns:
            Version hash
        """
        content = f"{self.model_id}:{self.version}:{self.path}:{json.dumps(self.config)}"
        return hashlib.sha256(content.encode()).hexdigest()

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary.

        Returns:
            Dictionary representation
        """
        return {
            "model_id": self.model_id,
            "version": self.version,
            "path": self.path,
            "config": self.config,
            "created_at": self.created_at.isoformat(),
            "hash": self.hash,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ModelVersion":
        """Create from dictionary.

        Args:
            data: Dictionary data

        Returns:
            Model version instance
        """
        return cls(
            model_id=data["model_id"],
            version=data["version"],
            path=data["path"],
            config=data["config"],
            created_at=datetime.fromisoformat(data["created_at"]),
        )


class ModelService(BaseService[Dict[str, Any]]):
    """Model service implementation."""

    def __init__(
        self,
        config: Config,
        publisher: PublisherService,
        metrics_manager: MetricsManager,
    ) -> None:
        """Initialize model service.

        Args:
            config: Application configuration
            publisher: Event publisher
            metrics_manager: Metrics manager
        """
        super().__init__(config, publisher, metrics_manager)
        self.models: Dict[str, SentenceTransformer] = {}
        self.versions: Dict[str, ModelVersion] = {}
        self.fallbacks: Dict[str, List[str]] = {}
        self._initialized = False

    async def initialize(self) -> None:
        """Initialize service resources."""
        if self._initialized:
            return

        try:
            # Load model versions
            await self._load_versions()

            # Initialize primary models
            await self._init_primary_models()

            # Initialize fallback models
            await self._init_fallback_models()

            self._initialized = True
            self.publisher.publish(
                Event(
                    type=EventType.SERVICE_STARTED,
                    timestamp=datetime.now(),
                    component="model_service",
                    description="Model service initialized",
                    metadata={
                        "models": list(self.models.keys()),
                        "versions": {k: v.version for k, v in self.versions.items()},
                    },
                )
            )

        except Exception as e:
            logger.error(f"Failed to initialize model service: {e}")
            raise ServiceException(
                message="Failed to initialize model service",
                service_name="ModelService",
                error_code="INIT_FAILED",
                details={"error": str(e)},
            )

    async def cleanup(self) -> None:
        """Clean up service resources."""
        self.models.clear()
        self.versions.clear()
        self.fallbacks.clear()
        self._initialized = False
        self.publisher.publish(
            Event(
                type=EventType.SERVICE_STOPPED,
                timestamp=datetime.now(),
                component="model_service",
                description="Model service stopped",
            )
        )

    async def health_check(self) -> Dict[str, Any]:
        """Check service health.

        Returns:
            Health check results
        """
        return {
            "service": "ModelService",
            "initialized": self._initialized,
            "models": list(self.models.keys()),
            "versions": {k: v.version for k, v in self.versions.items()},
            "fallbacks": self.fallbacks,
        }

    async def _load_versions(self) -> None:
        """Load model versions from disk."""
        version_file = self.config.model_dir / "versions.json"
        if version_file.exists():
            try:
                data = json.loads(version_file.read_text())
                self.versions = {
                    k: ModelVersion.from_dict(v) for k, v in data.items()
                }
                logger.info(f"Loaded {len(self.versions)} model versions")
            except Exception as e:
                logger.error(f"Failed to load model versions: {e}")
                self.versions = {}

    async def _save_versions(self) -> None:
        """Save model versions to disk."""
        version_file = self.config.model_dir / "versions.json"
        try:
            data = {k: v.to_dict() for k, v in self.versions.items()}
            version_file.write_text(json.dumps(data, indent=2))
            logger.info(f"Saved {len(self.versions)} model versions")
        except Exception as e:
            logger.error(f"Failed to save model versions: {e}")

    async def _init_primary_models(self) -> None:
        """Initialize primary models."""
        # Initialize bi-encoder
        try:
            model_id = "bi_encoder"
            model = await self._load_model(
                self.config.bi_encoder_model,
                model_id,
            )
            self.models[model_id] = model
            logger.info(f"Initialized primary bi-encoder: {model_id}")
        except Exception as e:
            logger.error(f"Failed to initialize primary bi-encoder: {e}")
            raise

        # Initialize cross-encoder
        try:
            model_id = "cross_encoder"
            model = await self._load_model(
                self.config.cross_encoder_model,
                model_id,
            )
            self.models[model_id] = model
            logger.info(f"Initialized cross-encoder: {model_id}")
        except Exception as e:
            logger.error(f"Failed to initialize cross-encoder: {e}")
            raise

    async def _init_fallback_models(self) -> None:
        """Initialize fallback models."""
        # Initialize bi-encoder fallback
        try:
            model_id = "bi_encoder_fallback"
            model = await self._load_model(
                self.config.bi_encoder_fallback,
                model_id,
            )
            self.models[model_id] = model
            self.fallbacks["bi_encoder"] = ["bi_encoder_fallback"]
            logger.info(f"Initialized bi-encoder fallback: {model_id}")
        except Exception as e:
            logger.error(f"Failed to initialize bi-encoder fallback: {e}")

        # Initialize multilingual model
        if hasattr(self.config, "multilingual_model"):
            try:
                model_id = "multilingual"
                model = await self._load_model(
                    self.config.multilingual_model,
                    model_id,
                )
                self.models[model_id] = model
                self.fallbacks["bi_encoder"].append("multilingual")
                logger.info(f"Initialized multilingual model: {model_id}")
            except Exception as e:
                logger.error(f"Failed to initialize multilingual model: {e}")

    async def _load_model(
        self,
        model_path: str,
        model_id: str,
    ) -> SentenceTransformer:
        """Load model from path.

        Args:
            model_path: Model path
            model_id: Model identifier

        Returns:
            Loaded model
        """
        try:
            self.publisher.publish(
                Event(
                    type=EventType.MODEL_LOADING_STARTED,
                    timestamp=datetime.now(),
                    component="model_service",
                    description=f"Loading model: {model_id}",
                    metadata={"model_id": model_id, "path": model_path},
                )
            )

            # Load model
            model = SentenceTransformer(model_path)

            # Verify model
            test_input = ["test sentence"]
            embeddings = model.encode(
                test_input,
                convert_to_tensor=True,
                show_progress_bar=False,
            )
            if embeddings is None or embeddings.shape[0] != len(test_input):
                raise ValueError("Model verification failed")

            # Create version
            version = ModelVersion(
                model_id=model_id,
                version=model.__version__,
                path=model_path,
                config=model.get_config_dict(),
                created_at=datetime.now(),
            )
            self.versions[model_id] = version

            # Save versions
            await self._save_versions()

            self.publisher.publish(
                Event(
                    type=EventType.MODEL_LOADING_COMPLETED,
                    timestamp=datetime.now(),
                    component="model_service",
                    description=f"Model loaded successfully: {model_id}",
                    metadata={"model_id": model_id, "version": version.version},
                )
            )

            return model

        except Exception as e:
            self.publisher.publish(
                Event(
                    type=EventType.MODEL_LOADING_FAILED,
                    timestamp=datetime.now(),
                    component="model_service",
                    description=f"Failed to load model: {model_id}",
                    error=str(e),
                    metadata={"model_id": model_id, "path": model_path},
                )
            )
            raise ServiceException(
                message=f"Failed to load model: {model_id}",
                service_name="ModelService",
                error_code="MODEL_LOAD_FAILED",
                details={"error": str(e), "model_id": model_id},
            )

    async def get_model(
        self,
        model_id: str,
        use_fallback: bool = True,
    ) -> SentenceTransformer:
        """Get model by ID.

        Args:
            model_id: Model identifier
            use_fallback: Whether to try fallback models

        Returns:
            Model instance
        """
        if not self._initialized:
            await self.initialize()

        try:
            # Try primary model
            if model_id in self.models:
                return self.models[model_id]

            # Try fallbacks
            if use_fallback and model_id in self.fallbacks:
                for fallback_id in self.fallbacks[model_id]:
                    if fallback_id in self.models:
                        logger.warning(f"Using fallback model: {fallback_id}")
                        self.metrics.increment(
                            "model_fallbacks",
                            1,
                            {"model_id": model_id, "fallback_id": fallback_id},
                        )
                        return self.models[fallback_id]

            raise ServiceException(
                message=f"Model not found: {model_id}",
                service_name="ModelService",
                error_code="MODEL_NOT_FOUND",
                details={"model_id": model_id},
            )

        except Exception as e:
            logger.error(f"Failed to get model: {e}")
            self.metrics.increment("model_errors", 1, {"model_id": model_id})
            raise ServiceException(
                message=f"Failed to get model: {model_id}",
                service_name="ModelService",
                error_code="MODEL_ACCESS_FAILED",
                details={"error": str(e), "model_id": model_id},
            )

    async def encode(
        self,
        texts: Union[str, List[str]],
        model_id: str = "bi_encoder",
        use_fallback: bool = True,
        **kwargs: Any,
    ) -> torch.Tensor:
        """Encode texts using model.

        Args:
            texts: Input texts
            model_id: Model identifier
            use_fallback: Whether to try fallback models
            **kwargs: Additional encoding arguments

        Returns:
            Encoded vectors
        """
        if not self._initialized:
            await self.initialize()

        try:
            # Get model
            model = await self.get_model(model_id, use_fallback)

            # Encode texts
            start_time = datetime.now()
            embeddings = model.encode(
                texts,
                convert_to_tensor=True,
                show_progress_bar=False,
                **kwargs,
            )

            # Update metrics
            duration = (datetime.now() - start_time).total_seconds()
            self.metrics.observe(
                "encoding_duration",
                duration,
                {"model_id": model_id},
            )
            self.metrics.increment(
                "texts_encoded",
                len(texts) if isinstance(texts, list) else 1,
                {"model_id": model_id},
            )

            return embeddings

        except Exception as e:
            logger.error(f"Failed to encode texts: {e}")
            self.metrics.increment(
                "encoding_errors",
                1,
                {"model_id": model_id},
            )
            raise ServiceException(
                message="Failed to encode texts",
                service_name="ModelService",
                error_code="ENCODING_FAILED",
                details={"error": str(e), "model_id": model_id},
            )


# Create service instance
model_service = ModelService(Config(), PublisherService(Config(), MetricsManager()), MetricsManager())

__all__ = ["ModelService", "ModelVersion", "model_service"] 