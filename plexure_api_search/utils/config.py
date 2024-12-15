"""Configuration module for the API search engine."""

import os
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv


class Config:
    """Singleton configuration class for the API search engine."""

    _instance: Optional["Config"] = None

    def __new__(cls) -> "Config":
        """Ensure only one instance is created."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        """Initialize configuration if not already initialized."""
        if self._initialized:
            return

        # Load environment variables
        load_dotenv()

        # API directory configuration
        self.api_dir = os.getenv("API_DIR", "assets/apis")

        # Pinecone configuration
        self.pinecone_api_key = os.getenv("PINECONE_API_KEY")
        self.pinecone_index = os.getenv("PINECONE_INDEX_NAME")
        self.pinecone_cloud = os.getenv("PINECONE_CLOUD", "aws")
        self.pinecone_region = os.getenv("PINECONE_REGION", "us-east-1")
        self.pinecone_dimension = int(os.getenv("PINECONE_DIMENSION", "384"))
        self.pinecone_metric = os.getenv("PINECONE_METRIC", "dotproduct")
        # Construct Pinecone environment
        self.pinecone_environment = (
            f"{self.pinecone_region}"  # Pinecone environment is just the region for AWS
        )

        # OpenRouter configuration
        self.openrouter_api_key = os.getenv("OPENROUTER_API_KEY")

        # Model configuration
        self.bi_encoder_model = os.getenv("BI_ENCODER_MODEL", "all-MiniLM-L6-v2")
        self.cross_encoder_model = os.getenv(
            "CROSS_ENCODER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2"
        )
        self.vector_dimension = (
            self.pinecone_dimension
        )  # Use same dimension as Pinecone
        self.pca_components = int(os.getenv("PCA_COMPONENTS", "128"))

        # Cache configuration
        self.cache_ttl = int(os.getenv("CACHE_TTL", "3600"))  # 1 hour default
        self.embedding_cache_ttl = int(
            os.getenv("EMBEDDING_CACHE_TTL", "86400")
        )  # 24 hours default

        # Directory configurations
        self.cache_dir = Path("data/.cache")
        self.metrics_dir = Path("data/.metrics")
        self.health_dir = Path("data/.health")

        # Create directories
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.metrics_dir.mkdir(parents=True, exist_ok=True)
        self.health_dir.mkdir(parents=True, exist_ok=True)

        # Validate required configuration
        if not self.pinecone_api_key:
            raise ValueError("PINECONE_API_KEY is required")
        if not self.pinecone_index:
            raise ValueError("PINECONE_INDEX_NAME is required")

        self._initialized = True

    @classmethod
    def get_instance(cls) -> "Config":
        """Get the singleton instance of the configuration."""
        return cls()


config_instance = Config.get_instance()

__all__ = ["config_instance"]
