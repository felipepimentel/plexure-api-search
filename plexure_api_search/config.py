"""Configuration management."""

import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

from dotenv import load_dotenv

logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()


@dataclass
class Config:
    """Application configuration."""

    # API Keys and Authentication
    huggingface_token: Optional[str] = os.getenv("HUGGINGFACE_TOKEN")

    # Directory Settings
    api_dir: Path = Path(os.getenv("API_DIR", "."))
    cache_dir: Path = Path(os.getenv("CACHE_DIR", ".cache"))
    health_dir: Path = Path(os.getenv("HEALTH_DIR", ".health"))

    # Pinecone Settings
    pinecone_api_key: str = os.getenv("PINECONE_API_KEY", "")
    pinecone_environment: str = os.getenv("PINECONE_ENVIRONMENT", "gcp-starter")
    pinecone_index_name: str = os.getenv("PINECONE_INDEX_NAME", "plexure-api-search")
    pinecone_cloud: str = os.getenv("PINECONE_CLOUD", "gcp")
    pinecone_region: str = os.getenv("PINECONE_REGION", "us-central1")

    # Embedding Models
    bi_encoder_model: str = os.getenv(
        "BI_ENCODER_MODEL", 
        "sentence-transformers/all-mpnet-base-v2"  # Best performing model for semantic search
    )
    bi_encoder_fallback: str = os.getenv(
        "BI_ENCODER_FALLBACK", 
        "sentence-transformers/all-MiniLM-L6-v2"  # Smaller but still effective
    )
    cross_encoder_model: str = os.getenv(
        "CROSS_ENCODER_MODEL", 
        "cross-encoder/stsb-roberta-base"  # Specifically for semantic textual similarity
    )
    multilingual_model: str = os.getenv(
        "MULTILINGUAL_MODEL", 
        "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"  # Good multilingual support
    )

    # Embedding Processing
    normalize_embeddings: bool = (
        os.getenv("NORMALIZE_EMBEDDINGS", "true").lower() == "true"
    )
    vector_dimension: int = int(os.getenv("VECTOR_DIMENSION", "384"))

    # Cache settings
    cache_ttl: int = int(os.getenv("CACHE_TTL", "3600"))
    embedding_cache_enabled: bool = (
        os.getenv("EMBEDDING_CACHE_ENABLED", "true").lower() == "true"
    )
    embedding_cache_ttl: int = int(
        os.getenv("EMBEDDING_CACHE_TTL", "86400")
    )  # 1 day default

    # Search settings
    search_max_results: int = int(os.getenv("SEARCH_MAX_RESULTS", "10"))
    search_min_score: float = float(os.getenv("SEARCH_MIN_SCORE", "0.1"))

    # Logging
    log_level: str = os.getenv("LOG_LEVEL", "INFO")

    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary.

        Returns:
            Dictionary of configuration values
        """
        return {key: getattr(self, key) for key in self.__annotations__}

    def validate(self) -> None:
        """Validate configuration values."""
        try:
            # Validate API keys
            if not self.huggingface_token:
                logger.warning(
                    "HUGGINGFACE_TOKEN not set. Some models may require authentication. "
                    "Please set HUGGINGFACE_TOKEN in your .env file or environment variables."
                )

            # Validate directories
            for directory in [self.api_dir, self.cache_dir, self.health_dir]:
                if not directory.exists():
                    directory.mkdir(parents=True, exist_ok=True)
                    logger.info(f"Created directory at {directory}")

            # Validate Pinecone settings
            if not self.pinecone_api_key:
                raise ValueError("Pinecone API key is required")
            if not self.pinecone_environment:
                raise ValueError("Pinecone environment is required")
            if not self.pinecone_index_name:
                raise ValueError("Pinecone index name is required")
            if not self.pinecone_cloud:
                raise ValueError("Pinecone cloud provider is required")
            if not self.pinecone_region:
                raise ValueError("Pinecone region is required")

            # Validate embedding models
            if not self.bi_encoder_model:
                raise ValueError("Bi-encoder model is required")
            if not self.bi_encoder_fallback:
                raise ValueError("Bi-encoder fallback model is required")
            if not self.cross_encoder_model:
                raise ValueError("Cross-encoder model is required")
            if not self.multilingual_model:
                raise ValueError("Multilingual model is required")

            # Validate vector dimension
            if self.vector_dimension <= 0:
                raise ValueError("Vector dimension must be positive")

            # Validate cache settings
            if self.cache_ttl <= 0:
                raise ValueError("Cache TTL must be positive")
            if self.embedding_cache_ttl <= 0:
                raise ValueError("Embedding cache TTL must be positive")

            # Validate search settings
            if self.search_max_results <= 0:
                raise ValueError("Maximum search results must be positive")
            if self.search_min_score < 0 or self.search_min_score > 1:
                raise ValueError("Search minimum score must be between 0 and 1")

        except Exception as e:
            logger.error(f"Configuration validation failed: {e}")
            raise


# Create global config instance
config_instance = Config()

# Validate configuration
config_instance.validate()

__all__ = ["config_instance", "Config"]
