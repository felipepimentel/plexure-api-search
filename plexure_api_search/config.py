"""Configuration management."""

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv

# Load environment variables
load_dotenv()


@dataclass
class Config:
    """Application configuration."""

    # Vector dimension settings
    vector_dimension: int = int(os.getenv("VECTOR_DIMENSION", "384"))

    # FAISS settings
    faiss_index_type: str = os.getenv("FAISS_INDEX_TYPE", "IVFFlat")
    faiss_nlist: int = int(os.getenv("FAISS_NLIST", "100"))

    # Pinecone settings
    pinecone_api_key: Optional[str] = os.getenv("PINECONE_API_KEY")
    pinecone_index_name: str = os.getenv("PINECONE_INDEX_NAME", "api-search")
    pinecone_region: str = os.getenv("PINECONE_REGION", "us-east-1")
    pinecone_cloud: str = os.getenv("PINECONE_CLOUD", "aws")

    # OpenRouter settings
    openrouter_api_key: Optional[str] = os.getenv("OPENROUTER_API_KEY")
    openrouter_model: str = os.getenv(
        "OPENROUTER_MODEL", 
        "mistral/mistral-small"  # Good balance between cost and performance
    )
    openrouter_max_tokens: int = int(os.getenv("OPENROUTER_MAX_TOKENS", "1024"))
    openrouter_temperature: float = float(os.getenv("OPENROUTER_TEMPERATURE", "0.7"))

    # Model settings
    bi_encoder_model: str = os.getenv("BI_ENCODER_MODEL", "all-MiniLM-L6-v2")
    cross_encoder_model: str = os.getenv(
        "CROSS_ENCODER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2"
    )

    # Cache settings
    embedding_cache_ttl: int = int(
        os.getenv("EMBEDDING_CACHE_TTL", "86400")
    )  # 24 hours

    # PCA settings
    pca_components: int = int(os.getenv("PCA_COMPONENTS", "128"))

    # API settings
    api_dir: str = os.getenv("API_DIR", "assets/apis")

    # Directory configurations
    cache_dir = Path("data/.cache")
    cache_ttl: int = int(os.getenv("CACHE_TTL", "3600"))  # 1 hour

    metrics_dir = Path("data/.metrics")
    health_dir = Path("data/.health")

    def __post_init__(self):
        """Validate configuration after initialization."""
        # Create cache directory if it doesn't exist
        os.makedirs(self.cache_dir, exist_ok=True)

        # Validate Pinecone settings
        if not self.pinecone_api_key:
            raise ValueError("PINECONE_API_KEY is required")

        # Validate OpenRouter settings
        if not self.openrouter_api_key:
            raise ValueError("OPENROUTER_API_KEY is required")


# Global configuration instance
config_instance = Config()

__all__ = ["config_instance", "Config"]
