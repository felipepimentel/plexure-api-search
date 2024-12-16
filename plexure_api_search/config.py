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
    openrouter_api_key: str = os.getenv("OPENROUTER_API_KEY", "")

    # OpenRouter Settings
    openrouter_model: str = os.getenv("OPENROUTER_MODEL", "mistralai/mistral-7b-instruct")
    openrouter_max_tokens: int = int(os.getenv("OPENROUTER_MAX_TOKENS", "1024"))
    openrouter_temperature: float = float(os.getenv("OPENROUTER_TEMPERATURE", "0.7"))

    # Query Expansion Settings
    max_query_expansions: int = int(os.getenv("MAX_QUERY_EXPANSIONS", "3"))
    min_expansion_length: int = int(os.getenv("MIN_EXPANSION_LENGTH", "3"))
    max_expansion_length: int = int(os.getenv("MAX_EXPANSION_LENGTH", "50"))
    expansion_strategy: str = os.getenv("EXPANSION_STRATEGY", "semantic")
    expansion_cache_ttl: int = int(os.getenv("EXPANSION_CACHE_TTL", "3600"))
    expansion_cache_enabled: bool = os.getenv("EXPANSION_CACHE_ENABLED", "true").lower() == "true"

    # Directory Settings
    api_dir: Path = Path(os.getenv("API_DIR", "assets/apis"))
    cache_dir: Path = Path(os.getenv("CACHE_DIR", ".cache"))
    health_dir: Path = Path(os.getenv("HEALTH_DIR", ".health"))

    # Pinecone Settings
    pinecone_api_key: str = os.getenv("PINECONE_API_KEY", "")
    pinecone_environment: str = os.getenv("PINECONE_ENVIRONMENT", "us-east-1")
    pinecone_index_name: str = os.getenv("PINECONE_INDEX_NAME", "plexure-api-search")
    pinecone_cloud: str = os.getenv("PINECONE_CLOUD", "aws")
    pinecone_region: str = os.getenv("PINECONE_REGION", "us-east-1")

    # Embedding Models
    bi_encoder_model: str = os.getenv("BI_ENCODER_MODEL", "bert-base-uncased")
    bi_encoder_fallback: str = os.getenv("BI_ENCODER_FALLBACK", "distilbert-base-uncased")
    cross_encoder_model: str = os.getenv("CROSS_ENCODER_MODEL", "bert-base-uncased")
    multilingual_model: str = os.getenv("MULTILINGUAL_MODEL", "bert-base-multilingual-uncased")

    # Embedding Processing
    normalize_embeddings: bool = (
        os.getenv("NORMALIZE_EMBEDDINGS", "true").lower() == "true"
    )
    pooling_strategy: str = os.getenv("POOLING_STRATEGY", "mean")
    max_seq_length: int = int(os.getenv("MAX_SEQ_LENGTH", "512"))
    vector_dimension: int = int(os.getenv("VECTOR_DIMENSION", "384"))

    # PCA Settings
    pca_enabled: bool = os.getenv("PCA_ENABLED", "true").lower() == "true"
    pca_components: int = int(os.getenv("PCA_COMPONENTS", "128"))
    pca_whiten: bool = os.getenv("PCA_WHITEN", "true").lower() == "true"
    pca_batch_size: int = int(os.getenv("PCA_BATCH_SIZE", "1000"))

    # Cache settings
    cache_ttl: int = int(os.getenv("CACHE_TTL", "3600"))
    semantic_cache_ttl: int = int(os.getenv("SEMANTIC_CACHE_TTL", "86400"))
    cache_compression_enabled: bool = (
        os.getenv("CACHE_COMPRESSION_ENABLED", "true").lower() == "true"
    )
    cache_compression_level: int = int(os.getenv("CACHE_COMPRESSION_LEVEL", "6"))

    # Embedding cache settings
    embedding_cache_enabled: bool = (
        os.getenv("EMBEDDING_CACHE_ENABLED", "true").lower() == "true"
    )
    embedding_cache_ttl: int = int(
        os.getenv("EMBEDDING_CACHE_TTL", "86400")
    )  # 1 day default

    # Cache prefetching settings
    cache_prefetch_enabled: bool = (
        os.getenv("CACHE_PREFETCH_ENABLED", "true").lower() == "true"
    )
    cache_prefetch_threshold: int = int(os.getenv("CACHE_PREFETCH_THRESHOLD", "5"))
    cache_prefetch_interval: int = int(os.getenv("CACHE_PREFETCH_INTERVAL", "60"))

    # Cache warm-up settings
    cache_warmup_enabled: bool = (
        os.getenv("CACHE_WARMUP_ENABLED", "true").lower() == "true"
    )
    cache_warmup_timeout: int = int(
        os.getenv("CACHE_WARMUP_TIMEOUT", "300")
    )  # 5 minutes

    # Hierarchical cache settings
    memory_cache_enabled: bool = (
        os.getenv("MEMORY_CACHE_ENABLED", "true").lower() == "true"
    )
    memory_cache_ttl: int = int(os.getenv("MEMORY_CACHE_TTL", "300"))  # 5 minutes
    memory_cache_max_size: int = int(
        os.getenv("MEMORY_CACHE_MAX_SIZE", "1000")
    )  # Max items

    disk_cache_enabled: bool = os.getenv("DISK_CACHE_ENABLED", "true").lower() == "true"
    disk_cache_ttl: int = int(os.getenv("DISK_CACHE_TTL", "3600"))  # 1 hour
    disk_cache_max_size: int = int(
        os.getenv("DISK_CACHE_MAX_SIZE", "10000")
    )  # Max items

    redis_cache_enabled: bool = (
        os.getenv("REDIS_CACHE_ENABLED", "false").lower() == "true"
    )
    redis_cache_ttl: int = int(os.getenv("REDIS_CACHE_TTL", "86400"))  # 1 day
    redis_cache_max_size: int = int(
        os.getenv("REDIS_CACHE_MAX_SIZE", "100000")
    )  # Max items

    # Redis settings
    redis_enabled: bool = os.getenv("REDIS_ENABLED", "false").lower() == "true"
    redis_host: str = os.getenv("REDIS_HOST", "localhost")
    redis_port: int = int(os.getenv("REDIS_PORT", "6379"))
    redis_password: Optional[str] = os.getenv("REDIS_PASSWORD")
    redis_cluster_mode: bool = (
        os.getenv("REDIS_CLUSTER_MODE", "false").lower() == "true"
    )
    redis_max_retries: int = int(os.getenv("REDIS_MAX_RETRIES", "3"))
    redis_retry_delay: float = float(os.getenv("REDIS_RETRY_DELAY", "0.1"))

    # Metrics settings
    metrics_enabled: bool = os.getenv("METRICS_ENABLED", "true").lower() == "true"
    metrics_backend: str = os.getenv("METRICS_BACKEND", "prometheus")
    metrics_port: int = int(os.getenv("METRICS_PORT", "9090"))

    # Search settings
    search_timeout: float = float(os.getenv("SEARCH_TIMEOUT", "5.0"))
    search_max_results: int = int(os.getenv("SEARCH_MAX_RESULTS", "100"))
    search_min_score: float = float(os.getenv("SEARCH_MIN_SCORE", "0.1"))
    search_cache_enabled: bool = (
        os.getenv("SEARCH_CACHE_ENABLED", "true").lower() == "true"
    )

    # Embedding settings
    embedding_model: str = os.getenv("EMBEDDING_MODEL", "BAAI/bge-large-en-v1.5")
    embedding_batch_size: int = int(os.getenv("EMBEDDING_BATCH_SIZE", "32"))
    embedding_cache_enabled: bool = (
        os.getenv("EMBEDDING_CACHE_ENABLED", "true").lower() == "true"
    )

    # Strategy weights
    bm25_weight: float = float(os.getenv("BM25_WEIGHT", "0.3"))
    vector_weight: float = float(os.getenv("VECTOR_WEIGHT", "0.7"))
    cross_encoder_weight: float = float(os.getenv("CROSS_ENCODER_WEIGHT", "0.5"))

    # Rate limiting
    rate_limit_enabled: bool = os.getenv("RATE_LIMIT_ENABLED", "true").lower() == "true"
    rate_limit_requests: int = int(os.getenv("RATE_LIMIT_REQUESTS", "100"))
    rate_limit_window: int = int(os.getenv("RATE_LIMIT_WINDOW", "60"))

    # Logging
    log_level: str = os.getenv("LOG_LEVEL", "INFO")
    log_format: str = os.getenv(
        "LOG_FORMAT", "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

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

            # Validate OpenRouter settings
            if not self.openrouter_api_key:
                logger.warning(
                    "OPENROUTER_API_KEY not set. Query expansion features will be limited. "
                    "Please set OPENROUTER_API_KEY in your .env file or environment variables."
                )
            if self.openrouter_max_tokens <= 0:
                raise ValueError("OpenRouter max tokens must be positive")
            if self.openrouter_temperature < 0 or self.openrouter_temperature > 1:
                raise ValueError("OpenRouter temperature must be between 0 and 1")

            # Validate query expansion settings
            if self.max_query_expansions <= 0:
                raise ValueError("Maximum query expansions must be positive")
            if self.min_expansion_length <= 0:
                raise ValueError("Minimum expansion length must be positive")
            if self.max_expansion_length <= self.min_expansion_length:
                raise ValueError("Maximum expansion length must be greater than minimum")
            if self.expansion_strategy not in ["semantic", "syntactic", "hybrid"]:
                raise ValueError("Invalid expansion strategy")
            if self.expansion_cache_ttl <= 0:
                raise ValueError("Expansion cache TTL must be positive")

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

            # Validate embedding processing
            if self.pooling_strategy not in ["mean", "max", "cls"]:
                raise ValueError("Invalid pooling strategy")
            if self.max_seq_length <= 0:
                raise ValueError("Max sequence length must be positive")
            if self.vector_dimension <= 0:
                raise ValueError("Vector dimension must be positive")

            # Validate PCA settings
            if self.pca_enabled:
                if self.pca_components <= 0:
                    raise ValueError("PCA components must be positive")
                if self.pca_components >= self.vector_dimension:
                    raise ValueError(
                        "PCA components must be less than vector dimension"
                    )
                if self.pca_batch_size <= 0:
                    raise ValueError("PCA batch size must be positive")

            # Validate cache directory
            if not self.cache_dir.exists():
                self.cache_dir.mkdir(parents=True, exist_ok=True)
                logger.info(f"Created cache directory at {self.cache_dir}")

            # Validate cache settings
            if self.cache_ttl <= 0:
                raise ValueError("Cache TTL must be positive")
            if self.semantic_cache_ttl <= 0:
                raise ValueError("Semantic cache TTL must be positive")
            if self.cache_compression_level < 1 or self.cache_compression_level > 9:
                raise ValueError("Cache compression level must be between 1 and 9")

            # Validate cache prefetching settings
            if self.cache_prefetch_enabled:
                if self.cache_prefetch_threshold <= 0:
                    raise ValueError("Cache prefetch threshold must be positive")
                if self.cache_prefetch_interval <= 0:
                    raise ValueError("Cache prefetch interval must be positive")

            # Validate cache warm-up settings
            if self.cache_warmup_enabled:
                if self.cache_warmup_timeout <= 0:
                    raise ValueError("Cache warm-up timeout must be positive")

            # Validate hierarchical cache settings
            if self.memory_cache_enabled:
                if self.memory_cache_ttl <= 0:
                    raise ValueError("Memory cache TTL must be positive")
                if self.memory_cache_max_size <= 0:
                    raise ValueError("Memory cache max size must be positive")

            if self.disk_cache_enabled:
                if self.disk_cache_ttl <= 0:
                    raise ValueError("Disk cache TTL must be positive")
                if self.disk_cache_max_size <= 0:
                    raise ValueError("Disk cache max size must be positive")

            if self.redis_cache_enabled:
                if self.redis_cache_ttl <= 0:
                    raise ValueError("Redis cache TTL must be positive")
                if self.redis_cache_max_size <= 0:
                    raise ValueError("Redis cache max size must be positive")

            # Validate Redis settings
            if self.redis_enabled:
                if not self.redis_host:
                    raise ValueError("Redis host is required when Redis is enabled")
                if self.redis_port <= 0:
                    raise ValueError("Redis port must be positive")
                if self.redis_max_retries < 0:
                    raise ValueError("Redis max retries must be non-negative")
                if self.redis_retry_delay <= 0:
                    raise ValueError("Redis retry delay must be positive")

            # Validate metrics settings
            if self.metrics_enabled:
                if self.metrics_port <= 0:
                    raise ValueError("Metrics port must be positive")
                if self.metrics_backend not in ["prometheus"]:
                    raise ValueError("Invalid metrics backend")

            # Validate search settings
            if self.search_timeout <= 0:
                raise ValueError("Search timeout must be positive")
            if self.search_max_results <= 0:
                raise ValueError("Search max results must be positive")
            if self.search_min_score < 0 or self.search_min_score > 1:
                raise ValueError("Search min score must be between 0 and 1")

            # Validate embedding settings
            if self.embedding_batch_size <= 0:
                raise ValueError("Embedding batch size must be positive")

            # Validate strategy weights
            if self.bm25_weight < 0 or self.bm25_weight > 1:
                raise ValueError("BM25 weight must be between 0 and 1")
            if self.vector_weight < 0 or self.vector_weight > 1:
                raise ValueError("Vector weight must be between 0 and 1")
            if self.cross_encoder_weight < 0 or self.cross_encoder_weight > 1:
                raise ValueError("Cross encoder weight must be between 0 and 1")

            # Validate rate limiting
            if self.rate_limit_enabled:
                if self.rate_limit_requests <= 0:
                    raise ValueError("Rate limit requests must be positive")
                if self.rate_limit_window <= 0:
                    raise ValueError("Rate limit window must be positive")

            # Validate logging
            if self.log_level not in ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]:
                raise ValueError("Invalid log level")

        except Exception as e:
            logger.error(f"Configuration validation failed: {e}")
            raise


# Global config instance
config_instance = Config()

# Validate configuration
config_instance.validate()

__all__ = ["config_instance", "Config"]
