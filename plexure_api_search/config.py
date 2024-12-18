"""Configuration management."""

import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

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
    model_dir: Path = Path(os.getenv("MODEL_DIR", ".models"))
    metrics_dir: Path = Path(os.getenv("METRICS_DIR", ".metrics"))

    # Pinecone Settings
    pinecone_api_key: str = os.getenv("PINECONE_API_KEY", "")
    pinecone_environment: str = os.getenv("PINECONE_ENVIRONMENT", "gcp-starter")
    pinecone_index_name: str = os.getenv("PINECONE_INDEX_NAME", "plexure-api-search")
    pinecone_cloud: str = os.getenv("PINECONE_CLOUD", "gcp")
    pinecone_region: str = os.getenv("PINECONE_REGION", "us-central1")

    # Pinecone Pool Settings
    pinecone_pool_min_size: int = int(os.getenv("PINECONE_POOL_MIN_SIZE", "2"))
    pinecone_pool_max_size: int = int(os.getenv("PINECONE_POOL_MAX_SIZE", "10"))
    pinecone_pool_max_idle_time: int = int(os.getenv("PINECONE_POOL_MAX_IDLE_TIME", "300"))
    pinecone_pool_cleanup_interval: int = int(os.getenv("PINECONE_POOL_CLEANUP_INTERVAL", "60"))

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
        "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"  # For multi-language support
    )

    # Vector Settings
    vector_dimension: int = int(os.getenv("VECTOR_DIMENSION", "384"))
    vector_metric: str = os.getenv("VECTOR_METRIC", "cosine")

    # Search Settings
    search_max_results: int = int(os.getenv("SEARCH_MAX_RESULTS", "10"))
    search_min_score: float = float(os.getenv("SEARCH_MIN_SCORE", "0.5"))
    search_timeout: int = int(os.getenv("SEARCH_TIMEOUT", "10"))
    search_cache_ttl: int = int(os.getenv("SEARCH_CACHE_TTL", "3600"))

    # Cache Settings
    cache_ttl: int = int(os.getenv("CACHE_TTL", "3600"))
    embedding_cache_ttl: int = int(os.getenv("EMBEDDING_CACHE_TTL", "86400"))
    cache_backend: str = os.getenv("CACHE_BACKEND", "disk")

    # Monitoring Settings
    enable_telemetry: bool = os.getenv("ENABLE_TELEMETRY", "true").lower() == "true"
    metrics_backend: str = os.getenv("METRICS_BACKEND", "prometheus")
    log_level: str = os.getenv("LOG_LEVEL", "INFO")

    # ZeroMQ Settings
    zmq_pub_port: int = int(os.getenv("ZMQ_PUB_PORT", "5555"))
    zmq_sub_port: int = int(os.getenv("ZMQ_SUB_PORT", "5556"))
    zmq_pub_address: str = os.getenv("ZMQ_PUB_ADDRESS", "tcp://*")
    zmq_sub_address: str = os.getenv("ZMQ_SUB_ADDRESS", "tcp://localhost")
    zmq_topic: str = os.getenv("ZMQ_TOPIC", "events")
    zmq_high_water_mark: int = int(os.getenv("ZMQ_HIGH_WATER_MARK", "1000"))
    zmq_reconnect_interval: int = int(os.getenv("ZMQ_RECONNECT_INTERVAL", "1000"))
    zmq_max_retries: int = int(os.getenv("ZMQ_MAX_RETRIES", "3"))

    # Service Settings
    service_name: str = os.getenv("SERVICE_NAME", "plexure-api-search")
    service_version: str = os.getenv("SERVICE_VERSION", "0.1.0")
    service_environment: str = os.getenv("SERVICE_ENVIRONMENT", "development")

    # Service Discovery Settings
    discovery_host: str = os.getenv("DISCOVERY_HOST", "localhost")
    discovery_port: int = int(os.getenv("DISCOVERY_PORT", "5557"))
    discovery_ttl: int = int(os.getenv("DISCOVERY_TTL", "30"))
    discovery_heartbeat_interval: int = int(os.getenv("DISCOVERY_HEARTBEAT_INTERVAL", "10"))
    discovery_cleanup_interval: int = int(os.getenv("DISCOVERY_CLEANUP_INTERVAL", "5"))

    # Circuit Breaker Settings
    circuit_failure_threshold: int = int(os.getenv("CIRCUIT_FAILURE_THRESHOLD", "5"))
    circuit_recovery_timeout: int = int(os.getenv("CIRCUIT_RECOVERY_TIMEOUT", "60"))
    circuit_half_open_timeout: int = int(os.getenv("CIRCUIT_HALF_OPEN_TIMEOUT", "30"))
    circuit_check_interval: int = int(os.getenv("CIRCUIT_CHECK_INTERVAL", "1"))

    # Service Registry Settings
    registry_host: str = os.getenv("REGISTRY_HOST", "localhost")
    registry_port: int = int(os.getenv("REGISTRY_PORT", "5558"))
    registry_ttl: int = int(os.getenv("REGISTRY_TTL", "30"))
    registry_cleanup_interval: int = int(os.getenv("REGISTRY_CLEANUP_INTERVAL", "5"))
    registry_max_services: int = int(os.getenv("REGISTRY_MAX_SERVICES", "100"))

    # Batch Processing Settings
    batch_size: int = int(os.getenv("BATCH_SIZE", "32"))
    batch_max_wait: float = float(os.getenv("BATCH_MAX_WAIT", "0.1"))
    batch_queue_size: int = int(os.getenv("BATCH_QUEUE_SIZE", "1000"))
    batch_workers: int = int(os.getenv("BATCH_WORKERS", "4"))

    # Vector Operation Settings
    vector_ops_batch_size: int = int(os.getenv("VECTOR_OPS_BATCH_SIZE", "1000"))
    vector_ops_num_threads: int = int(os.getenv("VECTOR_OPS_NUM_THREADS", "4"))
    vector_ops_quantization_bits: int = int(os.getenv("VECTOR_OPS_QUANTIZATION_BITS", "8"))
    vector_ops_use_gpu: bool = os.getenv("VECTOR_OPS_USE_GPU", "false").lower() == "true"

    # Performance Settings
    enable_connection_pooling: bool = os.getenv("ENABLE_CONNECTION_POOLING", "true").lower() == "true"
    enable_batch_processing: bool = os.getenv("ENABLE_BATCH_PROCESSING", "true").lower() == "true"
    enable_vector_quantization: bool = os.getenv("ENABLE_VECTOR_QUANTIZATION", "true").lower() == "true"
    use_numpy_optimization: bool = os.getenv("USE_NUMPY_OPTIMIZATION", "true").lower() == "true"
    use_parallel_processing: bool = os.getenv("USE_PARALLEL_PROCESSING", "true").lower() == "true"
    max_parallel_requests: int = int(os.getenv("MAX_PARALLEL_REQUESTS", "100"))
    request_timeout: int = int(os.getenv("REQUEST_TIMEOUT", "30"))
    rate_limit: int = int(os.getenv("RATE_LIMIT", "1000"))

    def validate(self) -> None:
        """Validate configuration values."""
        try:
            # Validate directories
            for dir_path in [
                self.api_dir,
                self.cache_dir,
                self.health_dir,
                self.model_dir,
                self.metrics_dir,
            ]:
                dir_path.mkdir(parents=True, exist_ok=True)

            # Validate Pinecone settings
            if not self.pinecone_api_key:
                raise ValueError("Pinecone API key is required")
            if not self.pinecone_environment:
                raise ValueError("Pinecone environment is required")
            if not self.pinecone_index_name:
                raise ValueError("Pinecone index name is required")

            # Validate Pinecone pool settings
            if self.pinecone_pool_min_size <= 0:
                raise ValueError("Pinecone pool minimum size must be positive")
            if self.pinecone_pool_max_size < self.pinecone_pool_min_size:
                raise ValueError("Pinecone pool maximum size must be greater than minimum size")
            if self.pinecone_pool_max_idle_time <= 0:
                raise ValueError("Pinecone pool maximum idle time must be positive")
            if self.pinecone_pool_cleanup_interval <= 0:
                raise ValueError("Pinecone pool cleanup interval must be positive")

            # Validate model settings
            if not self.bi_encoder_model:
                raise ValueError("Bi-encoder model is required")
            if not self.bi_encoder_fallback:
                raise ValueError("Bi-encoder fallback model is required")
            if not self.cross_encoder_model:
                raise ValueError("Cross-encoder model is required")
            if not self.multilingual_model:
                raise ValueError("Multilingual model is required")

            # Validate vector settings
            if self.vector_dimension <= 0:
                raise ValueError("Vector dimension must be positive")
            if self.vector_metric not in ["cosine", "euclidean", "dot"]:
                raise ValueError("Invalid vector metric")

            # Validate search settings
            if self.search_max_results <= 0:
                raise ValueError("Maximum search results must be positive")
            if not 0 <= self.search_min_score <= 1:
                raise ValueError("Search minimum score must be between 0 and 1")
            if self.search_timeout <= 0:
                raise ValueError("Search timeout must be positive")

            # Validate cache settings
            if self.cache_ttl <= 0:
                raise ValueError("Cache TTL must be positive")
            if self.embedding_cache_ttl <= 0:
                raise ValueError("Embedding cache TTL must be positive")
            if self.cache_backend not in ["disk", "memory", "redis"]:
                raise ValueError("Invalid cache backend")

            # Validate ZeroMQ settings
            if self.zmq_pub_port <= 0:
                raise ValueError("ZMQ publisher port must be positive")
            if self.zmq_sub_port <= 0:
                raise ValueError("ZMQ subscriber port must be positive")
            if self.zmq_high_water_mark <= 0:
                raise ValueError("ZMQ high water mark must be positive")
            if self.zmq_reconnect_interval <= 0:
                raise ValueError("ZMQ reconnect interval must be positive")
            if self.zmq_max_retries < 0:
                raise ValueError("ZMQ max retries must be non-negative")

            # Validate service settings
            if not self.service_name:
                raise ValueError("Service name is required")
            if not self.service_version:
                raise ValueError("Service version is required")
            if not self.service_environment:
                raise ValueError("Service environment is required")

            # Validate service discovery settings
            if not self.discovery_host:
                raise ValueError("Discovery host is required")
            if self.discovery_port <= 0:
                raise ValueError("Discovery port must be positive")
            if self.discovery_ttl <= 0:
                raise ValueError("Discovery TTL must be positive")
            if self.discovery_heartbeat_interval <= 0:
                raise ValueError("Discovery heartbeat interval must be positive")
            if self.discovery_cleanup_interval <= 0:
                raise ValueError("Discovery cleanup interval must be positive")

            # Validate circuit breaker settings
            if self.circuit_failure_threshold <= 0:
                raise ValueError("Circuit failure threshold must be positive")
            if self.circuit_recovery_timeout <= 0:
                raise ValueError("Circuit recovery timeout must be positive")
            if self.circuit_half_open_timeout <= 0:
                raise ValueError("Circuit half-open timeout must be positive")
            if self.circuit_check_interval <= 0:
                raise ValueError("Circuit check interval must be positive")

            # Validate service registry settings
            if not self.registry_host:
                raise ValueError("Registry host is required")
            if self.registry_port <= 0:
                raise ValueError("Registry port must be positive")
            if self.registry_ttl <= 0:
                raise ValueError("Registry TTL must be positive")
            if self.registry_cleanup_interval <= 0:
                raise ValueError("Registry cleanup interval must be positive")
            if self.registry_max_services <= 0:
                raise ValueError("Registry maximum services must be positive")

            # Validate batch processing settings
            if self.batch_size <= 0:
                raise ValueError("Batch size must be positive")
            if self.batch_max_wait <= 0:
                raise ValueError("Batch maximum wait time must be positive")
            if self.batch_queue_size <= 0:
                raise ValueError("Batch queue size must be positive")
            if self.batch_workers <= 0:
                raise ValueError("Number of batch workers must be positive")

            # Validate vector operation settings
            if self.vector_ops_batch_size <= 0:
                raise ValueError("Vector operations batch size must be positive")
            if self.vector_ops_num_threads <= 0:
                raise ValueError("Number of vector operations threads must be positive")
            if not 1 <= self.vector_ops_quantization_bits <= 32:
                raise ValueError("Vector operations quantization bits must be between 1 and 32")

            # Validate performance settings
            if self.max_parallel_requests <= 0:
                raise ValueError("Maximum parallel requests must be positive")
            if self.request_timeout <= 0:
                raise ValueError("Request timeout must be positive")
            if self.rate_limit <= 0:
                raise ValueError("Rate limit must be positive")

        except Exception as e:
            logger.error(f"Configuration validation failed: {e}")
            raise


# Create global config instance
config_instance = Config()

__all__ = ["Config", "config_instance"]
