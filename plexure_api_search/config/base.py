"""Base configuration classes and utilities."""

from enum import Enum
from typing import Dict, Optional, Any
from pathlib import Path

from pydantic import BaseModel

class Environment(str, Enum):
    """Environment enumeration."""
    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"

class ConfigurationError(Exception):
    """Base configuration error."""
    pass

class InvalidConfigurationError(ConfigurationError):
    """Invalid configuration error."""
    pass

class MissingConfigurationError(ConfigurationError):
    """Missing configuration error."""
    pass

class PathConfig(BaseModel):
    """Path configuration."""
    api_dir: Path
    cache_dir: Path
    health_dir: Path
    model_dir: Path
    metrics_dir: Path

class PineconeConfig(BaseModel):
    """Pinecone configuration."""
    api_key: str
    environment: str
    index_name: str
    cloud: str
    region: str
    pool_min_size: int
    pool_max_size: int
    pool_max_idle_time: int
    pool_cleanup_interval: int

class ModelConfig(BaseModel):
    """Model configuration."""
    bi_encoder: str
    bi_encoder_fallback: str
    cross_encoder: str
    multilingual: str
    huggingface_token: Optional[str] = None
    normalize_embeddings: bool = True

class VectorConfig(BaseModel):
    """Vector configuration."""
    dimension: int
    metric: str
    batch_size: int
    num_threads: int
    quantization_bits: int
    use_gpu: bool

class SearchConfig(BaseModel):
    """Search configuration."""
    max_results: int
    min_score: float
    timeout: int
    cache_ttl: int

class CacheConfig(BaseModel):
    """Cache configuration."""
    ttl: int
    embedding_ttl: int
    backend: str

class MonitoringConfig(BaseModel):
    """Monitoring configuration."""
    enable_telemetry: bool
    metrics_backend: str
    log_level: str
    publisher_port: int = 5555

class ServiceConfig(BaseModel):
    """Service configuration."""
    name: str
    version: str
    environment: Environment

class Config(BaseModel):
    """Complete configuration."""
    paths: PathConfig
    pinecone: PineconeConfig
    models: ModelConfig
    vectors: VectorConfig
    search: SearchConfig
    cache: CacheConfig
    monitoring: MonitoringConfig
    service: ServiceConfig

    # Path property getters
    @property
    def api_dir(self) -> Path:
        """Get API directory path."""
        return self.paths.api_dir

    @property
    def cache_dir(self) -> Path:
        """Get cache directory path."""
        return self.paths.cache_dir

    @property
    def health_dir(self) -> Path:
        """Get health directory path."""
        return self.paths.health_dir

    @property
    def model_dir(self) -> Path:
        """Get model directory path."""
        return self.paths.model_dir

    @property
    def metrics_dir(self) -> Path:
        """Get metrics directory path."""
        return self.paths.metrics_dir

    # Cache property getters
    @property
    def cache_ttl(self) -> int:
        """Get cache TTL."""
        return self.cache.ttl

    @property
    def embedding_cache_ttl(self) -> int:
        """Get embedding cache TTL."""
        return self.cache.embedding_ttl

    @property
    def cache_backend(self) -> str:
        """Get cache backend."""
        return self.cache.backend

    # Model property getters
    @property
    def bi_encoder_model(self) -> str:
        """Get bi-encoder model name."""
        return self.models.bi_encoder

    @property
    def bi_encoder_fallback(self) -> str:
        """Get bi-encoder fallback model name."""
        return self.models.bi_encoder_fallback

    @property
    def cross_encoder_model(self) -> str:
        """Get cross-encoder model name."""
        return self.models.cross_encoder

    @property
    def multilingual_model(self) -> str:
        """Get multilingual model name."""
        return self.models.multilingual

    @property
    def normalize_embeddings(self) -> bool:
        """Get whether to normalize embeddings."""
        return self.models.normalize_embeddings

    # Monitoring property getters
    @property
    def enable_telemetry(self) -> bool:
        """Get whether telemetry is enabled."""
        return self.monitoring.enable_telemetry

    @property
    def metrics_backend(self) -> str:
        """Get metrics backend."""
        return self.monitoring.metrics_backend

    @property
    def log_level(self) -> str:
        """Get log level."""
        return self.monitoring.log_level

    @property
    def publisher_port(self) -> int:
        """Get publisher port."""
        return self.monitoring.publisher_port

    def validate(self) -> None:
        """Validate configuration.
        
        This method is called after loading configuration to ensure
        all values are valid and consistent.
        
        Raises:
            ConfigurationError: If configuration is invalid.
        """
        try:
            # Create directories
            for path in [
                self.paths.api_dir,
                self.paths.cache_dir,
                self.paths.health_dir,
                self.paths.model_dir,
                self.paths.metrics_dir,
            ]:
                path.mkdir(parents=True, exist_ok=True)

            # Validate Pinecone config
            if self.pinecone.pool_min_size > self.pinecone.pool_max_size:
                raise InvalidConfigurationError(
                    "Pinecone minimum pool size must be less than maximum"
                )

            # Validate vector config
            if self.vectors.quantization_bits not in [1, 2, 4, 8]:
                raise InvalidConfigurationError(
                    "Vector quantization bits must be 1, 2, 4, or 8"
                )

            # Validate search config
            if not 0 <= self.search.min_score <= 1:
                raise InvalidConfigurationError(
                    "Search minimum score must be between 0 and 1"
                )

        except Exception as e:
            raise ConfigurationError(f"Configuration validation failed: {e}")

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Config':
        """Create configuration from dictionary.
        
        Args:
            data: Configuration dictionary.
            
        Returns:
            Configuration instance.
            
        Raises:
            ConfigurationError: If configuration is invalid.
        """
        try:
            return cls(**data)
        except Exception as e:
            raise ConfigurationError(f"Failed to create configuration: {e}")

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary.
        
        Returns:
            Configuration dictionary.
        """
        return self.dict() 