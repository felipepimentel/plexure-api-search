"""Configuration validation using Pydantic models."""

from typing import Dict, List, Optional, Union
from pydantic import BaseModel, Field, model_validator
from pathlib import Path
import re

from .base import Environment

class PathConfigModel(BaseModel):
    """Path configuration validation model."""
    api_dir: Path = Field(..., description="Directory containing API files")
    cache_dir: Path = Field(..., description="Directory for caching")
    health_dir: Path = Field(..., description="Directory for health checks")
    model_dir: Path = Field(..., description="Directory for model files")
    metrics_dir: Path = Field(..., description="Directory for metrics")

    @model_validator(mode='after')
    def validate_directories(self) -> 'PathConfigModel':
        """Validate directory exists or can be created."""
        for field, value in self:
            value.mkdir(parents=True, exist_ok=True)
        return self

class PineconeConfigModel(BaseModel):
    """Pinecone configuration validation model."""
    api_key: str = Field(..., description="Pinecone API key")
    environment: str = Field(..., description="Pinecone environment")
    index_name: str = Field(..., description="Pinecone index name")
    cloud: str = Field(..., description="Cloud provider")
    region: str = Field(..., description="Cloud region")
    pool_min_size: int = Field(ge=1, description="Minimum pool size")
    pool_max_size: int = Field(ge=1, description="Maximum pool size")
    pool_max_idle_time: int = Field(ge=1, description="Maximum idle time in seconds")
    pool_cleanup_interval: int = Field(ge=1, description="Cleanup interval in seconds")

    @model_validator(mode='after')
    def validate_pool_size(self) -> 'PineconeConfigModel':
        """Validate pool size configuration."""
        if self.pool_min_size > self.pool_max_size:
            raise ValueError("Minimum pool size must be less than maximum")
        return self

class ModelConfigModel(BaseModel):
    """Model configuration validation model."""
    bi_encoder: str = Field(..., description="Bi-encoder model name")
    bi_encoder_fallback: str = Field(..., description="Fallback bi-encoder model name")
    cross_encoder: str = Field(..., description="Cross-encoder model name")
    multilingual: str = Field(..., description="Multilingual model name")
    huggingface_token: Optional[str] = Field(None, description="HuggingFace API token")

    @model_validator(mode='after')
    def validate_model_names(self) -> 'ModelConfigModel':
        """Validate model name format."""
        for field, value in self:
            if not value or field == "huggingface_token":
                continue
            if not re.match(r"^[\w\-/]+$", value):
                raise ValueError(f"Invalid model name format: {value}")
        return self

class VectorConfigModel(BaseModel):
    """Vector configuration validation model."""
    dimension: int = Field(ge=1, description="Vector dimension")
    metric: str = Field(..., description="Distance metric")
    batch_size: int = Field(ge=1, description="Batch size")
    num_threads: int = Field(ge=1, description="Number of threads")
    quantization_bits: int = Field(ge=1, le=8, description="Quantization bits")
    use_gpu: bool = Field(description="Whether to use GPU")

class SearchConfigModel(BaseModel):
    """Search configuration validation model."""
    max_results: int = Field(ge=1, description="Maximum number of results")
    min_score: float = Field(ge=0, le=1, description="Minimum similarity score")
    timeout: int = Field(ge=1, description="Search timeout in seconds")
    cache_ttl: int = Field(ge=0, description="Cache TTL in seconds")

class CacheConfigModel(BaseModel):
    """Cache configuration validation model."""
    ttl: int = Field(ge=0, description="Cache TTL in seconds")
    embedding_ttl: int = Field(ge=0, description="Embedding cache TTL in seconds")
    backend: str = Field(..., description="Cache backend")

class MonitoringConfigModel(BaseModel):
    """Monitoring configuration validation model."""
    enable_telemetry: bool = Field(description="Whether to enable telemetry")
    metrics_backend: str = Field(..., description="Metrics backend")
    log_level: str = Field(..., description="Log level")

class ServiceConfigModel(BaseModel):
    """Service configuration validation model."""
    name: str = Field(..., description="Service name")
    version: str = Field(..., description="Service version")
    environment: Environment = Field(..., description="Service environment")

class ConfigModel(BaseModel):
    """Complete configuration validation model."""
    paths: PathConfigModel
    pinecone: PineconeConfigModel
    models: ModelConfigModel
    vectors: VectorConfigModel
    search: SearchConfigModel
    cache: CacheConfigModel
    monitoring: MonitoringConfigModel
    service: ServiceConfigModel

    class Config:
        """Pydantic model configuration."""
        arbitrary_types_allowed = True

def validate_config(config_dict: Dict) -> Dict:
    """Validate configuration dictionary.
    
    Args:
        config_dict: Configuration dictionary to validate.
        
    Returns:
        Validated configuration dictionary.
        
    Raises:
        ValueError: If configuration is invalid.
    """
    try:
        model = ConfigModel(**config_dict)
        return model.dict()
    except Exception as e:
        raise ValueError(f"Configuration validation failed: {e}")

def validate_config_file(path: Union[str, Path]) -> Dict:
    """Validate configuration file.
    
    Args:
        path: Path to configuration file.
        
    Returns:
        Validated configuration dictionary.
        
    Raises:
        ValueError: If configuration is invalid.
        FileNotFoundError: If file not found.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Configuration file not found: {path}")

    import yaml
    import json

    try:
        with open(path) as f:
            if path.suffix == ".yaml":
                config_dict = yaml.safe_load(f)
            elif path.suffix == ".json":
                config_dict = json.load(f)
            else:
                raise ValueError("Unsupported file format")
    except Exception as e:
        raise ValueError(f"Failed to load configuration file: {e}")

    return validate_config(config_dict) 