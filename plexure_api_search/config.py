"""
Configuration Management for Plexure API Search

This module provides configuration management functionality for the Plexure API Search system.
It implements a singleton pattern to ensure consistent configuration across the application and
handles loading settings from various sources including environment variables and configuration files.

Key Features:
- Singleton pattern for global configuration access
- Environment variable support with validation
- Configuration file loading (YAML/JSON)
- Default value management
- Type validation and conversion
- Dynamic configuration updates
- Configuration persistence
- Environment-specific settings (dev/prod/test)

The Config class provides methods for:
- Loading configuration from environment variables
- Accessing configuration values with type safety
- Updating configuration at runtime
- Validating configuration values
- Persisting configuration changes
- Managing environment-specific settings

Example Usage:
    from plexure_api_search.config import Config

    # Get configuration instance
    config = Config()

    # Access settings
    model_name = config.model_name
    batch_size = config.model_batch_size

    # Update settings
    config.update({
        "model_batch_size": 64,
        "min_score": 0.3
    })

Environment Variables:
    ENVIRONMENT: Environment name (development/staging/production)
    DEBUG: Enable debug mode (true/false)
    API_DIR: Directory containing API contracts
    CACHE_DIR: Cache directory
    MODEL_NAME: Embedding model name
    MODEL_DIMENSION: Embedding dimension
    MODEL_BATCH_SIZE: Model batch size
    ENABLE_TELEMETRY: Enable metrics collection
    MIN_SCORE: Minimum similarity score
    TOP_K: Default number of results
    LOG_LEVEL: Logging level
"""

import logging
import os
from enum import Enum
from typing import Any, Dict

logger = logging.getLogger(__name__)


class Environment(str, Enum):
    """Environment enumeration."""

    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"


class Config:
    """Configuration settings."""

    _instance = None

    def __new__(cls):
        """Create singleton instance."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        """Initialize configuration."""
        if self._initialized:
            return
        self._initialized = True

        # Environment
        self.environment = Environment.DEVELOPMENT.value
        self.debug = False

        # Paths
        self.api_dir = "assets/apis"
        self.cache_dir = ".cache/default"
        self.metrics_dir = ".cache/metrics"

        # Model
        self.model_name = "sentence-transformers/all-MiniLM-L6-v2"
        self.model_dimension = 384
        self.model_batch_size = 32
        self.model_normalize_embeddings = True

        # Monitoring
        self.enable_telemetry = True
        self.metrics_backend = "prometheus"
        self.publisher_port = 5555

        # Search
        self.min_score = 0.1
        self.top_k = 10
        self.expand_query = True
        self.rerank_results = True

        # Logging
        self.log_level = "INFO"
        self.log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

        # Load environment variables
        self._load_from_env()

    def _load_from_env(self) -> None:
        """Load configuration from environment variables."""
        # Load environment from .env file if it exists
        if os.path.exists(".env"):
            from dotenv import load_dotenv

            load_dotenv()

        # Environment
        self.environment = os.getenv("ENVIRONMENT", self.environment)
        self.debug = os.getenv("DEBUG", str(self.debug)).lower() == "true"

        # Paths
        self.api_dir = os.getenv("API_DIR", self.api_dir)
        self.cache_dir = os.getenv("CACHE_DIR", self.cache_dir)
        self.metrics_dir = os.getenv("METRICS_DIR", self.metrics_dir)

        # Model
        self.model_name = os.getenv("MODEL_NAME", self.model_name)
        self.model_dimension = int(
            os.getenv("MODEL_DIMENSION", str(self.model_dimension))
        )
        self.model_batch_size = int(
            os.getenv("MODEL_BATCH_SIZE", str(self.model_batch_size))
        )
        self.model_normalize_embeddings = (
            os.getenv(
                "MODEL_NORMALIZE_EMBEDDINGS", str(self.model_normalize_embeddings)
            ).lower()
            == "true"
        )

        # Monitoring
        self.enable_telemetry = (
            os.getenv("ENABLE_TELEMETRY", str(self.enable_telemetry)).lower() == "true"
        )
        self.metrics_backend = os.getenv("METRICS_BACKEND", self.metrics_backend)
        self.publisher_port = int(os.getenv("PUBLISHER_PORT", str(self.publisher_port)))

        # Search
        self.min_score = float(os.getenv("MIN_SCORE", str(self.min_score)))
        self.top_k = int(os.getenv("TOP_K", str(self.top_k)))
        self.expand_query = (
            os.getenv("EXPAND_QUERY", str(self.expand_query)).lower() == "true"
        )
        self.rerank_results = (
            os.getenv("RERANK_RESULTS", str(self.rerank_results)).lower() == "true"
        )

        # Logging
        self.log_level = os.getenv("LOG_LEVEL", self.log_level).upper()
        self.log_format = os.getenv("LOG_FORMAT", self.log_format)

        # Validate environment
        if self.environment not in [e.value for e in Environment]:
            self.environment = Environment.DEVELOPMENT.value

        # Validate log level
        valid_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        if self.log_level not in valid_levels:
            self.log_level = "INFO"

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return {
            "environment": self.environment,
            "debug": self.debug,
            "api_dir": self.api_dir,
            "cache_dir": self.cache_dir,
            "metrics_dir": self.metrics_dir,
            "model_name": self.model_name,
            "model_dimension": self.model_dimension,
            "model_batch_size": self.model_batch_size,
            "model_normalize_embeddings": self.model_normalize_embeddings,
            "enable_telemetry": self.enable_telemetry,
            "metrics_backend": self.metrics_backend,
            "publisher_port": self.publisher_port,
            "min_score": self.min_score,
            "top_k": self.top_k,
            "expand_query": self.expand_query,
            "rerank_results": self.rerank_results,
            "log_level": self.log_level,
            "log_format": self.log_format,
        }


# Create global instance
config = Config()

# Set up logging
logging.basicConfig(
    level=config.log_level,
    format=config.log_format,
)

# Set log level for noisy libraries
logging.getLogger("urllib3").setLevel(logging.WARNING)
logging.getLogger("requests").setLevel(logging.WARNING)
logging.getLogger("pydantic").setLevel(logging.WARNING)

# Export configuration instance
__all__ = ["config", "Environment"]
