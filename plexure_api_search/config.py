"""Configuration management."""

import logging
import os
from pathlib import Path
from typing import Optional

from pydantic import BaseModel, Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

logger = logging.getLogger(__name__)

class Config(BaseSettings):
    """Configuration settings."""

    # Environment
    env: str = Field(default="development")
    debug: bool = Field(default=False)

    # Paths
    api_dir: str = Field(default="assets/apis")
    cache_dir: str = Field(default=".cache/default")
    metrics_dir: str = Field(default=".cache/metrics")

    # Model
    model_name: str = Field(
        default="sentence-transformers/all-MiniLM-L6-v2",
    )

    # Search
    min_score: float = Field(default=0.1)
    top_k: int = Field(default=10)
    expand_query: bool = Field(default=True)
    rerank_results: bool = Field(default=True)

    # Metrics
    metrics_enabled: bool = Field(default=True)
    metrics_port: int = Field(default=8000)

    # Logging
    log_level: str = Field(default="INFO")
    log_format: str = Field(
        default="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    model_config = SettingsConfigDict(
        env_prefix="PLEXURE_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    @field_validator("log_level")
    def validate_log_level(cls, v: str) -> str:
        """Validate log level."""
        valid_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        if v.upper() not in valid_levels:
            raise ValueError(f"Invalid log level: {v}")
        return v.upper()

# Global instance
config_instance = Config()

# Set up logging
logging.basicConfig(
    level=config_instance.log_level,
    format=config_instance.log_format,
)

# Set log level for noisy libraries
logging.getLogger("urllib3").setLevel(logging.WARNING)
logging.getLogger("requests").setLevel(logging.WARNING)
logging.getLogger("pydantic").setLevel(logging.WARNING)
