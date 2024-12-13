"""Configuration management for Plexure API Search."""

import os
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

import yaml
from dotenv import load_dotenv
from rich.console import Console

# Load environment variables
load_dotenv()

# Default values
DEFAULT_API_DIR = os.getenv("API_DIR", "assets/apis")
DEFAULT_TOP_K = int(os.getenv("TOP_K", "10"))
DEFAULT_BATCH_SIZE = int(os.getenv("BATCH_SIZE", "100"))

# Model settings
DEFAULT_MODEL = os.getenv("MODEL_NAME", "sentence-transformers/all-MiniLM-L6-v2")
DEFAULT_DIMENSION = 384  # SentenceTransformer default dimension

# Pinecone settings
DEFAULT_PINECONE_CLOUD = os.getenv("PINECONE_CLOUD", "aws")
DEFAULT_PINECONE_REGION = os.getenv("PINECONE_REGION", "us-central1")

# HTTP method colors
METHOD_COLORS = {
    "GET": "green",
    "POST": "blue",
    "PUT": "yellow",
    "DELETE": "red",
    "PATCH": "magenta",
}

# Score adjustments for ranking
SCORE_ADJUSTMENTS = {
    "version_match": {"exact": 1.0, "major": 0.8, "minor": 0.6, "patch": 0.4},
    "method_match": 1.0,
    "path_match": {"exact": 1.0, "partial": 0.8, "similar": 0.6},
    "feature_match": {"exact": 1.0, "similar": 0.8},
    "tag_match": {"exact": 1.0, "similar": 0.8},
    "text_similarity": {"high": 0.9, "medium": 0.7, "low": 0.5},
}

# Feature settings
DEFAULT_FEATURES = {
    "has_auth": {"icon": "🔒", "label": "Auth Required", "style": "bold red"},
    "has_examples": {"icon": "📝", "label": "Has Examples", "style": "bold green"},
    "supports_pagination": {"icon": "📄", "label": "Paginated", "style": "bold blue"},
    "deprecated": {"icon": "⚠️", "label": "Deprecated", "style": "bold yellow"},
}


@dataclass
class Config:
    """Configuration settings."""

    # API directory settings
    api_dir: str = DEFAULT_API_DIR

    # Search settings
    model_name: str = DEFAULT_MODEL
    top_k: int = DEFAULT_TOP_K
    batch_size: int = DEFAULT_BATCH_SIZE
    embedding_dimension: int = DEFAULT_DIMENSION

    # Pinecone settings
    pinecone_api_key: str = os.getenv("PINECONE_API_KEY", "")
    pinecone_environment: str = os.getenv("PINECONE_ENVIRONMENT", "gcp-starter")
    pinecone_index: str = os.getenv("PINECONE_INDEX", "api-search")
    pinecone_cloud: str = DEFAULT_PINECONE_CLOUD
    pinecone_region: str = DEFAULT_PINECONE_REGION

    # Cache settings
    cache_dir: str = os.getenv("CACHE_DIR", ".cache")
    cache_ttl: int = int(os.getenv("CACHE_TTL", "3600"))

    # Monitoring settings
    log_level: str = os.getenv("LOG_LEVEL", "INFO")
    metrics_enabled: bool = os.getenv("METRICS_ENABLED", "true").lower() == "true"

    # Feature settings
    features: Dict[str, Dict[str, str]] = field(
        default_factory=lambda: DEFAULT_FEATURES
    )

    # Display settings
    method_colors: Dict[str, str] = field(default_factory=lambda: METHOD_COLORS)
    score_adjustments: Dict[str, Any] = field(default_factory=lambda: SCORE_ADJUSTMENTS)


class ConfigManager:
    """Configuration manager."""

    def __init__(self, config_file: Optional[str] = None):
        """Initialize configuration manager.

        Args:
            config_file: Optional path to configuration file.
        """
        self.console = Console()
        self.config_file = config_file or os.getenv("CONFIG_FILE", "config.yml")

    def load_config(self) -> Config:
        """Load configuration from file and environment.

        Returns:
            Configuration object.
        """
        config_dict = {}

        # Load from file if exists
        if os.path.exists(self.config_file):
            try:
                with open(self.config_file, "r") as f:
                    config_dict = yaml.safe_load(f) or {}
            except Exception as e:
                self.console.print(
                    f"[yellow]Warning: Failed to load config file: {e}[/]"
                )

        # Create config object with defaults
        config = Config()

        # Update with file values
        for key, value in config_dict.items():
            if hasattr(config, key):
                setattr(config, key, value)

        return config

    def save_config(self, config: Config) -> None:
        """Save configuration to file.

        Args:
            config: Configuration object to save.
        """
        # Convert config to dict
        config_dict = {key: getattr(config, key) for key in config.__annotations__}

        # Save to file
        try:
            with open(self.config_file, "w") as f:
                yaml.safe_dump(config_dict, f)
        except Exception as e:
            self.console.print(f"[red]Error saving config: {e}[/]")

    def show_config(self) -> None:
        """Display current configuration."""
        config = self.load_config()

        self.console.print("\n[bold]Current Configuration:[/]")
        for key in config.__annotations__:
            value = getattr(config, key)
            if "api_key" in key.lower():
                value = "****" if value else ""
            self.console.print(f"{key}: {value}")

    def set_config(self, key: str, value: str) -> None:
        """Set configuration value.

        Args:
            key: Configuration key to set.
            value: Value to set.
        """
        config = self.load_config()

        if not hasattr(config, key):
            raise ValueError(f"Invalid configuration key: {key}")

        # Convert value to correct type
        target_type = config.__annotations__[key]
        if target_type == bool:
            value = value.lower() == "true"
        elif target_type == int:
            value = int(value)

        setattr(config, key, value)
        self.save_config(config)
