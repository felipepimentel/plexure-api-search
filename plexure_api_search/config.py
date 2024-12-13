"""Configuration management for Plexure API Search."""

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import yaml
from dotenv import load_dotenv
from rich.console import Console

# Load environment variables
load_dotenv()

@dataclass
class Config:
    """Configuration settings."""
    
    # API directory settings
    api_dir: str = os.getenv("API_DIR", "./apis")
    
    # Search settings
    model_name: str = os.getenv(
        "MODEL_NAME",
        "sentence-transformers/all-MiniLM-L6-v2"
    )
    top_k: int = int(os.getenv("TOP_K", "5"))
    
    # Pinecone settings
    pinecone_api_key: str = os.getenv("PINECONE_API_KEY", "")
    pinecone_environment: str = os.getenv("PINECONE_ENVIRONMENT", "")
    pinecone_index: str = os.getenv("PINECONE_INDEX", "api-search")
    
    # Cache settings
    cache_dir: str = os.getenv("CACHE_DIR", ".cache")
    cache_ttl: int = int(os.getenv("CACHE_TTL", "3600"))
    
    # Monitoring settings
    log_level: str = os.getenv("LOG_LEVEL", "INFO")
    metrics_enabled: bool = os.getenv("METRICS_ENABLED", "true").lower() == "true"


class ConfigManager:
    """Configuration manager."""
    
    def __init__(self, config_file: Optional[str] = None):
        """Initialize configuration manager.
        
        Args:
            config_file: Optional path to configuration file.
        """
        self.console = Console()
        self.config_file = config_file or os.getenv(
            "CONFIG_FILE",
            "config.yml"
        )
        
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
        config_dict = {
            key: getattr(config, key)
            for key in config.__annotations__
        }
        
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