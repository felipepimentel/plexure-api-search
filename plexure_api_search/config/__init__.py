"""Configuration management package."""

from typing import Optional, Union
from pathlib import Path

from .base import (
    Config,
    Environment,
    ConfigurationError,
    InvalidConfigurationError,
    MissingConfigurationError,
)
from .loader import ConfigLoader, get_config
from .validation import validate_config, validate_config_file

__all__ = [
    "Config",
    "Environment",
    "ConfigurationError",
    "InvalidConfigurationError",
    "MissingConfigurationError",
    "ConfigLoader",
    "get_config",
    "validate_config",
    "validate_config_file",
    "init_config",
    "config_instance",
]

# Global configuration instance
_config: Optional[Config] = None

def init_config(
    env: Optional[Environment] = None,
    config_dir: Optional[Union[str, Path]] = None,
) -> Config:
    """Initialize configuration.
    
    This is the main entry point for configuration initialization.
    It ensures that configuration is loaded and validated only once.
    
    Args:
        env: Environment to load configuration for.
            Defaults to environment from ENVIRONMENT variable.
        config_dir: Directory containing configuration files.
            Defaults to current directory.
            
    Returns:
        Initialized configuration.
        
    Raises:
        ConfigurationError: If configuration initialization fails.
    """
    global _config
    
    if _config is not None:
        return _config
    
    try:
        # Create loader
        loader = ConfigLoader(config_dir)
        
        # Load and validate configuration
        config = loader.load(env)
        config.validate()
        
        # Store global instance
        _config = config
        return config
        
    except Exception as e:
        raise ConfigurationError(f"Configuration initialization failed: {e}")

def get_current_config() -> Config:
    """Get current configuration instance.
    
    Returns:
        Current configuration instance.
        
    Raises:
        ConfigurationError: If configuration not initialized.
    """
    if _config is None:
        raise ConfigurationError("Configuration not initialized")
    return _config

# Initialize default configuration instance
config_instance = init_config()