"""Configuration loader module."""

import os
from pathlib import Path
from typing import Dict, Optional, Union
import yaml
import json
from dotenv import load_dotenv

from .base import Config, Environment, ConfigurationError

class ConfigLoader:
    """Configuration loader class."""

    def __init__(self, config_dir: Optional[Union[str, Path]] = None):
        """Initialize configuration loader.
        
        Args:
            config_dir: Directory containing configuration files.
                Defaults to current directory.
        """
        self.config_dir = Path(config_dir or os.getcwd())
        
        # Load environment variables from .env file
        env_file = self.config_dir / ".env"
        if env_file.exists():
            load_dotenv(env_file)

    def load(self, env: Optional[Environment] = None) -> Config:
        """Load configuration.
        
        Args:
            env: Environment to load configuration for.
                Defaults to environment from ENVIRONMENT variable.
                
        Returns:
            Loaded configuration.
            
        Raises:
            ConfigurationError: If configuration loading fails.
        """
        try:
            # Determine environment
            env = env or Environment(
                os.getenv("ENVIRONMENT", Environment.DEVELOPMENT)
            )
            
            # Load configuration files
            config_data = self._load_config_files(env)
            
            # Create configuration instance
            config = Config.from_dict(config_data)
            
            return config
            
        except Exception as e:
            raise ConfigurationError(f"Failed to load configuration: {e}")

    def _load_config_files(self, env: Environment) -> Dict:
        """Load configuration files.
        
        Args:
            env: Environment to load configuration for.
            
        Returns:
            Configuration data.
            
        Raises:
            ConfigurationError: If configuration loading fails.
        """
        try:
            # Load base configuration
            base_config = self._load_file("config.yaml")
            
            # Load environment configuration
            env_config = self._load_file(f"config.{env}.yaml")
            
            # Merge configurations
            config = {**base_config, **env_config}
            
            # Add environment to service config
            if "service" not in config:
                config["service"] = {}
            config["service"]["environment"] = env
            
            return config
            
        except Exception as e:
            raise ConfigurationError(f"Failed to load configuration files: {e}")

    def _load_file(self, filename: str) -> Dict:
        """Load configuration file.
        
        Args:
            filename: Name of file to load.
            
        Returns:
            Configuration data.
            
        Raises:
            ConfigurationError: If file loading fails.
        """
        path = self.config_dir / filename
        if not path.exists():
            return {}
            
        try:
            with open(path) as f:
                if path.suffix == ".yaml":
                    return yaml.safe_load(f) or {}
                elif path.suffix == ".json":
                    return json.load(f)
                else:
                    raise ValueError("Unsupported file format")
                    
        except Exception as e:
            raise ConfigurationError(f"Failed to load {filename}: {e}")

def get_config(
    env: Optional[Environment] = None,
    config_dir: Optional[Union[str, Path]] = None,
) -> Config:
    """Get configuration.
    
    This is a convenience function that creates a loader
    and loads configuration in one step.
    
    Args:
        env: Environment to load configuration for.
            Defaults to environment from ENVIRONMENT variable.
        config_dir: Directory containing configuration files.
            Defaults to current directory.
            
    Returns:
        Loaded configuration.
        
    Raises:
        ConfigurationError: If configuration loading fails.
    """
    loader = ConfigLoader(config_dir)
    return loader.load(env) 