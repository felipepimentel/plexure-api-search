"""Plexure API Search package."""

import logging
from typing import Optional

from .config import Environment, init_config, get_current_config

__version__ = "0.1.0"

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

logger = logging.getLogger(__name__)

def initialize(env: Optional[Environment] = None) -> None:
    """Initialize the package.
    
    This is the main entry point for package initialization.
    It sets up configuration, logging, and other required components.
    
    Args:
        env: Environment to initialize for.
            Defaults to environment from ENVIRONMENT variable.
            
    Raises:
        Exception: If initialization fails.
    """
    try:
        # Initialize configuration
        config = init_config(env)
        
        # Configure logging
        logging.getLogger().setLevel(config.monitoring.log_level)
        
        logger.info(
            "Initialized Plexure API Search %s in %s environment",
            __version__,
            config.service.environment.value
        )
        
    except Exception as e:
        logger.error("Initialization failed: %s", e)
        raise

__all__ = [
    "__version__",
    "Environment",
    "get_current_config",
    "initialize",
]
