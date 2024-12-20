"""Initialization module."""

import logging
from importlib.metadata import version

from .config import config

logger = logging.getLogger(__name__)

__version__ = version("plexure-api-search")

def initialize() -> None:
    """Initialize the package."""
    try:
        # Get environment value
        env = config.environment
        
        # Log initialization
        logger.info(
            f"Initialized Plexure API Search {__version__} in {env} environment"
        )
    except Exception as e:
        logger.error(f"Failed to initialize: {e}")
        raise 