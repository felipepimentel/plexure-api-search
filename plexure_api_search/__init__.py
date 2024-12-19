"""Plexure API Search."""

import logging
from importlib.metadata import version

from .config import config_instance

__version__ = version("plexure-api-search")

logger = logging.getLogger(__name__)

# Log initialization
logger.info(
    f"Initialized Plexure API Search {__version__} in {config_instance.env} environment"
)

__all__ = [
    "__version__",
    "Environment",
    "get_current_config",
    "initialize",
]
