"""Plexure API Search."""

import logging
from importlib.metadata import version

__version__ = version("plexure-api-search")

logger = logging.getLogger(__name__)

__all__ = ["__version__"]
