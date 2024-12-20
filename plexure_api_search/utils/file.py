"""File utilities."""

import logging
from pathlib import Path
from typing import List, Optional

from ..config import config

logger = logging.getLogger(__name__)


def find_api_files(directory: Optional[str] = None) -> List[str]:
    """Find API contract files in directory.

    Args:
        directory: Directory to search in (default: config.api_dir)

    Returns:
        List of file paths
    """
    # Use configured directory if none provided
    if directory is None:
        directory = config.api_dir

    # Convert to Path
    directory = Path(directory)
    if not directory.exists():
        logger.warning(f"Directory does not exist: {directory}")
        return []

    # Find API files
    files = []
    for pattern in ["*.yaml", "*.yml", "*.json"]:
        files.extend(str(p) for p in directory.glob(pattern))

    # Sort for consistent ordering
    files.sort()

    return files
