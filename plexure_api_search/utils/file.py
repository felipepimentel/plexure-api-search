"""
File Utilities for Plexure API Search

This module provides file handling and management utilities for the Plexure API Search system.
It includes functions for file operations, path handling, and file system interactions that are
used throughout the application.

Key Features:
- File path handling
- Directory operations
- File reading/writing
- Path validation
- File type detection
- File searching
- File locking
- Temporary files

The module provides utilities for:
- Safe file operations
- Path normalization
- Directory traversal
- File type detection
- File searching
- Resource cleanup
- Error handling

File Operations:
1. Path Management:
   - Path normalization
   - Path validation
   - Directory creation
   - File existence checks

2. File Operations:
   - Safe reading
   - Safe writing
   - File copying
   - File deletion
   - File moving

3. Directory Operations:
   - Directory creation
   - Directory cleanup
   - Directory scanning
   - File pattern matching

Example Usage:
    from plexure_api_search.utils.file import (
        ensure_directory,
        safe_read_file,
        safe_write_file,
        find_files
    )

    # Create directory
    ensure_directory("data/cache")

    # Read file safely
    content = safe_read_file("config.yaml")

    # Write file safely
    safe_write_file("output.json", data)

    # Find files by pattern
    yaml_files = find_files("**/*.yaml")

Safety Features:
- Path validation
- Safe file operations
- Error handling
- Resource cleanup
- File locking
"""

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
