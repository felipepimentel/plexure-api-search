"""File handling utilities for API specifications and data management."""

import glob
import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import yaml

logger = logging.getLogger(__name__)


def find_api_files(
    directory: Optional[str] = None, extensions: Optional[List[str]] = None
) -> List[str]:
    """Find API specification files in directory.

    Args:
        directory: Root directory to search
        extensions: List of file extensions to include (default: ['.yaml', '.yml', '.json'])

    Returns:
        List of file paths
    """
    if extensions is None:
        extensions = [".yaml", ".yml", ".json"]

    files = []
    for ext in extensions:
        pattern = os.path.join(directory or "apis", f"**/*{ext}")
        files.extend(glob.glob(pattern, recursive=True))

    logger.info(f"Found {len(files)} API files in {directory or 'apis'}")
    return files


def load_api_file(file_path: Union[str, Path]) -> Optional[Dict]:
    """Load API specification from file.

    Args:
        file_path: Path to API specification file

    Returns:
        Loaded API specification or None if loading fails
    """
    try:
        file_path = Path(file_path)
        ext = file_path.suffix.lower()

        with open(file_path, "r", encoding="utf-8") as f:
            if ext in [".yaml", ".yml"]:
                data = yaml.safe_load(f)
            elif ext == ".json":
                data = json.load(f)
            else:
                logger.error(f"Unsupported file extension: {ext}")
                return None

        if not isinstance(data, dict):
            logger.error(f"Invalid API specification format in {file_path}")
            return None

        return data

    except Exception as e:
        logger.error(f"Error loading {file_path}: {e}")
        return None


def load_json(file_path: Union[str, Path]) -> Optional[Dict]:
    """Load and parse JSON file.

    Args:
        file_path: Path to JSON file

    Returns:
        Parsed JSON data or None if error
    """
    try:
        file_path = Path(file_path)
        with open(file_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"Error loading JSON file {file_path}: {e}")
        return None


def save_json(data: Any, file_path: Union[str, Path], create_dirs: bool = True) -> bool:
    """Save data to JSON file.

    Args:
        data: Data to save
        file_path: Path to save to
        create_dirs: Whether to create parent directories

    Returns:
        True if successful, False otherwise
    """
    try:
        file_path = Path(file_path)
        if create_dirs:
            file_path.parent.mkdir(parents=True, exist_ok=True)

        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        return True
    except Exception as e:
        logger.error(f"Error saving JSON file {file_path}: {e}")
        return False


def save_data(
    data: Dict,
    file_path: Union[str, Path],
    create_dirs: bool = True,
    format: Optional[str] = None,
) -> bool:
    """Save data to file in JSON or YAML format.

    Args:
        data: Data to save
        file_path: Path to save file
        create_dirs: Whether to create parent directories
        format: Optional format override ('json' or 'yaml')

    Returns:
        True if successful, False otherwise
    """
    try:
        file_path = Path(file_path)
        if create_dirs:
            file_path.parent.mkdir(parents=True, exist_ok=True)

        # Determine format
        if format:
            use_yaml = format.lower() in ["yaml", "yml"]
        else:
            use_yaml = file_path.suffix.lower() in [".yaml", ".yml"]

        with open(file_path, "w", encoding="utf-8") as f:
            if use_yaml:
                yaml.safe_dump(data, f, indent=2)
            else:  # Default to JSON
                json.dump(data, f, indent=2)
        return True

    except Exception as e:
        logger.error(f"Error saving file {file_path}: {e}")
        return False


def ensure_directory(path: Union[str, Path]) -> bool:
    """Ensure directory exists, create if necessary.

    Args:
        path: Directory path

    Returns:
        True if directory exists/created, False on error
    """
    try:
        Path(path).mkdir(parents=True, exist_ok=True)
        return True
    except Exception as e:
        logger.error(f"Error creating directory {path}: {e}")
        return False


def delete_file(file_path: Union[str, Path], missing_ok: bool = True) -> bool:
    """Delete a file.

    Args:
        file_path: Path to file
        missing_ok: Whether to ignore missing files

    Returns:
        True if deleted or missing (when missing_ok=True), False on error
    """
    try:
        Path(file_path).unlink(missing_ok=missing_ok)
        return True
    except Exception as e:
        logger.error(f"Error deleting file {file_path}: {e}")
        return False
