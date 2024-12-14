"""File handling utilities for API specifications and cache management."""

import glob
import json
import logging
import os
from typing import Dict, List, Optional

import yaml

logger = logging.getLogger(__name__)


class FileUtils:
    """File handling utilities for API specifications and cache files."""

    @staticmethod
    def find_api_files(
        directory: str, extensions: Optional[List[str]] = None
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
            pattern = os.path.join(directory, f"**/*{ext}")
            files.extend(glob.glob(pattern, recursive=True))

        logger.info(f"Found {len(files)} API files")
        return files

    @staticmethod
    def load_api_file(file_path: str) -> Optional[Dict]:
        """Load API specification from file.

        Args:
            file_path: Path to API specification file

        Returns:
            Loaded API specification or None if loading fails
        """
        try:
            _, ext = os.path.splitext(file_path)

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

    @staticmethod
    def save_cache(data: Dict, cache_file: str) -> bool:
        """Save data to cache file.

        Args:
            data: Data to cache
            cache_file: Path to cache file

        Returns:
            True if successful, False otherwise
        """
        try:
            os.makedirs(os.path.dirname(cache_file), exist_ok=True)
            with open(cache_file, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)
            return True
        except Exception as e:
            logger.error(f"Error saving cache: {e}")
            return False

    @staticmethod
    def load_cache(cache_file: str) -> Optional[Dict]:
        """Load data from cache file.

        Args:
            cache_file: Path to cache file

        Returns:
            Cached data or None if loading fails
        """
        try:
            if not os.path.exists(cache_file):
                return None

            with open(cache_file, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Error loading cache: {e}")
            return None 