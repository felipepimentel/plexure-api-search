"""Utility functions for file handling and common operations."""

import glob
import json
import logging
import os
from typing import Dict, List, Optional, Any

import yaml

logger = logging.getLogger(__name__)


class FileUtils:
    """File handling utilities."""

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


class APISpecUtils:
    """API specification utilities."""

    @staticmethod
    def extract_endpoints(spec: Dict) -> List[Dict]:
        """Extract endpoints from API specification.

        Args:
            spec: API specification dictionary

        Returns:
            List of endpoint dictionaries
        """
        endpoints = []

        try:
            # Extract API info
            api_name = spec.get("info", {}).get("title", "Unknown API")
            api_version = spec.get("info", {}).get("version", "1.0.0")

            # Get paths
            paths = spec.get("paths", {})

            # Process each path
            for path, path_data in paths.items():
                # Process each method
                for method, endpoint_data in path_data.items():
                    if method.upper() not in ["GET", "POST", "PUT", "DELETE", "PATCH"]:
                        continue

                    # Create endpoint object
                    endpoint = {
                        "api_name": api_name,
                        "api_version": api_version,
                        "path": path,
                        "method": method.upper(),
                        "description": endpoint_data.get("description", ""),
                        "summary": endpoint_data.get("summary", ""),
                        "parameters": endpoint_data.get("parameters", []),
                        "responses": endpoint_data.get("responses", {}),
                        "tags": endpoint_data.get("tags", []),
                        "operationId": endpoint_data.get("operationId", ""),
                        "deprecated": endpoint_data.get("deprecated", False),
                        "security": endpoint_data.get("security", []),
                        "requires_auth": bool(endpoint_data.get("security", [])),
                        "full_path": f"{api_name}{path}",
                    }

                    logger.info(f"Found endpoint: {method.upper()} {path}")
                    endpoints.append(endpoint)

        except Exception as e:
            logger.error(f"Error extracting endpoints: {e}")

        logger.info(f"Extracted {len(endpoints)} endpoints")
        return endpoints


class MetadataUtils:
    """Metadata handling utilities."""

    @staticmethod
    def sanitize_metadata(metadata: Dict) -> Dict:
        """Sanitize metadata for storage.

        Args:
            metadata: Raw metadata dictionary

        Returns:
            Sanitized metadata dictionary
        """
        def sanitize_value(value: Any) -> Any:
            """Sanitize a single value."""
            if isinstance(value, (str, int, float, bool)):
                return str(value)
            elif isinstance(value, list):
                # Convert list items to strings
                return [str(item) for item in value]
            elif isinstance(value, dict):
                # Convert dict to list of key:value strings
                return [f"{k}:{v}" for k, v in value.items()]
            else:
                return str(value)
                
        sanitized = {}
        
        # Process each field
        for key, value in metadata.items():
            try:
                if key == "parameters":
                    # Convert parameters to list of strings
                    sanitized[key] = [
                        f"{p.get('name', '')}:{p.get('in', '')}:{p.get('description', '')}"
                        for p in value
                    ]
                elif key == "responses":
                    # Convert responses to list of strings
                    sanitized[key] = [
                        f"{code}:{resp.get('description', '')}"
                        for code, resp in value.items()
                    ]
                elif key == "security":
                    # Convert security to list of strings
                    sanitized[key] = [str(s) for s in value] if value else []
                else:
                    sanitized[key] = sanitize_value(value)
                    
            except Exception as e:
                logger.error(f"Error sanitizing metadata field {key}: {e}")
                sanitized[key] = str(value)
                
        return sanitized
        
    @staticmethod
    def create_endpoint_id(endpoint: Dict) -> str:
        """Create unique ID for endpoint."""
        id_parts = [
            str(endpoint.get("api_name", "")),
            str(endpoint.get("method", "")),
            str(endpoint.get("path", "")),
        ]
        
        endpoint_id = "_".join(id_parts)
        return endpoint_id.replace("/", "_").replace("{", "").replace("}", "")
