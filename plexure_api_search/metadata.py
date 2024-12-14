"""Utilities for handling and sanitizing API metadata."""

import logging
from typing import Any, Dict

logger = logging.getLogger(__name__)


class Metadata:
    """Metadata handling utilities for API endpoints."""

    @staticmethod
    def sanitize_metadata(metadata: Dict) -> Dict:
        """Sanitize metadata for storage.

        Args:
            metadata: Raw metadata dictionary

        Returns:
            Sanitized metadata dictionary with standardized values
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
        """Create unique ID for endpoint.

        Args:
            endpoint: Endpoint dictionary with metadata

        Returns:
            Unique identifier string for the endpoint
        """
        id_parts = [
            str(endpoint.get("api_name", "")),
            str(endpoint.get("method", "")),
            str(endpoint.get("path", "")),
        ]

        endpoint_id = "_".join(id_parts)
        return endpoint_id.replace("/", "_").replace("{", "").replace("}", "")
