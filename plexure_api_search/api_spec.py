"""Utilities for processing API specifications and extracting endpoints."""

import logging
from typing import Dict, List

logger = logging.getLogger(__name__)


class APISpec:
    """API specification utilities for endpoint extraction and processing."""

    @staticmethod
    def extract_endpoints(spec: Dict) -> List[Dict]:
        """Extract endpoints from API specification.

        Args:
            spec: API specification dictionary

        Returns:
            List of endpoint dictionaries with metadata
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

                    logger.debug(f"Found endpoint: {method.upper()} {path}")
                    endpoints.append(endpoint)

        except Exception as e:
            logger.error(f"Error extracting endpoints: {e}")

        logger.info(f"Extracted {len(endpoints)} endpoints")
        return endpoints
