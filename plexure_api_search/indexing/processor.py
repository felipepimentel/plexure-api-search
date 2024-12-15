"""API specification processor for indexing."""

import logging
from typing import Dict, List, Optional

from ..utils.file import FileUtils, find_apis, load_api

logger = logging.getLogger(__name__)


class APIProcessor:
    """API specification processor for indexing."""

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

    @staticmethod
    def sanitize_metadata(metadata: Dict) -> Dict:
        """Sanitize metadata for storage.

        Args:
            metadata: Raw metadata dictionary

        Returns:
            Sanitized metadata dictionary with standardized values
        """
        def sanitize_value(value: any) -> any:
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

    def process_api_files(self, directory: Optional[str] = None) -> List[Dict]:
        """Process all API files in directory.

        Args:
            directory: Directory containing API files (default: settings.paths.api_dir)

        Returns:
            List of processed endpoints with metadata
        """
        all_endpoints = []

        # Find API files
        api_files = find_apis(directory)

        # Process each file
        for file_path in api_files:
            try:
                # Load API spec
                spec = load_api(file_path)
                if not spec:
                    continue

                # Extract endpoints
                endpoints = self.extract_endpoints(spec)

                # Process each endpoint
                for endpoint in endpoints:
                    # Sanitize metadata
                    sanitized = self.sanitize_metadata(endpoint)
                    
                    # Add endpoint ID
                    sanitized["id"] = self.create_endpoint_id(endpoint)
                    
                    all_endpoints.append(sanitized)

            except Exception as e:
                logger.error(f"Error processing file {file_path}: {e}")

        logger.info(f"Processed {len(all_endpoints)} total endpoints")
        return all_endpoints


# Global processor instance
processor = APIProcessor()

__all__ = ["processor", "APIProcessor"] 