"""API contract parser."""

import json
import logging
from pathlib import Path
from typing import Dict, List, Any

import yaml

logger = logging.getLogger(__name__)

class APIContractParser:
    """Parser for API contracts."""

    def parse_contract(self, file_path: str) -> List[Dict[str, Any]]:
        """Parse API contract file.
        
        Args:
            file_path: Path to API contract file
            
        Returns:
            List of endpoint metadata dicts
        """
        try:
            # Load contract
            file_path = Path(file_path)
            logger.info(f"Parsing contract: {file_path}")
            
            with open(file_path) as f:
                if file_path.suffix in ['.yaml', '.yml']:
                    spec_data = yaml.safe_load(f)
                elif file_path.suffix == '.json':
                    spec_data = json.load(f)
                else:
                    raise ValueError(f"Unsupported file type: {file_path.suffix}")

            # Extract endpoints
            endpoints = []
            paths = spec_data.get("paths", {})
            logger.info(f"Found {len(paths)} paths")

            for path, path_data in paths.items():
                if not isinstance(path_data, dict):
                    logger.warning(f"Invalid path data for {path}")
                    continue

                logger.debug(f"Processing path: {path}")
                logger.debug(f"Path data keys: {list(path_data.keys())}")

                # Get path-level parameters
                path_params = path_data.get("parameters", [])

                for method, endpoint_data in path_data.items():
                    if method == "parameters" or not isinstance(endpoint_data, dict):
                        continue

                    logger.debug(f"Processing method: {method}")
                    logger.debug(f"Endpoint data keys: {list(endpoint_data.keys())}")

                    # Combine path and endpoint parameters
                    parameters = path_params.copy()
                    parameters.extend(endpoint_data.get("parameters", []))

                    # Create endpoint metadata
                    endpoint = {
                        "path": path,
                        "method": method.upper(),
                        "description": endpoint_data.get("description", ""),
                        "summary": endpoint_data.get("summary", ""),
                        "tags": endpoint_data.get("tags", []),
                        "parameters": parameters,
                        "responses": endpoint_data.get("responses", {}),
                        "operationId": endpoint_data.get("operationId", ""),
                        "security": endpoint_data.get("security", []),
                    }

                    # Clean up empty fields
                    endpoint = {k: v for k, v in endpoint.items() if v}

                    endpoints.append(endpoint)
                    logger.debug(f"Added endpoint: {method.upper()} {path}")

            logger.info(f"Parsed {len(endpoints)} endpoints")
            return endpoints

        except Exception as e:
            logger.error(f"Failed to parse contract {file_path}: {e}")
            raise 