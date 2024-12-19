"""API contract indexing and vector creation."""

import json
import logging
import os
from pathlib import Path
from typing import Dict, List, Optional, Any

import numpy as np
import yaml
from dependency_injector.wiring import inject
from dependency_injector.providers import Provider

from ..config import config_instance
from ..monitoring.metrics import MetricsManager
from ..services.models import model_service
from ..services.vector_store import vector_store
from ..utils.file import find_api_files

logger = logging.getLogger(__name__)

class APIIndexer:
    """API contract indexer."""

    def __init__(self):
        """Initialize indexer."""
        self.metrics = MetricsManager()
        self.initialized = False
        self.next_id = 0

    def initialize(self) -> None:
        """Initialize indexer."""
        if self.initialized:
            return

        try:
            # Initialize services
            model_service.initialize()
            vector_store.initialize()
            self.initialized = True
            logger.info("Indexer initialized")

        except Exception as e:
            logger.error(f"Failed to initialize indexer: {e}")
            raise

    def cleanup(self) -> None:
        """Clean up indexer."""
        model_service.cleanup()
        vector_store.cleanup()
        self.initialized = False
        logger.info("Indexer cleaned up")

    def index_directory(
        self,
        directory: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Index all API files in directory.
        
        Args:
            directory: Directory to process (default: config.api_dir)
            
        Returns:
            Dictionary with indexing results
        """
        if not self.initialized:
            self.initialize()

        try:
            # Start timer
            start_time = self.metrics.start_timer()

            # Find API files
            files = find_api_files(directory)
            if not files:
                logger.warning("No API files found")
                return {
                    "total_files": 0,
                    "total_endpoints": 0,
                    "failed_files": [],
                    "indexed_apis": [],
                }

            # Process each file
            total_endpoints = 0
            failed_files = []
            indexed_apis = []

            for file_path in files:
                try:
                    # Load API spec
                    with open(file_path) as f:
                        if file_path.endswith('.yaml') or file_path.endswith('.yml'):
                            spec_data = yaml.safe_load(f)
                        elif file_path.endswith('.json'):
                            spec_data = json.load(f)
                        else:
                            continue

                    # Extract endpoints
                    endpoints = []
                    paths = spec_data.get("paths", {})
                    
                    # Get global parameters if any
                    global_parameters = spec_data.get("parameters", {})
                    
                    for path, path_data in paths.items():
                        if not isinstance(path_data, dict):
                            continue
                            
                        # Get path-level parameters
                        path_parameters = path_data.get("parameters", [])
                        
                        for method, endpoint_data in path_data.items():
                            if method == "parameters" or not isinstance(endpoint_data, dict):
                                continue

                            # Extract endpoint information
                            description = endpoint_data.get("description", "")
                            if not description:
                                description = endpoint_data.get("summary", "")

                            # Combine parameters from different levels
                            parameters = []
                            
                            # Add global parameters
                            for param_name, param_data in global_parameters.items():
                                if isinstance(param_data, dict):
                                    param = {
                                        "name": param_name,
                                        "type": param_data.get("type", param_data.get("schema", {}).get("type", "")),
                                        "description": param_data.get("description", ""),
                                        "required": param_data.get("required", False),
                                        "in": param_data.get("in", "")
                                    }
                                    parameters.append(param)
                            
                            # Add path parameters
                            for param in path_parameters:
                                if isinstance(param, dict):
                                    param_copy = param.copy()
                                    if "schema" in param_copy:
                                        schema = param_copy.pop("schema")
                                        if isinstance(schema, dict):
                                            param_copy["type"] = schema.get("type", "")
                                    parameters.append(param_copy)
                            
                            # Add operation parameters
                            for param in endpoint_data.get("parameters", []):
                                if isinstance(param, dict):
                                    param_copy = param.copy()
                                    if "schema" in param_copy:
                                        schema = param_copy.pop("schema")
                                        if isinstance(schema, dict):
                                            param_copy["type"] = schema.get("type", "")
                                    parameters.append(param_copy)

                            # Create endpoint metadata with consistent structure
                            endpoint = {
                                "path": path,
                                "method": method.upper(),
                                "description": description,
                                "tags": endpoint_data.get("tags", []),
                                "parameters": parameters,
                                "responses": endpoint_data.get("responses", {}),
                                "operationId": endpoint_data.get("operationId", ""),
                                "summary": endpoint_data.get("summary", ""),
                                "security": endpoint_data.get("security", [])
                            }
                            endpoints.append(endpoint)

                    # Index endpoints
                    if endpoints:
                        logger.debug(f"Indexing {len(endpoints)} endpoints from {file_path}")
                        self.index_endpoints(endpoints)
                        total_endpoints += len(endpoints)
                        indexed_apis.append({
                            "path": file_path,
                            "endpoints": len(endpoints),
                        })
                    else:
                        failed_files.append({
                            "path": file_path,
                            "error": "No valid endpoints found",
                        })

                except Exception as e:
                    logger.error(f"Failed to index file {file_path}: {e}")
                    failed_files.append({
                        "path": file_path,
                        "error": str(e),
                    })

            # Stop timer
            self.metrics.stop_timer(start_time, "indexing")

            return {
                "total_files": len(files),
                "total_endpoints": total_endpoints,
                "failed_files": failed_files,
                "indexed_apis": indexed_apis,
            }

        except Exception as e:
            logger.error(f"Failed to index directory: {e}")
            self.metrics.increment("indexing_errors")
            raise

    def index_endpoints(self, endpoints: List[Dict[str, Any]]) -> None:
        """Index API endpoints.
        
        Args:
            endpoints: List of endpoint dictionaries
        """
        if not self.initialized:
            self.initialize()

        try:
            # Start timer
            start_time = self.metrics.start_timer()

            # Create text list
            texts = []
            for endpoint in endpoints:
                text = f"{endpoint['method']} {endpoint['path']} {endpoint.get('summary', '')} {endpoint.get('description', '')}"
                texts.append(text)
                logger.debug(f"Added text: {text}")

            # Generate embeddings in batch
            embeddings = model_service.encode(texts)
            logger.debug(f"Generated embeddings shape: {embeddings.shape}")
            logger.debug(f"Generated embeddings type: {embeddings.dtype}")
            logger.debug(f"Generated embeddings sample: {embeddings[0][:5]}")

            # Store vectors
            if len(embeddings) > 0:
                vector_store.store_vectors(embeddings)

            # Stop timer
            self.metrics.stop_timer(start_time, "indexing")
            self.metrics.increment(
                "endpoints_indexed",
                {"count": len(endpoints)},
            )

        except Exception as e:
            logger.error(f"Failed to index endpoints: {e}")
            self.metrics.increment("indexing_errors")
            raise

# Global instance
api_indexer = APIIndexer()
