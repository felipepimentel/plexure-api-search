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
        self.endpoint_metadata = {}

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

                    logger.debug(f"Processing file: {file_path}")
                    logger.debug(f"Spec data keys: {list(spec_data.keys())}")

                    # Extract endpoints
                    endpoints = []
                    paths = spec_data.get("paths", {})
                    
                    logger.debug(f"Found {len(paths)} paths in {file_path}")
                    
                    for path, path_data in paths.items():
                        if not isinstance(path_data, dict):
                            logger.warning(f"Invalid path data for {path}: {path_data}")
                            continue
                            
                        logger.debug(f"Processing path: {path}")
                        logger.debug(f"Path data keys: {list(path_data.keys())}")
                        
                        for method, endpoint_data in path_data.items():
                            if method == "parameters" or not isinstance(endpoint_data, dict):
                                continue

                            logger.debug(f"Processing method: {method}")
                            logger.debug(f"Endpoint data keys: {list(endpoint_data.keys())}")

                            # Extract endpoint information
                            description = endpoint_data.get("description", "")
                            if not description:
                                description = endpoint_data.get("summary", "")

                            # Create endpoint metadata with consistent structure
                            endpoint = {
                                "path": path,
                                "method": method.upper(),
                                "description": description,
                                "tags": endpoint_data.get("tags", []),
                                "parameters": [],  # We'll handle parameters later
                                "responses": endpoint_data.get("responses", {}),
                                "operationId": endpoint_data.get("operationId", ""),
                                "summary": endpoint_data.get("summary", ""),
                                "security": endpoint_data.get("security", [])
                            }
                            endpoints.append(endpoint)
                            logger.debug(f"Added endpoint: {method.upper()} {path}")

                    # Index endpoints
                    if endpoints:
                        logger.debug(f"Indexing {len(endpoints)} endpoints from {file_path}")
                        try:
                            self.index_endpoints(endpoints)
                            total_endpoints += len(endpoints)
                            indexed_apis.append({
                                "path": file_path,
                                "endpoints": len(endpoints),
                            })
                        except Exception as e:
                            logger.error(f"Failed to index endpoints from {file_path}: {e}")
                            failed_files.append({
                                "path": file_path,
                                "error": str(e),
                            })
                    else:
                        logger.warning(f"No valid endpoints found in {file_path}")
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

            # Skip if no endpoints
            if not endpoints:
                logger.warning("No endpoints to index")
                return

            # Log endpoint info
            logger.debug(f"Processing {len(endpoints)} endpoints")
            for endpoint in endpoints:
                logger.debug(f"Endpoint: {endpoint['method']} {endpoint['path']}")

            # Create text list and IDs
            texts = []
            ids = []
            for endpoint in endpoints:
                text = f"{endpoint['method']} {endpoint['path']} {endpoint.get('summary', '')} {endpoint.get('description', '')}"
                texts.append(text)
                ids.append(self.next_id)
                self.next_id += 1
                logger.debug(f"Added text: {text}")
                logger.debug(f"Added ID: {ids[-1]}")

            # Log texts and IDs
            logger.debug(f"Number of texts: {len(texts)}")
            logger.debug(f"Number of IDs: {len(ids)}")
            logger.debug(f"Sample text: {texts[0] if texts else 'No texts'}")
            logger.debug(f"Sample ID: {ids[0] if ids else 'No IDs'}")
            logger.debug(f"ID range: {min(ids) if ids else 'N/A'} to {max(ids) if ids else 'N/A'}")

            # Generate embeddings in batch
            embeddings = model_service.encode(texts)
            logger.debug(f"Generated embeddings shape: {embeddings.shape}")
            logger.debug(f"Generated embeddings type: {embeddings.dtype}")
            logger.debug(f"Generated embeddings sample: {embeddings[0][:5] if len(embeddings) > 0 else 'No embeddings'}")

            # Store vectors
            if len(embeddings) > 0:
                logger.debug(f"Storing {len(embeddings)} vectors with {len(ids)} IDs")
                # Convert IDs to numpy array
                ids_array = np.array(ids, dtype=np.int64)
                # Ensure contiguous arrays
                embeddings = np.ascontiguousarray(embeddings)
                ids_array = np.ascontiguousarray(ids_array)
                # Store vectors
                vector_store.store_vectors(embeddings, ids_array)

                # Store endpoint metadata
                for i, endpoint in enumerate(endpoints):
                    self.endpoint_metadata[ids[i]] = {
                        "method": endpoint["method"],
                        "path": endpoint["path"],
                        "description": endpoint.get("description", ""),
                        "summary": endpoint.get("summary", ""),
                        "parameters": endpoint.get("parameters", []),
                        "responses": endpoint.get("responses", {}),
                        "tags": endpoint.get("tags", []),
                    }

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
