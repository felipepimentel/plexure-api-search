"""API contract indexing and vector creation."""

import json
import logging
import os
import time
from datetime import datetime
import traceback
from typing import Any, Dict, List, Optional

import yaml

from ..config import config_instance
from ..search.vectorizer import TripleVectorizer
from ..integrations import pinecone_instance
from ..monitoring.events import Event, EventType, publisher
from ..utils.file import find_api_files
from .validation import DataValidator

logger = logging.getLogger(__name__)


class APIIndexer:
    """Handles API contract indexing and vector creation."""

    def __init__(self):
        """Initialize indexer."""
        self.client = pinecone_instance
        self.vectorizer = TripleVectorizer()
        self.validator = DataValidator()

        # Start publisher
        try:
            publisher.start()
            time.sleep(0.1)  # Allow time for connection
            logger.debug("Publisher started for indexer")
        except Exception as e:
            logger.warning(f"Failed to start publisher: {e}")

    def _safe_load_yaml(self, file_path: str) -> Optional[Dict[str, Any]]:
        """Safely load YAML file.
        
        Args:
            file_path: Path to YAML file
            
        Returns:
            Loaded YAML data or None if failed
        """
        try:
            publisher.emit(Event(
                type=EventType.INDEXING_FILE_STARTED,
                timestamp=datetime.now(),
                component="indexer",
                description=f"Loading file: {file_path}"
            ))
            
            with open(file_path, 'r') as f:
                data = yaml.safe_load(f)
            
            publisher.emit(Event(
                type=EventType.INDEXING_FILE_COMPLETED,
                timestamp=datetime.now(),
                component="indexer",
                description=f"Successfully loaded file: {file_path}"
            ))
            
            return data
            
        except Exception as e:
            logger.error(f"Error loading file {file_path}: {e}")
            logger.error(traceback.format_exc())
            
            publisher.emit(Event(
                type=EventType.INDEXING_FILE_FAILED,
                timestamp=datetime.now(),
                component="indexer",
                description=f"Failed to load file: {file_path}",
                error=str(e),
                success=False
            ))
            
            return None

    def index_directory(
        self,
        directory: Optional[str] = None,
        force: bool = False,
        validate: bool = True,
    ) -> Dict[str, Any]:
        """Index all API files in directory.

        Args:
            directory: Directory to process (default: settings.paths.api_dir)
            force: Whether to force reindexing
            validate: Whether to validate data before indexing

        Returns:
            Dictionary with indexing results
        """
        start_time = time.time()
        try:
            publisher.emit(Event(
                type=EventType.INDEXING_STARTED,
                timestamp=datetime.now(),
                component="indexer",
                description="Starting API indexing"
            ))

            # Find API files
            files = find_api_files(directory)
            if not files:
                logger.warning("No API files found")
                publisher.emit(Event(
                    type=EventType.INDEXING_COMPLETED,
                    timestamp=datetime.now(),
                    component="indexer",
                    description="No API files found",
                    duration_ms=(time.time() - start_time) * 1000,
                    metadata={"total_files": 0}
                ))
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

            for i, file_path in enumerate(files):
                file_start_time = time.time()
                try:
                    # Load API spec
                    spec_data = self._safe_load_yaml(file_path)
                    if not spec_data:
                        continue

                    # Index endpoints
                    endpoints = self.index_file(spec_data, force=force)
                    if endpoints:
                        total_endpoints += len(endpoints)
                        indexed_apis.append({
                            "path": file_path,
                            "endpoints": len(endpoints),
                        })

                        publisher.emit(Event(
                            type=EventType.INDEXING_FILE_COMPLETED,
                            timestamp=datetime.now(),
                            component="indexer",
                            description=(
                                f"Successfully indexed {len(endpoints)} endpoints from {file_path}"
                            ),
                            duration_ms=(time.time() - file_start_time) * 1000,
                            metadata={
                                "file": file_path,
                                "endpoints": len(endpoints),
                                "progress": (i + 1) / len(files),
                            },
                        ))
                    else:
                        failed_files.append({
                            "path": file_path,
                            "error": "No valid endpoints found",
                        })
                        publisher.emit(Event(
                            type=EventType.INDEXING_FILE_FAILED,
                            timestamp=datetime.now(),
                            component="indexer",
                            description=f"No valid endpoints found in {file_path}",
                            duration_ms=(time.time() - file_start_time) * 1000,
                            success=False,
                            metadata={
                                "file": file_path,
                                "error": "No valid endpoints found",
                            },
                        ))

                except Exception as e:
                    failed_files.append({
                        "path": file_path,
                        "error": str(e),
                    })
                    publisher.emit(Event(
                        type=EventType.INDEXING_FILE_FAILED,
                        timestamp=datetime.now(),
                        component="indexer",
                        description=f"Failed to index file: {file_path}",
                        duration_ms=(time.time() - file_start_time) * 1000,
                        success=False,
                        error=str(e),
                        metadata={
                            "file": file_path,
                            "error": str(e),
                        },
                    ))

            # Emit completion event
            duration_ms = (time.time() - start_time) * 1000
            publisher.emit(Event(
                type=EventType.INDEXING_COMPLETED,
                timestamp=datetime.now(),
                component="indexer",
                description="API indexing completed",
                duration_ms=duration_ms,
                metadata={
                    "total_files": len(files),
                    "total_endpoints": total_endpoints,
                    "failed_files": len(failed_files),
                },
            ))

            return {
                "total_files": len(files),
                "total_endpoints": total_endpoints,
                "failed_files": failed_files,
                "indexed_apis": indexed_apis,
            }

        except Exception as e:
            # Emit failure event
            duration_ms = (time.time() - start_time) * 1000
            publisher.emit(Event(
                type=EventType.INDEXING_FAILED,
                timestamp=datetime.now(),
                component="indexer",
                description="API indexing failed",
                duration_ms=duration_ms,
                success=False,
                error=str(e),
            ))
            
            logger.error(f"Indexing failed: {e}")
            logger.error(traceback.format_exc())
            return {
                "total_files": 0,
                "total_endpoints": 0,
                "failed_files": [],
                "indexed_apis": [],
                "error": str(e),
            }
        finally:
            # Stop publisher
            publisher.stop()

    def _simplify_metadata(self, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Simplify metadata to be compatible with Pinecone.
        
        Args:
            metadata: Original metadata
            
        Returns:
            Simplified metadata with only simple types
        """
        simplified = {}
        
        # Copy simple fields
        simplified["endpoint"] = metadata.get("endpoint", "")
        simplified["method"] = metadata.get("method", "")
        simplified["description"] = metadata.get("description", "")
        
        # Convert tags to comma-separated string
        tags = metadata.get("tags", [])
        simplified["tags"] = ",".join(str(tag) for tag in tags) if tags else ""
        
        # Extract parameter names and types
        parameters = metadata.get("parameters", [])
        param_names = []
        param_types = []
        for param in parameters:
            if isinstance(param, dict):
                param_names.append(param.get("name", ""))
                param_types.append(param.get("in", ""))
        simplified["parameter_names"] = ",".join(param_names)
        simplified["parameter_types"] = ",".join(param_types)
        
        # Extract response codes
        responses = metadata.get("responses", {})
        response_codes = list(str(code) for code in responses.keys())
        simplified["response_codes"] = ",".join(response_codes)
        
        return simplified

    def index_file(
        self,
        spec_data: Dict[str, Any],
        force: bool = False,
    ) -> List[Dict[str, Any]]:
        """Index a single API specification file.

        Args:
            spec_data: OpenAPI specification data
            force: Whether to force reindexing

        Returns:
            List of indexed endpoints
        """
        try:
            endpoints = []
            paths = spec_data.get("paths", {})

            for path, path_data in paths.items():
                for method, endpoint_data in path_data.items():
                    if not isinstance(endpoint_data, dict):
                        continue

                    # Extract endpoint information
                    description = endpoint_data.get("description", "")
                    if not description:
                        description = endpoint_data.get("summary", "")

                    # Create endpoint metadata
                    endpoint_metadata = {
                        "endpoint": path,
                        "method": method.upper(),
                        "description": description,
                        "tags": endpoint_data.get("tags", []),
                        "parameters": endpoint_data.get("parameters", []),
                        "responses": endpoint_data.get("responses", {}),
                    }

                    # Create vector
                    try:
                        from ..search.vectorizer import Triple
                        triple = Triple(
                            endpoint=path,
                            method=method.upper(),
                            description=description
                        )
                        vector = self.vectorizer.vectorize(triple)

                        # Create vector entry with simplified metadata
                        vector_entry = {
                            "id": f"{method.upper()}:{path}",
                            "values": vector.tolist(),
                            "metadata": self._simplify_metadata(endpoint_metadata),
                        }

                        endpoints.append(vector_entry)

                    except Exception as e:
                        logger.error(f"Failed to vectorize endpoint {path}: {e}")
                        continue

            # Store vectors in batches
            if endpoints:
                self.client.upsert_vectors(endpoints)

            return endpoints

        except Exception as e:
            logger.error(f"Failed to index file: {e}")
            logger.error(traceback.format_exc())
            return []
