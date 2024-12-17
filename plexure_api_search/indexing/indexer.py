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
from ..monitoring.events import Event, EventType, event_manager
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

    def _safe_load_yaml(self, file_path: str) -> Optional[Dict[str, Any]]:
        """Safely load YAML/JSON file.

        Args:
            file_path: Path to file

        Returns:
            Loaded data or None if error
        """
        try:
            logger.debug(f"Loading file: {file_path}")

            # Check file exists
            if not os.path.exists(file_path):
                logger.error(f"File not found: {file_path}")
                return None

            # Check file extension
            ext = os.path.splitext(file_path)[1].lower()
            if ext not in [".yaml", ".yml", ".json"]:
                logger.error(f"Unsupported file extension: {ext}")
                return None

            # Read file content
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()

            # Try to load as YAML first
            try:
                if ext in [".yaml", ".yml"]:
                    data = yaml.safe_load(content)
                else:
                    data = json.loads(content)

                # Validate basic structure
                if not isinstance(data, dict):
                    logger.error(
                        f"Invalid file format - root must be an object: {file_path}"
                    )
                    return None

                if "paths" not in data:
                    logger.error(f"Invalid API spec - missing paths: {file_path}")
                    return None

                if not isinstance(data["paths"], dict):
                    logger.error(
                        f"Invalid API spec - paths must be an object: {file_path}"
                    )
                    return None

                logger.debug(f"Successfully loaded file: {file_path}")
                return data

            except yaml.YAMLError as e:
                logger.error(f"YAML parsing error in {file_path}: {e}")
                if hasattr(e, "problem_mark"):
                    mark = e.problem_mark
                    logger.error(
                        f"Error position: line {mark.line + 1}, column {mark.column + 1}"
                    )
                return None

            except json.JSONDecodeError as e:
                logger.error(f"JSON parsing error in {file_path}: {e}")
                return None

        except Exception as e:
            logger.error(f"Error loading file {file_path}: {e}")
            logger.error(traceback.format_exc())
            return None

    def _resolve_ref(self, ref_data: Any, spec_data: Dict) -> Optional[Dict]:
        """Resolve OpenAPI reference.

        Args:
            ref_data: Reference data (can be string or dict with $ref)
            spec_data: Full OpenAPI specification

        Returns:
            Resolved reference data or None if not found
        """
        try:
            # Handle dict with $ref
            if isinstance(ref_data, dict) and "$ref" in ref_data:
                ref = ref_data["$ref"]
            # Handle string ref
            elif isinstance(ref_data, str):
                ref = ref_data
            else:
                return None

            # Skip if not a reference
            if not isinstance(ref, str) or not ref.startswith("#/"):
                return None

            # Parse reference path
            parts = ref.split("/")
            if parts[0] == "#":
                parts = parts[1:]

            # Traverse the spec data
            current = spec_data
            for part in parts:
                if not isinstance(current, dict):
                    return None
                current = current.get(part)
                if current is None:
                    return None

            # Return only if result is a dict
            return current if isinstance(current, dict) else None

        except Exception as e:
            logger.debug(f"Failed to resolve reference: {e}")
            return None

    def index_directory(
        self,
        force: bool = False,
        validate: bool = True,
    ) -> Dict[str, Any]:
        """Index all API files in directory."""
        start_time = time.time()
        event_manager.emit(Event(
            type=EventType.INDEXING_STARTED,
            timestamp=datetime.now(),
            component="indexer",
            description="Starting API indexing process",
            metadata={"force": force, "validate": validate}
        ))
        
        try:
            directory = config_instance.api_dir
            logger.info(f"Indexing APIs from directory: {directory}")

            # Find API files
            files = find_api_files(directory)
            if not files:
                logger.warning("No API files found")
                event_manager.emit(Event(
                    type=EventType.INDEXING_COMPLETED,
                    timestamp=datetime.now(),
                    component="indexer",
                    description="No API files found",
                    success=False,
                    duration_ms=(time.time() - start_time) * 1000
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
            all_endpoints = []

            for i, file_path in enumerate(files):
                file_start_time = time.time()
                event_manager.emit(Event(
                    type=EventType.INDEXING_FILE_STARTED,
                    timestamp=datetime.now(),
                    component="indexer",
                    description=f"Processing file: {file_path}",
                    metadata={"file": file_path, "progress": (i / len(files)) * 100}
                ))
                
                try:
                    logger.info(f"Processing file: {file_path}")
                    # Load API spec
                    spec_data = self._safe_load_yaml(file_path)
                    if not spec_data:
                        failed_files.append({
                            "path": file_path,
                            "error": "Failed to load file or invalid format",
                        })
                        continue

                    # Validate spec data
                    if not isinstance(spec_data, dict):
                        logger.error(f"Invalid API spec format in {file_path}")
                        failed_files.append({
                            "path": file_path,
                            "error": "Invalid API spec format - not a dictionary",
                        })
                        continue

                    if "paths" not in spec_data:
                        logger.error(f"No paths found in API spec {file_path}")
                        failed_files.append({
                            "path": file_path,
                            "error": "No paths found in API spec",
                        })
                        continue

                    # Extract endpoints
                    try:
                        endpoints = self.index_file(spec_data, force=force)
                        if endpoints:
                            total_endpoints += len(endpoints)
                            all_endpoints.extend(endpoints)

                            # Record success
                            indexed_apis.append({
                                "path": file_path,
                                "api_name": spec_data.get("info", {}).get(
                                    "title", "Unknown API"
                                ),
                                "version": spec_data.get("info", {}).get(
                                    "version", "1.0.0"
                                ),
                                "endpoints": len(endpoints),
                            })
                            logger.info(
                                f"Successfully indexed {len(endpoints)} endpoints from {file_path}"
                            )
                            event_manager.emit(Event(
                                type=EventType.INDEXING_FILE_COMPLETED,
                                timestamp=datetime.now(),
                                component="indexer",
                                description=f"Successfully indexed file: {file_path}",
                                metadata={
                                    "file": file_path,
                                    "endpoints": len(endpoints),
                                    "progress": ((i + 1) / len(files)) * 100
                                },
                                duration_ms=(time.time() - file_start_time) * 1000,
                                success=True
                            ))
                        else:
                            failed_files.append({
                                "path": file_path,
                                "error": "No valid endpoints found",
                            })
                            event_manager.emit(Event(
                                type=EventType.INDEXING_FILE_FAILED,
                                timestamp=datetime.now(),
                                component="indexer",
                                description=f"Failed to index file: {file_path}",
                                error="No valid endpoints found",
                                metadata={"file": file_path},
                                success=False
                            ))

                    except Exception as e:
                        logger.error(f"Failed to index file {file_path}: {e}")
                        failed_files.append({
                            "path": file_path,
                            "error": str(e),
                        })
                        event_manager.emit(Event(
                            type=EventType.INDEXING_FILE_FAILED,
                            timestamp=datetime.now(),
                            component="indexer",
                            description=f"Failed to index file: {file_path}",
                            error=str(e),
                            metadata={"file": file_path},
                            success=False
                        ))
                        continue

                except Exception as e:
                    logger.error(f"Failed to process file {file_path}: {e}")
                    failed_files.append({
                        "path": file_path,
                        "error": str(e),
                    })
                    event_manager.emit(Event(
                        type=EventType.INDEXING_FILE_FAILED,
                        timestamp=datetime.now(),
                        component="indexer",
                        description=f"Failed to process file: {file_path}",
                        error=str(e),
                        metadata={"file": file_path},
                        success=False
                    ))
                    continue

            # Emit completion event
            duration_ms = (time.time() - start_time) * 1000
            event_manager.emit(Event(
                type=EventType.INDEXING_COMPLETED,
                timestamp=datetime.now(),
                component="indexer",
                description="API indexing completed",
                metadata={
                    "total_files": len(files),
                    "total_endpoints": total_endpoints,
                    "failed_files": len(failed_files),
                },
                duration_ms=duration_ms,
                success=True
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
            event_manager.emit(Event(
                type=EventType.INDEXING_FAILED,
                timestamp=datetime.now(),
                component="indexer",
                description="API indexing failed",
                error=str(e),
                duration_ms=duration_ms,
                success=False
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
