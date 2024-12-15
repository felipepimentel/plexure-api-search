"""API contract indexing and vector creation."""

import logging
import traceback
from typing import Any, Dict, List, Optional

import yaml

from ..config import config_instance
from ..embedding.embeddings import TripleVectorizer
from ..storage import pinecone_instance
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

    def _safe_load_yaml(self, file_path: str) -> Optional[Dict]:
        """Safely load YAML file with error handling.

        Args:
            file_path: Path to YAML file

        Returns:
            Loaded YAML data or None if error
        """
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f)
                if not isinstance(data, dict):
                    logger.error(f"Invalid YAML format in {file_path}")
                    return None
                return data
        except Exception as e:
            logger.error(f"Error loading YAML file {file_path}: {e}")
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
            parts = ref.split('/')
            if parts[0] == '#':
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
        """Index all API files in directory.

        Args:
            force: Whether to force reindexing
            validate: Whether to validate data before indexing

        Returns:
            Dictionary with indexing results
        """
        try:
            directory = config_instance.api_dir
            # Find API files
            files = find_api_files(directory)
            if not files:
                logger.warning("No API files found")
                return {
                    "total_files": 0,
                    "total_endpoints": 0,
                    "failed_files": [],
                    "indexed_apis": [],
                    "quality_metrics": None,
                }

            # Process each file
            total_endpoints = 0
            failed_files = []
            indexed_apis = []
            all_endpoints = []

            for file_path in files:
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

                    # Extract endpoints
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

                except Exception as e:
                    logger.error(f"Error processing {file_path}: {e}")
                    logger.error(f"Traceback: {traceback.format_exc()}")
                    failed_files.append({
                        "path": file_path,
                        "error": str(e),
                        "traceback": traceback.format_exc()
                    })

            # Calculate quality metrics if validated
            quality_metrics = None
            if validate and all_endpoints:
                quality_metrics = self.validator.calculate_quality_metrics(
                    all_endpoints
                )

            return {
                "total_files": len(files),
                "total_endpoints": total_endpoints,
                "failed_files": failed_files,
                "indexed_apis": indexed_apis,
                "quality_metrics": quality_metrics.to_dict()
                if quality_metrics
                else None,
            }

        except Exception as e:
            logger.error(f"Indexing failed: {e}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            raise

    def index_file(
        self, data: Dict[str, Any], force: bool = False
    ) -> List[Dict[str, Any]]:
        """Index single API specification file.

        Args:
            data: API specification data
            force: Whether to force reindexing

        Returns:
            List of indexed endpoints
        """
        try:
            # Extract endpoints from spec
            endpoints = []
            paths = data.get("paths", {})
            if not isinstance(paths, dict):
                logger.error("Invalid paths object in API spec")
                return []

            # Get API info
            info = data.get("info", {})
            if not isinstance(info, dict):
                info = {}

            api_name = info.get("title", "Unknown API")
            api_version = info.get("version", "1.0.0")

            # Get servers info for base path
            servers = data.get("servers", [])
            base_path = ""
            if servers and isinstance(servers, list) and len(servers) > 0:
                server = servers[0]
                if isinstance(server, dict):
                    base_path = server.get("url", "").rstrip("/")

            # Process each path
            for path, path_data in paths.items():
                try:
                    if not isinstance(path_data, dict):
                        logger.warning(f"Skipping invalid path data for {path}")
                        continue

                    full_path = f"{base_path}{path}".rstrip("/")

                    # Process each method
                    for method, endpoint_data in path_data.items():
                        try:
                            if not isinstance(endpoint_data, dict):
                                logger.warning(
                                    f"Skipping invalid endpoint data for {method} {path}"
                                )
                                continue

                            if method.upper() not in {"GET", "POST", "PUT", "DELETE", "PATCH"}:
                                continue

                            # Handle $ref in endpoint data
                            if "$ref" in endpoint_data:
                                resolved_data = self._resolve_ref(endpoint_data, data)
                                if resolved_data:
                                    endpoint_data.update(resolved_data)

                            # Create endpoint metadata
                            endpoint = {
                                "path": full_path,
                                "method": method.upper(),
                                "description": endpoint_data.get("description", ""),
                                "summary": endpoint_data.get("summary", ""),
                                "api_name": api_name,
                                "api_version": api_version,
                                "parameters": self._extract_parameters(endpoint_data, data),
                                "responses": self._extract_responses(endpoint_data, data),
                                "tags": endpoint_data.get("tags", []),
                                "requires_auth": bool(endpoint_data.get("security", [])),
                                "deprecated": endpoint_data.get("deprecated", False),
                            }

                            # Create vector representation
                            vector = self.vectorizer.vectorize(endpoint)
                            combined_vector = vector.to_combined_vector()

                            # Create unique ID
                            endpoint_id = (
                                f"{api_name}_{api_version}_{method}_{full_path}".lower()
                            )

                            # Check if exists and force flag
                            if not force:
                                try:
                                    stats = self.client._verify_connection()
                                    if endpoint_id in stats.get("vectors", {}).get("ids", []):
                                        logger.debug(
                                            f"Skipping existing endpoint: {endpoint_id}"
                                        )
                                        endpoints.append(endpoint)
                                        continue
                                except Exception as e:
                                    logger.warning(f"Error checking existing endpoint: {e}")

                            # Upsert to vector store
                            try:
                                # Ensure vector is a list
                                vector_values = combined_vector if isinstance(combined_vector, list) else combined_vector.tolist()
                                
                                self.client.upsert_vectors(
                                    vectors=[{
                                        "id": endpoint_id,
                                        "values": vector_values,
                                        "metadata": endpoint,
                                    }]
                                )
                                endpoints.append(endpoint)
                            except Exception as e:
                                logger.error(f"Error upserting endpoint {endpoint_id}: {e}")

                        except Exception as e:
                            logger.error(f"Error processing endpoint {method} {path}: {e}")
                            logger.error(f"Traceback: {traceback.format_exc()}")

                except Exception as e:
                    logger.error(f"Error processing path {path}: {e}")
                    logger.error(f"Traceback: {traceback.format_exc()}")

            logger.info(
                f"Indexed {len(endpoints)} endpoints from {api_name} v{api_version}"
            )
            return endpoints

        except Exception as e:
            logger.error(f"Failed to index file: {e}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            raise

    def _extract_parameters(self, endpoint_data: Dict, spec_data: Dict) -> List[str]:
        """Extract parameter information from endpoint data.

        Args:
            endpoint_data: Endpoint specification data
            spec_data: Full OpenAPI specification

        Returns:
            List of parameter descriptions
        """
        try:
            parameters = []
            
            # Handle OpenAPI 3.x parameters
            for param in endpoint_data.get("parameters", []):
                try:
                    # Skip if not valid parameter
                    if not isinstance(param, (dict, str)):
                        continue

                    # Resolve reference if needed
                    if isinstance(param, str) or (isinstance(param, dict) and "$ref" in param):
                        resolved = self._resolve_ref(param, spec_data)
                        if not resolved:
                            continue
                        param = resolved

                    # Extract parameter info
                    param_type = param.get("in", "query")
                    param_name = param.get("name", "")
                    param_desc = param.get("description", "")
                    required = "required" if param.get("required", False) else "optional"
                    
                    # Handle schema
                    schema = param.get("schema", {})
                    if isinstance(schema, dict):
                        # Resolve schema reference if needed
                        if "$ref" in schema:
                            resolved = self._resolve_ref(schema, spec_data)
                            if resolved:
                                schema = resolved

                        param_data_type = schema.get("type", "string")
                        parameters.append(f"{param_name}:{param_type}:{param_data_type}:{param_desc} ({required})")

                except Exception as e:
                    logger.error(f"Error processing parameter: {e}")
                    logger.error(f"Parameter data: {param}")
                    continue

            # Handle OpenAPI 3.x requestBody
            try:
                request_body = endpoint_data.get("requestBody", {})
                if isinstance(request_body, (dict, str)):
                    # Resolve reference if needed
                    if isinstance(request_body, str) or (isinstance(request_body, dict) and "$ref" in request_body):
                        resolved = self._resolve_ref(request_body, spec_data)
                        if resolved:
                            request_body = resolved
                    
                    if isinstance(request_body, dict):
                        content = request_body.get("content", {})
                        if isinstance(content, dict):
                            for content_type, content_schema in content.items():
                                if not isinstance(content_schema, dict):
                                    continue
                                    
                                schema = content_schema.get("schema", {})
                                if isinstance(schema, dict):
                                    # Resolve schema reference if needed
                                    if "$ref" in schema:
                                        resolved = self._resolve_ref(schema, spec_data)
                                        if resolved:
                                            schema = resolved

                                    if "properties" in schema:
                                        for prop_name, prop in schema["properties"].items():
                                            if not isinstance(prop, dict):
                                                continue
                                                
                                            prop_type = prop.get("type", "object")
                                            prop_desc = prop.get("description", "")
                                            parameters.append(f"{prop_name}:body:{prop_type}:{prop_desc}")

            except Exception as e:
                logger.error(f"Error processing request body: {e}")
                logger.error(f"Request body data: {request_body}")

            return parameters

        except Exception as e:
            logger.error(f"Error extracting parameters: {e}")
            logger.error(f"Endpoint data: {endpoint_data}")
            return []

    def _extract_responses(self, endpoint_data: Dict, spec_data: Dict) -> List[str]:
        """Extract response information from endpoint data.

        Args:
            endpoint_data: Endpoint specification data
            spec_data: Full OpenAPI specification

        Returns:
            List of response descriptions
        """
        try:
            responses = []
            for status_code, response in endpoint_data.get("responses", {}).items():
                try:
                    # Skip if not valid response
                    if not isinstance(response, (dict, str)):
                        continue

                    # Resolve reference if needed
                    if isinstance(response, str) or (isinstance(response, dict) and "$ref" in response):
                        resolved = self._resolve_ref(response, spec_data)
                        if not resolved:
                            continue
                        response = resolved

                    desc = response.get("description", "")
                    
                    # Handle OpenAPI 3.x response schema
                    content = response.get("content", {})
                    if isinstance(content, dict):
                        for content_type, content_schema in content.items():
                            if isinstance(content_schema, dict) and "schema" in content_schema:
                                schema = content_schema["schema"]
                                if isinstance(schema, dict):
                                    # Resolve schema reference if needed
                                    if "$ref" in schema:
                                        resolved = self._resolve_ref(schema, spec_data)
                                        if resolved:
                                            schema = resolved

                                    response_type = schema.get("type", "object")
                                    desc = f"{desc} ({response_type})"
                                    break

                    responses.append(f"{status_code}: {desc}")

                except Exception as e:
                    logger.error(f"Error processing response {status_code}: {e}")
                    logger.error(f"Response data: {response}")
                    continue

            return responses

        except Exception as e:
            logger.error(f"Error extracting responses: {e}")
            logger.error(f"Endpoint data: {endpoint_data}")
            return []
