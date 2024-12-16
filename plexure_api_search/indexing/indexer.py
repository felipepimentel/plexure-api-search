"""API contract indexing and vector creation."""

import json
import logging
import os
import traceback
from typing import Any, Dict, List, Optional

import yaml

from ..config import config_instance
from ..embedding.embeddings import TripleVectorizer
from ..enrichment import LLMEnricher
from ..integrations import pinecone_instance
from ..integrations.llm.openrouter_client import OpenRouterClient
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
        self.llm = OpenRouterClient()
        self.enricher = LLMEnricher(self.llm)

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
        """Index all API files in directory.

        Args:
            force: Whether to force reindexing
            validate: Whether to validate data before indexing

        Returns:
            Dictionary with indexing results
        """
        try:
            directory = config_instance.api_dir
            logger.info(f"Indexing APIs from directory: {directory}")

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
                        else:
                            logger.warning(f"No endpoints found in {file_path}")
                            failed_files.append({
                                "path": file_path,
                                "error": "No endpoints found",
                            })

                    except Exception as e:
                        logger.error(f"Error indexing file {file_path}: {e}")
                        logger.error(f"Traceback: {traceback.format_exc()}")
                        failed_files.append({
                            "path": file_path,
                            "error": str(e),
                            "traceback": traceback.format_exc(),
                        })

                except Exception as e:
                    logger.error(f"Error processing {file_path}: {e}")
                    logger.error(f"Traceback: {traceback.format_exc()}")
                    failed_files.append({
                        "path": file_path,
                        "error": str(e),
                        "traceback": traceback.format_exc(),
                    })

            # Calculate quality metrics if validated
            quality_metrics = None
            if validate and all_endpoints:
                try:
                    quality_metrics = self.validator.calculate_quality_metrics(
                        all_endpoints
                    )
                except Exception as e:
                    logger.error(f"Error calculating quality metrics: {e}")
                    logger.error(f"Traceback: {traceback.format_exc()}")

            # Log summary
            logger.info(f"Total files processed: {len(files)}")
            logger.info(f"Total endpoints indexed: {total_endpoints}")
            logger.info(f"Failed files: {len(failed_files)}")
            logger.info(f"Successfully indexed APIs: {len(indexed_apis)}")

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
                            # Skip non-HTTP method keys
                            if method.upper() not in {
                                "GET",
                                "POST",
                                "PUT",
                                "DELETE",
                                "PATCH",
                            }:
                                continue

                            if not isinstance(endpoint_data, dict):
                                logger.warning(
                                    f"Skipping invalid endpoint data for {method} {path}"
                                )
                                continue

                            # Handle $ref in endpoint data
                            if "$ref" in endpoint_data:
                                resolved_data = self._resolve_ref(endpoint_data, data)
                                if resolved_data:
                                    endpoint_data.update(resolved_data)

                            # Merge path-level parameters with method-level parameters
                            merged_data = endpoint_data.copy()
                            if "parameters" in path_data:
                                path_params = path_data["parameters"]
                                method_params = endpoint_data.get("parameters", [])
                                merged_data["parameters"] = path_params + method_params

                            # Create endpoint metadata
                            endpoint = {
                                "path": full_path,
                                "method": method.upper(),
                                "description": merged_data.get("description", ""),
                                "summary": merged_data.get("summary", ""),
                                "api_name": api_name,
                                "api_version": api_version,
                                "parameters": self._extract_parameters(
                                    merged_data, data
                                ),
                                "responses": self._extract_responses(merged_data, data),
                                "tags": merged_data.get("tags", []),
                                "requires_auth": bool(merged_data.get("security", [])),
                                "deprecated": merged_data.get("deprecated", False),
                            }

                            # Create unique ID
                            endpoint_id = (
                                f"{api_name}_{api_version}_{method}_{full_path}".lower()
                                .replace("/", "_")
                                .replace("{", "")
                                .replace("}", "")
                            )
                            endpoint["id"] = endpoint_id

                            # Enrich endpoint with LLM-generated content
                            try:
                                enriched_endpoint = self.enricher.enrich_endpoint(endpoint)
                                if "enriched" in enriched_endpoint:
                                    # Serialize enriched data as JSON string
                                    endpoint["enriched"] = json.dumps(
                                        enriched_endpoint["enriched"]
                                    )
                                logger.info(
                                    f"Enriched endpoint {method} {path} with LLM content"
                                )
                            except Exception as e:
                                logger.error(
                                    f"Failed to enrich endpoint {method} {path}: {e}"
                                )

                            # Create vector representation
                            vector = self.vectorizer.vectorize(endpoint)
                            combined_vector = vector.to_combined_vector()

                            # Check if exists and force flag
                            if not force:
                                try:
                                    stats = self.client._verify_connection()
                                    if endpoint_id in stats.get("vectors", {}).get(
                                        "ids", []
                                    ):
                                        logger.debug(
                                            f"Skipping existing endpoint: {endpoint_id}"
                                        )
                                        endpoints.append(endpoint)
                                        continue
                                except Exception as e:
                                    logger.warning(
                                        f"Error checking existing endpoint: {e}"
                                    )

                            # Upsert to vector store
                            try:
                                # Ensure vector is a list
                                vector_values = (
                                    combined_vector
                                    if isinstance(combined_vector, list)
                                    else combined_vector.tolist()
                                )

                                self.client.upsert_vectors(
                                    vectors=[
                                        {
                                            "id": endpoint_id,
                                            "values": vector_values,
                                            "metadata": endpoint,
                                        }
                                    ]
                                )
                                endpoints.append(endpoint)
                                logger.info(
                                    f"Successfully indexed endpoint {method} {path}"
                                )
                            except Exception as e:
                                logger.error(
                                    f"Error upserting endpoint {endpoint_id}: {e}"
                                )
                                logger.error(
                                    f"Vector values type: {type(vector_values)}"
                                )
                                logger.error(
                                    f"Vector values shape: {len(vector_values)}"
                                )
                                logger.error(
                                    f"Endpoint metadata: {json.dumps(endpoint, indent=2)}"
                                )

                        except Exception as e:
                            logger.error(
                                f"Error processing endpoint {method} {path}: {e}"
                            )
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
                    if isinstance(param, str) or (
                        isinstance(param, dict) and "$ref" in param
                    ):
                        resolved = self._resolve_ref(param, spec_data)
                        if not resolved:
                            continue
                        param = resolved

                    # Extract parameter info
                    param_type = param.get("in", "query")
                    param_name = param.get("name", "")
                    param_desc = param.get("description", "")
                    required = (
                        "required" if param.get("required", False) else "optional"
                    )

                    # Handle schema
                    schema = param.get("schema", {})
                    if isinstance(schema, dict):
                        # Resolve schema reference if needed
                        if "$ref" in schema:
                            resolved = self._resolve_ref(schema, spec_data)
                            if resolved:
                                schema = resolved

                        param_data_type = schema.get("type", "string")
                        parameters.append(
                            f"{param_name}:{param_type}:{param_data_type}:{param_desc} ({required})"
                        )

                except Exception as e:
                    logger.error(f"Error processing parameter: {e}")
                    logger.error(f"Parameter data: {param}")
                    continue

            # Handle OpenAPI 3.x requestBody
            try:
                request_body = endpoint_data.get("requestBody", {})
                if isinstance(request_body, (dict, str)):
                    # Resolve reference if needed
                    if isinstance(request_body, str) or (
                        isinstance(request_body, dict) and "$ref" in request_body
                    ):
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
                                        for prop_name, prop in schema[
                                            "properties"
                                        ].items():
                                            if not isinstance(prop, dict):
                                                continue

                                            prop_type = prop.get("type", "object")
                                            prop_desc = prop.get("description", "")
                                            parameters.append(
                                                f"{prop_name}:body:{prop_type}:{prop_desc}"
                                            )

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
                    if isinstance(response, str) or (
                        isinstance(response, dict) and "$ref" in response
                    ):
                        resolved = self._resolve_ref(response, spec_data)
                        if not resolved:
                            continue
                        response = resolved

                    desc = response.get("description", "")

                    # Handle OpenAPI 3.x response schema
                    content = response.get("content", {})
                    if isinstance(content, dict):
                        for content_type, content_schema in content.items():
                            if (
                                isinstance(content_schema, dict)
                                and "schema" in content_schema
                            ):
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
