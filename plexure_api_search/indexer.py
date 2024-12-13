"""API contract indexer with triple vector embeddings."""

import glob
import json
import logging
import os
import time
from typing import Dict, List

import yaml
from pinecone import Pinecone, ServerlessSpec

from .config import API_DIR
from .embeddings import TripleVectorizer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class APIIndexer:
    """Indexes API contracts with triple vector embeddings."""

    def __init__(
        self,
        index_name: str,
        api_key: str,
        environment: str,
        cloud: str = "aws",
        region: str = "us-east-1",
        api_dir: str = API_DIR,
    ):
        """Initialize indexer.

        Args:
            index_name: Name of the Pinecone index.
            api_key: Pinecone API key.
            environment: Pinecone environment.
            cloud: Cloud provider (aws, gcp, azure).
            region: Cloud region.
            api_dir: Directory containing API contracts (defaults to config.API_DIR).
        """
        self.api_dir = api_dir
        self.vectorizer = TripleVectorizer()

        print("\nInitializing Pinecone...")
        print(f"Index: {index_name}")
        print(f"Environment: {environment}")
        print(f"Region: {region}")
        print(f"Cloud: {cloud}")
        print(f"API Directory: {api_dir}")

        # Initialize Pinecone
        try:
            # Initialize Pinecone with the API key
            pc = Pinecone(api_key=api_key)

            # Check if index exists
            existing_indexes = pc.list_indexes().names()
            print(f"\nExisting indexes: {existing_indexes}")

            if index_name not in existing_indexes:
                print(f"\nCreating new index: {index_name}")
                pc.create_index(
                    name=index_name,
                    dimension=384,  # Matches our TripleVector dimension
                    metric="dotproduct",
                    spec=ServerlessSpec(cloud=cloud, region=region),
                )
                # Wait for index to be ready
                print("Waiting for index to be ready...")
                time.sleep(10)
            else:
                print(f"\nUsing existing index: {index_name}")

            # Get index instance
            self.index = pc.Index(index_name)

            # Verify connection
            try:
                stats = self.index.describe_index_stats()
                print("\nIndex stats:")
                print(f"Total vectors: {stats.get('total_vector_count', 0)}")
                print(f"Dimension: {stats.get('dimension', 384)}")
            except Exception as e:
                print(f"Warning: Could not get index stats: {e}")

        except Exception as e:
            raise RuntimeError(f"Failed to initialize Pinecone: {str(e)}")

    def _load_api_files(self) -> List[Dict]:
        """Load API contract files.

        Returns:
            List of API contract data.
        """
        api_files = []

        # Find all YAML/JSON files
        yaml_files = glob.glob(os.path.join(self.api_dir, "**/*.yaml"), recursive=True)
        yaml_files.extend(
            glob.glob(os.path.join(self.api_dir, "**/*.yml"), recursive=True)
        )
        json_files = glob.glob(os.path.join(self.api_dir, "**/*.json"), recursive=True)

        print(f"Found {len(yaml_files)} YAML files and {len(json_files)} JSON files")

        # Load YAML files
        for file_path in yaml_files:
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    data = yaml.safe_load(f)
                if data and isinstance(data, dict):
                    api_files.append({"file_path": file_path, "data": data})
            except Exception as e:
                print(f"Error loading {file_path}: {e}")

        # Load JSON files
        for file_path in json_files:
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                if data and isinstance(data, dict):
                    api_files.append({"file_path": file_path, "data": data})
            except Exception as e:
                print(f"Error loading {file_path}: {e}")

        return api_files

    def _extract_endpoints(self, api_spec: Dict) -> List[Dict]:
        """Extract endpoints from API specification."""
        endpoints = []

        try:
            # Extract API info
            api_name = api_spec.get("info", {}).get("title", "Unknown API")
            api_version = api_spec.get("info", {}).get("version", "1.0.0")

            # Get paths
            paths = api_spec.get("paths", {})

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

                    logger.info(f"Found endpoint: {method.upper()} {path}")
                    endpoints.append(endpoint)

        except Exception as e:
            logger.error(f"Error extracting endpoints: {str(e)}")
            raise

        logger.info(f"Extracted {len(endpoints)} endpoints")
        return endpoints

    def _prepare_metadata(self, endpoint: Dict) -> Dict:
        """Prepare metadata for indexing."""
        return {
            "api_name": str(endpoint["api_name"]),
            "api_version": str(endpoint["api_version"]),
            "path": str(endpoint["path"]),
            "method": str(endpoint["method"]),
            "description": str(endpoint["description"]),
            "summary": str(endpoint["summary"]),
            "parameters": [str(p) for p in endpoint["parameters"]],
            "responses": [str(r) for r in endpoint["responses"]],
            "tags": [str(t) for t in endpoint["tags"]],
            "operationId": str(endpoint["operationId"]),
            "deprecated": str(endpoint["deprecated"]).lower(),
            "security": [str(s) for s in endpoint["security"]],
            "requires_auth": str(endpoint["requires_auth"]).lower(),
            "full_path": str(endpoint["full_path"]),
        }

    def index_apis(self, force: bool = False) -> None:
        """Index API contracts."""
        try:
            logger.info("Starting API indexation...")

            if force:
                logger.info("Force reindex requested. Cleaning up...")
                try:
                    # Delete all vectors without namespace
                    self.index.delete(delete_all=True)
                    time.sleep(2)  # Wait for deletion
                except Exception as e:
                    logger.warning(f"Could not clear index: {e}")

            # Find API files
            api_files = glob.glob(
                os.path.join(self.api_dir, "**/*.yaml"), recursive=True
            )
            logger.info(f"Found {len(api_files)} API files to process")

            total_vectors = 0
            batch_size = 10
            vectors_batch = []

            # Process each API file
            for api_file in api_files:
                try:
                    # Load and parse API spec
                    with open(api_file) as f:
                        api_spec = yaml.safe_load(f)

                    # Extract endpoints
                    endpoints = self._extract_endpoints(api_spec)

                    # Process each endpoint
                    for endpoint in endpoints:
                        try:
                            # Create vector
                            vector = self.vectorizer.vectorize(endpoint)
                            combined_vector = vector.to_combined_vector()

                            # Create unique ID
                            vector_id = f"{endpoint['api_name']}_{endpoint['method']}_{endpoint['path']}"
                            vector_id = (
                                vector_id.replace("/", "_")
                                .replace("{", "")
                                .replace("}", "")
                            )

                            # Prepare metadata
                            metadata = self._prepare_metadata(endpoint)

                            # Create vector entry
                            vector_entry = {
                                "id": vector_id,
                                "values": combined_vector.tolist(),
                                "metadata": metadata,
                            }

                            vectors_batch.append(vector_entry)

                            # Upsert when batch is full
                            if len(vectors_batch) >= batch_size:
                                try:
                                    # Upsert without namespace
                                    self.index.upsert(vectors=vectors_batch)
                                    total_vectors += len(vectors_batch)
                                    logger.info(
                                        f"Upserted batch of {len(vectors_batch)} vectors. Total: {total_vectors}"
                                    )
                                    vectors_batch = []
                                except Exception as e:
                                    logger.error(f"Batch upsert failed: {e}")
                                    # Try one by one
                                    for v in vectors_batch:
                                        try:
                                            self.index.upsert(vectors=[v])
                                            total_vectors += 1
                                        except Exception as e:
                                            logger.error(
                                                f"Single vector upsert failed: {e}"
                                            )
                                    vectors_batch = []

                        except Exception as e:
                            logger.error(f"Error processing endpoint: {e}")
                            continue

                except Exception as e:
                    logger.error(f"Error processing file {api_file}: {e}")
                    continue

            # Upsert remaining vectors
            if vectors_batch:
                try:
                    self.index.upsert(vectors=vectors_batch)
                    total_vectors += len(vectors_batch)
                    logger.info(
                        f"Upserted final batch of {len(vectors_batch)} vectors. Total: {total_vectors}"
                    )
                except Exception as e:
                    logger.error(f"Final batch upsert failed: {e}")
                    # Try one by one
                    for v in vectors_batch:
                        try:
                            self.index.upsert(vectors=[v])
                            total_vectors += 1
                        except Exception as e:
                            logger.error(f"Single vector upsert failed: {e}")

            # Verify final state
            time.sleep(2)  # Wait for index to update
            stats = self.index.describe_index_stats()
            final_count = stats.get("total_vector_count", 0)
            logger.info(f"Final vector count: {final_count}")

            if final_count != total_vectors:
                logger.warning(
                    f"Vector count mismatch. Expected: {total_vectors}, Found: {final_count}"
                )

        except Exception as e:
            logger.error(f"Indexing failed: {e}")
            raise
