"""API contract indexer with triple vector embeddings."""

import glob
import json
import os
import time
from typing import Dict, List
import logging

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
        """Extract endpoints from API specification.
        
        Args:
            api_spec: Loaded API specification dictionary
            
        Returns:
            List of endpoint dictionaries
        """
        endpoints = []
        
        try:
            # Extract API info
            api_name = api_spec.get("info", {}).get("title", "Unknown API")
            api_version = api_spec.get("info", {}).get("version", "1.0.0")
            
            # Get paths
            paths = api_spec.get("paths", {})
            
            # Process each path
            for path, path_data in paths.items():
                # Process each method (GET, POST, etc)
                for method, endpoint_data in path_data.items():
                    if method.upper() not in ["GET", "POST", "PUT", "DELETE", "PATCH"]:
                        continue
                        
                    # Extract parameters
                    parameters = endpoint_data.get("parameters", [])
                    
                    # Extract responses
                    responses = endpoint_data.get("responses", {})
                    
                    # Extract security requirements
                    security = endpoint_data.get("security", [])
                    
                    # Get description, falling back to summary if empty
                    description = endpoint_data.get("description", "")
                    if not description.strip():
                        description = endpoint_data.get("summary", "")
                    
                    # Create endpoint object with rich metadata
                    endpoint = {
                        "api_name": api_name,
                        "api_version": api_version,
                        "path": path,
                        "method": method.upper(),
                        "description": description,
                        "summary": endpoint_data.get("summary", ""),
                        "parameters": parameters,
                        "responses": responses,
                        "tags": endpoint_data.get("tags", []),
                        "operationId": endpoint_data.get("operationId", ""),
                        "deprecated": str(endpoint_data.get("deprecated", False)),
                        "security": security,
                        "full_path": f"{api_name}/{path}",
                        "requires_auth": str(bool(security)).lower()
                    }
                    
                    logger.info(f"Found endpoint: {method.upper()} {path}")
                    endpoints.append(endpoint)
                    
        except Exception as e:
            logger.error(f"Error extracting endpoints: {str(e)}")
            raise
            
        logger.info(f"Extracted {len(endpoints)} endpoints")
        return endpoints

    def _sanitize_metadata(self, metadata: Dict) -> Dict:
        """Sanitize metadata to match Pinecone requirements.
        
        Args:
            metadata: Original metadata dictionary
            
        Returns:
            Sanitized metadata dictionary
        """
        sanitized = {}
        
        for key, value in metadata.items():
            if key == "parameters":
                # Convert parameters to list of strings
                param_list = []
                for param in value:
                    param_str = f"{param.get('name', '')}:{param.get('in', '')}:{param.get('description', '')}"
                    param_list.append(param_str)
                sanitized[key] = param_list
            elif key == "responses":
                # Convert responses to list of strings
                resp_list = []
                for code, details in value.items():
                    resp_str = f"{code}:{details.get('description', '')}"
                    resp_list.append(resp_str)
                sanitized[key] = resp_list
            elif key == "security":
                # Convert security to string
                sanitized[key] = str(value)
            elif isinstance(value, (str, int, float, bool)):
                # Keep primitive types as is
                sanitized[key] = value
            elif isinstance(value, list):
                # Convert list items to strings
                sanitized[key] = [str(item) for item in value]
            else:
                # Convert other types to string
                sanitized[key] = str(value)
                
        return sanitized

    def index_apis(self, force: bool = False) -> None:
        """Index API contracts.
        
        Args:
            force: Whether to force reindexing of all APIs.
        """
        try:
            logger.info("Starting API indexation...")
            
            if force:
                logger.info("Force reindex requested. Cleaning up...")
                try:
                    logger.info("Deleting existing index...")
                    self.index.delete(delete_all=True)
                    logger.info("Waiting for deletion to complete...")
                    time.sleep(5)
                    
                    # Verify deletion
                    stats = self.index.describe_index_stats()
                    total_vectors = stats.get("total_vector_count", 0)
                    logger.info(f"Vectors after deletion: {total_vectors}")
                    
                    if total_vectors > 0:
                        logger.warning("Index not fully cleared. Retrying deletion...")
                        self.index.delete(delete_all=True)
                        time.sleep(5)
                        stats = self.index.describe_index_stats()
                        logger.info(f"Vectors after second deletion attempt: {stats.get('total_vector_count', 0)}")
                except Exception as e:
                    logger.warning(f"Could not clear index: {e}")
                    logger.info("Continuing with indexing...")
            
            api_files = glob.glob(os.path.join(self.api_dir, "**/*.yaml"), recursive=True)
            logger.info(f"Found {len(api_files)} API files to process")
            
            all_vectors = []
            successful_upserts = 0
            
            for api_file in api_files:
                try:
                    with open(api_file) as f:
                        api_spec = yaml.safe_load(f)
                        
                    endpoints = self._extract_endpoints(api_spec)
                    
                    # Processar endpoints em lotes menores
                    batch_size = 10  # Reduzindo o tamanho do lote
                    current_batch = []
                    
                    for i, endpoint in enumerate(endpoints):
                        try:
                            triple_vector = self.vectorizer.vectorize(endpoint)
                            combined_vector = triple_vector.to_combined_vector()
                            
                            sanitized_metadata = self._sanitize_metadata(endpoint)
                            
                            vector = {
                                "id": f"{os.path.basename(api_file)}_{i}",
                                "values": combined_vector.tolist(),
                                "metadata": sanitized_metadata
                            }
                            
                            current_batch.append(vector)
                            logger.info(f"Created vector for endpoint {i}: {endpoint['method']} {endpoint['path']}")
                            
                            # Quando o lote estiver cheio, fazer o upsert
                            if len(current_batch) >= batch_size:
                                try:
                                    logger.info(f"Upserting batch of {len(current_batch)} vectors...")
                                    self.index.upsert(vectors=current_batch)
                                    successful_upserts += len(current_batch)
                                    logger.info(f"Successfully upserted batch. Total: {successful_upserts}")
                                    
                                    # Verificar se os vetores foram realmente inseridos
                                    time.sleep(1)  # Dar tempo para o índice atualizar
                                    stats = self.index.describe_index_stats()
                                    logger.info(f"Current vector count: {stats.get('total_vector_count', 0)}")
                                    
                                    current_batch = []  # Limpar o lote atual
                                except Exception as e:
                                    logger.error(f"Error upserting batch: {str(e)}")
                                    logger.error("Will retry with smaller batch size")
                                    
                                    # Tentar upsert um por um
                                    for v in current_batch:
                                        try:
                                            self.index.upsert(vectors=[v])
                                            successful_upserts += 1
                                            time.sleep(0.5)
                                        except Exception as e:
                                            logger.error(f"Error upserting single vector: {str(e)}")
                                    
                                    current_batch = []
                        
                        except Exception as e:
                            logger.error(f"Error processing endpoint {i}: {str(e)}")
                            continue
                    
                    # Upsert qualquer vetor restante no lote
                    if current_batch:
                        try:
                            logger.info(f"Upserting final batch of {len(current_batch)} vectors...")
                            self.index.upsert(vectors=current_batch)
                            successful_upserts += len(current_batch)
                            logger.info(f"Successfully upserted final batch. Total: {successful_upserts}")
                        except Exception as e:
                            logger.error(f"Error upserting final batch: {str(e)}")
                            # Tentar upsert um por um
                            for v in current_batch:
                                try:
                                    self.index.upsert(vectors=[v])
                                    successful_upserts += 1
                                    time.sleep(0.5)
                                except Exception as e:
                                    logger.error(f"Error upserting single vector: {str(e)}")
                
                except Exception as e:
                    logger.error(f"Error processing API file {api_file}: {str(e)}")
                    continue
            
            # Verificar estado final
            logger.info("\nVerifying final index state...")
            time.sleep(2)  # Dar mais tempo para o índice atualizar
            stats = self.index.describe_index_stats()
            total_vectors = stats.get("total_vector_count", 0)
            
            if total_vectors != successful_upserts:
                logger.warning(f"Vector count mismatch. Expected: {successful_upserts}, Found: {total_vectors}")
                logger.warning("Retrying verification after delay...")
                time.sleep(5)
                stats = self.index.describe_index_stats()
                total_vectors = stats.get("total_vector_count", 0)
                logger.info(f"Final vector count after delay: {total_vectors}")
                
        except Exception as e:
            logger.error(f"Fatal error during indexing: {str(e)}")
            raise
