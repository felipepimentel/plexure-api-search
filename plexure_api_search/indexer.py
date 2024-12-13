"""API Indexer module for ingesting and indexing API contracts."""

import hashlib
import json
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Union

import httpx
import numpy as np
import yaml
from pinecone import Pinecone, ServerlessSpec
from rich.console import Console
from rich.progress import Progress
from sentence_transformers import SentenceTransformer
from sklearn.decomposition import PCA

from plexure_api_search.config import Config, ConfigManager


def flatten_metadata(metadata: Dict[str, Any]) -> Dict[str, Union[str, int, float, bool, List[str]]]:
    """Flatten metadata to be compatible with Pinecone.
    
    Args:
        metadata: Original metadata dictionary.
        
    Returns:
        Flattened metadata dictionary.
    """
    flattened = {}
    
    for key, value in metadata.items():
        if isinstance(value, (str, int, float, bool)):
            flattened[key] = value
        elif isinstance(value, list):
            if all(isinstance(x, str) for x in value):
                flattened[key] = value
            else:
                flattened[key] = [str(x) for x in value]
        elif isinstance(value, dict):
            for sub_key, sub_value in value.items():
                new_key = f"{key}_{sub_key}"
                if isinstance(sub_value, (str, int, float, bool)):
                    flattened[new_key] = sub_value
                elif isinstance(sub_value, list):
                    if all(isinstance(x, str) for x in sub_value):
                        flattened[new_key] = sub_value
                    else:
                        flattened[new_key] = [str(x) for x in sub_value]
        else:
            flattened[key] = str(value)
            
    return flattened


class EndpointCategory(Enum):
    """Categories for API endpoints."""

    AUTHENTICATION = "Authentication"
    USERS = "Users"
    DATA = "Data"
    CONFIGURATION = "Configuration"
    MONITORING = "Monitoring"
    UNKNOWN = "Unknown"


@dataclass
class EndpointSignature:
    """Unique signature for an API endpoint."""

    path: str
    method: str
    version: str

    def __hash__(self) -> int:
        return hash(f"{self.path}:{self.method}:{self.version}")


class DataValidator:
    """Validates and normalizes API data."""

    @staticmethod
    def normalize_path(path: str) -> str:
        """Normalize API path."""
        # Remove leading/trailing slashes
        path = path.strip("/")

        # Replace multiple slashes with single slash
        path = "/".join(filter(None, path.split("/")))

        # Normalize parameter names
        path = path.replace("{id}", "{identifier}")

        return f"/{path}"

    @staticmethod
    def normalize_version(version: str) -> str:
        """Normalize API version."""
        if not version:
            return "1.0.0"

        # Extract version numbers
        numbers = [int(n) for n in re.findall(r"\d+", version)]

        # Pad with zeros
        while len(numbers) < 3:
            numbers.append(0)

        return ".".join(map(str, numbers[:3]))

    @staticmethod
    def normalize_method(method: str) -> str:
        """Normalize HTTP method."""
        method = method.upper()
        valid_methods = {"GET", "POST", "PUT", "DELETE", "PATCH"}
        return method if method in valid_methods else "GET"

    @staticmethod
    def validate_and_normalize(api_data: Dict[str, Any]) -> Dict[str, Any]:
        """Validate and normalize API data."""
        # Ensure required fields
        api_data["path"] = DataValidator.normalize_path(api_data.get("path", "/"))
        api_data["method"] = DataValidator.normalize_method(
            api_data.get("method", "GET")
        )
        api_data["api_version"] = DataValidator.normalize_version(
            api_data.get("api_version", "")
        )

        # Clean and normalize text fields
        api_data["description"] = (
            api_data.get("description", "").strip() or "No description provided."
        )
        api_data["summary"] = (
            api_data.get("summary", "").strip() or api_data["description"][:100]
        )

        # Normalize tags
        api_data["tags"] = [
            tag.strip().lower()
            for tag in api_data.get("tags", [])
            if tag and tag.strip()
        ]

        # Normalize parameters
        api_data["parameters"] = [
            f"{p.get('name', '')}:{p.get('schema', {}).get('type', 'string')}"
            for p in api_data.get("parameters", [])
            if p.get("name")
        ]

        return api_data


class MetadataEnricher:
    """Enriches API metadata using LLM."""

    def __init__(self, config: Config):
        """Initialize metadata enricher.
        
        Args:
            config: Configuration object.
        """
        self.config = config

    def _infer_category(self, api_data: Dict[str, Any]) -> EndpointCategory:
        """Infer endpoint category from metadata."""
        path = api_data["path"].lower()
        description = api_data["description"].lower()

        if any(term in path + description for term in ["auth", "login", "token"]):
            return EndpointCategory.AUTHENTICATION
        elif any(term in path + description for term in ["user", "profile", "account"]):
            return EndpointCategory.USERS
        elif any(term in path + description for term in ["config", "setting"]):
            return EndpointCategory.CONFIGURATION
        elif any(
            term in path + description for term in ["monitor", "health", "metric"]
        ):
            return EndpointCategory.MONITORING
        elif any(term in path + description for term in ["data", "record", "item"]):
            return EndpointCategory.DATA
        else:
            return EndpointCategory.UNKNOWN

    def _infer_features(self, api_data: Dict[str, Any]) -> Dict[str, bool]:
        """Infer endpoint features from metadata."""
        path = api_data["path"].lower()
        description = api_data["description"].lower()
        method = api_data["method"]

        return {
            "has_auth": any(
                term in path + description for term in ["auth", "token", "secure"]
            ),
            "has_examples": bool(api_data.get("examples") or "example" in description),
            "supports_pagination": any(
                term in description for term in ["page", "limit", "offset"]
            )
            or method == "GET"
            and bool(api_data.get("parameters")),
            "deprecated": "deprecat" in description
            or api_data.get("deprecated", False),
        }

    def enrich_metadata(self, api_data: Dict[str, Any]) -> Dict[str, Any]:
        """Enrich API metadata with inferred insights."""
        try:
            # Infer basic metadata
            category = self._infer_category(api_data)
            features = self._infer_features(api_data)

            # Add inferred metadata
            api_data["features"] = features
            api_data["category"] = category.value
            
            # Flatten metadata for Pinecone
            return flatten_metadata(api_data)

        except Exception as e:
            print(f"Metadata enrichment failed: {e}")
            return api_data


class APIIndexer:
    """Handles API contract ingestion and indexing."""

    def __init__(self, api_dir: Optional[str] = None, config: Optional[Config] = None):
        """Initialize API indexer.
        
        Args:
            api_dir: Optional directory containing API contracts.
            config: Optional configuration object.
        """
        self.config = config or ConfigManager().load_config()
        self.api_dir = api_dir or self.config.api_dir
        self.console = Console()
        self.model = SentenceTransformer(self.config.model_name)

        # Initialize Pinecone
        self.pc = Pinecone(api_key=self.config.pinecone_api_key)

        # Initialize PCA for embedding compression
        self.pca = PCA(n_components=self.config.embedding_dimension)
        self.index_cache_file = Path(self.api_dir) / ".index_cache.json"

        # Ensure index exists
        if self.config.pinecone_index not in self.pc.list_indexes().names():
            # Create index with serverless spec
            self.pc.create_index(
                name=self.config.pinecone_index,
                dimension=self.config.embedding_dimension,
                metric="cosine",
                spec=ServerlessSpec(
                    cloud=self.config.pinecone_cloud,
                    region=self.config.pinecone_region
                )
            )

        self.index = self.pc.Index(self.config.pinecone_index)

        # Initialize helpers
        self.validator = DataValidator()
        self.enricher = MetadataEnricher(self.config)

        # Load or initialize index cache
        self.index_cache = self._load_index_cache()

        # Track indexed endpoints
        self.indexed_signatures: Set[EndpointSignature] = set()

    def _load_index_cache(self) -> Dict[str, Any]:
        """Load index cache from file."""
        if self.index_cache_file.exists():
            try:
                with open(self.index_cache_file, "r") as f:
                    return json.load(f)
            except Exception as e:
                self.console.print(
                    f"[yellow]Warning: Failed to load index cache: {e}[/]"
                )

        return {
            "last_indexed": {},
            "file_hashes": {},
            "embedding_stats": {"mean": None, "std": None},
        }

    def _save_index_cache(self) -> None:
        """Save index cache to file."""
        try:
            with open(self.index_cache_file, "w") as f:
                json.dump(self.index_cache, f)
        except Exception as e:
            self.console.print(f"[yellow]Warning: Failed to save index cache: {e}[/]")

    def _get_file_hash(self, file_path: str) -> str:
        """Calculate file hash."""
        hasher = hashlib.sha256()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hasher.update(chunk)
        return hasher.hexdigest()

    def _needs_indexing(self, file_path: str) -> bool:
        """Check if file needs to be indexed."""
        current_hash = self._get_file_hash(file_path)
        last_hash = self.index_cache["file_hashes"].get(file_path)

        if last_hash != current_hash:
            self.index_cache["file_hashes"][file_path] = current_hash
            return True

        last_indexed = self.index_cache["last_indexed"].get(file_path, 0)
        return time.time() - last_indexed > 86400  # 24 hours

    def _create_embedding(self, endpoint: Dict[str, Any]) -> np.ndarray:
        """Create rich embedding for endpoint."""
        # Create rich text representation
        text = f"""
        Path: {endpoint["path"]}
        Method: {endpoint["method"]}
        Version: {endpoint["api_version"]}
        Category: {endpoint.get("category", "")}
        Description: {endpoint.get("description", "")}
        Summary: {endpoint.get("summary", "")}
        Tags: {", ".join(endpoint.get("tags", []))}
        Usage Notes: {endpoint.get("usage_notes", "")}
        Best Practices: {endpoint.get("best_practices", "")}
        """

        # Get embedding
        embedding = self.model.encode(text)

        # Normalize embedding
        if self.index_cache["embedding_stats"]["mean"] is not None:
            embedding = (
                embedding - self.index_cache["embedding_stats"]["mean"]
            ) / self.index_cache["embedding_stats"]["std"]

        return embedding

    def _process_api_file(self, file_path: str) -> List[Dict[str, Any]]:
        """Process a single API file and extract endpoints."""
        if not self._needs_indexing(file_path):
            return []

        endpoints = []

        try:
            with open(file_path, "r") as f:
                api_spec = yaml.safe_load(f)

            if not isinstance(api_spec, dict):
                return []

            # Extract API info
            info = api_spec.get("info", {})
            api_name = info.get("title", os.path.splitext(os.path.basename(file_path))[0])
            version = info.get("version", "1.0.0")
            api_description = info.get("description", "")

            # Process paths
            paths = api_spec.get("paths", {})
            for path, methods in paths.items():
                if not isinstance(methods, dict):
                    continue
                    
                for method, details in methods.items():
                    if not isinstance(details, dict):
                        continue
                        
                    if method.upper() not in ["GET", "POST", "PUT", "DELETE", "PATCH"]:
                        continue

                    # Get operation details
                    summary = details.get("summary", "")
                    description = details.get("description", "") or summary or api_description
                    tags = details.get("tags", [])
                    operation_id = details.get("operationId", "")

                    # Create base endpoint data
                    endpoint_data = {
                        "path": path,
                        "method": method.upper(),
                        "description": description,
                        "summary": summary,
                        "parameters": details.get("parameters", []),
                        "tags": tags,
                        "api_version": version,
                        "api_name": api_name,
                        "endpoint": f"{method.upper()} {path}",
                        "operation_id": operation_id,
                        "file_path": file_path,
                        "last_modified": datetime.fromtimestamp(
                            os.path.getmtime(file_path)
                        ).isoformat(),
                    }

                    # Validate and normalize
                    endpoint_data = self.validator.validate_and_normalize(
                        endpoint_data
                    )

                    # Create signature
                    signature = EndpointSignature(
                        path=endpoint_data["path"],
                        method=endpoint_data["method"],
                        version=endpoint_data["api_version"],
                    )

                    # Skip if already indexed
                    if signature in self.indexed_signatures:
                        continue

                    # Enrich with metadata
                    endpoint_data = self.enricher.enrich_metadata(endpoint_data)

                    endpoints.append(endpoint_data)
                    self.indexed_signatures.add(signature)

            # Update index cache
            self.index_cache["last_indexed"][file_path] = time.time()

        except Exception as e:
            self.console.print(f"[red]Error processing {file_path}: {str(e)}[/]")

        return endpoints

    def index_apis(self, force_reindex: bool = False) -> None:
        """Index all API contracts with improved processing."""
        try:
            # Check if index exists and has data
            stats = self.index.describe_index_stats()
            
            # If force_reindex, delete all vectors
            if force_reindex and stats["total_vector_count"] > 0:
                self.index.delete(delete_all=True)
                # Reset cache
                self.index_cache = {
                    "last_indexed": {},
                    "file_hashes": {},
                    "embedding_stats": {"mean": None, "std": None},
                }
                self.indexed_signatures.clear()
            elif not force_reindex and stats["total_vector_count"] > 0:
                self.console.print("\nUsing existing Pinecone index")
                return

            self.console.print(
                f"\nIndexing API contracts from directory: {self.api_dir}"
            )

            # Collect all API files
            api_files = []
            for root, _, files in os.walk(self.api_dir):
                for file in files:
                    if file.endswith((".yaml", ".yml")):
                        api_files.append(os.path.join(root, file))

            if not api_files:
                self.console.print("[yellow]No API files found.[/]")
                return

            # Process files in parallel
            all_endpoints = []
            with ThreadPoolExecutor(max_workers=4) as executor:
                with Progress() as progress:
                    task = progress.add_task(
                        "[cyan]Processing API files...", total=len(api_files)
                    )

                    for endpoints in executor.map(self._process_api_file, api_files):
                        all_endpoints.extend(endpoints)
                        progress.advance(task)

            if not all_endpoints:
                self.console.print("[green]All files are up to date.[/]")
                return

            # Create and process embeddings
            embeddings = []
            for endpoint in all_endpoints:
                embedding = self._create_embedding(endpoint)
                embeddings.append(embedding)

            # Update embedding statistics
            embeddings_array = np.array(embeddings)
            self.index_cache["embedding_stats"]["mean"] = embeddings_array.mean(
                axis=0
            ).tolist()
            self.index_cache["embedding_stats"]["std"] = embeddings_array.std(
                axis=0
            ).tolist()

            # Compress embeddings if needed
            if len(embeddings[0]) > self.config.embedding_dimension:
                embeddings = self.pca.fit_transform(embeddings)

            # Index endpoints in batches
            with Progress() as progress:
                task = progress.add_task(
                    "[cyan]Indexing endpoints...", total=len(all_endpoints)
                )

                for i in range(0, len(all_endpoints), self.config.batch_size):
                    batch = all_endpoints[i : i + self.config.batch_size]
                    batch_embeddings = embeddings[i : i + self.config.batch_size]

                    # Print first endpoint metadata for debugging
                    if i == 0:
                        self.console.print("\nFirst endpoint metadata:")
                        endpoint = batch[0]
                        self.console.print(f"API Name: {endpoint.get('api_name', 'N/A')}")
                        self.console.print(f"Version: {endpoint.get('api_version', 'N/A')}")
                        self.console.print(f"Endpoint: {endpoint.get('endpoint', 'N/A')}")
                        self.console.print(f"Description: {endpoint.get('description', 'N/A')}")

                    vectors = [
                        (
                            f"{endpoint['method']}_{endpoint['path']}",
                            embedding.tolist(),
                            flatten_metadata({
                                "api_name": endpoint["api_name"],
                                "api_version": endpoint["api_version"],
                                "endpoint": endpoint["endpoint"],
                                "description": endpoint["description"],
                                "summary": endpoint["summary"],
                                "method": endpoint["method"],
                                "path": endpoint["path"],
                                "tags": endpoint["tags"],
                            }),
                        )
                        for endpoint, embedding in zip(batch, batch_embeddings)
                    ]

                    # Print first vector metadata for debugging
                    if i == 0:
                        self.console.print("\nFirst vector metadata:")
                        self.console.print(vectors[0][2])

                    # Upsert batch to Pinecone
                    self.index.upsert(vectors=vectors)
                    progress.advance(task, len(batch))

            # Save index cache
            self._save_index_cache()

            self.console.print(
                f"\n[green]Successfully indexed {len(all_endpoints)} endpoints.[/]"
            )

        except Exception as e:
            self.console.print(f"[red]Error during indexing: {str(e)}[/]")


def main():
    """Main function for running the indexer directly."""
    import argparse

    parser = argparse.ArgumentParser(description="Index API contracts")
    parser.add_argument(
        "--api-dir",
        default=None,
        help="Directory containing API contracts"
    )
    parser.add_argument(
        "--force-reindex",
        action="store_true",
        help="Force reindexing even if index exists",
    )

    args = parser.parse_args()

    indexer = APIIndexer(api_dir=args.api_dir)
    indexer.index_apis(force_reindex=args.force_reindex)


if __name__ == "__main__":
    main()
