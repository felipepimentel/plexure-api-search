"""API Indexer module for ingesting and indexing API contracts."""

import os
import yaml
import json
import time
import hashlib
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional, Set, Tuple
import numpy as np
from sentence_transformers import SentenceTransformer
from pinecone import Pinecone
from rich.console import Console
from rich.progress import Progress
from sklearn.decomposition import PCA
import httpx
from dataclasses import dataclass
from enum import Enum
import re

from plexure_api_search.constants import (
    OPENROUTER_URL,
    PINECONE_ENVIRONMENT,
    PINECONE_API_KEY,
    PINECONE_INDEX_NAME,
    OPENROUTER_API_KEY,
    DEFAULT_API_DIR,
    DEFAULT_BATCH_SIZE,
    LLM_MODEL,
    LLM_TEMPERATURE,
    HTTP_HEADERS,
    SENTENCE_TRANSFORMER_MODEL,
    PINECONE_DIMENSION
)

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
        numbers = [int(n) for n in re.findall(r'\d+', version)]
        
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
        api_data["method"] = DataValidator.normalize_method(api_data.get("method", "GET"))
        api_data["api_version"] = DataValidator.normalize_version(api_data.get("api_version", ""))
        
        # Clean and normalize text fields
        api_data["description"] = api_data.get("description", "").strip() or "No description provided."
        api_data["summary"] = api_data.get("summary", "").strip() or api_data["description"][:100]
        
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
    
    def __init__(self, headers: Dict[str, str]):
        self.headers = headers
        
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
        elif any(term in path + description for term in ["monitor", "health", "metric"]):
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
            "has_auth": any(term in path + description for term in ["auth", "token", "secure"]),
            "has_examples": bool(api_data.get("examples") or "example" in description),
            "supports_pagination": any(term in description for term in ["page", "limit", "offset"]) or
                                method == "GET" and bool(api_data.get("parameters")),
            "deprecated": "deprecat" in description or api_data.get("deprecated", False)
        }
    
    def enrich_metadata(self, api_data: Dict[str, Any]) -> Dict[str, Any]:
        """Enrich API metadata with LLM-generated insights."""
        try:
            # First, infer basic metadata
            category = self._infer_category(api_data)
            features = self._infer_features(api_data)
            
            # Create context for LLM
            context = {
                "path": api_data["path"],
                "method": api_data["method"],
                "version": api_data["api_version"],
                "description": api_data["description"],
                "summary": api_data["summary"],
                "parameters": api_data["parameters"],
                "tags": api_data["tags"],
                "inferred_category": category.value,
                "inferred_features": features
            }
            
            # Get LLM enrichment
            response = httpx.post(
                OPENROUTER_URL,
                headers=self.headers,
                json={
                    "model": LLM_MODEL,
                    "messages": [
                        {
                            "role": "system",
                            "content": """You are an API documentation expert. Analyze the API endpoint and provide enriched metadata.

Example input:
path: /users
method: GET
description: Retrieve a list of users.

Example output:
features:
    has_auth: true
    supports_pagination: true
    deprecated: false
category: Users
summary: Fetches user data with pagination support.
usage_notes: Use the 'page' parameter for pagination.
best_practices: Filter results to minimize response size."""
                        },
                        {"role": "user", "content": json.dumps(context)}
                    ],
                    "temperature": LLM_TEMPERATURE["metadata_enrichment"]
                }
            )
            
            if response.status_code == 200:
                enriched = response.json()["choices"][0]["message"]["content"]
                try:
                    enriched_data = yaml.safe_load(enriched)
                    
                    # Validate enriched data
                    required_keys = {"features", "summary", "usage_notes"}
                    if all(key in enriched_data for key in required_keys):
                        # Merge inferred and LLM-generated metadata
                        api_data.update(enriched_data)
                        api_data["features"].update(features)  # Prefer inferred features
                        api_data["category"] = category.value
                        return api_data
                        
                except Exception as e:
                    print(f"Failed to parse LLM response: {e}")
            
            # Fallback to inferred metadata
            api_data["features"] = features
            api_data["category"] = category.value
            return api_data
            
        except Exception as e:
            print(f"Metadata enrichment failed: {e}")
            return api_data

class APIIndexer:
    """Handles API contract ingestion and indexing."""
    
    def __init__(self, api_dir: str = DEFAULT_API_DIR):
        self.api_dir = api_dir
        self.console = Console()
        self.model = SentenceTransformer(SENTENCE_TRANSFORMER_MODEL)
        self.headers = {
            **HTTP_HEADERS,
            "Authorization": f"Bearer {OPENROUTER_API_KEY}"
        }
        
        # Initialize Pinecone
        self.pc = Pinecone(api_key=PINECONE_API_KEY)
        
        # Initialize PCA for embedding compression
        self.pca = PCA(n_components=PINECONE_DIMENSION)
        self.index_cache_file = Path(api_dir) / ".index_cache.json"
        
        # Ensure index exists
        if PINECONE_INDEX_NAME not in self.pc.list_indexes().names():
            self.pc.create_index(
                name=PINECONE_INDEX_NAME,
                dimension=PINECONE_DIMENSION,
                metric='cosine'
            )
        
        self.index = self.pc.Index(PINECONE_INDEX_NAME)
        
        # Initialize helpers
        self.validator = DataValidator()
        self.enricher = MetadataEnricher(self.headers)
        
        # Load or initialize index cache
        self.index_cache = self._load_index_cache()
        
        # Track indexed endpoints
        self.indexed_signatures: Set[EndpointSignature] = set()

    def _load_index_cache(self) -> Dict[str, Any]:
        """Load index cache from file."""
        if self.index_cache_file.exists():
            try:
                with open(self.index_cache_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                self.console.print(f"[yellow]Warning: Failed to load index cache: {e}[/]")
        
        return {
            "last_indexed": {},
            "file_hashes": {},
            "embedding_stats": {
                "mean": None,
                "std": None
            }
        }

    def _save_index_cache(self) -> None:
        """Save index cache to file."""
        try:
            with open(self.index_cache_file, 'w') as f:
                json.dump(self.index_cache, f)
        except Exception as e:
            self.console.print(f"[yellow]Warning: Failed to save index cache: {e}[/]")

    def _get_file_hash(self, file_path: str) -> str:
        """Calculate file hash."""
        hasher = hashlib.sha256()
        with open(file_path, 'rb') as f:
            for chunk in iter(lambda: f.read(4096), b''):
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
        Path: {endpoint['path']}
        Method: {endpoint['method']}
        Version: {endpoint['api_version']}
        Category: {endpoint.get('category', '')}
        Description: {endpoint.get('description', '')}
        Summary: {endpoint.get('summary', '')}
        Tags: {', '.join(endpoint.get('tags', []))}
        Usage Notes: {endpoint.get('usage_notes', '')}
        Best Practices: {endpoint.get('best_practices', '')}
        """
        
        # Get embedding
        embedding = self.model.encode(text)
        
        # Normalize embedding
        if self.index_cache["embedding_stats"]["mean"] is not None:
            embedding = (embedding - self.index_cache["embedding_stats"]["mean"]) / self.index_cache["embedding_stats"]["std"]
        
        return embedding

    def _process_api_file(self, file_path: str) -> List[Dict[str, Any]]:
        """Process a single API file and extract endpoints."""
        if not self._needs_indexing(file_path):
            return []
            
        endpoints = []
        
        try:
            with open(file_path, 'r') as f:
                api_spec = yaml.safe_load(f)
            
            # Extract version from file path or spec
            version = api_spec.get("version", "1.0.0")
            
            # Process paths
            paths = api_spec.get("paths", {})
            for path, methods in paths.items():
                for method, details in methods.items():
                    if method.upper() in ["GET", "POST", "PUT", "DELETE", "PATCH"]:
                        # Create base endpoint data
                        endpoint_data = {
                            "path": path,
                            "method": method.upper(),
                            "description": details.get("description", ""),
                            "summary": details.get("summary", ""),
                            "parameters": details.get("parameters", []),
                            "tags": details.get("tags", []),
                            "api_version": version,
                            "file_path": file_path,
                            "last_modified": datetime.fromtimestamp(os.path.getmtime(file_path)).isoformat()
                        }
                        
                        # Validate and normalize
                        endpoint_data = self.validator.validate_and_normalize(endpoint_data)
                        
                        # Create signature
                        signature = EndpointSignature(
                            path=endpoint_data["path"],
                            method=endpoint_data["method"],
                            version=endpoint_data["api_version"]
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
            if not force_reindex and stats['total_vector_count'] > 0:
                self.console.print("\nUsing existing Pinecone index")
                return
            
            self.console.print(f"\nIndexing API contracts from directory: {self.api_dir}")
            
            # Collect all API files
            api_files = []
            for root, _, files in os.walk(self.api_dir):
                for file in files:
                    if file.endswith(('.yaml', '.yml')):
                        api_files.append(os.path.join(root, file))
            
            if not api_files:
                self.console.print("[yellow]No API files found.[/]")
                return
            
            # Process files in parallel
            all_endpoints = []
            with ThreadPoolExecutor(max_workers=4) as executor:
                with Progress() as progress:
                    task = progress.add_task("[cyan]Processing API files...", total=len(api_files))
                    
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
            self.index_cache["embedding_stats"]["mean"] = embeddings_array.mean(axis=0).tolist()
            self.index_cache["embedding_stats"]["std"] = embeddings_array.std(axis=0).tolist()
            
            # Compress embeddings if needed
            if len(embeddings[0]) > PINECONE_DIMENSION:
                embeddings = self.pca.fit_transform(embeddings)
            
            # Index endpoints in batches
            with Progress() as progress:
                task = progress.add_task("[cyan]Indexing endpoints...", total=len(all_endpoints))
                
                for i in range(0, len(all_endpoints), DEFAULT_BATCH_SIZE):
                    batch = all_endpoints[i:i + DEFAULT_BATCH_SIZE]
                    batch_embeddings = embeddings[i:i + DEFAULT_BATCH_SIZE]
                    
                    vectors = [
                        (
                            f"{endpoint['method']}_{endpoint['path']}",
                            embedding.tolist(),
                            endpoint
                        )
                        for endpoint, embedding in zip(batch, batch_embeddings)
                    ]
                    
                    # Upsert batch to Pinecone
                    self.index.upsert(vectors=vectors)
                    progress.advance(task, len(batch))
            
            # Save index cache
            self._save_index_cache()
            
            self.console.print(f"\n[green]Successfully indexed {len(all_endpoints)} endpoints.[/]")
            
        except Exception as e:
            self.console.print(f"[red]Error during indexing: {str(e)}[/]")

def main():
    """Main function for running the indexer directly."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Index API contracts')
    parser.add_argument('--api-dir', default=DEFAULT_API_DIR, help='Directory containing API contracts')
    parser.add_argument('--force-reindex', action='store_true', help='Force reindexing even if index exists')
    
    args = parser.parse_args()
    
    indexer = APIIndexer(api_dir=args.api_dir)
    indexer.index_apis(force_reindex=args.force_reindex)

if __name__ == '__main__':
    main()
