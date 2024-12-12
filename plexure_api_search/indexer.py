"""API Indexer module for ingesting and indexing API contracts."""

import os
import yaml
import json
import httpx
from typing import Dict, List, Any
from sentence_transformers import SentenceTransformer
from pinecone import Pinecone
from rich.console import Console
from rich.progress import Progress

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
    PINECONE_DIMENSION,
    ERROR_MESSAGES
)

class APIIndexer:
    """Handles API contract ingestion and indexing."""
    
    def __init__(self, api_dir: str = DEFAULT_API_DIR):
        self.api_dir = api_dir
        self.console = Console()
        
        # Validate environment variables
        if not PINECONE_API_KEY:
            raise ValueError(ERROR_MESSAGES["api_key_missing"].format(key="PINECONE_API_KEY"))
        if not OPENROUTER_API_KEY:
            raise ValueError(ERROR_MESSAGES["api_key_missing"].format(key="OPENROUTER_API_KEY"))
        
        self.model = SentenceTransformer(SENTENCE_TRANSFORMER_MODEL)
        self.headers = {
            **HTTP_HEADERS,
            "Authorization": f"Bearer {OPENROUTER_API_KEY}"
        }
        
        # Initialize Pinecone
        self.pc = Pinecone(api_key=PINECONE_API_KEY)
        
        # Ensure index exists
        if PINECONE_INDEX_NAME not in self.pc.list_indexes().names():
            self.pc.create_index(
                name=PINECONE_INDEX_NAME,
                dimension=PINECONE_DIMENSION,
                metric='cosine'
            )
        
        self.index = self.pc.Index(PINECONE_INDEX_NAME)

    def _enrich_api_metadata(self, api_data: Dict[str, Any]) -> Dict[str, Any]:
        """Enrich API metadata using LLM."""
        try:
            # Create context for the LLM
            context = {
                "path": api_data.get("path", ""),
                "method": api_data.get("method", ""),
                "description": api_data.get("description", ""),
                "parameters": api_data.get("parameters", []),
                "responses": api_data.get("responses", {}),
                "tags": api_data.get("tags", [])
            }
            
            response = httpx.post(
                OPENROUTER_URL,
                headers=self.headers,
                json={
                    "model": LLM_MODEL,
                    "messages": [
                        {
                            "role": "system",
                            "content": """You are an API expert analyzing OpenAPI endpoints. Your task is to extract and infer key metadata.

TASK:
Analyze the API endpoint and provide structured metadata about its characteristics.

RESPONSE FORMAT (YAML):
features:
    has_auth: boolean  # Requires authentication
    has_examples: boolean  # Has usage examples
    supports_pagination: boolean  # Supports pagination
    deprecated: boolean  # Is deprecated
category: string  # Main category (e.g., Users, Authentication, Data)
summary: string  # One-line technical summary
usage_notes: string  # Key points for usage
best_practices: string  # Implementation recommendations"""
                        },
                        {"role": "user", "content": json.dumps(context)}
                    ],
                    "temperature": LLM_TEMPERATURE["metadata_enrichment"]
                }
            )
            
            if response.status_code == 200:
                content = response.json()["choices"][0]["message"]["content"]
                # Clean up the YAML content
                content = content.replace("```yaml", "").replace("```", "").strip()
                enriched_data = yaml.safe_load(content)
                api_data.update(enriched_data)
                
        except Exception as e:
            self.console.print(f"[yellow]Warning: API enrichment failed: {str(e)}[/]")
            
        return api_data

    def _process_api_file(self, file_path: str) -> List[Dict[str, Any]]:
        """Process a single API file and extract endpoints."""
        endpoints = []
        
        try:
            with open(file_path, 'r') as f:
                api_spec = yaml.safe_load(f)
            
            # Extract version from file path or spec
            version = "1.0.0"  # Default version
            if "version" in api_spec:
                version = api_spec["version"]
            
            # Process paths
            paths = api_spec.get("paths", {})
            for path, methods in paths.items():
                for method, details in methods.items():
                    if method.upper() in ["GET", "POST", "PUT", "DELETE", "PATCH"]:
                        endpoint_data = {
                            "path": path,
                            "method": method.upper(),
                            "description": details.get("description", ""),
                            "summary": details.get("summary", ""),
                            "parameters": [
                                f"{p['name']}:{p['schema']['type']}"
                                for p in details.get("parameters", [])
                            ],
                            "tags": details.get("tags", []),
                            "api_version": version
                        }
                        
                        # Enrich with LLM-generated metadata
                        endpoint_data = self._enrich_api_metadata(endpoint_data)
                        endpoints.append(endpoint_data)
            
        except Exception as e:
            self.console.print(f"[red]Error processing {file_path}: {str(e)}[/]")
        
        return endpoints

    def index_apis(self, force_reindex: bool = False) -> None:
        """Index all API contracts from the specified directory."""
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
            
            # Process each API file
            all_endpoints = []
            with Progress() as progress:
                task = progress.add_task("[cyan]Processing API files...", total=len(api_files))
                
                for file_path in api_files:
                    endpoints = self._process_api_file(file_path)
                    all_endpoints.extend(endpoints)
                    progress.advance(task)
            
            # Index endpoints in batches
            batch_size = DEFAULT_BATCH_SIZE
            with Progress() as progress:
                task = progress.add_task("[cyan]Indexing endpoints...", total=len(all_endpoints))
                
                for i in range(0, len(all_endpoints), batch_size):
                    batch = all_endpoints[i:i + batch_size]
                    vectors = []
                    
                    for endpoint in batch:
                        # Create embedding from description and path
                        text = f"{endpoint['path']} {endpoint['method']} {endpoint['description']} {endpoint.get('summary', '')}"
                        vector = self.model.encode(text).tolist()
                        
                        vectors.append((
                            f"{endpoint['method']}_{endpoint['path']}",
                            vector,
                            endpoint
                        ))
                    
                    # Upsert batch to Pinecone
                    self.index.upsert(vectors=[(id, vec, meta) for id, vec, meta in vectors])
                    progress.advance(task, len(batch))
            
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