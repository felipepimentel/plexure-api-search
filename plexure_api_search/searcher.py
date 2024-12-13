"""Search engine module for API contracts."""

import glob
import os
import re
from typing import Any, Dict, List, Optional, Union

import numpy as np
import yaml
from pinecone import Pinecone, ServerlessSpec
from rich.console import Console
from rich.table import Table
from sentence_transformers import SentenceTransformer

from plexure_api_search.cache import SearchCache
from plexure_api_search.config import Config, ConfigManager
from plexure_api_search.metrics import MetricsCalculator
from plexure_api_search.monitoring import Logger
from plexure_api_search.validation import DataValidator


def unflatten_metadata(metadata: Dict[str, Any]) -> Dict[str, Any]:
    """Unflatten metadata from Pinecone format.

    Args:
        metadata: Flattened metadata dictionary.

    Returns:
        Unflattened metadata dictionary.
    """
    unflattened = {}

    for key, value in metadata.items():
        if "_" in key:
            parent_key, child_key = key.split("_", 1)
            if parent_key not in unflattened:
                unflattened[parent_key] = {}
            unflattened[parent_key][child_key] = value
        else:
            unflattened[key] = value

    return unflattened


class SearchEngine:
    """Search engine for API contracts."""

    def __init__(self, config: Optional[Config] = None):
        """Initialize the search engine.

        Args:
            config: Optional configuration object.
        """
        self.config = config or ConfigManager().load_config()
        self.logger = Logger()
        self.cache = SearchCache()
        self.metrics = MetricsCalculator()
        self.validator = DataValidator()
        self.console = Console()

        # Initialize the embedding model
        self.model = SentenceTransformer(self.config.model_name)

        # Initialize Pinecone
        self.pc = Pinecone(api_key=self.config.pinecone_api_key)

        # Ensure index exists
        if self.config.pinecone_index not in self.pc.list_indexes().names():
            # Create index with serverless spec
            self.pc.create_index(
                name=self.config.pinecone_index,
                dimension=self.config.embedding_dimension,
                metric="cosine",
                spec=ServerlessSpec(
                    cloud=self.config.pinecone_cloud, region=self.config.pinecone_region
                ),
            )

        self.index = self.pc.Index(self.config.pinecone_index)

        # Load API data
        self.api_data = self._load_api_data()

    def search(self, query: str, top_k: int = 5) -> List[Dict[str, Union[str, float]]]:
        """Search for API endpoints matching the query.

        Args:
            query: The search query.
            top_k: Number of top results to return.

        Returns:
            List of search results with metadata.
        """
        try:
            # Check cache first
            cached_results = self.cache.get_search_results(query)
            if cached_results:
                self.logger.info(f"Cache hit for query: {query}")
                return cached_results

            # Process version-specific queries
            if self._is_version_query(query):
                return self._handle_version_query(query)

            # Get query embedding
            query_embedding = self.model.encode(query)

            # Search in vector store
            results = self._vector_search(query_embedding, top_k)

            # Enrich results with metadata
            enriched_results = self._enrich_results(results)

            # Cache results
            self.cache.store_search_results(query, enriched_results)

            # Log metrics
            self.metrics.log_search_metrics(query, len(enriched_results))

            return enriched_results

        except Exception as e:
            self.logger.error(f"Search failed: {str(e)}")
            raise

    def display_results(self, results: List[Dict[str, Union[str, float]]]) -> None:
        """Display search results in a formatted table.

        Args:
            results: List of search results to display.
        """
        table = Table(title="API Search Results")

        table.add_column("Score", justify="right", style="cyan")
        table.add_column("API", style="green")
        table.add_column("Version", style="yellow")
        table.add_column("Endpoint", style="blue")
        table.add_column("Description", style="white")

        for result in results:
            table.add_row(
                f"{result['score']:.3f}",
                result.get("api_name", "N/A"),
                result.get("version", "N/A"),
                result.get("endpoint", "N/A"),
                result.get("description", "N/A"),
            )

        self.console.print(table)

    def _is_version_query(self, query: str) -> bool:
        """Check if query is about API versions.

        Args:
            query: The search query.

        Returns:
            True if query is about versions, False otherwise.
        """
        version_patterns = [
            r"version[s]?\s+\d+(\.\d+)*",
            r"v\d+(\.\d+)*",
            r"apis?\s+in\s+v\d+(\.\d+)*",
            r"quantas?\s+apis?\s+(?:estão\s+)?(?:na\s+)?versão\s+\d+(\.\d+)*",
        ]

        return any(re.search(pattern, query.lower()) for pattern in version_patterns)

    def _handle_version_query(self, query: str) -> List[Dict[str, Union[str, float]]]:
        """Handle version-specific queries.

        Args:
            query: The version-related query.

        Returns:
            List of APIs matching the version criteria.
        """
        # Extract version number from query
        version_match = re.search(r"\d+(\.\d+)*", query)
        if not version_match:
            return []

        version = version_match.group()

        # Get all APIs with matching version
        results = self._get_apis_by_version(version)

        # Format results
        formatted_results = []
        for api in results:
            formatted_results.append({
                "score": 1.0,
                "api_name": api["name"],
                "version": api["version"],
                "endpoint": api["endpoint"],
                "description": f"API version {api['version']}",
            })

        return formatted_results

    def _vector_search(self, query_embedding: np.ndarray, top_k: int) -> List[Dict]:
        """Perform vector search using embeddings.

        Args:
            query_embedding: The query embedding vector.
            top_k: Number of results to return.

        Returns:
            List of search results.
        """
        # Query Pinecone
        results = self.index.query(
            vector=query_embedding.tolist(), top_k=top_k, include_metadata=True
        )

        # Format results
        formatted_results = []
        for match in results.matches:
            # Unflatten metadata
            metadata = unflatten_metadata(match.metadata)

            formatted_results.append({"score": match.score, "metadata": metadata})

        return formatted_results

    def _get_apis_by_version(self, version: str) -> List[Dict]:
        """Get all APIs with matching version.

        Args:
            version: Version to filter by.

        Returns:
            List of APIs with matching version.
        """
        matching_apis = []

        for api in self.api_data:
            api_version = str(api.get("version", ""))
            if self._version_matches(api_version, version):
                # Load the API file to get endpoint information
                try:
                    with open(api["file_path"], "r") as f:
                        data = yaml.safe_load(f)
                    
                    # Get the first endpoint path as an example
                    paths = data.get("paths", {})
                    endpoint = next(iter(paths.keys())) if paths else "/"
                    
                    matching_apis.append({
                        "name": api.get("name", "Unknown"),
                        "version": api_version,
                        "description": api.get("description", ""),
                        "endpoint": endpoint
                    })
                except Exception as e:
                    self.logger.error(f"Failed to load endpoint data from {api['file_path']}: {e}")
                    matching_apis.append({
                        "name": api.get("name", "Unknown"),
                        "version": api_version,
                        "description": api.get("description", ""),
                        "endpoint": "/"
                    })

        return matching_apis

    def _version_matches(self, api_version: str, query_version: str) -> bool:
        """Check if API version matches query version.

        Args:
            api_version: API version string.
            query_version: Query version string.

        Returns:
            True if versions match, False otherwise.
        """
        # Clean and normalize versions
        api_parts = api_version.strip().split(".")
        query_parts = query_version.strip().split(".")

        # Pad with zeros for comparison
        while len(api_parts) < len(query_parts):
            api_parts.append("0")
        while len(query_parts) < len(api_parts):
            query_parts.append("0")

        # Compare each part
        return api_parts == query_parts

    def _load_api_data(self) -> List[Dict]:
        """Load API data from files.

        Returns:
            List of API data dictionaries.
        """
        api_data = []
        api_dir = self.config.api_dir

        # Find all YAML files
        yaml_files = glob.glob(os.path.join(api_dir, "**/*.yml"), recursive=True)
        yaml_files.extend(glob.glob(os.path.join(api_dir, "**/*.yaml"), recursive=True))

        for file_path in yaml_files:
            try:
                with open(file_path, "r") as f:
                    data = yaml.safe_load(f)

                if isinstance(data, dict):
                    # Extract API name from file path
                    api_name = os.path.splitext(os.path.basename(file_path))[0]

                    # Add API data
                    api_data.append({
                        "name": api_name,
                        "version": data.get("version", "1.0.0"),
                        "description": data.get("description", ""),
                        "file_path": file_path,
                    })

            except Exception as e:
                self.logger.error(f"Failed to load API data from {file_path}: {e}")

        return api_data

    def _enrich_results(
        self, results: List[Dict]
    ) -> List[Dict[str, Union[str, float]]]:
        """Enrich search results with additional metadata.

        Args:
            results: Raw search results.

        Returns:
            Enriched results with metadata.
        """
        enriched = []
        for result in results:
            # Get metadata directly from the match
            metadata = result.get("metadata", {})
            
            # Handle nested API info
            api_info = metadata.get("api", {})
            
            enriched.append({
                "score": result.get("score", 0.0),
                "api_name": api_info.get("name", "N/A"),
                "version": api_info.get("version", "N/A"),
                "endpoint": metadata.get("endpoint", "N/A"),
                "description": metadata.get("description", "") or metadata.get("summary", "No description provided."),
            })
        return enriched
