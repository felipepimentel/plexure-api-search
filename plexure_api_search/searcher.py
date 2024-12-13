"""Search engine module for API contracts."""

import re
from typing import Dict, List, Optional, Union

import numpy as np
from rich.console import Console
from rich.table import Table
from sentence_transformers import SentenceTransformer

from plexure_api_search.cache import SearchCache
from plexure_api_search.config import Config
from plexure_api_search.metrics import MetricsCalculator
from plexure_api_search.monitoring import Logger
from plexure_api_search.validation import DataValidator


class SearchEngine:
    """Search engine for API contracts."""

    def __init__(self, config: Optional[Config] = None):
        """Initialize the search engine.
        
        Args:
            config: Optional configuration object.
        """
        self.config = config or Config()
        self.logger = Logger()
        self.cache = SearchCache()
        self.metrics = MetricsCalculator()
        self.validator = DataValidator()
        self.console = Console()
        
        # Initialize the embedding model
        self.model = SentenceTransformer(self.config.model_name)
        
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
                result.get('api_name', 'N/A'),
                result.get('version', 'N/A'),
                result.get('endpoint', 'N/A'),
                result.get('description', 'N/A')
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
            r'version[s]?\s+\d+',
            r'v\d+',
            r'apis?\s+in\s+v\d+',
            r'quantas?\s+apis?\s+(?:estão\s+)?(?:na\s+)?versão\s+\d+'
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
        version_match = re.search(r'\d+', query)
        if not version_match:
            return []
            
        version = version_match.group()
        
        # Get all APIs with matching version
        results = self._get_apis_by_version(version)
        
        # Format results
        formatted_results = []
        for api in results:
            formatted_results.append({
                'score': 1.0,
                'api_name': api['name'],
                'version': api['version'],
                'endpoint': 'N/A',
                'description': f"API version {api['version']}"
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
        # TODO: Implement actual vector search using Pinecone
        # This is a placeholder
        return []
        
    def _get_apis_by_version(self, version: str) -> List[Dict]:
        """Get all APIs matching a specific version.
        
        Args:
            version: The version to filter by.
            
        Returns:
            List of APIs with matching version.
        """
        # TODO: Implement actual version filtering
        # This is a placeholder
        return []
        
    def _enrich_results(self, results: List[Dict]) -> List[Dict[str, Union[str, float]]]:
        """Enrich search results with additional metadata.
        
        Args:
            results: Raw search results.
            
        Returns:
            Enriched results with metadata.
        """
        enriched = []
        for result in results:
            metadata = result.get('metadata', {})
            enriched.append({
                'score': result.get('score', 0.0),
                'api_name': metadata.get('api_name', 'N/A'),
                'version': metadata.get('version', 'N/A'),
                'endpoint': metadata.get('endpoint', 'N/A'),
                'description': metadata.get('description', 'N/A')
            })
        return enriched
