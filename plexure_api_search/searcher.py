"""Advanced API search with triple vector embeddings and contextual boosting."""

import time
import json
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
from pinecone import Pinecone, ServerlessSpec
from .embeddings import TripleVectorizer
from .boosting import ContextualBooster
from .understanding import ZeroShotUnderstanding
from .expansion import QueryExpander
from .quality import QualityMetrics, SearchResult, SearchEvaluation


class APISearcher:
    """Advanced API search engine with multiple strategies."""
    
    def __init__(
        self,
        index_name: str,
        api_key: str,
        environment: str,
        cloud: str = "aws",
        region: str = "us-east-1"
    ):
        """Initialize searcher with advanced features.
        
        Args:
            index_name: Name of the Pinecone index.
            api_key: Pinecone API key.
            environment: Pinecone environment.
            cloud: Cloud provider (aws, gcp, azure).
            region: Cloud region.
        """
        if not api_key:
            raise ValueError("Pinecone API key is required")
        if not index_name:
            raise ValueError("Pinecone index name is required")
        if not environment:
            raise ValueError("Pinecone environment is required")
            
        # Initialize Pinecone
        try:
            # Initialize Pinecone with the API key
            pc = Pinecone(api_key=api_key)
            
            # Get index instance
            self.index = pc.Index(index_name)
            
            # Verify connection
            try:
                stats = self.index.describe_index_stats()
                print(f"\nIndex stats:")
                print(f"Total vectors: {stats.get('total_vector_count', 0)}")
                print(f"Dimension: {stats.get('dimension', 384)}")
            except Exception as e:
                print(f"Warning: Could not get index stats: {e}")
                
        except Exception as e:
            raise RuntimeError(f"Failed to initialize Pinecone: {str(e)}")
            
        # Initialize advanced components
        self.vectorizer = TripleVectorizer()
        self.booster = ContextualBooster()
        self.understanding = ZeroShotUnderstanding()
        self.expander = QueryExpander()
        self.metrics = QualityMetrics()
        
        # Search configuration
        self.top_k = 10
        self.min_score = 0.5
        
    def search(
        self,
        query: str,
        filters: Optional[Dict] = None,
        include_metadata: bool = True
    ) -> Tuple[List[Dict], SearchEvaluation]:
        """Perform advanced API search.
        
        Args:
            query: Search query string.
            filters: Optional filters for search.
            include_metadata: Whether to include metadata in results.
            
        Returns:
            Tuple of (search results, evaluation metrics).
        """
        start_time = time.time()
        
        try:
            # Expand query
            expanded = self.expander.expand_query(query)
            
            # Generate triple vectors for query
            query_vectors = self.vectorizer.vectorize({
                'path': '',
                'method': '',
                'description': query,
                'summary': expanded.original_query
            })
            
            # Get contextual weights
            weights = self.booster.adjust_weights(query)
            
            # Combine vectors with weights and convert to list
            combined_vector = query_vectors.to_combined_vector(weights.to_dict())
            if hasattr(combined_vector, 'tolist'):
                combined_vector = combined_vector.tolist()
            
            # Debug print query vector
            print("\nSearch query vector:")
            print(f"Length: {len(combined_vector)}")
            print(f"Sample: {combined_vector[:5]}...")
            
            # Perform search
            search_results = self.index.query(
                vector=combined_vector,
                top_k=self.top_k,
                include_metadata=include_metadata,
                filter=filters
            )
            
            print(f"\nFound {len(search_results.matches)} matches")
            
            # Process results
            results = []
            search_eval_results = []
            
            for i, match in enumerate(search_results.matches, 1):
                try:
                    # Extract metadata
                    metadata = match.metadata if include_metadata else {}
                    
                    # Debug print raw metadata
                    print(f"\nRaw metadata for result {i}:")
                    print(json.dumps(metadata, indent=2))
                    
                    # Create result object with rich metadata
                    result = {
                        'endpoint_id': match.id,
                        'score': match.score,
                        'api_name': metadata.get('api_name', 'N/A'),
                        'api_version': metadata.get('api_version', 'N/A'),
                        'path': metadata.get('path', 'N/A'),
                        'method': metadata.get('method', 'N/A'),
                        'description': metadata.get('description', 'N/A'),
                        'summary': metadata.get('summary', ''),
                        'parameters': metadata.get('parameters', []),
                        'responses': metadata.get('responses', {}),
                        'tags': metadata.get('tags', []),
                        'operationId': metadata.get('operationId', ''),
                        'deprecated': metadata.get('deprecated', 'false') == 'true',
                        'security': metadata.get('security', []),
                        'requires_auth': metadata.get('requires_auth', 'false') == 'true',
                        'full_path': metadata.get('full_path', 'N/A')
                    }
                    
                    # Debug print processed result
                    print(f"\nProcessed result {i}:")
                    print(json.dumps(result, indent=2))
                    
                    results.append(result)
                    
                    # Create evaluation result
                    search_eval_results.append(SearchResult(
                        endpoint_id=match.id,
                        score=match.score,
                        rank=i,
                        is_relevant=match.score >= self.min_score
                    ))
                    
                except Exception as e:
                    print(f"Error processing search result {i}: {e}")
                    print(f"Raw match data: {match}")
                    continue
            
            # Calculate latency
            latency_ms = (time.time() - start_time) * 1000
            
            # Evaluate search quality
            evaluation = self.metrics.evaluate_search(
                query=query,
                results=search_eval_results,
                total_relevant=len([r for r in search_eval_results if r.is_relevant]),
                latency_ms=latency_ms
            )
            
            # Record search for contextual learning
            self.booster.record_search(query)
            
            return results, evaluation
            
        except Exception as e:
            print(f"Search error: {e}")
            raise RuntimeError(f"Search failed: {str(e)}")
            
    def update_feedback(
        self,
        query: str,
        endpoint_id: str,
        is_relevant: bool,
        score: float = 1.0
    ) -> None:
        """Update feedback for search results.
        
        Args:
            query: Original search query.
            endpoint_id: ID of the endpoint.
            is_relevant: Whether the result was relevant.
            score: Feedback score (0 to 1).
        """
        try:
            # Update contextual booster
            self.booster.update_feedback(query, score if is_relevant else 0.0)
        except Exception as e:
            raise RuntimeError(f"Failed to update feedback: {str(e)}")
            
    def get_quality_metrics(self) -> Dict[str, float]:
        """Get current quality metrics.
        
        Returns:
            Dictionary of quality metrics.
        """
        try:
            return self.metrics.get_average_metrics()
        except Exception as e:
            raise RuntimeError(f"Failed to get quality metrics: {str(e)}")
            
    def get_metric_trends(self) -> Dict[str, List[float]]:
        """Get metric trends over time.
        
        Returns:
            Dictionary of metric trends.
        """
        try:
            return self.metrics.get_metric_trends()
        except Exception as e:
            raise RuntimeError(f"Failed to get metric trends: {str(e)}")
        
    def analyze_query(self, query: str) -> Dict[str, Any]:
        """Analyze a search query.
        
        Args:
            query: Search query string.
            
        Returns:
            Dictionary with query analysis.
        """
        try:
            # Expand query
            expanded = self.expander.expand_query(query)
            
            # Get contextual weights
            weights = self.booster.adjust_weights(query)
            
            return {
                'original_query': query,
                'semantic_variants': expanded.semantic_variants,
                'technical_mappings': expanded.technical_mappings,
                'use_cases': expanded.use_cases,
                'weights': expanded.weights,
                'contextual_weights': weights.to_dict()
            }
        except Exception as e:
            raise RuntimeError(f"Query analysis failed: {str(e)}")
