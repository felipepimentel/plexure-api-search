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
from .pinecone_client import PineconeClient


class APISearcher:
    """Advanced API search engine with multiple strategies."""
    
    def __init__(self, pinecone_client: PineconeClient):
        """Initialize searcher.
        
        Args:
            pinecone_client: PineconeClient instance
        """
        self.client = pinecone_client
        self.vectorizer = TripleVectorizer()
        self.booster = ContextualBooster()
        self.understanding = ZeroShotUnderstanding()
        self.expander = QueryExpander()
        self.metrics = QualityMetrics()
        self.top_k = 10
        
    def search(
        self,
        query: str,
        filters: Optional[Dict] = None,
        include_metadata: bool = True
    ) -> List[Dict]:
        """Search for API endpoints."""
        try:
            query_vector = self.vectorizer.vectorize_query(query)
            results = self.client.search_vectors(
                query_vector=query_vector,
                top_k=self.top_k,
                filters=filters,
                include_metadata=include_metadata
            )
            return self._process_results(results)
        except Exception as e:
            logger.error(f"Search failed: {e}")
            raise
        
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
