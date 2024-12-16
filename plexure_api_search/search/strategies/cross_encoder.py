"""Cross-encoder reranking strategy."""

import logging
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from ...embedding.embeddings import ModernEmbeddings
from ..cache import ScoreData, SemanticScoreCache
from ..storage import EndpointData
from .base import BaseSearchStrategy, SearchConfig, SearchResult, StrategyFactory

logger = logging.getLogger(__name__)


@StrategyFactory.register("cross_encoder")
class CrossEncoderStrategy(BaseSearchStrategy):
    """Cross-encoder reranking strategy."""

    def __init__(self, config: Optional[SearchConfig] = None):
        """Initialize cross-encoder strategy.

        Args:
            config: Optional strategy configuration
        """
        super().__init__(config)
        self.score_cache = SemanticScoreCache("cross_encoder")

    def _prepare_endpoint_text(self, endpoint: EndpointData) -> str:
        """Prepare endpoint text for cross-encoder.

        Args:
            endpoint: Endpoint data

        Returns:
            Formatted endpoint text
        """
        try:
            parts = []
            
            # Add method and path
            parts.append(f"{endpoint.method} {endpoint.path}")
            
            # Add description
            if endpoint.description:
                parts.append(endpoint.description)
                
            # Add parameters
            if endpoint.parameters:
                param_text = "Parameters: " + ", ".join(str(p) for p in endpoint.parameters)
                parts.append(param_text)
                
            # Add responses
            if endpoint.responses:
                response_text = "Responses: " + ", ".join(str(r) for r in endpoint.responses)
                parts.append(response_text)
                
            return " | ".join(parts)

        except Exception as e:
            logger.error(f"Error preparing endpoint text: {e}")
            return ""

    def _rerank_results(
        self,
        query: str,
        results: List[Tuple[EndpointData, float]],
        top_k: int = 10,
    ) -> List[Tuple[EndpointData, float, float]]:
        """Rerank results using cross-encoder.

        Args:
            query: Search query
            results: Initial search results (endpoint, score)
            top_k: Number of results to return

        Returns:
            List of (endpoint, combined_score, cross_encoder_score)
        """
        try:
            if not results:
                return []
                
            # Prepare text pairs for cross-encoder
            pairs = []
            for endpoint, _ in results:
                endpoint_text = self._prepare_endpoint_text(endpoint)
                if endpoint_text:
                    pairs.append((query, endpoint_text))
                    
            if not pairs:
                return [(e, s, 0.0) for e, s in results[:top_k]]
                
            # Get cross-encoder scores
            cross_encoder_scores = self.embeddings.compute_similarity_batch(
                pairs,
                strategy="cross_encoder",
            )
            
            # Combine with original scores
            reranked = []
            for (endpoint, original_score), cross_score in zip(results, cross_encoder_scores):
                # Combine cross-encoder score with original score
                combined_score = (original_score * 0.3 + cross_score * 0.7)
                reranked.append((endpoint, combined_score, cross_score))
                
            # Sort by combined score
            reranked.sort(key=lambda x: x[1], reverse=True)
            return reranked[:top_k]

        except Exception as e:
            logger.error(f"Error reranking results: {e}")
            return [(e, s, 0.0) for e, s in results[:top_k]]

    def search(
        self,
        query: str,
        top_k: int = 10,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[SearchResult]:
        """Execute cross-encoder search strategy.

        Args:
            query: Search query
            top_k: Number of results to return
            filters: Optional filters to apply

        Returns:
            Search results
        """
        try:
            # Check cache first
            cached_scores = self.score_cache.get_scores(query, filters)
            if cached_scores:
                # Convert cached scores to results
                results = []
                for score_data in cached_scores.values():
                    endpoint = self.store.get_endpoint(score_data.endpoint_id)
                    if endpoint:
                        results.append(
                            self._convert_to_search_result(
                                endpoint=endpoint,
                                score=score_data.score,
                                strategy="cross_encoder",
                                extra_metadata=score_data.components,
                            )
                        )
                return sorted(results, key=lambda x: x.score, reverse=True)[:top_k]
            
            # Get initial results using base vector search
            initial_results = self.store.search_endpoints(
                query=query,
                top_k=top_k * 2,  # Get more results for reranking
                filters=filters,
            )
            
            # Rerank using cross-encoder
            reranked = self._rerank_results(query, initial_results, top_k=top_k)
            
            # Create score data for caching
            score_data = {}
            for endpoint, combined_score, cross_score in reranked:
                # Get original score from initial results
                original_score = next(
                    score for e, score in initial_results
                    if e.id == endpoint.id
                )
                
                score_data[endpoint.id] = ScoreData(
                    endpoint_id=endpoint.id,
                    score=combined_score,
                    strategy="cross_encoder",
                    components={
                        "cross_encoder_score": cross_score,
                        "original_score": original_score,
                    },
                    timestamp=time.time(),
                )
                
            # Cache scores
            self.score_cache.set_scores(query, score_data, filters)
            
            # Convert to SearchResult objects
            search_results = []
            for endpoint, combined_score, cross_score in reranked:
                # Get original score from initial results
                original_score = next(
                    score for e, score in initial_results
                    if e.id == endpoint.id
                )
                
                search_results.append(
                    self._convert_to_search_result(
                        endpoint=endpoint,
                        score=combined_score,
                        strategy="cross_encoder",
                        extra_metadata={
                            "cross_encoder_score": cross_score,
                            "original_score": original_score,
                        },
                    )
                )
                
            return search_results

        except Exception as e:
            logger.error(f"Cross-encoder search failed: {e}")
            return [] 