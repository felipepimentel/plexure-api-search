"""Sparse-Dense hybrid search strategy."""

import logging
import time
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
from rank_bm25 import BM25Okapi

from ...embedding.embeddings import ModernEmbeddings
from ..cache import ScoreData, SemanticScoreCache
from ..storage import EndpointData
from .base import BaseSearchStrategy, SearchConfig, SearchResult, StrategyFactory

logger = logging.getLogger(__name__)


@StrategyFactory.register("sparse_dense")
class SparseDenseStrategy(BaseSearchStrategy):
    """Sparse-Dense hybrid search strategy."""

    def __init__(self, config: Optional[SearchConfig] = None):
        """Initialize sparse-dense strategy.

        Args:
            config: Optional strategy configuration
        """
        super().__init__(config)
        self.bm25_index = None
        self.corpus = []
        self.doc_mapping = {}
        self.score_cache = SemanticScoreCache("sparse_dense")

    def _tokenize(self, text: str) -> List[str]:
        """Tokenize text for BM25.

        Args:
            text: Input text

        Returns:
            List of tokens
        """
        # Simple tokenization - can be enhanced
        return text.lower().split()

    def _prepare_document(self, endpoint: EndpointData) -> str:
        """Prepare endpoint document for BM25.

        Args:
            endpoint: Endpoint data

        Returns:
            Formatted document text
        """
        try:
            parts = []
            
            # Add method and path
            parts.extend([
                endpoint.method,
                endpoint.path,
            ])
            
            # Add description
            if endpoint.description:
                parts.append(endpoint.description)
                
            # Add parameters
            if endpoint.parameters:
                parts.extend(str(p) for p in endpoint.parameters)
                
            # Add responses
            if endpoint.responses:
                parts.extend(str(r) for r in endpoint.responses)
                
            return " ".join(str(p) for p in parts if p)

        except Exception as e:
            logger.error(f"Error preparing document: {e}")
            return ""

    def _index_endpoints(self, endpoints: List[EndpointData]) -> None:
        """Index endpoints for BM25.

        Args:
            endpoints: List of endpoints to index
        """
        try:
            # Prepare corpus and mapping
            self.corpus = []
            self.doc_mapping = {}
            
            for i, endpoint in enumerate(endpoints):
                doc = self._prepare_document(endpoint)
                if doc:
                    tokenized = self._tokenize(doc)
                    self.corpus.append(tokenized)
                    self.doc_mapping[i] = endpoint
                    
            # Create BM25 index
            if self.corpus:
                self.bm25_index = BM25Okapi(self.corpus)
                logger.info(f"Indexed {len(endpoints)} endpoints for BM25")
            else:
                logger.warning("No endpoints indexed for BM25")

        except Exception as e:
            logger.error(f"Error indexing endpoints: {e}")
            self.bm25_index = None
            self.corpus = []
            self.doc_mapping = {}

    def _get_sparse_scores(
        self,
        query: str,
        endpoints: List[EndpointData],
        top_k: int = 100,
    ) -> Dict[str, float]:
        """Get BM25 scores for query.

        Args:
            query: Search query
            endpoints: List of endpoints to score
            top_k: Number of results to return

        Returns:
            Dictionary of endpoint ID to BM25 score
        """
        try:
            # Index endpoints if needed
            if not self.bm25_index:
                self._index_endpoints(endpoints)
                
            if not self.bm25_index:
                return {}
                
            # Get BM25 scores
            tokenized_query = self._tokenize(query)
            scores = self.bm25_index.get_scores(tokenized_query)
            
            # Get top k results
            top_indices = np.argsort(scores)[-top_k:][::-1]
            
            # Create score mapping
            results = {}
            for idx in top_indices:
                endpoint = self.doc_mapping[idx]
                score = float(scores[idx])
                if score > 0:
                    results[endpoint.id] = score
                    
            return results

        except Exception as e:
            logger.error(f"Error getting sparse scores: {e}")
            return {}

    def _get_dense_scores(
        self,
        query: str,
        endpoints: List[EndpointData],
        top_k: int = 100,
    ) -> Dict[str, float]:
        """Get dense embedding scores for query.

        Args:
            query: Search query
            endpoints: List of endpoints to score
            top_k: Number of results to return

        Returns:
            Dictionary of endpoint ID to similarity score
        """
        try:
            if not endpoints:
                return {}
                
            # Get query embedding
            query_vector = self.embeddings.get_embeddings(query)
            
            # Get endpoint embeddings
            endpoint_texts = [
                self._prepare_document(endpoint)
                for endpoint in endpoints
            ]
            endpoint_vectors = self.embeddings.get_embeddings(endpoint_texts)
            
            # Calculate similarities
            scores = {}
            for endpoint, vector in zip(endpoints, endpoint_vectors):
                similarity = np.dot(query_vector, vector) / (
                    np.linalg.norm(query_vector) * np.linalg.norm(vector)
                )
                if similarity > 0:
                    scores[endpoint.id] = float(similarity)
                    
            # Sort and limit
            sorted_scores = dict(
                sorted(
                    scores.items(),
                    key=lambda x: x[1],
                    reverse=True,
                )[:top_k]
            )
            
            return sorted_scores

        except Exception as e:
            logger.error(f"Error getting dense scores: {e}")
            return {}

    def _combine_scores(
        self,
        sparse_scores: Dict[str, float],
        dense_scores: Dict[str, float],
        sparse_weight: float = 0.3,
        dense_weight: float = 0.7,
    ) -> Dict[str, Dict[str, float]]:
        """Combine sparse and dense scores.

        Args:
            sparse_scores: BM25 scores by endpoint ID
            dense_scores: Dense scores by endpoint ID
            sparse_weight: Weight for sparse scores
            dense_weight: Weight for dense scores

        Returns:
            Combined scores with components
        """
        try:
            # Normalize scores
            if sparse_scores:
                max_sparse = max(sparse_scores.values())
                sparse_scores = {
                    k: v / max_sparse
                    for k, v in sparse_scores.items()
                }
                
            if dense_scores:
                max_dense = max(dense_scores.values())
                dense_scores = {
                    k: v / max_dense
                    for k, v in dense_scores.items()
                }
                
            # Combine scores
            combined = {}
            all_ids = set(sparse_scores) | set(dense_scores)
            
            for endpoint_id in all_ids:
                sparse_score = sparse_scores.get(endpoint_id, 0.0)
                dense_score = dense_scores.get(endpoint_id, 0.0)
                
                # Calculate combined score
                combined_score = (
                    sparse_score * sparse_weight +
                    dense_score * dense_weight
                )
                
                combined[endpoint_id] = {
                    "score": combined_score,
                    "sparse_score": sparse_score,
                    "dense_score": dense_score,
                }
                
            return combined

        except Exception as e:
            logger.error(f"Error combining scores: {e}")
            return {}

    def search(
        self,
        query: str,
        top_k: int = 10,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[SearchResult]:
        """Execute sparse-dense hybrid search.

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
                                strategy="sparse_dense",
                                extra_metadata=score_data.components,
                            )
                        )
                return sorted(results, key=lambda x: x.score, reverse=True)[:top_k]
            
            # Get all endpoints
            endpoints = []
            for endpoint_id in self.store.endpoints:
                endpoint = self.store.get_endpoint(endpoint_id)
                if endpoint:
                    # Apply filters if provided
                    if filters and not self.store._apply_filters(endpoint, filters):
                        continue
                    endpoints.append(endpoint)
            
            if not endpoints:
                return []
                
            # Get sparse (BM25) scores
            sparse_scores = self._get_sparse_scores(
                query,
                endpoints,
                top_k=top_k * 2,
            )
            
            # Get dense (embedding) scores
            dense_scores = self._get_dense_scores(
                query,
                endpoints,
                top_k=top_k * 2,
            )
            
            # Combine scores
            combined_scores = self._combine_scores(
                sparse_scores,
                dense_scores,
                sparse_weight=self.config.bm25_weight,
                dense_weight=self.config.vector_weight,
            )
            
            # Sort by combined score
            sorted_results = sorted(
                combined_scores.items(),
                key=lambda x: x[1]["score"],
                reverse=True,
            )
            
            # Create score data for caching
            score_data = {}
            for endpoint_id, scores in sorted_results:
                score_data[endpoint_id] = ScoreData(
                    endpoint_id=endpoint_id,
                    score=scores["score"],
                    strategy="sparse_dense",
                    components={
                        "sparse_score": scores["sparse_score"],
                        "dense_score": scores["dense_score"],
                    },
                    timestamp=time.time(),
                )
                
            # Cache scores
            self.score_cache.set_scores(query, score_data, filters)
            
            # Convert to SearchResult objects
            search_results = []
            for endpoint_id, scores in sorted_results[:top_k]:
                # Get endpoint data
                endpoint = self.store.get_endpoint(endpoint_id)
                if not endpoint:
                    continue
                    
                search_results.append(
                    self._convert_to_search_result(
                        endpoint=endpoint,
                        score=scores["score"],
                        strategy="sparse_dense",
                        extra_metadata={
                            "sparse_score": scores["sparse_score"],
                            "dense_score": scores["dense_score"],
                        },
                    )
                )
                
            return search_results

        except Exception as e:
            logger.error(f"Failed to execute sparse-dense search: {e}")
            return [] 