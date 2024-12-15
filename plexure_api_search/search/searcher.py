"""Advanced API search with triple vector embeddings and contextual boosting."""

import logging
from typing import Any, Dict, List, Optional, Union

from ..embedding.embeddings import TripleVectorizer
from ..storage import pinecone_instance
from ..utils.cache import DiskCache
from ..config import config_instance
from .boosting import ContextualBooster
from .expansion import QueryExpander
from .quality import QualityMetrics
from .search_models import SearchResult
from .understanding import ZeroShotUnderstanding

logger = logging.getLogger(__name__)

search_cache = DiskCache[List[Dict[str, Union[str, float]]]](
    namespace="search",
    ttl=config_instance.cache_ttl,  # 1 hour
)


class APISearcher:
    """Advanced API search engine with multiple strategies."""

    def __init__(self, top_k: int = 10):
        """Initialize searcher.

        Args:
            top_k: Number of results to return (default: 10)
        """
        self.client = pinecone_instance
        self.vectorizer = TripleVectorizer()
        self.booster = ContextualBooster()
        self.understanding = ZeroShotUnderstanding()
        self.expander = QueryExpander()
        self.metrics = QualityMetrics()
        self.top_k = top_k

    def search(
        self,
        query: str,
        filters: Optional[Dict] = None,
        include_metadata: bool = True,
        use_cache: bool = True,
    ) -> List[SearchResult]:
        """Enhanced search with caching and reranking.

        Args:
            query: Search query string
            filters: Optional filters to apply
            include_metadata: Whether to include metadata
            use_cache: Whether to use result caching

        Returns:
            List of search results
        """
        try:
            # Check cache first
            if use_cache:
                cached_results = search_cache.get(query)
                if cached_results:
                    return [SearchResult.from_dict(r) for r in cached_results]

            # Vector search
            query_vector = self.vectorizer.vectorize_query(query)
            results = self.client.search_vectors(
                query_vector=query_vector,
                top_k=self.top_k * 2,  # Increase for reranking
                filters=filters,
                include_metadata=include_metadata,
            )

            # Process results
            processed = self._process_results(results)

            # Rerank results
            reranked = self._rerank_results(query, processed)

            # Cache results
            if use_cache:
                search_cache.set(query, [r.to_dict() for r in reranked])

            return reranked

        except Exception as e:
            logger.error(f"Search failed: {e}")
            raise

    def _process_results(self, results: Dict[str, Any]) -> List[SearchResult]:
        """Process search results from Pinecone.

        Args:
            results: Raw results from Pinecone search

        Returns:
            List of processed SearchResult objects
        """
        processed_results = []

        try:
            matches = results.get("matches", [])
            for i, match in enumerate(matches):
                try:
                    result = SearchResult.from_pinecone_match(match, rank=i + 1)
                    processed_results.append(result)
                except Exception as e:
                    logger.error(f"Error processing match: {e}")
                    continue

        except Exception as e:
            logger.error(f"Error processing results: {e}")

        return processed_results

    def _rerank_results(
        self, query: str, results: List[SearchResult]
    ) -> List[SearchResult]:
        """Rerank results using semantic similarity and boosting.

        Args:
            query: Original search query
            results: List of results to rerank

        Returns:
            Reranked results
        """
        if not results:
            return results

        # Create query-result pairs for reranking
        pairs = [(query, result.description) for result in results]

        # Get reranking scores
        scores = self.vectorizer.cross_encoder.predict(pairs)

        # Apply boosting
        boosted_scores = self.booster.adjust_scores(query, scores)

        # Sort results by new scores
        reranked = sorted(
            zip(results, boosted_scores), key=lambda x: x[1], reverse=True
        )

        # Update scores and ranks
        final_results = []
        for i, (result, score) in enumerate(reranked[: self.top_k]):
            result.score = score
            result.rank = i + 1
            final_results.append(result)

        return final_results

    def update_feedback(
        self, query: str, endpoint_id: str, is_relevant: bool, score: float = 1.0
    ) -> None:
        """Update feedback for search results.

        Args:
            query: Original search query
            endpoint_id: ID of the endpoint
            is_relevant: Whether the result was relevant
            score: Feedback score (0 to 1)
        """
        try:
            # Update contextual booster
            self.booster.update_feedback(query, score if is_relevant else 0.0)

            # Clear cache for this query to force re-ranking with new feedback
            search_cache.delete(query)

        except Exception as e:
            raise RuntimeError(f"Failed to update feedback: {str(e)}")

    def get_quality_metrics(self) -> Dict[str, float]:
        """Get current quality metrics.

        Returns:
            Dictionary of quality metrics
        """
        try:
            return self.metrics.get_average_metrics()
        except Exception as e:
            raise RuntimeError(f"Failed to get quality metrics: {str(e)}")

    def get_metric_trends(self) -> Dict[str, List[float]]:
        """Get metric trends over time.

        Returns:
            Dictionary of metric trends
        """
        try:
            return self.metrics.get_metric_trends()
        except Exception as e:
            raise RuntimeError(f"Failed to get metric trends: {str(e)}")

    def analyze_query(self, query: str) -> Dict[str, Any]:
        """Analyze a search query.

        Args:
            query: Search query string

        Returns:
            Dictionary with query analysis
        """
        try:
            # Expand query
            expanded = self.expander.expand_query(query)

            # Get contextual weights
            weights = self.booster.adjust_weights(query)

            return {
                "original_query": query,
                "semantic_variants": expanded.semantic_variants,
                "technical_mappings": expanded.technical_mappings,
                "use_cases": expanded.use_cases,
                "weights": expanded.weights,
                "contextual_weights": weights.to_dict(),
            }
        except Exception as e:
            raise RuntimeError(f"Query analysis failed: {str(e)}")
