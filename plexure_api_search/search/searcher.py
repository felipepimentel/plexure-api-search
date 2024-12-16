"""Advanced API search with triple vector embeddings and contextual boosting."""

import logging
import traceback
from typing import Any, Dict, List, Optional

from ..config import config_instance
from ..embedding.embeddings import TripleVectorizer
from ..integrations import pinecone_instance
from ..integrations.llm.openrouter_client import OpenRouterClient
from ..utils.cache import DiskCache
from .boosting import ContextualBooster
from .expansion import QueryExpander
from .quality import QualityMetrics
from .search_models import SearchResult
from .understanding import ZeroShotUnderstanding

logger = logging.getLogger(__name__)

search_cache = DiskCache[Dict[str, Any]](
    namespace="search",
    ttl=config_instance.cache_ttl,  # 1 hour
)


class APISearcher:
    """Advanced API search engine with multiple strategies."""

    def __init__(self, top_k: int = 10, use_cache: bool = True):
        """Initialize searcher.

        Args:
            top_k: Number of results to return (default: 10)
            use_cache: Whether to use caching (default: True)
        """
        self.client = pinecone_instance
        self.vectorizer = TripleVectorizer()
        self.booster = ContextualBooster()
        self.understanding = ZeroShotUnderstanding()
        self.expander = QueryExpander()
        self.metrics = QualityMetrics()
        self.llm = OpenRouterClient(use_cache=use_cache)
        self.top_k = top_k
        self.use_cache = use_cache

    def search(
        self,
        query: str,
        filters: Optional[Dict] = None,
        include_metadata: bool = True,
        use_cache: bool = True,
        enhance_results: bool = True,
    ) -> Dict[str, Any]:
        """Enhanced search with caching, reranking, and LLM enhancement.

        Args:
            query: Search query string
            filters: Optional filters to apply
            include_metadata: Whether to include metadata
            use_cache: Whether to use result caching
            enhance_results: Whether to enhance results with LLM analysis

        Returns:
            Search results with optional LLM enhancement
        """
        try:
            # Check cache first
            if use_cache:
                cached_results = search_cache.get(query)
                if cached_results:
                    logger.debug("Using cached results")
                    return cached_results

            logger.debug("Performing vector search")
            # Vector search
            query_vector = self.vectorizer.vectorize_query(query)
            results = self.client.search_vectors(
                query_vector=query_vector,
                top_k=self.top_k * 2,  # Increase for reranking
                filters=filters,
                include_metadata=include_metadata,
            )

            logger.debug("Processing search results")
            # Process results
            processed = self._process_results(results)
            if not processed:
                logger.warning("No results found")
                return {
                    "query": query,
                    "results": [],
                    "related_queries": [],
                }

            logger.debug("Reranking results")
            # Rerank results
            reranked = self._rerank_results(query, processed)

            # Convert reranked results to dict for JSON serialization
            reranked_dicts = [result.to_dict() for result in reranked]
            logger.debug(f"Found {len(reranked_dicts)} results")

            # Initialize final results
            final_results = {
                "query": query,
                "results": reranked_dicts,
                "related_queries": [],
            }

            # Get related queries
            try:
                logger.debug("Getting related queries")
                related_queries = self.llm.suggest_related_queries(query)
                if related_queries and isinstance(related_queries, list):
                    # Ensure each query has required fields
                    validated_queries = []
                    for q in related_queries:
                        if isinstance(q, dict) and all(k in q for k in ["query", "category", "description", "score"]):
                            validated_queries.append(q)
                    if validated_queries:
                        final_results["related_queries"] = validated_queries
                        logger.debug(f"Found {len(validated_queries)} related queries")
                    else:
                        logger.debug("No valid related queries found")
                else:
                    logger.debug("No related queries found")
            except Exception as e:
                logger.error(f"Failed to get related queries: {e}")
                logger.error(traceback.format_exc())

            # Enhance results with LLM if requested
            if enhance_results and reranked_dicts:
                try:
                    logger.debug("Enhancing results with LLM")
                    enhanced = self.llm.enhance_search_results(query, reranked_dicts)
                    if "analysis" in enhanced:
                        final_results["analysis"] = enhanced["analysis"]
                        logger.info(f"Enhanced search results for query: {query}")
                    else:
                        logger.warning("No analysis returned from LLM")
                except Exception as e:
                    logger.error(f"Failed to enhance search results: {e}")
                    logger.error(traceback.format_exc())

            # Cache results
            if use_cache:
                logger.debug("Caching results")
                search_cache.set(query, final_results)

            return final_results

        except Exception as e:
            logger.error(f"Search failed: {e}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            # Return empty results on error
            return {
                "query": query,
                "results": [],
                "related_queries": [],
            }

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
                    # Create SearchResult from Pinecone match
                    result = SearchResult.from_pinecone_match(match, rank=i + 1)
                    processed_results.append(result)
                except Exception as e:
                    logger.error(f"Error processing match: {e}")
                    logger.error(f"Match data: {match}")
                    continue

        except Exception as e:
            logger.error(f"Error processing results: {e}")
            logger.error(f"Results data: {results}")

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
        pairs = []
        for result in results:
            pairs.append((query, result.description))

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
