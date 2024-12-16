"""Advanced API search with triple vector embeddings and contextual boosting."""

import logging
import traceback
from typing import Any, Dict, List, Optional
from datetime import datetime

from ..config import config_instance
from ..embedding.embeddings import TripleVectorizer
from ..integrations import pinecone_instance
from ..integrations.llm.openrouter_client import OpenRouterClient
from ..utils.cache import DiskCache
from .boosting import ContextualBooster, BusinessContext, BusinessValue
from .expansion import QueryExpander
from .quality import QualityMetrics
from .search_models import SearchResult
from .understanding import ZeroShotUnderstanding
from ..monitoring import metrics_manager

logger = logging.getLogger(__name__)

search_cache = DiskCache[Dict[str, Any]](
    namespace="search",
    ttl=config_instance.cache_ttl,  # 1 hour
)


class BusinessInsight:
    """Business insight for search results."""

    def __init__(
        self,
        title: str,
        description: str,
        impact_score: float,
        recommendations: List[str],
        metrics: Dict[str, float],
        category: str,
    ):
        """Initialize business insight.

        Args:
            title: Insight title
            description: Detailed description
            impact_score: Business impact score (0-1)
            recommendations: List of actionable recommendations
            metrics: Related business metrics
            category: Insight category
        """
        self.title = title
        self.description = description
        self.impact_score = impact_score
        self.recommendations = recommendations
        self.metrics = metrics
        self.category = category
        self.timestamp = datetime.now()


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
        self.metrics = metrics_manager
        self.quality_metrics = QualityMetrics()
        self._init_components()

    def _init_components(self):
        """Initialize search components."""
        from .vectorizer import APIVectorizer
        from .boosting import ContextualBooster
        from .rag import RAGEnhancer
        
        self.vectorizer = APIVectorizer()
        self.booster = ContextualBooster()
        self.rag = RAGEnhancer()

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

    def _generate_business_insights(
        self,
        query: str,
        results: List[Dict],
        context: BusinessContext,
    ) -> List[BusinessInsight]:
        """Generate business insights for search results.

        Args:
            query: Search query
            results: Search results
            context: Business context

        Returns:
            List of business insights
        """
        insights = []

        try:
            # Analyze result patterns
            endpoint_types = self._analyze_endpoint_types(results)
            business_values = self._analyze_business_values(results, context)
            usage_patterns = self._analyze_usage_patterns(results)

            # Generate implementation insights
            if context.integration_complexity > 0.7:
                insights.append(
                    BusinessInsight(
                        title="Complex Integration Detected",
                        description="These endpoints require careful implementation planning.",
                        impact_score=0.8,
                        recommendations=[
                            "Start with a proof of concept",
                            "Plan for incremental implementation",
                            "Set up comprehensive error handling",
                            "Implement thorough testing",
                        ],
                        metrics={
                            "complexity_score": context.integration_complexity,
                            "estimated_effort": business_values["avg_implementation_effort"],
                        },
                        category="implementation",
                    )
                )

            # Generate revenue insights
            high_value_endpoints = [
                r for r in results
                if self.booster.calculate_business_value(
                    r["id"], context
                ).revenue_impact > 0.7
            ]
            if high_value_endpoints:
                insights.append(
                    BusinessInsight(
                        title="High Revenue Potential Identified",
                        description="Several endpoints show significant revenue potential.",
                        impact_score=0.9,
                        recommendations=[
                            "Prioritize implementation of revenue-generating endpoints",
                            "Set up revenue tracking",
                            "Plan for scalability",
                            "Consider premium features",
                        ],
                        metrics={
                            "potential_revenue_impact": business_values["avg_revenue_impact"],
                            "adoption_probability": business_values["avg_adoption_rate"],
                        },
                        category="revenue",
                    )
                )

            # Generate efficiency insights
            if "efficiency" in endpoint_types:
                insights.append(
                    BusinessInsight(
                        title="Optimization Opportunities Found",
                        description="These endpoints can improve operational efficiency.",
                        impact_score=0.7,
                        recommendations=[
                            "Identify current bottlenecks",
                            "Measure baseline performance",
                            "Plan phased optimization",
                            "Set up monitoring",
                        ],
                        metrics={
                            "efficiency_gain": business_values["avg_efficiency_score"],
                            "cost_saving_potential": business_values["avg_cost_savings"],
                        },
                        category="efficiency",
                    )
                )

            # Generate customer experience insights
            if context.query_intent == "customer":
                insights.append(
                    BusinessInsight(
                        title="Customer Experience Enhancement",
                        description="These endpoints can improve customer satisfaction.",
                        impact_score=0.8,
                        recommendations=[
                            "Map customer journey touchpoints",
                            "Plan A/B testing",
                            "Set up user feedback collection",
                            "Monitor customer satisfaction metrics",
                        ],
                        metrics={
                            "satisfaction_impact": business_values["avg_customer_impact"],
                            "engagement_potential": business_values["avg_engagement_score"],
                        },
                        category="customer",
                    )
                )

            # Generate compliance insights
            high_risk_endpoints = [
                r for r in results
                if self.booster.calculate_business_value(
                    r["id"], context
                ).compliance_risk > 0.6
            ]
            if high_risk_endpoints:
                insights.append(
                    BusinessInsight(
                        title="Compliance Considerations",
                        description="Some endpoints require careful compliance handling.",
                        impact_score=0.8,
                        recommendations=[
                            "Review regulatory requirements",
                            "Implement audit logging",
                            "Plan compliance monitoring",
                            "Document compliance measures",
                        ],
                        metrics={
                            "risk_level": business_values["avg_compliance_risk"],
                            "audit_readiness": business_values["avg_audit_score"],
                        },
                        category="compliance",
                    )
                )

            # Generate market trend insights
            if business_values["avg_market_demand"] > 0.7:
                insights.append(
                    BusinessInsight(
                        title="High Market Demand",
                        description="These endpoints align with current market trends.",
                        impact_score=0.7,
                        recommendations=[
                            "Accelerate implementation timeline",
                            "Monitor competitor offerings",
                            "Plan for scalability",
                            "Consider market expansion",
                        ],
                        metrics={
                            "market_demand": business_values["avg_market_demand"],
                            "competitive_advantage": business_values["avg_market_position"],
                        },
                        category="market",
                    )
                )

            # Sort insights by impact score
            insights.sort(key=lambda x: x.impact_score, reverse=True)

        except Exception as e:
            logger.error(f"Failed to generate business insights: {e}")

        return insights

    def _analyze_endpoint_types(self, results: List[Dict]) -> List[str]:
        """Analyze types of endpoints in results."""
        types = set()
        for result in results:
            if "transaction" in result["path"].lower():
                types.add("transaction")
            if "report" in result["path"].lower():
                types.add("analytics")
            if "user" in result["path"].lower():
                types.add("user_management")
            if "optimize" in result["path"].lower():
                types.add("efficiency")
        return list(types)

    def _analyze_business_values(
        self,
        results: List[Dict],
        context: BusinessContext,
    ) -> Dict[str, float]:
        """Analyze business values across results."""
        values = []
        for result in results:
            value = self.booster.calculate_business_value(result["id"], context)
            values.append(value)

        return {
            "avg_revenue_impact": np.mean([v.revenue_impact for v in values]),
            "avg_adoption_rate": np.mean([v.adoption_rate for v in values]),
            "avg_implementation_effort": np.mean([v.implementation_effort for v in values]),
            "avg_customer_impact": np.mean([v.customer_impact for v in values]),
            "avg_strategic_alignment": np.mean([v.strategic_alignment for v in values]),
            "avg_compliance_risk": np.mean([v.compliance_risk for v in values]),
            "avg_market_demand": np.mean([v.market_demand for v in values]),
            "avg_efficiency_score": np.mean([1 - v.implementation_effort for v in values]),
            "avg_cost_savings": np.mean([v.strategic_alignment * (1 - v.implementation_effort) for v in values]),
            "avg_engagement_score": np.mean([v.customer_impact * v.adoption_rate for v in values]),
            "avg_audit_score": np.mean([1 - v.compliance_risk for v in values]),
            "avg_market_position": np.mean([v.market_demand * v.strategic_alignment for v in values]),
        }

    def _analyze_usage_patterns(self, results: List[Dict]) -> Dict[str, Any]:
        """Analyze usage patterns in results."""
        patterns = {
            "popular_endpoints": [],
            "trending_endpoints": [],
            "high_value_endpoints": [],
            "complex_endpoints": [],
        }

        for result in results:
            # Analyze based on usage statistics
            if result.get("usage_count", 0) > 1000:
                patterns["popular_endpoints"].append(result["id"])
            if result.get("usage_trend", 0) > 0.5:
                patterns["trending_endpoints"].append(result["id"])
            if result.get("business_value", 0) > 0.7:
                patterns["high_value_endpoints"].append(result["id"])
            if result.get("complexity_score", 0) > 0.7:
                patterns["complex_endpoints"].append(result["id"])

        return patterns

    def search(
        self,
        query: str,
        filters: Optional[Dict] = None,
        include_metadata: bool = True,
        use_cache: bool = True,
        enhance_results: bool = True,
    ) -> Dict[str, Any]:
        """Perform API search with business insights.

        Args:
            query: Search query
            filters: Optional search filters
            include_metadata: Include additional metadata
            use_cache: Use search cache
            enhance_results: Enhance results with business insights

        Returns:
            Search results with business insights
        """
        try:
            # Get base search results
            results = super().search(
                query=query,
                filters=filters,
                include_metadata=include_metadata,
                use_cache=use_cache,
            )

            if not enhance_results:
                return results

            # Get business context
            context = self.booster._analyze_query_intent(query)

            # Generate business insights
            insights = self._generate_business_insights(
                query=query,
                results=results["results"],
                context=context,
            )

            # Add insights to results
            enhanced_results = {
                **results,
                "business_insights": [
                    {
                        "title": insight.title,
                        "description": insight.description,
                        "impact_score": insight.impact_score,
                        "recommendations": insight.recommendations,
                        "metrics": insight.metrics,
                        "category": insight.category,
                    }
                    for insight in insights
                ],
                "business_context": {
                    "query_intent": context.query_intent,
                    "user_segment": context.user_segment,
                    "use_case": context.use_case,
                    "industry_vertical": context.industry_vertical,
                    "integration_complexity": context.integration_complexity,
                    "time_sensitivity": context.time_sensitivity,
                },
            }

            return enhanced_results

        except Exception as e:
            logger.error(f"Search failed: {e}")
            return {
                "query": query,
                "results": [],
                "error": str(e),
            }
