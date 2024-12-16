"""Query enhancement using LLM for better search results."""

import json
import logging
from typing import Any, Dict, List, Optional

from ..integrations.llm.base import LLMProvider
from ..utils.json_utils import clean_json_string

logger = logging.getLogger(__name__)

# Business-focused suggestion categories
BUSINESS_CATEGORIES = {
    "quick_start": {
        "priority": 0.9,
        "description": "Essential endpoints for getting started",
        "value_prop": "Fastest path to implementation"
    },
    "revenue_generating": {
        "priority": 0.8,
        "description": "Endpoints that directly drive revenue",
        "value_prop": "Direct business value generation"
    },
    "cost_saving": {
        "priority": 0.7,
        "description": "Endpoints that optimize operations",
        "value_prop": "Operational cost reduction"
    },
    "customer_experience": {
        "priority": 0.8,
        "description": "Endpoints that enhance user experience",
        "value_prop": "Improved customer satisfaction"
    },
    "integration": {
        "priority": 0.6,
        "description": "Endpoints for system integration",
        "value_prop": "Seamless system connectivity"
    },
    "analytics": {
        "priority": 0.7,
        "description": "Endpoints for business insights",
        "value_prop": "Data-driven decision making"
    },
    "compliance": {
        "priority": 0.8,
        "description": "Endpoints for regulatory compliance",
        "value_prop": "Risk mitigation and compliance"
    },
    "optimization": {
        "priority": 0.6,
        "description": "Endpoints for performance optimization",
        "value_prop": "Enhanced system efficiency"
    }
}

class QueryEnhancer:
    """Handles query enhancement and suggestions using LLM."""

    def __init__(self, llm_provider: LLMProvider, use_cache: bool = True):
        """Initialize enhancer.

        Args:
            llm_provider: LLM provider instance
            use_cache: Whether to use caching
        """
        self.llm_provider = llm_provider
        self.use_cache = use_cache

    def enhance_search_results(
        self, query: str, results: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Enhance search results with LLM analysis.

        Args:
            query: Original search query
            results: List of search results

        Returns:
            Enhanced results with LLM analysis
        """
        try:
            # Create cache key
            cache_key = None
            if self.use_cache:
                cache_key = f"enhance_{query}_{len(results)}"

            # Build enhancement prompt
            prompt = f"""Analyze these API search results for the query: "{query}"

Results:
{json.dumps(results, indent=2)}

Please provide:
1. A summary of the most relevant endpoints
2. How these endpoints might be used together
3. Suggested query refinements
4. Code examples showing endpoint usage
5. Alternative approaches using these endpoints

Format the response as JSON with these keys:
- result_summary
- integration_patterns
- query_suggestions
- code_examples
- alternative_approaches"""

            # Get LLM response
            response = self.llm_provider.call(
                prompt=prompt,
                cache_key=cache_key,
                temperature=0.5,  # More focused for search enhancement
            )

            # Extract and parse content
            if "error" not in response:
                try:
                    content = response["choices"][0]["message"]["content"]
                    # Clean and parse JSON
                    cleaned_content = clean_json_string(content)
                    enhanced_data = json.loads(cleaned_content)
                    return {"results": results, "analysis": enhanced_data}
                except (json.JSONDecodeError, KeyError) as e:
                    logger.error(f"Failed to parse LLM response: {e}")
                    logger.error(f"Raw content: {content}")

            return {"results": results}

        except Exception as e:
            logger.error(f"Search enhancement failed: {e}")
            return {"results": results}

    def suggest_related_queries(self, query: str) -> List[Dict[str, Any]]:
        """Generate business-focused query suggestions.

        Args:
            query: Original search query

        Returns:
            List of query suggestions with business context
        """
        try:
            # Check cache
            cache_key = f"business_suggestions:{query}"
            cached = self.cache.get(cache_key)
            if cached:
                return cached

            # Get LLM suggestions
            prompt = self._create_business_prompt(query)
            response = self.llm.call(
                prompt=prompt,
                temperature=0.3,
                max_tokens=1000
            )

            if "error" in response:
                logger.error(f"Failed to get suggestions: {response['error']}")
                return []

            content = response["choices"][0]["message"]["content"]
            
            try:
                suggestions = eval(content)  # Convert string to list of dicts
            except:
                logger.error(f"Failed to parse suggestions: {content}")
                return []

            # Validate and enhance suggestions
            valid_suggestions = []
            for i, suggestion in enumerate(suggestions):
                try:
                    # Basic validation
                    required_fields = {"query", "category", "description", "score"}
                    if not all(field in suggestion for field in required_fields):
                        continue

                    # Validate category
                    category = suggestion["category"]
                    if category not in BUSINESS_CATEGORIES:
                        continue

                    # Enhance with business metadata
                    enhanced = {
                        "query": suggestion["query"],
                        "category": category,
                        "description": suggestion["description"],
                        "score": float(suggestion["score"]),
                        "value_proposition": BUSINESS_CATEGORIES[category]["value_prop"],
                        "business_priority": BUSINESS_CATEGORIES[category]["priority"],
                        "implementation_complexity": self._estimate_complexity(suggestion["query"]),
                        "estimated_time_to_value": self._estimate_time_to_value(category),
                        "suggested_next_steps": self._get_next_steps(category),
                    }

                    valid_suggestions.append(enhanced)

                except Exception as e:
                    logger.error(f"Error processing suggestion {i}: {e}")
                    continue

            # Sort by business priority and score
            valid_suggestions.sort(
                key=lambda x: (
                    x["business_priority"],
                    x["score"]
                ),
                reverse=True
            )

            # Cache results
            if valid_suggestions:
                self.cache.set(cache_key, valid_suggestions)

            return valid_suggestions

        except Exception as e:
            logger.error(f"Failed to generate suggestions: {e}")
            return []

    def _create_business_prompt(self, query: str) -> str:
        """Create business-focused prompt for query enhancement.

        Args:
            query: Original search query

        Returns:
            Business-focused prompt
        """
        return f"""Analyze this API search query from a business perspective: "{query}"

Consider these aspects:
1. What business goals might the user be trying to achieve?
2. Which API endpoints would provide the most business value?
3. What related functionality could increase adoption or revenue?
4. How can we help the user implement revenue-generating features?

For each suggestion, provide:
- A clear business-focused query
- A relevant category from: {list(BUSINESS_CATEGORIES.keys())}
- A description of the business value
- A confidence score (0-1)

Format: List of JSON objects with fields:
- query: string
- category: string
- description: string
- score: float
- value_proposition: string
"""

    def _estimate_complexity(self, query: str) -> str:
        """Estimate implementation complexity."""
        # Simple heuristic based on query complexity
        words = query.lower().split()
        if any(w in words for w in ["simple", "basic", "get"]):
            return "Low"
        elif any(w in words for w in ["integrate", "workflow", "process"]):
            return "Medium"
        else:
            return "High"

    def _estimate_time_to_value(self, category: str) -> str:
        """Estimate time to realize business value."""
        estimates = {
            "quick_start": "1-2 days",
            "revenue_generating": "1-2 weeks",
            "cost_saving": "2-4 weeks",
            "customer_experience": "1-3 weeks",
            "integration": "2-4 weeks",
            "analytics": "1-2 weeks",
            "compliance": "2-4 weeks",
            "optimization": "1-3 weeks"
        }
        return estimates.get(category, "2-4 weeks")

    def _get_next_steps(self, category: str) -> List[str]:
        """Get suggested next steps based on category."""
        next_steps = {
            "quick_start": [
                "Review API documentation",
                "Test endpoint with sample data",
                "Implement basic error handling"
            ],
            "revenue_generating": [
                "Analyze revenue potential",
                "Plan integration with payment system",
                "Set up monitoring for transactions"
            ],
            "cost_saving": [
                "Identify current inefficiencies",
                "Calculate potential savings",
                "Plan gradual rollout"
            ],
            "customer_experience": [
                "Define success metrics",
                "Plan A/B testing",
                "Set up user feedback collection"
            ],
            "integration": [
                "Review system requirements",
                "Plan data synchronization",
                "Set up error handling"
            ],
            "analytics": [
                "Define key metrics",
                "Plan data collection",
                "Set up dashboards"
            ],
            "compliance": [
                "Review regulatory requirements",
                "Plan audit trail",
                "Set up compliance monitoring"
            ],
            "optimization": [
                "Identify performance bottlenecks",
                "Plan optimization strategy",
                "Set up performance monitoring"
            ]
        }
        return next_steps.get(category, []) 