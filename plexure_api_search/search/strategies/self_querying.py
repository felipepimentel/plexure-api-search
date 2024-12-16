"""Self-querying search strategy with query understanding."""

import json
import logging
from typing import Any, Dict, List, Optional, Tuple

from ...integrations.llm.openrouter_client import OpenRouterClient
from .base import BaseSearchStrategy, SearchConfig, SearchResult, StrategyFactory

logger = logging.getLogger(__name__)


@StrategyFactory.register("self_querying")
class SelfQueryingStrategy(BaseSearchStrategy):
    """Self-querying strategy with query understanding."""

    def __init__(self, config: Optional[SearchConfig] = None):
        """Initialize self-querying strategy.

        Args:
            config: Optional strategy configuration
        """
        super().__init__(config)
        self.llm = OpenRouterClient()

    def _analyze_query(self, query: str) -> Dict[str, Any]:
        """Analyze query to understand intent and constraints.

        Args:
            query: Search query

        Returns:
            Query analysis with intent and constraints
        """
        prompt = f"""Analyze this API search query: "{query}"

Break it down into these components in JSON format:
{{
    "intent": "primary search intent",
    "http_method": "expected HTTP method if specified",
    "resource_type": "type of API resource",
    "constraints": {{
        "required_params": ["list of required parameters"],
        "optional_params": ["list of optional parameters"],
        "response_format": "expected response format",
        "auth_required": boolean,
        "version_constraints": "any version requirements"
    }},
    "use_case": "intended use case",
    "importance": {{
        "method": 0-1 score,
        "path": 0-1 score,
        "params": 0-1 score,
        "response": 0-1 score,
        "auth": 0-1 score
    }}
}}"""

        try:
            response = self.llm.call(
                prompt=prompt,
                temperature=0.3,
                cache_key=f"analyze_{query}",
            )

            if "error" in response:
                logger.error(f"Query analysis failed: {response['error']}")
                return {}

            content = response["choices"][0]["message"]["content"]
            return json.loads(content)

        except Exception as e:
            logger.error(f"Error analyzing query: {e}")
            return {}

    def _generate_structured_query(
        self,
        analysis: Dict[str, Any],
    ) -> Tuple[str, Dict[str, float]]:
        """Generate structured query from analysis.

        Args:
            analysis: Query analysis

        Returns:
            Tuple of (structured query, importance weights)
        """
        try:
            # Extract components
            intent = analysis.get("intent", "")
            method = analysis.get("http_method", "")
            resource = analysis.get("resource_type", "")
            constraints = analysis.get("constraints", {})
            use_case = analysis.get("use_case", "")
            
            # Build structured query
            query_parts = []
            
            if method:
                query_parts.append(f"HTTP {method}")
            if resource:
                query_parts.append(resource)
            if intent:
                query_parts.append(intent)
                
            # Add constraints
            required_params = constraints.get("required_params", [])
            if required_params:
                query_parts.append(f"requires {', '.join(required_params)}")
                
            response_format = constraints.get("response_format")
            if response_format:
                query_parts.append(f"returns {response_format}")
                
            if constraints.get("auth_required"):
                query_parts.append("requires authentication")
                
            if use_case:
                query_parts.append(f"for {use_case}")
                
            # Get importance weights
            importance = analysis.get("importance", {})
            weights = {
                "method": importance.get("method", 0.5),
                "path": importance.get("path", 0.5),
                "params": importance.get("params", 0.5),
                "response": importance.get("response", 0.5),
                "auth": importance.get("auth", 0.5),
            }
            
            return " ".join(query_parts), weights

        except Exception as e:
            logger.error(f"Error generating structured query: {e}")
            return "", {}

    def _apply_weights(
        self,
        results: List[Dict[str, Any]],
        weights: Dict[str, float],
        analysis: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        """Apply importance weights to results.

        Args:
            results: Search results
            weights: Importance weights
            analysis: Query analysis

        Returns:
            Weighted results
        """
        try:
            weighted_results = []
            
            for result in results:
                score = result["score"]
                method = result["method"]
                path = result["path"]
                
                # Apply method weight if specified
                if analysis.get("http_method"):
                    if method.upper() == analysis["http_method"].upper():
                        score *= (1 + weights["method"])
                    else:
                        score *= (1 - weights["method"])
                        
                # Apply path weight based on resource match
                if analysis.get("resource_type"):
                    if analysis["resource_type"].lower() in path.lower():
                        score *= (1 + weights["path"])
                        
                # Apply parameter weight
                required_params = analysis.get("constraints", {}).get("required_params", [])
                if required_params:
                    param_match = any(
                        p.lower() in str(result.get("parameters", [])).lower()
                        for p in required_params
                    )
                    if param_match:
                        score *= (1 + weights["params"])
                        
                # Apply response format weight
                expected_format = analysis.get("constraints", {}).get("response_format")
                if expected_format:
                    format_match = expected_format.lower() in str(result.get("responses", [])).lower()
                    if format_match:
                        score *= (1 + weights["response"])
                        
                # Apply auth weight
                auth_required = analysis.get("constraints", {}).get("auth_required")
                if auth_required is not None:
                    if auth_required == result.get("requires_auth", False):
                        score *= (1 + weights["auth"])
                        
                result["score"] = score
                weighted_results.append(result)
                
            return weighted_results

        except Exception as e:
            logger.error(f"Error applying weights: {e}")
            return results

    def search(
        self,
        query: str,
        top_k: int = 10,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[SearchResult]:
        """Execute self-querying search strategy.

        Args:
            query: Search query
            top_k: Number of results to return
            filters: Optional filters to apply

        Returns:
            Search results
        """
        try:
            # Analyze query
            analysis = self._analyze_query(query)
            if not analysis:
                return []
                
            # Generate structured query
            structured_query, weights = self._generate_structured_query(analysis)
            if not structured_query:
                return []
                
            # Perform base search
            results = []  # TODO: Implement base vector search
            
            # Apply importance weights
            weighted_results = self._apply_weights(results, weights, analysis)
            
            # Convert to SearchResult objects
            search_results = []
            for result in weighted_results:
                search_results.append(
                    SearchResult(
                        id=result["id"],
                        score=float(result["score"]),
                        method=result["method"],
                        path=result["path"],
                        description=result["description"],
                        api_name=result["api_name"],
                        api_version=result["api_version"],
                        metadata={
                            **result.get("metadata", {}),
                            "analysis": analysis,
                            "structured_query": structured_query,
                        },
                        strategy="self_querying",
                    )
                )
                
            return search_results[:top_k]

        except Exception as e:
            logger.error(f"Self-querying search failed: {e}")
            return [] 