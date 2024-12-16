"""Query enhancement using LLM for better search results."""

import json
import logging
from typing import Any, Dict, List, Optional

from ..integrations.llm.base import LLMProvider
from ..utils.json_utils import clean_json_string

logger = logging.getLogger(__name__)


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
        """Generate related search queries.

        Args:
            query: Original search query

        Returns:
            List of related queries with metadata
        """
        try:
            # Create cache key
            cache_key = None
            if self.use_cache:
                cache_key = f"suggest_{query}"

            # Build suggestion prompt with strict JSON format requirement
            system_prompt = (
                "You are a JSON formatting assistant. You ONLY output valid JSON arrays. "
                "Never output numbered lists, text, or explanations. "
                "Only respond with the exact JSON array requested."
            )

            user_prompt = (
                "{"
                f'"task": "Generate 5 related queries for: {query}",'
                '"format": "json_array",'
                '"required_fields": ["query", "category", "description", "score"],'
                '"valid_categories": ["alternative", "related", "use_case", "implementation", "error"],'
                '"example": [{"query": "example", "category": "alternative", "description": "why", "score": 0.9}]'
                "}"
            )

            # Get LLM response
            response = self.llm_provider.call(
                prompt=user_prompt,
                system_prompt=system_prompt,
                cache_key=cache_key,
                temperature=0.7,
                max_tokens=512,
            )

            # Extract suggestions
            if "error" not in response and "choices" in response:
                try:
                    content = response["choices"][0]["message"]["content"].strip()
                    logger.debug(f"Raw suggestion content: {content}")

                    # Clean JSON content and ensure it starts with [
                    content = clean_json_string(content)
                    if not content.startswith("["):
                        logger.error("Content does not start with [")
                        raise ValueError("Invalid JSON array format")
                    logger.debug(f"Cleaned suggestion content: {content}")

                    # Parse JSON
                    suggestions = json.loads(content)
                    logger.debug(f"Parsed suggestions: {json.dumps(suggestions, indent=2)}")

                    # Validate suggestions
                    if not isinstance(suggestions, list):
                        logger.error("Response was not a JSON array")
                        raise ValueError("Response must be a JSON array")

                    valid_suggestions = []
                    required_fields = {"query", "category", "description", "score"}
                    valid_categories = {
                        "alternative",
                        "related",
                        "use_case",
                        "implementation",
                        "error",
                    }

                    for i, suggestion in enumerate(suggestions[:5]):
                        try:
                            # Validate all required fields are present
                            if not isinstance(suggestion, dict):
                                logger.error(f"Suggestion {i} is not a dictionary: {suggestion}")
                                continue

                            if not all(field in suggestion for field in required_fields):
                                logger.error(f"Suggestion {i} missing required fields: {suggestion}")
                                continue

                            # Validate field types and values
                            if not isinstance(suggestion["query"], str):
                                logger.error(f"Suggestion {i} query is not a string: {suggestion['query']}")
                                continue

                            if not isinstance(suggestion["category"], str) or suggestion["category"] not in valid_categories:
                                logger.error(f"Suggestion {i} has invalid category: {suggestion['category']}")
                                continue

                            if not isinstance(suggestion["description"], str):
                                logger.error(f"Suggestion {i} description is not a string: {suggestion['description']}")
                                continue

                            try:
                                score = float(suggestion["score"])
                                if not 0 <= score <= 1:
                                    logger.error(f"Suggestion {i} score not between 0 and 1: {score}")
                                    continue
                            except (TypeError, ValueError):
                                logger.error(f"Suggestion {i} score is not a valid float: {suggestion['score']}")
                                continue

                            # If all validation passes, add to valid suggestions
                            valid_suggestions.append({
                                "query": str(suggestion["query"]),
                                "category": str(suggestion["category"]),
                                "description": str(suggestion["description"]),
                                "score": float(suggestion["score"]),
                            })

                        except Exception as e:
                            logger.error(f"Error validating suggestion {i}: {e}")
                            continue

                    if valid_suggestions:
                        return valid_suggestions

                except (json.JSONDecodeError, KeyError, ValueError) as e:
                    logger.error(f"Failed to process suggestions: {e}")
                    logger.error(f"Content: {content}")

            # Fallback suggestions
            return [
                {
                    "query": query,
                    "category": "original",
                    "description": "Original search query",
                    "score": 1.0,
                },
                {
                    "query": f"{query} example",
                    "category": "use_case",
                    "description": "Example usage",
                    "score": 0.9,
                },
            ]

        except Exception as e:
            logger.error(f"Query suggestion failed: {e}")
            # Return minimal fallback
            return [
                {
                    "query": str(query),
                    "category": "original",
                    "description": "Original search query",
                    "score": 1.0,
                }
            ] 