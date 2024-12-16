"""OpenRouter LLM client for API enrichment and search enhancement."""

import json
import logging
from typing import Any, Dict, List, Optional

import httpx
import traceback

from ...config import config_instance
from ...utils.cache import DiskCache

logger = logging.getLogger(__name__)

# Cache for LLM responses
llm_cache = DiskCache[Dict[str, Any]](
    namespace="llm_responses",
    ttl=config_instance.cache_ttl * 24,  # Cache LLM responses for longer
)


class OpenRouterClient:
    """Client for OpenRouter LLM API integration."""

    def __init__(self, use_cache: bool = True):
        """Initialize OpenRouter client.
        
        Args:
            use_cache: Whether to use caching for LLM responses
        """
        self.api_key = config_instance.openrouter_api_key
        self.model = config_instance.openrouter_model
        self.max_tokens = config_instance.openrouter_max_tokens
        self.temperature = config_instance.openrouter_temperature
        self.base_url = "https://openrouter.ai/api/v1"
        self.use_cache = use_cache
        
        # Headers required by OpenRouter
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "HTTP-Referer": "https://github.com/plexure/plexure-api-search",  # Required by OpenRouter
            "X-Title": "Plexure API Search",  # Required by OpenRouter
            "X-Custom-Auth": "plexure-api-search",  # Recommended by OpenRouter
            "Content-Type": "application/json",
        }
        
        self.http_client = httpx.Client(
            headers=self.headers,
            timeout=30.0,
        )

    def _call_llm(
        self, 
        prompt: str,
        cache_key: Optional[str] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """Make a call to OpenRouter API.

        Args:
            prompt: The prompt to send to the LLM
            cache_key: Optional cache key for response
            **kwargs: Additional parameters for the API call

        Returns:
            LLM response
        """
        try:
            # Check cache first if enabled
            if self.use_cache and cache_key:
                cached = llm_cache.get(cache_key)
                if cached is not None:
                    return cached

            # Prepare request
            data = {
                "model": self.model,
                "messages": [
                    {
                        "role": "system",
                        "content": kwargs.get("system_prompt", "You are an AI assistant specialized in analyzing and documenting APIs. You provide technical, accurate, and concise responses focused on API functionality and best practices.")
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                "max_tokens": kwargs.get("max_tokens", self.max_tokens),
                "temperature": kwargs.get("temperature", self.temperature),
                "top_p": 1,
                "stream": False,
                "transforms": ["middle-out"],  # OpenRouter optimization
            }

            # Log request details for debugging
            logger.debug(f"Request URL: {self.base_url}/chat/completions")
            logger.debug(f"Request headers: {self.headers}")
            logger.debug(f"Request body: {json.dumps(data, indent=2)}")

            # Make request
            response = self.http_client.post(
                f"{self.base_url}/chat/completions",
                json=data
            )
            
            # Log response for debugging if there's an error
            if response.status_code != 200:
                logger.error(f"OpenRouter API error response: {response.text}")
                
            response.raise_for_status()
            result = response.json()

            # Cache response if enabled and key provided
            if self.use_cache and cache_key:
                llm_cache.set(cache_key, result)

            return result

        except Exception as e:
            logger.error(f"OpenRouter API call failed: {e}")
            if isinstance(e, httpx.HTTPError):
                logger.error(f"Response content: {e.response.content if hasattr(e, 'response') else 'No response content'}")
            return {"error": str(e)}

    def _clean_json_string(self, content: str) -> str:
        """Clean JSON string by removing invalid characters and normalizing newlines.
        
        Args:
            content: Raw JSON string
            
        Returns:
            Cleaned JSON string
        """
        # Remove any markdown code block markers
        content = content.strip()
        if content.startswith("```json"):
            content = content[7:]
        if content.endswith("```"):
            content = content[:-3]
        content = content.strip()
        
        # Replace multiple newlines with single newline
        content = "\n".join(line.strip() for line in content.split("\n"))
        
        # Escape newlines in strings
        in_string = False
        result = []
        i = 0
        while i < len(content):
            char = content[i]
            if char == '"' and (i == 0 or content[i-1] != '\\'):
                in_string = not in_string
            if in_string and char in '\n\r\t':
                if char == '\n':
                    result.append('\\n')
                elif char == '\r':
                    result.append('\\r')
                elif char == '\t':
                    result.append('\\t')
            else:
                result.append(char)
            i += 1
            
        return "".join(result)

    def enrich_endpoint(self, endpoint_data: Dict[str, Any]) -> Dict[str, Any]:
        """Enrich endpoint data with LLM-generated content.

        Args:
            endpoint_data: Raw endpoint data

        Returns:
            Enriched endpoint data
        """
        try:
            # Create cache key
            cache_key = None
            if self.use_cache:
                cache_key = f"enrich_{endpoint_data['method']}_{endpoint_data['path']}"

            # Build enrichment prompt
            prompt = f"""Analyze this API endpoint and provide enriched information:

Method: {endpoint_data['method']}
Path: {endpoint_data['path']}
Description: {endpoint_data.get('description', '')}
Parameters: {json.dumps(endpoint_data.get('parameters', []), indent=2)}
Responses: {json.dumps(endpoint_data.get('responses', []), indent=2)}

Please provide:
1. A more detailed description with use cases
2. Common usage patterns
3. Best practices for using this endpoint
4. Related endpoints that might be useful
5. Example request/response pairs
6. Potential error scenarios and how to handle them

Format the response as JSON with these keys:
- detailed_description (string)
- use_cases (string)
- best_practices (string)
- related_endpoints (string)
- examples (object with request/response)
- error_scenarios (string)

Important: Do not use newlines within text values. Use periods or semicolons for separation."""

            # Get LLM response
            response = self._call_llm(
                prompt=prompt,
                cache_key=cache_key,
                temperature=0.7,  # More creative for enrichment
            )

            # Extract and parse content
            if "error" not in response:
                try:
                    content = response["choices"][0]["message"]["content"]
                    # Clean and parse JSON
                    cleaned_content = self._clean_json_string(content)
                    logger.debug(f"Cleaned JSON content: {cleaned_content}")
                    
                    enriched_data = json.loads(cleaned_content)
                    endpoint_data.update({
                        "enriched": enriched_data
                    })
                except (json.JSONDecodeError, KeyError) as e:
                    logger.error(f"Failed to parse LLM response: {e}")
                    logger.error(f"Raw content: {content}")
                    logger.error(f"Cleaned content: {cleaned_content}")

            return endpoint_data

        except Exception as e:
            logger.error(f"Endpoint enrichment failed: {e}")
            return endpoint_data

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
            response = self._call_llm(
                prompt=prompt,
                cache_key=cache_key,
                temperature=0.5,  # More focused for search enhancement
            )

            # Extract and parse content
            if "error" not in response:
                try:
                    content = response["choices"][0]["message"]["content"]
                    # Clean the content to ensure it's valid JSON
                    content = content.strip()
                    if content.startswith("```json"):
                        content = content[7:]
                    if content.endswith("```"):
                        content = content[:-3]
                    content = content.strip()
                    
                    enhanced_data = json.loads(content)
                    return {
                        "results": results,
                        "analysis": enhanced_data
                    }
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
                'You are a JSON formatting assistant. You ONLY output valid JSON arrays. '
                'Never output numbered lists, text, or explanations. '
                'Only respond with the exact JSON array requested.'
            )

            user_prompt = (
                '{'
                f'"task": "Generate 5 related queries for: {query}",'
                '"format": "json_array",'
                '"required_fields": ["query", "category", "description", "score"],'
                '"valid_categories": ["alternative", "related", "use_case", "implementation", "error"],'
                '"example": [{"query": "example", "category": "alternative", "description": "why", "score": 0.9}]'
                '}'
            )

            # Get LLM response
            response = self._call_llm(
                prompt=user_prompt,
                cache_key=cache_key,
                temperature=0.7,
                max_tokens=512,
                system_prompt=system_prompt,
            )

            # Extract suggestions
            if "error" not in response and "choices" in response:
                try:
                    content = response["choices"][0]["message"]["content"].strip()
                    logger.debug(f"Raw suggestion content: {content}")
                    
                    # Clean JSON content and ensure it starts with [
                    content = self._clean_json_string(content)
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
                    valid_categories = {"alternative", "related", "use_case", "implementation", "error"}
                    
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
                                "score": float(suggestion["score"])
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
                    "score": 1.0
                },
                {
                    "query": f"{query} example",
                    "category": "use_case",
                    "description": "Example usage",
                    "score": 0.9
                }
            ]

        except Exception as e:
            logger.error(f"Query suggestion failed: {e}")
            logger.error(traceback.format_exc())
            # Return minimal fallback
            return [{
                "query": str(query),
                "category": "original",
                "description": "Original search query",
                "score": 1.0
            }]
