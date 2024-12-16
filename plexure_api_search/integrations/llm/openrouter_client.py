"""OpenRouter LLM client for API enrichment and search enhancement."""

import json
import logging
import re
import traceback
from typing import Any, Dict, List, Optional

import httpx

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
        self, prompt: str, cache_key: Optional[str] = None, **kwargs
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

            # Prepare request with strict JSON formatting instructions
            system_prompt = """You are a JSON-only API documentation assistant.
CRITICAL RULES:
1. You MUST ONLY output valid JSON objects
2. NEVER include any explanatory text or markdown
3. ALL property names MUST be in double quotes
4. ALL string values MUST be properly escaped
5. NO trailing commas allowed
6. ALL arrays MUST have at least one item
7. ALL arrays MUST end with proper closing bracket
8. ALL objects MUST end with proper closing brace
9. NEVER include comments or explanations
10. ALWAYS validate your response is parseable JSON before returning

Example valid response format:
{
    "detailed_description": "This is a description",
    "use_cases": ["Use case 1"],
    "best_practices": ["Best practice 1"],
    "related_endpoints": ["Related endpoint 1"],
    "parameters_description": {
        "required": ["Required param 1"],
        "optional": ["Optional param 1"]
    },
    "response_scenarios": {
        "success": "Success scenario",
        "errors": ["Error scenario 1"]
    }
}"""

            data = {
                "model": self.model,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt},
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
                f"{self.base_url}/chat/completions", json=data
            )

            # Log response for debugging if there's an error
            if response.status_code != 200:
                logger.error(f"OpenRouter API error response: {response.text}")
                response.raise_for_status()

            # Get raw response content
            raw_content = response.text
            logger.debug(f"Raw response content: {raw_content}")

            # Parse response JSON
            result = response.json()

            # Extract and clean the actual content from the response
            if "choices" in result and result["choices"]:
                content = result["choices"][0]["message"]["content"]
                logger.debug(f"Extracted content: {content}")

                # Clean the content
                cleaned_content = self._clean_json_string(content)
                logger.debug(f"Cleaned content: {cleaned_content}")

                try:
                    # Parse the cleaned content
                    parsed_content = json.loads(cleaned_content)

                    # Validate required fields
                    required_fields = {
                        "detailed_description",
                        "use_cases",
                        "best_practices",
                        "related_endpoints",
                        "parameters_description",
                        "response_scenarios",
                    }

                    if not all(field in parsed_content for field in required_fields):
                        logger.error("Missing required fields in parsed content")
                        raise ValueError("Invalid response structure")

                    # Update the response with cleaned content
                    result["choices"][0]["message"]["content"] = parsed_content

                    # Cache response if enabled and key provided
                    if self.use_cache and cache_key:
                        llm_cache.set(cache_key, result)

                    return result
                except (json.JSONDecodeError, ValueError) as e:
                    logger.error(f"Failed to parse cleaned content: {e}")
                    logger.error(f"Content that failed to parse: {cleaned_content}")
                    return {
                        "choices": [
                            {
                                "message": {
                                    "content": {
                                        "detailed_description": "Failed to parse enriched data",
                                        "use_cases": ["No data available"],
                                        "best_practices": ["No data available"],
                                        "related_endpoints": ["No data available"],
                                        "parameters_description": {
                                            "required": ["No data available"],
                                            "optional": ["No data available"],
                                        },
                                        "response_scenarios": {
                                            "success": "No data available",
                                            "errors": ["No data available"],
                                        },
                                    }
                                }
                            }
                        ]
                    }

            return result

        except Exception as e:
            logger.error(f"OpenRouter API call failed: {e}")
            logger.error(f"Full traceback: {traceback.format_exc()}")
            if isinstance(e, httpx.HTTPError):
                logger.error(
                    f"Response content: {e.response.content if hasattr(e, 'response') else 'No response content'}"
                )
            return {
                "choices": [
                    {
                        "message": {
                            "content": {
                                "detailed_description": "API call failed",
                                "use_cases": ["No data available"],
                                "best_practices": ["No data available"],
                                "related_endpoints": ["No data available"],
                                "parameters_description": {
                                    "required": ["No data available"],
                                    "optional": ["No data available"],
                                },
                                "response_scenarios": {
                                    "success": "No data available",
                                    "errors": ["No data available"],
                                },
                            }
                        }
                    }
                ]
            }

    def _clean_json_string(self, content: str) -> str:
        """Clean and fix malformed JSON string.

        Args:
            content: Raw JSON string

        Returns:
            Cleaned and valid JSON string
        """
        try:
            # Log original content for debugging
            logger.debug(f"Original content to clean: {content}")

            # Remove any markdown code block markers
            content = content.strip()
            if content.startswith("```json"):
                content = content[7:]
            elif content.startswith("```"):
                content = content[3:]
            if content.endswith("```"):
                content = content[:-3]
            content = content.strip()

            logger.debug(f"Content after markdown cleanup: {content}")

            # Handle cases where content might be wrapped in additional quotes
            if content.startswith('"') and content.endswith('"'):
                try:
                    # Try to parse as a JSON string that contains JSON
                    decoded = json.loads(content)
                    if isinstance(decoded, str):
                        content = decoded
                except json.JSONDecodeError:
                    pass

            # Basic cleanup
            content = re.sub(r"\n\s*\n", "\n", content)  # Remove empty lines
            content = re.sub(r"^\s*{\s*\n", "{", content)  # Clean start
            content = re.sub(r"\n\s*}\s*$", "}", content)  # Clean end

            # Fix common JSON formatting issues
            content = re.sub(
                r'(?<!\\)"(?![\s,}\]])', r'\\"', content
            )  # Fix unescaped quotes
            content = re.sub(r",(\s*[}\]])", r"\1", content)  # Remove trailing commas
            content = re.sub(
                r'([{,])\s*([^"\s]+)\s*:', r'\1"\2":', content
            )  # Quote unquoted keys
            content = re.sub(
                r':\s*([^"\s\d{[][\w.-]+)', r':"\1"', content
            )  # Quote unquoted string values

            logger.debug(f"Content after basic cleanup: {content}")

            # Try to parse and reformat
            try:
                parsed = json.loads(content)
                return json.dumps(parsed, indent=2)
            except json.JSONDecodeError as e:
                logger.debug(f"Initial JSON parse failed: {e}")

                # Try to extract just the JSON object if there's extra text
                json_pattern = r"({[\s\S]*})"
                matches = re.findall(json_pattern, content)
                if matches:
                    for potential_json in matches:
                        try:
                            parsed = json.loads(potential_json)
                            return json.dumps(parsed, indent=2)
                        except json.JSONDecodeError:
                            continue

                # If still failing, return a minimal valid JSON structure
                logger.error("Failed to parse JSON, returning minimal structure")
                return json.dumps(
                    {
                        "detailed_description": "Failed to parse enriched data",
                        "use_cases": [],
                        "best_practices": [],
                        "related_endpoints": [],
                        "parameters_description": {"required": [], "optional": []},
                        "response_scenarios": {
                            "success": "No data available",
                            "errors": [],
                        },
                    },
                    indent=2,
                )

        except Exception as e:
            logger.error(f"Error cleaning JSON string: {e}")
            logger.error(f"Full traceback: {traceback.format_exc()}")
            return json.dumps({})

    def _clean_content(self, content: str) -> str:
        """Clean content by removing angle brackets from URLs and fixing JSON format."""
        # Remove angle brackets from URLs
        content = re.sub(r"<(http[^>]+)>", r"\1", content)

        # Try to parse and reformat as JSON
        try:
            parsed = json.loads(content)
            return json.dumps(parsed, indent=2)
        except json.JSONDecodeError:
            logger.warning("Failed to parse content as JSON, returning as is")
            return content

    def clean_json_content(self, content: str) -> Dict[Any, Any]:
        """
        Clean and parse JSON content with improved error handling and sanitization.
        """
        try:
            # Log original content for debugging
            self.logger.debug(f"Original content to clean: {content}")

            # Basic string cleanup
            content = self._sanitize_string(content)
            self.logger.debug(f"Content after basic cleanup: {content}")

            # Fix JSON formatting
            content = self._fix_json_formatting(content)
            self.logger.debug(f"Content after JSON formatting fixes: {content}")

            # Validate JSON
            if not self._validate_json(content):
                self.logger.error("Invalid JSON structure")
                return self._get_minimal_structure()

            # Parse JSON
            return json.loads(content)

        except Exception as e:
            self.logger.error(f"Failed to parse JSON: {str(e)}")
            return self._get_minimal_structure()

    def _sanitize_string(self, content: str) -> str:
        """
        Sanitize string content before JSON parsing.
        """
        # Remove any BOM or special characters
        content = content.strip().lstrip("\ufeff")

        # Replace problematic quotes
        content = content.replace("'", '"')
        content = content.replace(
            """, '"')
        content = content.replace(""",
            '"',
        )

        # Fix escaped quotes
        content = content.replace('\\"', '"')
        content = content.replace('"', '"')

        return content

    def _fix_json_formatting(self, content: str) -> str:
        """
        Fix common JSON formatting issues.
        """
        try:
            # Remove any trailing/leading whitespace
            content = content.strip()

            # Fix property names without quotes
            content = re.sub(
                r"([{,])\s*([a-zA-Z_][a-zA-Z0-9_]*)\s*:", r'\1"\2":', content
            )

            # Fix single quotes to double quotes, but not within content
            content = re.sub(r"'([^']*)':", r'"\1":', content)

            # Fix boolean values
            content = content.replace("true", "true").replace("True", "true")
            content = content.replace("false", "false").replace("False", "false")
            content = content.replace("null", "null").replace("None", "null")

            # Fix arrays and objects that are quoted as strings
            content = re.sub(r'"(\[.*?\])"', r"\1", content)
            content = re.sub(r'"(\{.*?\})"', r"\1", content)

            # Remove escaped quotes within strings
            content = re.sub(r'\\"', '"', content)

            # Fix trailing commas
            content = re.sub(r",(\s*[}\]])", r"\1", content)

            # Ensure the content is a valid JSON object
            if not content.startswith("{"):
                content = "{" + content
            if not content.endswith("}"):
                content = content + "}"

            # Validate JSON structure
            json.loads(content)
            return content

        except Exception as e:
            self.logger.error(f"Error fixing JSON formatting: {str(e)}")
            self.logger.debug(f"Problematic content: {content}")
            return "{}"

    def _get_minimal_structure(self) -> Dict[str, Any]:
        """
        Return a minimal fallback structure when JSON parsing fails.
        """
        return {
            "detailed_description": "Failed to parse enriched data",
            "use_cases": [],
            "best_practices": [],
            "related_endpoints": [],
            "parameters_description": {"required": [], "optional": []},
            "response_scenarios": {"success": "No data available", "errors": []},
        }

    def enrich_endpoint_content(self, content: Dict[str, Any]) -> Dict[str, Any]:
        """
        Enrich endpoint content with improved error handling.

        Args:
            content: Dictionary containing endpoint content

        Returns:
            Dict containing enriched content
        """
        try:
            # Convert content to JSON string with proper formatting
            content_str = json.dumps(content, ensure_ascii=False, indent=2)

            # Clean and parse the content
            cleaned_content = self.clean_json_content(content_str)

            # Validate required fields
            if not self._validate_enriched_content(cleaned_content):
                logger.warning("Enriched content validation failed")
                return self._get_minimal_structure()

            return cleaned_content

        except Exception as e:
            logger.error(f"Error enriching endpoint content: {str(e)}")
            return self._get_minimal_structure()

    def _validate_enriched_content(self, content: Dict[str, Any]) -> bool:
        """
        Validate that enriched content has required fields.
        """
        required_fields = [
            "detailed_description",
            "use_cases",
            "best_practices",
            "related_endpoints",
        ]

        return all(field in content for field in required_fields)

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

            # Build enrichment prompt with strict JSON format
            prompt = f"""Analyze this API endpoint and provide enriched documentation in STRICT JSON format:

Method: {endpoint_data["method"]}
Path: {endpoint_data["path"]}
Description: {endpoint_data.get("description", "")}
Parameters: {json.dumps(endpoint_data.get("parameters", []), indent=2)}
Responses: {json.dumps(endpoint_data.get("responses", []), indent=2)}

IMPORTANT: Your response MUST be a valid JSON object with EXACTLY this structure:
{{
    "detailed_description": "string",
    "use_cases": ["string"],
    "best_practices": ["string"],
    "related_endpoints": ["string"],
    "parameters_description": {{
        "required": ["string"],
        "optional": ["string"]
    }},
    "response_scenarios": {{
        "success": "string",
        "errors": ["string"]
    }}
}}

Rules:
1. ONLY output the JSON object, no other text
2. ALL string values must be properly escaped
3. ALL property names must be in double quotes
4. NO trailing commas
5. NO comments or explanations
6. Arrays must contain at least one item
7. ALL arrays must end with a proper closing bracket
8. ALL objects must end with a proper closing brace"""

            # Get LLM response
            response = self._call_llm(
                prompt=prompt,
                cache_key=cache_key,
                temperature=0.3,  # Lower temperature for more focused documentation
            )

            # Extract and parse content
            if "error" not in response and "choices" in response:
                try:
                    content = response["choices"][0]["message"]["content"]
                    # Clean and parse JSON
                    cleaned_content = self._clean_json_string(content)
                    logger.debug(f"Cleaned JSON content: {cleaned_content}")

                    try:
                        enriched_data = json.loads(cleaned_content)
                        # Validate required fields
                        required_fields = {
                            "detailed_description",
                            "use_cases",
                            "best_practices",
                            "related_endpoints",
                            "parameters_description",
                            "response_scenarios",
                        }

                        if not all(field in enriched_data for field in required_fields):
                            logger.error("Missing required fields in enriched data")
                            raise ValueError("Invalid enriched data structure")

                        endpoint_data.update({"enriched": enriched_data})
                    except (json.JSONDecodeError, ValueError) as e:
                        logger.error(f"Failed to parse enriched data: {e}")
                        logger.error(f"Raw content: {content}")
                        logger.error(f"Cleaned content: {cleaned_content}")
                        endpoint_data.update({
                            "enriched": {
                                "detailed_description": "Failed to parse enriched data",
                                "use_cases": ["No data available"],
                                "best_practices": ["No data available"],
                                "related_endpoints": ["No data available"],
                                "parameters_description": {
                                    "required": ["No data available"],
                                    "optional": ["No data available"],
                                },
                                "response_scenarios": {
                                    "success": "No data available",
                                    "errors": ["No data available"],
                                },
                            }
                        })

                except Exception as e:
                    logger.error(f"Failed to process enriched data: {e}")
                    logger.error(traceback.format_exc())

            return endpoint_data

        except Exception as e:
            logger.error(f"Endpoint enrichment failed: {e}")
            logger.error(traceback.format_exc())
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
                    logger.debug(
                        f"Parsed suggestions: {json.dumps(suggestions, indent=2)}"
                    )

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
                                logger.error(
                                    f"Suggestion {i} is not a dictionary: {suggestion}"
                                )
                                continue

                            if not all(
                                field in suggestion for field in required_fields
                            ):
                                logger.error(
                                    f"Suggestion {i} missing required fields: {suggestion}"
                                )
                                continue

                            # Validate field types and values
                            if not isinstance(suggestion["query"], str):
                                logger.error(
                                    f"Suggestion {i} query is not a string: {suggestion['query']}"
                                )
                                continue

                            if (
                                not isinstance(suggestion["category"], str)
                                or suggestion["category"] not in valid_categories
                            ):
                                logger.error(
                                    f"Suggestion {i} has invalid category: {suggestion['category']}"
                                )
                                continue

                            if not isinstance(suggestion["description"], str):
                                logger.error(
                                    f"Suggestion {i} description is not a string: {suggestion['description']}"
                                )
                                continue

                            try:
                                score = float(suggestion["score"])
                                if not 0 <= score <= 1:
                                    logger.error(
                                        f"Suggestion {i} score not between 0 and 1: {score}"
                                    )
                                    continue
                            except (TypeError, ValueError):
                                logger.error(
                                    f"Suggestion {i} score is not a valid float: {suggestion['score']}"
                                )
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
            logger.error(traceback.format_exc())
            # Return minimal fallback
            return [
                {
                    "query": str(query),
                    "category": "original",
                    "description": "Original search query",
                    "score": 1.0,
                }
            ]

    def _validate_json(self, content: str) -> bool:
        """
        Validate if string is valid JSON.
        """
        try:
            if not content:
                return False

            # Try to parse as JSON
            json.loads(content)

            # Check for required structure
            if not all(
                key in content
                for key in [
                    '"detailed_description"',
                    '"use_cases"',
                    '"best_practices"',
                    '"related_endpoints"',
                ]
            ):
                return False

            return True

        except Exception:
            return False
