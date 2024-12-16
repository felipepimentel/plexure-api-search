"""LLM-based API endpoint enrichment."""

import json
import logging
from typing import Any, Dict, Optional, Union

from ..integrations.llm.base import LLMProvider
from ..utils.json_utils import clean_json_string, get_minimal_enrichment_structure

logger = logging.getLogger(__name__)


class LLMEnricher:
    """Handles API endpoint enrichment using LLM."""

    def __init__(self, llm_provider: LLMProvider, use_cache: bool = True):
        """Initialize enricher.

        Args:
            llm_provider: LLM provider instance
            use_cache: Whether to use caching
        """
        self.llm_provider = llm_provider
        self.use_cache = use_cache

    def _adapt_llm_response(self, data: Union[Dict[str, Any], str]) -> Dict[str, Any]:
        """Adapt LLM response to expected format.

        Args:
            data: Raw LLM response data

        Returns:
            Adapted data in expected format
        """
        try:
            # If data is string, try to parse it
            if isinstance(data, str):
                try:
                    data = json.loads(data)
                except json.JSONDecodeError:
                    logger.error("Failed to parse string data as JSON")
                    return get_minimal_enrichment_structure()

            # If data is not a dict after parsing, return minimal structure
            if not isinstance(data, dict):
                logger.error(f"Data is not a dictionary: {type(data)}")
                return get_minimal_enrichment_structure()

            # Start with the fields that are already correct
            adapted = {
                "detailed_description": str(data.get("detailed_description", "No description available")),
                "use_cases": [str(x) for x in data.get("use_cases", ["No use cases available"])],
                "best_practices": [str(x) for x in data.get("best_practices", ["No best practices available"])],
                "related_endpoints": [str(x) for x in data.get("related_endpoints", ["No related endpoints available"])],
            }

            # Extract parameters description
            adapted["parameters_description"] = {
                "required": ["No required parameters"],
                "optional": ["No optional parameters"]
            }

            # Extract response scenarios from examples and error_scenarios
            success_example = None
            if "examples" in data and isinstance(data["examples"], list):
                for example in data["examples"]:
                    if not isinstance(example, dict):
                        continue
                    response = example.get("response", {})
                    if not isinstance(response, dict):
                        continue
                    status = str(response.get("status", ""))
                    if status.startswith("2"):
                        success_example = str(response.get("description", "Success"))
                        break

            error_scenarios = []
            if "error_scenarios" in data and isinstance(data["error_scenarios"], list):
                error_scenarios = [str(x) for x in data["error_scenarios"]]
            if not error_scenarios:
                error_scenarios = ["No error scenarios available"]

            adapted["response_scenarios"] = {
                "success": success_example or "Success",
                "errors": error_scenarios
            }

            return adapted

        except Exception as e:
            logger.error(f"Error adapting LLM response: {e}")
            logger.error(f"Data type: {type(data)}")
            logger.error(f"Data content: {data}")
            return get_minimal_enrichment_structure()

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
            prompt = f"""Analyze this API endpoint and provide enriched documentation in strict JSON format:

Method: {endpoint_data["method"]}
Path: {endpoint_data["path"]}
Description: {endpoint_data.get("description", "")}
Parameters: {json.dumps(endpoint_data.get("parameters", []), indent=2)}
Responses: {json.dumps(endpoint_data.get("responses", []), indent=2)}

Return a JSON object with these exact fields. Use ONLY the structure shown below:

{{
    "detailed_description": "A single line description of what the endpoint does",
    "use_cases": [
        "First use case in a single line",
        "Second use case in a single line"
    ],
    "best_practices": [
        "First best practice in a single line",
        "Second best practice in a single line"
    ],
    "related_endpoints": [
        "/api/v1/first-related-endpoint",
        "/api/v1/second-related-endpoint"
    ],
    "examples": [
        {{
            "request": {{
                "description": "Example request description"
            }},
            "response": {{
                "status": "200",
                "description": "Example response description"
            }}
        }}
    ],
    "error_scenarios": [
        "First error scenario in a single line",
        "Second error scenario in a single line"
    ]
}}

CRITICAL FORMATTING RULES:
1. Return ONLY the JSON object above, no additional text
2. Do not modify the structure - use exactly the same field names and nesting
3. Property names must be exactly as shown (e.g. "detailed_description", not "description")
4. All strings must be in double quotes (") - never use single quotes (')
5. Keep all text in a single line - no line breaks or newlines in text
6. Do not escape quotes in property names
7. Only escape quotes in string values when absolutely necessary
8. Each array must have at least one item
9. Do not add any fields that are not shown in the example
10. Do not add any comments or explanations - just the JSON object"""

            # Get LLM response
            response = self.llm_provider.call(
                prompt=prompt,
                cache_key=cache_key,
                temperature=0.3,  # Lower temperature for more focused documentation
            )

            # Log raw response for debugging
            logger.debug(f"Raw LLM response: {json.dumps(response, indent=2)}")

            # Check for error in response
            if "error" in response:
                logger.error(f"LLM provider returned error: {response['error']}")
                endpoint_data.update({"enriched": get_minimal_enrichment_structure()})
                return endpoint_data

            # Check for missing choices
            if "choices" not in response or not response["choices"]:
                logger.error("LLM response missing choices")
                endpoint_data.update({"enriched": get_minimal_enrichment_structure()})
                return endpoint_data

            try:
                # Extract content
                content = response["choices"][0]["message"]["content"]
                logger.debug(f"Raw content from LLM: {content}")

                # Clean and parse JSON
                cleaned_content = clean_json_string(content)
                logger.debug(f"Cleaned JSON content: {cleaned_content}")

                try:
                    # Parse JSON
                    raw_data = json.loads(cleaned_content)
                    logger.debug(f"Parsed raw data: {json.dumps(raw_data, indent=2)}")

                    # Adapt the response to our expected format
                    enriched_data = self._adapt_llm_response(raw_data)
                    logger.debug(f"Adapted enriched data: {json.dumps(enriched_data, indent=2)}")

                    # Validate required fields
                    required_fields = {
                        "detailed_description",
                        "use_cases",
                        "best_practices",
                        "related_endpoints",
                        "parameters_description",
                        "response_scenarios",
                    }

                    # Check each required field
                    missing_fields = [field for field in required_fields if field not in enriched_data]
                    if missing_fields:
                        logger.error(f"Missing required fields in enriched data: {missing_fields}")
                        raise ValueError(f"Invalid enriched data structure - missing fields: {missing_fields}")

                    # Validate nested structures
                    if not isinstance(enriched_data["use_cases"], list) or not enriched_data["use_cases"]:
                        logger.error("use_cases must be a non-empty array")
                        raise ValueError("use_cases must be a non-empty array")

                    if not isinstance(enriched_data["best_practices"], list) or not enriched_data["best_practices"]:
                        logger.error("best_practices must be a non-empty array")
                        raise ValueError("best_practices must be a non-empty array")

                    if not isinstance(enriched_data["related_endpoints"], list) or not enriched_data["related_endpoints"]:
                        logger.error("related_endpoints must be a non-empty array")
                        raise ValueError("related_endpoints must be a non-empty array")

                    if not isinstance(enriched_data["parameters_description"], dict):
                        logger.error("parameters_description must be an object")
                        raise ValueError("parameters_description must be an object")

                    if not isinstance(enriched_data["response_scenarios"], dict):
                        logger.error("response_scenarios must be an object")
                        raise ValueError("response_scenarios must be an object")

                    # Update endpoint data with validated enriched data
                    endpoint_data.update({"enriched": enriched_data})
                    logger.info(f"Successfully enriched endpoint {endpoint_data['method']} {endpoint_data['path']}")

                except (json.JSONDecodeError, ValueError) as e:
                    logger.error(f"Failed to parse or validate enriched data: {e}")
                    logger.error(f"Raw content: {content}")
                    logger.error(f"Cleaned content: {cleaned_content}")
                    endpoint_data.update({"enriched": get_minimal_enrichment_structure()})

            except Exception as e:
                logger.error(f"Failed to process enriched data: {e}")
                endpoint_data.update({"enriched": get_minimal_enrichment_structure()})

            return endpoint_data

        except Exception as e:
            logger.error(f"Endpoint enrichment failed: {e}")
            endpoint_data.update({"enriched": get_minimal_enrichment_structure()})
            return endpoint_data 