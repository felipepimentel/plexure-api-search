import argparse
import json
import os
import random
import time
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urlparse

import httpx
import yaml
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec
from sentence_transformers import SentenceTransformer

# Load environment variables from .env file
load_dotenv()

# Default configuration constants
DEFAULT_API_DIR = "assets/apis"
DEFAULT_MIN_SCORE = 0.5
DEFAULT_TOP_K = 5
DEFAULT_EXAMPLE_QUERY = "find pet by id"
VALID_HTTP_METHODS = ["GET", "POST", "PUT", "DELETE", "PATCH", "HEAD", "OPTIONS"]

# OpenRouter configuration
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"


class QueryUnderstanding:
    """Natural language query understanding using LLM"""

    def __init__(self):
        self.headers = {
            "Authorization": f"Bearer {OPENROUTER_API_KEY}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://github.com/felipepimentel/plexure-api-search",
        }

    def parse_natural_query(self, query: str) -> Dict[str, Any]:
        """
        Convert natural language query into structured search parameters

        Args:
            query: Natural language query (e.g., "find APIs with highest availability")

        Returns:
            Dictionary with structured search parameters
        """
        system_prompt = """You are an API search assistant. Convert natural language queries into structured search parameters.
        Focus on extracting:
        - Main search terms
        - Filters (method, availability, latency, error rate)
        - Sorting preferences
        - Special requirements
        
        Respond in JSON format with these fields:
        {
            "query": "main search terms",
            "filters": {
                "method": null or HTTP method,
                "min_availability": null or float,
                "max_latency": null or int,
                "max_error_rate": null or float,
                "lifecycle_state": null or string
            },
            "sort_by": null or field name,
            "sort_order": "asc" or "desc"
        }
        """

        try:
            with httpx.Client(timeout=10.0) as client:
                response = client.post(
                    OPENROUTER_URL,
                    headers=self.headers,
                    json={
                        "model": "mistralai/mistral-7b-instruct",  # Fast and cost-effective
                        "messages": [
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": query},
                        ],
                    },
                )

                if response.status_code == 200:
                    result = response.json()
                    structured_query = json.loads(
                        result["choices"][0]["message"]["content"]
                    )
                    return structured_query
                else:
                    print(f"Warning: LLM query failed: {response.text}")
                    # Fallback to basic search
                    return {
                        "query": query,
                        "filters": {},
                        "sort_by": None,
                        "sort_order": None,
                    }

        except Exception as e:
            print(f"Warning: LLM processing failed: {str(e)}")
            # Fallback to basic search
            return {"query": query, "filters": {}, "sort_by": None, "sort_order": None}


class ResultsExplanation:
    """Natural language explanation of search results using LLM"""

    def __init__(self):
        self.headers = {
            "Authorization": f"Bearer {OPENROUTER_API_KEY}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://github.com/yourusername/plexure-api-search",
        }

    def explain_results(
        self, query: str, results: List[Dict[str, Any]], original_intent: Dict[str, Any]
    ) -> str:
        """
        Generate a natural language explanation of search results

        Args:
            query: Original natural language query
            results: List of search results
            original_intent: Structured understanding of the original query

        Returns:
            Natural language explanation of results
        """
        # Create a summary of the results
        results_summary = self._create_results_summary(results)

        system_prompt = """You are an API search assistant helping developers find the right APIs.
        Analyze the search results and provide a friendly, concise explanation that includes:
        1. A summary of what was found
        2. How well the results match the user's requirements
        3. Important metrics and characteristics
        4. Recommendations or warnings if relevant
        
        Keep the tone professional but conversational. Focus on the most relevant information for the user's needs.
        If there are trade-offs or alternatives, mention them briefly."""

        user_prompt = f"""Original query: {query}

Structured intent:
{json.dumps(original_intent, indent=2)}

Search results summary:
{results_summary}

Provide a natural language explanation of these results, focusing on how well they match the user's needs."""

        try:
            with httpx.Client(timeout=10.0) as client:
                response = client.post(
                    OPENROUTER_URL,
                    headers=self.headers,
                    json={
                        "model": "mistralai/mistral-7b-instruct",
                        "messages": [
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": user_prompt},
                        ],
                    },
                )

                if response.status_code == 200:
                    result = response.json()
                    return result["choices"][0]["message"]["content"]
                else:
                    return self._generate_fallback_explanation(results)

        except Exception as e:
            print(f"Warning: LLM explanation failed: {str(e)}")
            return self._generate_fallback_explanation(results)

    def _create_results_summary(self, results: List[Dict[str, Any]]) -> str:
        """Create a structured summary of search results for the LLM"""
        summary = []

        for i, result in enumerate(results, 1):
            summary.append(f"Result {i}:")
            summary.append(
                f"- API: {result['api']} v{result.get('api_version', 'N/A')}"
            )
            summary.append(f"- Endpoint: {result['method']} {result['path']}")
            summary.append(f"- Match Score: {result['score']:.4f}")

            if "latency_ms" in result:
                summary.append(f"- Latency: {result['latency_ms']}ms")
            if "availability" in result:
                summary.append(f"- Availability: {result['availability'] * 100:.2f}%")
            if "error_rate" in result:
                summary.append(f"- Error Rate: {result['error_rate'] * 100:.2f}%")

            summary.append(f"- Lifecycle: {result.get('lifecycle_state', 'Unknown')}")
            summary.append(
                f"- Description: {result.get('description', 'No description available')}"
            )
            summary.append("")

        return "\n".join(summary)

    def _generate_fallback_explanation(self, results: List[Dict[str, Any]]) -> str:
        """Generate a basic explanation when LLM fails"""
        if not results:
            return "No APIs found matching your criteria."

        explanation = [f"Found {len(results)} matching APIs:"]

        for result in results:
            explanation.append(
                f"\n• {result['api']} v{result.get('api_version', 'N/A')}"
            )
            explanation.append(f"  {result['method']} {result['path']}")

            if "availability" in result:
                explanation.append(
                    f"  Availability: {result['availability'] * 100:.2f}%"
                )
            if "latency_ms" in result:
                explanation.append(f"  Latency: {result['latency_ms']}ms")

        return "\n".join(explanation)


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="API Search Engine - Search and explore APIs using semantic search"
    )

    # Index management arguments
    index_group = parser.add_argument_group("Index Management")
    index_group.add_argument(
        "--force-reindex",
        action="store_true",
        help="Force reindexing of all APIs even if index exists",
    )
    index_group.add_argument(
        "--skip-index",
        action="store_true",
        help="Skip indexing completely and use existing index",
    )

    # Search arguments
    search_group = parser.add_argument_group("Search Options")
    search_group.add_argument(
        "--query",
        type=str,
        help="Search query (if not provided, will run example searches)",
    )
    search_group.add_argument(
        "--natural",
        action="store_true",
        help='Process query as natural language (e.g., "find APIs with highest availability")',
    )
    search_group.add_argument(
        "--method", choices=VALID_HTTP_METHODS, help="Filter by HTTP method"
    )
    search_group.add_argument(
        "--min-score",
        type=float,
        default=DEFAULT_MIN_SCORE,
        help=f"Minimum similarity score (0-1), default: {DEFAULT_MIN_SCORE}",
    )
    search_group.add_argument(
        "--top-k",
        type=int,
        default=DEFAULT_TOP_K,
        help=f"Number of results to return, default: {DEFAULT_TOP_K}",
    )

    # API directory
    parser.add_argument(
        "--api-dir",
        type=str,
        default=DEFAULT_API_DIR,
        help=f"Directory containing API contracts, default: {DEFAULT_API_DIR}",
    )

    return parser.parse_args()


def display_results(
    results: List[Dict[str, Any]],
    title: str,
    query: Optional[str] = None,
    original_intent: Optional[Dict[str, Any]] = None,
):
    """Display search results with rich metadata and natural language explanation"""
    if not results:
        print(f"\n{title}:")
        print("No results found.")
        return

    # If we have the original query and intent, generate a natural explanation
    if query and original_intent:
        explanation = ResultsExplanation()
        friendly_explanation = explanation.explain_results(
            query, results, original_intent
        )
        print("\n" + "=" * 80)
        print("Search Analysis:")
        print("-" * 40)
        print(friendly_explanation)
        print("=" * 80 + "\n")

    print(f"\n{title}")
    print("=" * 80)
    print(f"Found {len(results)} results:\n")

    for i, result in enumerate(results, 1):
        print(f"Result {i}:")
        print("-" * 40)

        # Basic API Information
        print(f"🔍 API: {result['api']} v{result.get('api_version', 'N/A')}")
        print(f"📍 Endpoint: {result['method']} {result['path']}")
        if result.get("summary"):
            print(f"📝 Summary: {result['summary']}")
        if result.get("description"):
            print(f"ℹ️  Description: {result['description']}")
        print(f" Match Score: {result['score']:.4f}")

        # Lifecycle Information
        print("\n📊 Status:")
        lifecycle_state = result.get("lifecycle_state", "Unknown")
        state_emoji = {
            "stable": "✅",
            "beta": "����",
            "deprecated": "⚠️",
            "proposal": "💡",
        }.get(lifecycle_state, "❓")
        print(f"   State: {state_emoji} {lifecycle_state}")

        if result.get("deprecated", False):
            print("   ⚠️  DEPRECATED API")

        if last_updated := result.get("last_updated"):
            print(f"   📅 Last Updated: {last_updated}")

        # Performance Metrics (if available)
        if any(key in result for key in ["latency_ms", "error_rate", "availability"]):
            print("\n📈 Performance Metrics:")
            if "latency_ms" in result:
                print(f"   ⚡ Latency: {result['latency_ms']}ms")
            if "p95_latency_ms" in result:
                print(f"   ⚡ P95 Latency: {result['p95_latency_ms']}ms")
            if "error_rate" in result:
                print(f"   ❌ Error Rate: {result['error_rate'] * 100:.2f}%")
            if "availability" in result:
                print(f"   🟢 Availability: {result['availability'] * 100:.2f}%")

        # Technical Details
        print("\n🔧 Technical Details:")
        params_count = result.get("parameters_count", 0)
        required_params = result.get("required_parameters", 0)
        print(f"   Parameters: {params_count} total, {required_params} required")

        if result.get("has_auth"):
            print("   🔒 Authentication Required")

        if complexity := result.get("complexity_score"):
            print(f"   📊 Complexity Score: {complexity:.2f}")

        # Tags and Categories
        if tags := result.get("tags", []):
            print("\n🏷️  Tags:")
            print(f"   {', '.join(tags)}")

        if category := result.get("category"):
            print(f"   📁 Category: {category}")

        # Related APIs
        if related_apis := result.get("related_apis", []):
            print("\n🔗 Related APIs:")
            for related in related_apis:
                rel_type = related.get("relationship_type", "related")
                print(f"   • {related['api']} ({rel_type})")

        # Usage Examples
        print("\n📚 Usage Examples:")
        if curl_example := result.get("curl_example"):
            print("   cURL:")
            print(f"   {curl_example}")

        if python_example := result.get("usage_example"):
            print("\n   Python:")
            print(f"   {python_example}")

        # Alerts
        if alerts := get_api_alerts(result):
            print("\n⚠️  Alerts:")
            for alert in alerts:
                print(f"   • {alert}")

        print("\n" + "=" * 80 + "\n")


def get_api_alerts(api_info: Dict[str, Any]) -> List[str]:
    """
    Generate alerts for API issues based on various metrics and states

    Args:
        api_info: Dictionary containing API metadata and metrics

    Returns:
        List of alert messages
    """
    alerts = []

    # Deprecation alert
    if api_info.get("deprecated"):
        alerts.append("This API is deprecated. Consider using an alternative version.")

    # Performance alerts
    if api_info.get("latency_ms", 0) > 100:
        alerts.append(
            "High latency detected. This API might not be suitable for time-sensitive operations."
        )

    if api_info.get("error_rate", 0) > 0.01:
        alerts.append("High error rate detected. Monitor API reliability closely.")

    if api_info.get("availability", 1) < 0.99:
        alerts.append(
            "Low availability detected. Consider using a more reliable alternative."
        )

    # Update alerts
    last_updated = api_info.get("last_updated")
    if last_updated and isinstance(last_updated, str):
        try:
            last_updated_date = datetime.fromisoformat(last_updated)
            if (datetime.now() - last_updated_date).days > 365:
                alerts.append(
                    "This API hasn't been updated in over a year. It might be outdated."
                )
        except ValueError:
            pass  # Skip date comparison if format is invalid

    # Complexity alerts
    if api_info.get("complexity_score", 0) > 3.0:
        alerts.append(
            "High complexity score. Consider breaking down the request into smaller operations."
        )

    # Authentication alerts
    if api_info.get("has_auth") and not api_info.get("security_schemes"):
        alerts.append(
            "Authentication required but security scheme not well documented."
        )

    # Parameter alerts
    params_count = api_info.get("parameters_count", 0)
    required_params = api_info.get("required_parameters", 0)
    if params_count > 5:
        alerts.append(
            f"Complex endpoint with {params_count} parameters. Consider simplifying."
        )
    if required_params > 3:
        alerts.append(
            f"High number of required parameters ({required_params}). Consider making some optional."
        )

    # Response alerts
    if not api_info.get("success_responses"):
        alerts.append("No success response codes documented.")
    if not api_info.get("error_responses"):
        alerts.append("No error responses documented. Error handling might be unclear.")

    # Documentation alerts
    if not api_info.get("description"):
        alerts.append("Missing endpoint description. Consider improving documentation.")
    if not api_info.get("examples") and not api_info.get("has_schema"):
        alerts.append("No examples or schema provided. Usage might be unclear.")

    return alerts


class APIEnrichment:
    """Enrich API metadata using LLM"""

    def __init__(self):
        self.headers = {
            "Authorization": f"Bearer {OPENROUTER_API_KEY}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://github.com/yourusername/plexure-api-search",
        }
        self.max_retries = 3
        self.retry_delay = 1  # seconds
        self.system_prompt = """You are an API documentation expert. Analyze the API endpoint information and generate enriched documentation.
        
        IMPORTANT: You must respond with a valid JSON object containing the following fields:
        {
            "description": "Enhanced description of the endpoint",
            "examples": [
                {
                    "title": "Example title",
                    "request": {
                        "curl": "curl command",
                        "python": "python code"
                    },
                    "response": {
                        "status": 200,
                        "body": {}
                    }
                }
            ],
            "responses": {
                "200": {
                    "description": "Success response description",
                    "examples": [{"content": {}}],
                    "schema": {}
                },
                "400": {
                    "description": "Error response description",
                    "examples": [{"content": {}}],
                    "schema": {}
                }
            },
            "use_cases": ["Use case 1", "Use case 2"],
            "business_domains": ["Domain 1", "Domain 2"],
            "integration_tips": ["Tip 1", "Tip 2"],
            "best_practices": ["Practice 1", "Practice 2"],
            "security_considerations": ["Security note 1", "Security note 2"],
            "performance_implications": {
                "expected_latency": "Expected latency range",
                "throughput": "Expected throughput",
                "resource_usage": "Resource usage notes"
            },
            "common_errors": [
                {
                    "code": "error_code",
                    "message": "error message",
                    "solution": "how to fix"
                }
            ],
            "related_endpoints": [
                {
                    "path": "related endpoint path",
                    "method": "HTTP method",
                    "relationship": "relationship type"
                }
            ]
        }

        IMPORTANT NOTES:
        1. Ensure all JSON is properly formatted with correct commas and brackets
        2. Do not include any explanatory text outside the JSON object
        3. All string values must be properly escaped
        4. Arrays must have matching brackets
        5. Objects must have matching braces
        6. No trailing commas allowed
        7. Use double quotes for all keys and string values
        """
    
    def enrich_endpoint(self, endpoint: Dict[str, Any]) -> Dict[str, Any]:
        """
        Enrich endpoint metadata using LLM analysis with retry mechanism
        
        Args:
            endpoint: Original endpoint metadata
            
        Returns:
            Enriched endpoint metadata
        """
        for attempt in range(self.max_retries):
            try:
                enriched = self._try_enrich_endpoint(endpoint)
                return enriched
            except json.JSONDecodeError as e:
                print(f"Warning: Invalid JSON from LLM (attempt {attempt + 1}/{self.max_retries})")
                print(f"Error details: {str(e)}")
                if attempt < self.max_retries - 1:
                    print(f"Retrying in {self.retry_delay} seconds...")
                    time.sleep(self.retry_delay)
                    # Adjust the prompt for next attempt
                    if "response_format" not in self.system_prompt:
                        self.system_prompt = self._add_format_reminder(self.system_prompt)
                else:
                    print("Max retries reached, using fallback enrichment")
                    return self._generate_fallback_enrichment(endpoint)
            except Exception as e:
                print(f"Warning: LLM enrichment failed: {str(e)}")
                return self._generate_fallback_enrichment(endpoint)
    
    def _try_enrich_endpoint(self, endpoint: Dict[str, Any]) -> Dict[str, Any]:
        """Single attempt to enrich endpoint metadata"""
        # Create context from existing endpoint data
        endpoint_context = self._create_endpoint_context(endpoint)
        
        with httpx.Client(timeout=15.0) as client:
            response = client.post(
                OPENROUTER_URL,
                headers=self.headers,
                json={
                    "model": "mistralai/mistral-7b-instruct",
                    "messages": [
                        {"role": "system", "content": self.system_prompt},
                        {"role": "user", "content": endpoint_context}
                    ],
                    "response_format": {"type": "json_object"},
                    "temperature": 0.1  # Even lower temperature for more consistent JSON
                }
            )
            
            if response.status_code == 200:
                try:
                    result = response.json()
                    content = result['choices'][0]['message']['content']
                    
                    # Print raw content for debugging
                    print("\nRaw LLM response:")
                    print("=" * 40)
                    print(content[:500])
                    print("=" * 40)
                    
                    # Clean and validate JSON
                    content = self._clean_json_content(content)
                    
                    # Parse and validate against expected schema
                    enrichment = json.loads(content)
                    self._validate_enrichment_schema(enrichment)
                    
                    return self._merge_enrichment(endpoint, enrichment)
                except json.JSONDecodeError as e:
                    print(f"\nJSON Error Details:")
                    print(f"Error position: line {e.lineno}, column {e.colno}")
                    print(f"Error message: {e.msg}")
                    # Print the problematic line with a pointer to the error
                    if e.doc:
                        lines = e.doc.split('\n')
                        if 0 <= e.lineno - 1 < len(lines):
                            print("\nProblematic line:")
                            print(lines[e.lineno - 1])
                            print(' ' * (e.colno - 1) + '^')
                    raise
            else:
                raise Exception(f"LLM request failed: {response.status_code} - {response.text}")
    
    def _clean_json_content(self, content: str) -> str:
        """Clean and validate JSON content"""
        if isinstance(content, str):
            # Remove any markdown code block markers
            content = content.strip()
            if content.startswith('```json'):
                content = content[7:]
            if content.startswith('```'):
                content = content[3:]
            if content.endswith('```'):
                content = content[:-3]
            content = content.strip()
            
            # Remove any comments
            content_lines = content.split('\n')
            content_lines = [line for line in content_lines if not line.strip().startswith('//')]
            content = '\n'.join(content_lines)
            
            # Fix common JSON issues
            content = content.replace('\\n', ' ')  # Replace newlines in strings
            content = content.replace('\\"', '"')  # Fix double escaped quotes
            content = content.replace('",}', '"}')  # Remove trailing commas
            content = content.replace(',]', ']')    # Remove trailing commas in arrays
            
            # Fix property names not in quotes
            import re
            def quote_unquoted_keys(match):
                key = match.group(1)
                # Don't add quotes if already quoted
                if key.startswith('"') and key.endswith('"'):
                    return match.group(0)
                return f'"{key}":'
            
            # Match property names that aren't in quotes
            content = re.sub(r'([a-zA-Z_][a-zA-Z0-9_]*):', quote_unquoted_keys, content)
            
            # Fix single quotes to double quotes
            content = content.replace("'", '"')
            
            # Fix spaces in property names
            content = re.sub(r'"([^"]+)":', lambda m: f'"{m.group(1).replace(" ", "_")}":', content)
            
            # Remove any non-standard JSON syntax
            content = re.sub(r'\/\*.*?\*\/', '', content, flags=re.DOTALL)  # Remove C-style comments
            content = re.sub(r'[\t\n\r]+', ' ', content)  # Normalize whitespace
            content = re.sub(r',\s*([}\]])', r'\1', content)  # Remove trailing commas
            
            # Ensure proper array and object closure
            brackets = []
            for char in content:
                if char in '{[':
                    brackets.append(char)
                elif char in '}]':
                    if not brackets:
                        continue  # Skip unmatched closing brackets
                    if (char == '}' and brackets[-1] == '{') or (char == ']' and brackets[-1] == '['):
                        brackets.pop()
            
            # Close any unclosed brackets
            while brackets:
                last = brackets.pop()
                content += '}' if last == '{' else ']'
            
            try:
                # Try to parse and re-serialize to ensure valid JSON
                parsed = json.loads(content)
                content = json.dumps(parsed)
            except json.JSONDecodeError as e:
                print(f"Warning: JSON cleaning failed. Error: {str(e)}")
                print(f"Problematic content: {content[:200]}...")
                raise
            
        return content
    
    def _validate_enrichment_schema(self, enrichment: Dict[str, Any]):
        """Validate enrichment data against expected schema"""
        required_fields = [
            "description",
            "examples",
            "responses",
            "use_cases",
            "business_domains",
            "integration_tips",
            "best_practices",
            "security_considerations",
            "performance_implications",
            "common_errors",
            "related_endpoints"
        ]
        
        missing_fields = [field for field in required_fields if field not in enrichment]
        if missing_fields:
            raise ValueError(f"Missing required fields in enrichment: {missing_fields}")
    
    def _add_format_reminder(self, prompt: str) -> str:
        """Add extra formatting reminders to the prompt"""
        reminder = """

ADDITIONAL FORMATTING REQUIREMENTS:
- Response must be a single valid JSON object
- No text before or after the JSON object
- Use double quotes for strings
- No trailing commas
- No comments allowed
- Arrays and objects must be properly closed
"""
        return prompt + reminder
    
    def _create_endpoint_context(self, endpoint: Dict[str, Any]) -> str:
        """Create context for LLM from endpoint data"""
        # Format the context in a more structured way
        context = f"""Please analyze this API endpoint and generate enriched documentation:

API Information:
- Name: {endpoint['api']}
- Version: {endpoint.get('api_version', 'N/A')}
- Method: {endpoint['method']} {endpoint['path']}

Summary: {endpoint.get('summary', 'N/A')}

Description: {endpoint.get('description', 'N/A')}

Parameters:
{json.dumps(endpoint.get('parameters', []), indent=2)}

Response Codes:
{json.dumps(endpoint.get('responses', []), indent=2)}

Security:
{json.dumps(endpoint.get('security_schemes', []), indent=2)}

Additional Context:
- Tags: {', '.join(endpoint.get('tags', []))}
- Deprecated: {endpoint.get('deprecated', False)}
- Operation ID: {endpoint.get('operationId', 'N/A')}

Please generate comprehensive documentation including examples, use cases, and best practices.
Remember to respond with valid JSON as specified in the format above."""

        return context
    
    def _merge_enrichment(self, original: Dict[str, Any], enrichment: Dict[str, Any]) -> Dict[str, Any]:
        """Merge enriched data with original endpoint data"""
        enriched = original.copy()

        # Enhance description if generated one is better
        if len(enrichment.get("description", "")) > len(enriched.get("description", "")):
            enriched["description"] = enrichment["description"]

        # Add generated examples if none exist
        if not enriched.get("examples"):
            enriched["examples"] = enrichment.get("examples", [])

        # Add synthetic response examples
        if enrichment.get("responses"):
            for response in enriched.get("responses", []):
                response_code = response["code"]
                if response_code in enrichment["responses"]:
                    response["examples"] = enrichment["responses"][response_code].get("examples", [])
                    response["schema"] = enrichment["responses"][response_code].get("schema", {})

        # Add enriched metadata
        enriched.update({
            "use_cases": enrichment.get("use_cases", []),
            "business_domains": enrichment.get("business_domains", []),
            "integration_tips": enrichment.get("integration_tips", []),
            "best_practices": enrichment.get("best_practices", []),
            "security_considerations": enrichment.get("security_considerations", []),
            "performance_implications": enrichment.get("performance_implications", {}),
            "common_errors": enrichment.get("common_errors", []),
            "related_endpoints": enrichment.get("related_endpoints", []),
        })

        return enriched

    def _generate_fallback_enrichment(self, endpoint: Dict[str, Any]) -> Dict[str, Any]:
        """Generate basic enrichment when LLM fails"""
        enriched = endpoint.copy()

        # Add basic synthetic data
        method = endpoint["method"].upper()
        path = endpoint["path"]

        # Generate example responses if missing
        if not any(r.get("examples") for r in enriched.get("responses", [])):
            for response in enriched.get("responses", []):
                code = response["code"]
                if code.startswith("2"):
                    response["examples"] = [
                        {
                            "content_type": "application/json",
                            "example": self._generate_success_example(method, path),
                        }
                    ]
                elif code.startswith("4"):
                    response["examples"] = [
                        {
                            "content_type": "application/json",
                            "example": {
                                "error": "Bad Request",
                                "message": "Invalid input parameters",
                                "details": {
                                    "field": "id",
                                    "issue": "Must be a positive integer",
                                },
                            },
                        }
                    ]
                elif code.startswith("5"):
                    response["examples"] = [
                        {
                            "content_type": "application/json",
                            "example": {
                                "error": "Internal Server Error",
                                "message": "An unexpected error occurred",
                                "reference_id": "err_123456",
                            },
                        }
                    ]

        return enriched

    def _generate_success_example(self, method: str, path: str) -> Dict[str, Any]:
        """Generate basic success response example"""
        if "user" in path.lower():
            return {
                "id": 123,
                "username": "john.doe",
                "email": "john.doe@example.com",
                "created_at": "2024-01-01T00:00:00Z",
            }
        elif "product" in path.lower():
            return {"id": 456, "name": "Sample Product", "price": 29.99, "stock": 100}
        else:
            return {
                "id": 789,
                "name": "Generic Resource",
                "status": "active",
                "timestamp": "2024-01-01T00:00:00Z",
            }


class APISearchEngine:
    def __init__(self):
        # Get environment variables
        self.pinecone_api_key = os.getenv("PINECONE_API_KEY")
        self.pinecone_environment = os.getenv("PINECONE_ENVIRONMENT")
        self.index_name = os.getenv("PINECONE_INDEX_NAME")

        if (
            not self.pinecone_api_key
            or not self.pinecone_environment
            or not self.index_name
        ):
            raise ValueError(
                "Environment variables PINECONE_API_KEY, PINECONE_ENVIRONMENT and PINECONE_INDEX_NAME are required"
            )

        # Initialize Pinecone
        pc = Pinecone(api_key=self.pinecone_api_key)

        # Initialize or get Pinecone index
        if self.index_name not in pc.list_indexes().names():
            pc.create_index(
                name=self.index_name,
                dimension=384,  # dimension of all-MiniLM-L6-v2 embeddings
                metric="dotproduct",  # Changed from cosine to dotproduct to support hybrid search
                spec=ServerlessSpec(cloud="aws", region=self.pinecone_environment),
            )

        # Initialize the embedding model
        self.model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
        self.index = pc.Index(self.index_name)

        self.query_understanding = QueryUnderstanding()
        self.enrichment = APIEnrichment()

    def load_api_contracts(self, api_dir: str) -> List[Dict[str, Any]]:
        """Load all API contracts from YAML files in the specified directory and subdirectories"""
        api_contracts = []

        print(f"\nStarting API contracts loading from directory: {api_dir}")

        for root, dirs, files in os.walk(api_dir):
            print(f"\nProcessing directory: {root}")

            # Filter YAML files only
            yaml_files = [f for f in files if f.endswith((".yaml", ".yml"))]

            for yaml_file in yaml_files:
                file_path = os.path.join(root, yaml_file)
                print(f"Loading file: {file_path}")

                try:
                    with open(file_path, "r", encoding="utf-8") as f:
                        contract = yaml.safe_load(f)
                        if contract:  # Check if file is not empty
                            api_contracts.append({
                                "file": os.path.relpath(file_path, api_dir),
                                "contract": contract,
                            })
                            print(f"✓ File {yaml_file} loaded successfully")
                        else:
                            print(f"✗ File {yaml_file} is empty or malformed")
                except Exception as e:
                    print(f" Error loading {yaml_file}: {str(e)}")

        print(f"\nTotal of {len(api_contracts)} API contracts loaded")

        # Debug: show loaded files
        if api_contracts:
            print("\nLoaded files:")
            for contract in api_contracts:
                print(f"- {contract['file']}")
        else:
            print("\nNo YAML files found!")
            print(f"Directory {api_dir} contents:")
            for root, dirs, files in os.walk(api_dir):
                print(f"\nDirectory: {root}")
                print("Files:", files)
                print("Subdirectories:", dirs)

        return api_contracts

    def extract_endpoints(
        self, contracts: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Extract and enrich endpoints from API contracts"""
        endpoints = []

        for contract in contracts:
            filename = contract["file"]
            api_spec = contract["contract"]

            # Extract API-level information
            api_info = api_spec.get("info", {})
            api_title = api_info.get("title", os.path.basename(filename))
            api_version = api_info.get("version", "1.0.0")
            api_description = api_info.get("description", "")

            # Extract contact information
            contact = api_info.get("contact", {})
            contact_name = contact.get("name", "")
            contact_email = contact.get("email", "")

            # Extract license information
            license_info = api_info.get("license", {})
            license_name = license_info.get("name", "")
            license_url = license_info.get("url", "")

            # Extract server information
            servers = api_spec.get("servers", [])
            base_path = ""
            host = "api.example.com"
            if servers:
                server_url = servers[0].get("url", "")
                if server_url:
                    parsed = urlparse(server_url)
                    host = parsed.netloc or host
                    base_path = parsed.path or base_path

            # Extract tag descriptions
            tags_info = api_spec.get("tags", [])
            tag_descriptions = {
                tag["name"]: tag.get("description", "")
                for tag in tags_info
                if "name" in tag
            }

            # Process each path
            paths = api_spec.get("paths", {})
            for path, path_info in paths.items():
                # Get path-level parameters
                path_parameters = path_info.get("parameters", [])

                # Process each operation (GET, POST, etc.)
                for method, operation in path_info.items():
                    if method == "parameters":
                        continue

                    # Combine path and operation parameters
                    parameters = path_parameters.copy()
                    parameters.extend(operation.get("parameters", []))

                    # Format parameters
                    formatted_params = []
                    for param in parameters:
                        if isinstance(param, dict):
                            formatted_params.append({
                                "name": param.get("name", ""),
                                "in": param.get("in", ""),
                                "description": param.get("description", ""),
                                "required": param.get("required", False),
                                "type": param.get("schema", {}).get("type", "string"),
                            })

                    # Format responses
                    formatted_responses = []
                    for code, response in operation.get("responses", {}).items():
                        formatted_responses.append({
                            "code": code,
                            "description": response.get("description", ""),
                            "schema": response.get("schema", {}),
                            "examples": response.get("examples", {}),
                        })

                    # Extract operation tags
                    op_tags = operation.get("tags", [])

                    # Create endpoint metadata
                    endpoint = {
                        "api": api_title,
                        "api_version": api_version,
                        "api_description": api_description,
                        "file": filename,
                        "host": host,
                        "base_path": base_path,
                        "path": path,
                        "method": method.upper(),
                        "summary": operation.get("summary", ""),
                        "description": operation.get("description", ""),
                        "operationId": operation.get("operationId", ""),
                        "tags": op_tags,
                        "tag_descriptions": [
                            tag_descriptions.get(tag, "") for tag in op_tags
                        ],
                        "parameters": formatted_params,
                        "responses": formatted_responses,
                        "security_schemes": operation.get("security", []),
                        "contact": {"name": contact_name, "email": contact_email},
                        "license": {"name": license_name, "url": license_url},
                        "deprecated": operation.get("deprecated", False),
                    }

                    # Enrich endpoint metadata using LLM
                    enriched_endpoint = self.enrichment.enrich_endpoint(endpoint)

                    # Create rich searchable text
                    searchable_text = f"""
                    API: {enriched_endpoint["api"]} v{enriched_endpoint["api_version"]}
                    {enriched_endpoint["api_description"]}
                    
                    Endpoint: {enriched_endpoint["method"]} {enriched_endpoint["path"]}
                    Summary: {enriched_endpoint["summary"]}
                    Description: {enriched_endpoint["description"]}
                    
                    Tags: {" ".join(enriched_endpoint["tags"])}
                    Tag Descriptions: {" ".join(enriched_endpoint["tag_descriptions"])}
                    
                    Parameters:
                    {" ".join(f"{p['name']} ({p['in']}): {p['description']}" for p in enriched_endpoint["parameters"])}
                    
                    Responses:
                    {" ".join(f"{r['code']}: {r['description']}" for r in enriched_endpoint["responses"])}
                    
                    Use Cases:
                    {" ".join(enriched_endpoint.get("use_cases", []))}
                    
                    Business Domains:
                    {" ".join(enriched_endpoint.get("business_domains", []))}
                    
                    Integration Tips:
                    {" ".join(enriched_endpoint.get("integration_tips", []))}
                    
                    Best Practices:
                    {" ".join(enriched_endpoint.get("best_practices", []))}
                    """

                    enriched_endpoint["searchable_text"] = " ".join(
                        searchable_text.split()
                    )
                    endpoints.append(enriched_endpoint)

        return endpoints

    def index_endpoints(self, endpoints: List[Dict[str, Any]]):
        """Index endpoints with enriched metadata"""
        batch_size = 100

        # First, let's organize endpoints by API for easy access to API parent information
        apis_info = {}
        for endpoint in endpoints:
            api_key = f"{endpoint['api']}_{endpoint['api_version']}"
            if api_key not in apis_info:
                apis_info[api_key] = {
                    "title": endpoint["api"],
                    "version": endpoint["api_version"],
                    "description": endpoint.get("api_description", ""),
                    "base_path": endpoint.get("base_path", ""),
                    "host": endpoint.get("host", ""),
                    "contact": endpoint.get("contact", {}),
                    "license": endpoint.get("license", {}),
                    "global_tags": endpoint.get("tag_descriptions", []),
                    "endpoints_count": 0,
                    "methods_distribution": {},
                    "tags_distribution": set(),
                }

            # Update API statistics
            apis_info[api_key]["endpoints_count"] += 1
            apis_info[api_key]["methods_distribution"][endpoint["method"]] = (
                apis_info[api_key]["methods_distribution"].get(endpoint["method"], 0)
                + 1
            )
            apis_info[api_key]["tags_distribution"].update(endpoint.get("tags", []))

        for i in range(0, len(endpoints), batch_size):
            batch = endpoints[i : i + batch_size]
            vectors = []

            for idx, endpoint in enumerate(batch):
                # Enrich searchable text with more context
                searchable_text = self._create_enriched_text(endpoint, apis_info)
                vector = self.model.encode(searchable_text)

                # Get API info
                api_key = f"{endpoint['api']}_{endpoint['api_version']}"
                api_info = apis_info[api_key]

                # Calculate endpoint complexity score
                complexity_score = self._calculate_endpoint_complexity(endpoint)

                # Enhanced metadata for better filtering and display
                metadata = {
                    # API-level information
                    "api": endpoint["api"],
                    "api_version": endpoint["api_version"],
                    "api_description": endpoint.get("api_description", ""),
                    "api_base_path": endpoint.get("base_path", ""),
                    "api_total_endpoints": api_info["endpoints_count"],
                    "api_contact_name": api_info["contact"].get("name", ""),
                    "api_contact_email": api_info["contact"].get("email", ""),
                    # Endpoint-specific information
                    "file": endpoint["file"],
                    "path": endpoint["path"],
                    "full_path": f"{endpoint.get('base_path', '')}{endpoint['path']}",
                    "method": endpoint["method"],
                    "summary": endpoint["summary"],
                    "description": endpoint["description"],
                    "operationId": endpoint["operationId"],
                    # Tags and categorization
                    "tags": endpoint.get("tags", []),
                    "category": self._categorize_endpoint(endpoint),
                    "global_tags": api_info["global_tags"],
                    # Technical details
                    "parameters_count": len(endpoint.get("parameters", [])),
                    "required_parameters": sum(
                        1
                        for p in endpoint.get("parameters", [])
                        if p.get("required", False)
                    ),
                    "query_parameters": sum(
                        1
                        for p in endpoint.get("parameters", [])
                        if p.get("in") == "query"
                    ),
                    "path_parameters": sum(
                        1
                        for p in endpoint.get("parameters", [])
                        if p.get("in") == "path"
                    ),
                    "body_parameters": sum(
                        1
                        for p in endpoint.get("parameters", [])
                        if p.get("in") == "body"
                    ),
                    # Security and authentication
                    "has_auth": any(
                        p.get("in") == "header" and "auth" in p.get("name", "").lower()
                        for p in endpoint.get("parameters", [])
                    ),
                    "security_schemes": endpoint.get("security", []),
                    # Response information
                    "response_codes": [
                        r["code"] for r in endpoint.get("responses", [])
                    ],
                    "success_response": any(
                        r["code"] in ["200", "201", "204"]
                        for r in endpoint.get("responses", [])
                    ),
                    "supports_pagination": any(
                        "page" in p.get("name", "").lower()
                        or "limit" in p.get("name", "").lower()
                        for p in endpoint.get("parameters", [])
                    ),
                    # Status and metrics
                    "deprecated": endpoint.get("deprecated", False),
                    "complexity_score": complexity_score,
                    "has_examples": any(
                        r.get("examples") for r in endpoint.get("responses", [])
                    ),
                    "has_schema": any(
                        r.get("schema") for r in endpoint.get("responses", [])
                    ),
                }

                vectors.append((f"endpoint_{i + idx}", vector.tolist(), metadata))

            self.index.upsert(vectors=vectors)

    def _create_enriched_text(
        self, endpoint: Dict[str, Any], apis_info: Dict[str, Any]
    ) -> str:
        """Create enriched searchable text for better semantic search"""
        api_key = f"{endpoint['api']}_{endpoint['api_version']}"
        api_info = apis_info[api_key]

        sections = [
            # API Context
            f"API: {endpoint['api']} version {endpoint['api_version']}",
            f"API Description: {api_info.get('description', '')}",
            f"API Base Path: {api_info.get('base_path', '')}",
            # Endpoint Basic Info
            f"Endpoint: {endpoint['method']} {endpoint['path']}",
            f"Summary: {endpoint['summary']}",
            f"Description: {endpoint['description']}",
            # Tags and Categories
            f"Tags: {' '.join(endpoint.get('tags', []))}",
            f"Tag Descriptions: {' '.join(api_info.get('global_tags', []))}",
            f"Category: {self._categorize_endpoint(endpoint)}",
            # Parameters
            "Parameters: "
            + " ".join(
                f"{p['name']} ({p['in']}): {p['description']} {'(required)' if p.get('required') else '(optional)'}"
                for p in endpoint.get("parameters", [])
            ),
            # Response Information
            "Responses: "
            + " ".join(
                f"{r['code']}: {r['description']}"
                for r in endpoint.get("responses", [])
            ),
            # Examples and Schemas (if available)
            "Examples: "
            + " ".join(
                f"{r['code']}: {str(r.get('examples', ''))}"
                for r in endpoint.get("responses", [])
                if r.get("examples")
            ),
            # Security Information
            "Security: " + " ".join(str(s) for s in endpoint.get("security", [])),
            # Contact Information
            f"Contact: {api_info['contact'].get('name', '')} {api_info['contact'].get('email', '')}",
        ]

        return " ".join(filter(None, sections))

    def _calculate_endpoint_complexity(self, endpoint: Dict[str, Any]) -> float:
        """Calculate endpoint complexity score based on various factors"""
        score = 0.0

        # Parameter complexity
        params = endpoint.get("parameters", [])
        score += len(params) * 0.2  # Base score for each parameter
        score += (
            sum(1 for p in params if p.get("required", False)) * 0.3
        )  # Additional score for required parameters
        score += (
            sum(1 for p in params if p.get("in") == "body") * 0.4
        )  # Body parameters are more complex

        # Response complexity
        responses = endpoint.get("responses", [])
        score += len(responses) * 0.2  # Base score for each response type
        score += (
            sum(1 for r in responses if r.get("schema")) * 0.3
        )  # Complex response schemas

        # Security complexity
        if endpoint.get("security"):
            score += 1.0

        # Path complexity
        path = endpoint.get("path", "")
        score += path.count("{") * 0.3  # Path parameters add complexity
        score += path.count("/") * 0.1  # Deeper paths are more complex

        return round(score, 2)

    def _categorize_endpoint(self, endpoint: Dict[str, Any]) -> str:
        """Categorize endpoint based on its characteristics"""
        path = endpoint["path"].lower()
        method = endpoint["method"].upper()

        if method == "GET":
            if "{" in path:
                return "retrieve_single"
            return "list"
        elif method == "POST":
            return "create"
        elif method == "PUT":
            return "update"
        elif method == "DELETE":
            return "delete"
        elif method == "PATCH":
            return "partial_update"

        return "other"

    def search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Search for endpoints matching the query"""
        # Generate embedding for the query
        query_embedding = self.model.encode(query)

        # Search in Pinecone (without hybrid search for now)
        results = self.index.query(
            vector=query_embedding.tolist(), top_k=top_k, include_metadata=True
        )

        return self._format_results(results)

    def _format_results(self, results: Dict) -> List[Dict[str, Any]]:
        """Format search results"""
        formatted_results = []
        for match in results["matches"]:
            metadata = match.get("metadata", {})
            formatted_result = {
                "score": match.get("score", 0.0),
                "api": metadata.get("api", "Unknown API"),
                "api_version": metadata.get("api_version", ""),
                "file": metadata.get("file", "Unknown File"),
                "path": metadata.get("path", "Unknown Path"),
                "method": metadata.get("method", "Unknown Method"),
                "summary": metadata.get("summary", ""),
                "description": metadata.get("description", ""),
                "operationId": metadata.get("operationId", ""),
                "tags": metadata.get("tags", []),
                "parameters": metadata.get("parameters", []),
                "responses": metadata.get("responses", []),
                "contact": metadata.get("contact", {}),
                "host": metadata.get("host", ""),
            }

            # Add usage information
            formatted_result["curl_example"] = self.generate_curl_example(
                formatted_result
            )
            formatted_result["usage_example"] = self.generate_usage_example(
                formatted_result
            )

            formatted_results.append(formatted_result)

        return formatted_results

    def generate_curl_example(self, endpoint: Dict[str, Any]) -> str:
        """Generate a cURL example for the endpoint"""
        host = endpoint.get("host", "api.example.com")
        path = endpoint.get("path", "/")
        method = endpoint.get("method", "GET")

        curl = f"curl -X {method} https://{host}{path}"

        # Add parameters as examples
        for param in endpoint.get("parameters", []):
            if param.get("in") == "header":
                curl += f" -H '{param['name']}: value'"
            elif param.get("in") == "query":
                if "?" not in curl:
                    curl += "?"
                else:
                    curl += "&"
                curl += f"{param['name']}=value"

        return curl

    def generate_usage_example(self, endpoint: Dict[str, Any]) -> str:
        """Generate a Python usage example for the endpoint"""
        method = endpoint.get("method", "GET").lower()
        path = endpoint.get("path", "/")

        # Use double curly braces to escape f-string syntax in the example code
        example = f"""import requests

response = requests.{method}(
    "https://{endpoint.get("host", "api.example.com")}{path}",
    headers={{
        "Content-Type": "application/json"
    }}
)

if response.status_code == 200:
    data = response.json()
    print(data)
else:
    print(f"Error: {{response.status_code}}")
"""
        return example

    def initialize_with_contracts(self, api_dir: str, force_update: bool = False):
        """
        Initialize the search engine with API contracts from a directory

        Args:
            api_dir: Directory containing API contracts
            force_update: If True, forces reindexing even if index exists
        """
        print(f"\nStarting initialization with API contracts from directory: {api_dir}")

        # Check current index state
        try:
            stats = self.index.describe_index_stats()
            total_vectors = stats.get("total_vector_count", 0)
            print("\nCurrent index statistics:")
            print(f"Total vectors: {total_vectors}")

            if total_vectors > 0 and not force_update:
                print(
                    "\nIndex already contains data. Use force_update=True to force update."
                )
                print(
                    "Example: search_engine.initialize_with_contracts('assets/apis', force_update=True)"
                )
                return

            if force_update:
                print("\nForcing index update...")
                # Optional: Clear existing index
                # self.index.delete(delete_all=True)

        except Exception as e:
            print(f"Error checking index statistics: {str(e)}")
            return

        print("\nLoading and indexing API contracts...")

        # Load API contracts
        contracts = self.load_api_contracts(api_dir)
        if not contracts:
            print("No API contracts found!")
            return

        print("\nContracts found:")
        for contract in contracts:
            print(f"- {contract['file']}")

        # Extract endpoints
        endpoints = self.extract_endpoints(contracts)
        if not endpoints:
            print("No endpoints extracted from contracts!")
            return

        # Index endpoints
        self.index_endpoints(endpoints)

        # Check final index state
        final_stats = self.index.describe_index_stats()
        final_vectors = final_stats.get("total_vector_count", 0)
        print("\nIndexing completed!")
        print(f"Total vectors after indexing: {final_vectors}")
        if total_vectors > 0:
            print(f"Difference: {final_vectors - total_vectors} new vectors")

    def search_with_filters(
        self,
        query: str,
        method: Optional[str] = None,
        tags: Optional[List[str]] = None,
        min_score: float = 0.5,
        filter_conditions: Optional[Dict[str, Any]] = None,
        top_k: int = 5,
    ) -> List[Dict[str, Any]]:
        """
        Search for endpoints with advanced filtering options

        Args:
            query: Search query string
            method: Filter by HTTP method (GET, POST, etc)
            tags: Filter by API tags
            min_score: Minimum similarity score (0-1)
            filter_conditions: Additional filter conditions for Pinecone
            top_k: Number of results to return

        Returns:
            List of matching endpoints with metadata
        """
        # Generate embedding for the query
        query_embedding = self.model.encode(query).tolist()

        # Build filter conditions
        final_filter = {}
        if filter_conditions:
            final_filter.update(filter_conditions)

        if method:
            final_filter["method"] = method

        if tags:
            final_filter["tags"] = {"$in": tags}

        # Search in Pinecone with filters
        results = self.index.query(
            vector=query_embedding,
            top_k=top_k,
            include_metadata=True,
            filter=final_filter,
            score_threshold=min_score,
        )

        # Format results
        return self._format_results(results)

    def search_with_related(
        self,
        query: str,
        include_dependencies: bool = True,
        include_alternatives: bool = True,
    ) -> List[Dict[str, Any]]:
        """
        Search for related APIs

        Args:
            query: Search query
            include_dependencies: Include dependencies
            include_alternatives: Include alternative APIs
        """
        # Generate embedding for the query
        query_embedding = self.model.encode(query)

        # Search in Pinecone (without hybrid search for now)
        results = self.index.query(
            vector=query_embedding.tolist(), top_k=5, include_metadata=True
        )

        return self._format_results(results)

    def _calculate_update_metrics(self, endpoint: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate update metrics for an endpoint"""
        # In a real scenario, this would come from version control history
        current_time = datetime.now()

        # Simulate last update (random date within last 2 years)
        days_ago = random.randint(0, 730)  # Up to 2 years ago
        last_updated = current_time - timedelta(days=days_ago)

        return {
            "last_updated": last_updated.isoformat(),  # Keep ISO format for display
            "last_updated_timestamp": last_updated.timestamp(),  # Add timestamp for filtering
            "update_frequency": random.randint(1, 12),  # Updates per year
        }

    def smart_search(
        self, natural_query: str, top_k: int = 5
    ) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        """
        Perform smart search using natural language understanding

        Returns:
            Tuple of (results, original_intent)
        """
        # Parse natural query
        structured_query = self.query_understanding.parse_natural_query(natural_query)

        # Build filter conditions
        filter_conditions = {}

        if structured_query["filters"]:
            if structured_query["filters"]["method"]:
                filter_conditions["method"] = structured_query["filters"]["method"]

            if structured_query["filters"]["min_availability"]:
                filter_conditions["availability"] = {
                    "$gte": structured_query["filters"]["min_availability"]
                }

            if structured_query["filters"]["max_latency"]:
                filter_conditions["latency_ms"] = {
                    "$lte": structured_query["filters"]["max_latency"]
                }

            if structured_query["filters"]["max_error_rate"]:
                filter_conditions["error_rate"] = {
                    "$lte": structured_query["filters"]["max_error_rate"]
                }

            if structured_query["filters"]["lifecycle_state"]:
                filter_conditions["lifecycle_state"] = structured_query["filters"][
                    "lifecycle_state"
                ]

        # Perform search
        results = self.search_with_filters(
            query=structured_query["query"],
            filter_conditions=filter_conditions,
            top_k=top_k,
        )

        # Sort results if specified
        if structured_query["sort_by"]:
            reverse = structured_query["sort_order"] == "desc"
            results.sort(
                key=lambda x: x.get(structured_query["sort_by"], 0), reverse=reverse
            )

        return results, structured_query


if __name__ == "__main__":
    # Parse command line arguments
    args = parse_args()

    # Initialize the search engine
    search_engine = APISearchEngine()

    # Handle indexing based on arguments
    if not args.skip_index:
        print(f"\nInitializing with API contracts from directory: {args.api_dir}")
        search_engine.initialize_with_contracts(
            args.api_dir, force_update=args.force_reindex
        )
    else:
        print("\nSkipping indexing, using existing index...")

    # If query is provided, do a single search
    if args.query:
        print(f"\nSearching for: {args.query}")

        if args.natural:
            # Use natural language processing
            results, original_intent = search_engine.smart_search(
                natural_query=args.query, top_k=args.top_k
            )
            display_results(results, "Search Results", args.query, original_intent)
        else:
            # Use traditional search with filters
            results = search_engine.search_with_filters(
                query=args.query,
                method=args.method,
                min_score=args.min_score,
                top_k=args.top_k,
            )
            display_results(results, "Search Results")
    else:
        # Run example natural language searches
        print("\nExample 1: High Availability Search")
        results, intent = search_engine.smart_search(
            "find APIs with highest availability"
        )
        display_results(
            results,
            "High Availability APIs",
            "find APIs with highest availability",
            intent,
        )

        print("\nExample 2: Low Latency Search")
        results = search_engine.smart_search("show me the fastest authentication APIs")
        display_results(results, "Fast Authentication APIs")

        print("\nExample 3: Stable APIs Search")
        results = search_engine.smart_search(
            "find stable user management APIs with good reliability"
        )
        display_results(results, "Stable & Reliable APIs")

        print("\nExample 4: Recent Updates Search")
        results = search_engine.smart_search("show recently updated payment APIs")
        display_results(results, "Recently Updated Payment APIs")

        print("\nExample 5: Complex Requirements")
        results = search_engine.smart_search(
            "find GET APIs for user data with less than 50ms latency and 99.9% availability"
        )
        display_results(results, "Optimized User APIs")
