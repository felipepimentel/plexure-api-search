"""Constants module for the Plexure API Search tool."""

import os
from pathlib import Path

from dotenv import load_dotenv
from rich.console import Console

console = Console()

# Load environment variables from .env file
env_path = Path(__file__).parent.parent / ".env"
if env_path.exists():
    load_dotenv(env_path)
    console.print(f"[green]Loaded environment variables from {env_path}[/]")
else:
    console.print(f"[yellow]Warning: .env file not found at {env_path}[/]")

# Debug: Print environment variables (without exposing sensitive values)
console.print("\n[bold]Environment Variables Status:[/]")
for var in [
    "OPENROUTER_API_KEY",
    "PINECONE_API_KEY",
    "PINECONE_ENVIRONMENT",
    "PINECONE_INDEX_NAME",
]:
    value = os.getenv(var)
    if value:
        console.print(f"[green]✓[/] {var}: {'*' * 8}{value[-4:] if value else ''}")
    else:
        console.print(f"[red]✗[/] {var}: Not set")

# API and Environment Configuration
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_ENVIRONMENT = os.getenv("PINECONE_ENVIRONMENT", "gcp-starter")
PINECONE_INDEX_NAME = os.getenv(
    "PINECONE_INDEX_NAME", "plexure-api-search-small-dotproduct"
)
PINECONE_DIMENSION = 384  # Dimension for all-MiniLM-L6-v2

# Model Configuration
SENTENCE_TRANSFORMER_MODEL = "all-MiniLM-L6-v2"
LLM_MODEL = "mistralai/mistral-7b-instruct"
LLM_TEMPERATURE = {
    "query_analysis": 0.1,
    "relevance_explanation": 0.3,
    "search_summary": 0.3,
    "metadata_enrichment": 0.1,
}

# Default Values
DEFAULT_API_DIR = "assets/apis"
DEFAULT_TOP_K = 5
DEFAULT_BATCH_SIZE = 100

# LLM Prompts
QUERY_ANALYSIS_PROMPT = """You are an API search expert. Your task is to analyze API search queries and extract structured information to improve search results.

TASK:
Analyze the query to extract search parameters and intent. Focus on technical aspects that can improve API search.

RESPONSE FORMAT:
Respond with a JSON object containing:
{
    "intent": {
        "primary_goal": "What user wants to achieve",
        "secondary_goals": ["Additional objectives"],
        "context": "Technical context of the request"
    },
    "search_parameters": {
        "version": "Specific version mentioned or null",
        "method": "HTTP method if mentioned or null",
        "path_contains": ["Key path segments"],
        "features": ["Specific features mentioned"],
        "metadata_filters": {
            "has_auth": boolean if auth is relevant,
            "deprecated": boolean if deprecation is relevant,
            "has_examples": boolean if examples are relevant
        }
    },
    "enhanced_query": "Technical search query optimized for API search",
    "ranking_priorities": [
        "List of aspects to prioritize in ranking",
        "e.g. version_match, path_relevance, description_match"
    ]
}"""

RELEVANCE_EXPLANATION_PROMPT = """You are an API expert explaining why a search result is relevant to a user's query.

TASK:
Explain why this API endpoint matches the user's search criteria and how well it satisfies their needs.

GUIDELINES:
1. Focus on specific matches between the query and the endpoint
2. Highlight key features that align with user's requirements
3. Note any potential limitations or considerations
4. Reference OpenAPI specifications when relevant
5. Keep explanation concise but informative

FORMAT:
Write a 2-3 sentence explanation that:
1. States the main reason for relevance
2. Highlights specific matching features
3. Notes any important considerations

Example:
"This API endpoint /users/{id} (GET method, version 2.0) is highly relevant as it provides direct user data access matching your authentication requirements. It supports both basic and OAuth2 authentication, includes comprehensive examples, and follows the latest REST best practices. However, note that this endpoint requires additional scopes for sensitive data access."

Keep technical but clear. Use markdown for formatting key terms."""

SEARCH_SUMMARY_PROMPT = """You are an API expert providing search result summaries. Your task is to create a clear, concise summary of API search results.

GUIDELINES:
1. Start with key statistics (total results, version distribution)
2. Highlight most relevant findings
3. Mention any patterns or themes
4. Note important characteristics
5. Suggest next steps if relevant

FORMAT YOUR RESPONSE IN SECTIONS:
📊 OVERVIEW
[Key statistics and high-level summary]

🎯 MOST RELEVANT
[Top 1-2 most relevant results and why]

🔍 PATTERNS
[Common patterns or themes found]

💡 SUGGESTIONS
[Brief suggestions for refining search if needed]

Keep each section brief but informative. Use technical language appropriately."""

# HTTP Method Colors
METHOD_COLORS = {
    "GET": "green",
    "POST": "blue",
    "PUT": "yellow",
    "DELETE": "red",
    "PATCH": "magenta",
    "HEAD": "cyan",
    "OPTIONS": "white",
}

# Score Adjustments
SCORE_ADJUSTMENTS = {
    "version_match": 0.3,
    "method_match": 0.2,
    "path_match": 0.15,
    "feature_match": 0.1,
    "metadata_match": 0.1,
}

# Feature Icons
FEATURE_ICONS = {
    "has_auth": {"icon": "🔒", "style": "bold blue", "label": "Auth Required"},
    "has_examples": {"icon": "📝", "style": "bold green", "label": "Has Examples"},
    "supports_pagination": {
        "icon": "📄",
        "style": "bold yellow",
        "label": "Supports Pagination",
    },
    "deprecated": {"icon": "⚠️", "style": "bold red", "label": "Deprecated"},
}

# HTTP Headers
HTTP_HEADERS = {
    "HTTP-Referer": "https://github.com/pimentel/plexure-api-search",
    "X-Title": "Plexure API Search",
}

# Valid HTTP Methods
VALID_HTTP_METHODS = ["GET", "POST", "PUT", "DELETE", "PATCH", "HEAD", "OPTIONS"]

# Table Formatting
TABLE_SETTINGS = {"key_column_width": 15, "key_column_style": "bold cyan"}

# Error Messages
ERROR_MESSAGES = {
    "no_results": "[yellow]No matching endpoints found.[/]",
    "query_analysis_failed": "[yellow]Warning: Query analysis failed: {error}[/]",
    "relevance_failed": "[yellow]Warning: Relevance explanation failed: {error}[/]",
    "summary_failed": "[yellow]Warning: Summary generation failed: {error}[/]",
    "search_failed": "[red]Error during search: {error}[/]",
    "api_key_missing": "[red]Error: {key} environment variable is not set[/]",
}
