"""Constants module for API search functionality."""

import os
from typing import Dict, Any

# Environment variables and defaults
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_ENVIRONMENT = os.getenv("PINECONE_ENVIRONMENT", "gcp-starter")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "api-search")
PINECONE_DIMENSION = 384  # SentenceTransformer default dimension

# API endpoints
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"

# Default values
DEFAULT_API_DIR = "api_contracts"
DEFAULT_TOP_K = 10
DEFAULT_BATCH_SIZE = 100

# Model settings
SENTENCE_TRANSFORMER_MODEL = "all-MiniLM-L6-v2"
LLM_MODEL = "openai/gpt-3.5-turbo"

# Temperature settings for different LLM tasks
LLM_TEMPERATURE = {
    "query_analysis": 0.1,
    "metadata_enrichment": 0.2,
    "relevance_explanation": 0.3,
    "search_summary": 0.4
}

# HTTP headers
HTTP_HEADERS = {
    "HTTP-Referer": "https://github.com/cursor-ai",
    "X-Title": "Cursor AI"
}

# Score adjustments for ranking
SCORE_ADJUSTMENTS = {
    "version_match": {
        "exact": 1.0,
        "major": 0.8,
        "minor": 0.6,
        "patch": 0.4
    },
    "method_match": 1.0,
    "path_match": {
        "exact": 1.0,
        "partial": 0.8,
        "similar": 0.6
    },
    "feature_match": {
        "exact": 1.0,
        "similar": 0.8
    },
    "tag_match": {
        "exact": 1.0,
        "similar": 0.8
    },
    "text_similarity": {
        "high": 0.9,
        "medium": 0.7,
        "low": 0.5
    }
}

# Feature icons and styles
FEATURE_ICONS = {
    "has_auth": {
        "icon": "🔒",
        "label": "Auth Required",
        "style": "bold red"
    },
    "has_examples": {
        "icon": "📝",
        "label": "Has Examples",
        "style": "bold green"
    },
    "supports_pagination": {
        "icon": "📄",
        "label": "Paginated",
        "style": "bold blue"
    },
    "deprecated": {
        "icon": "⚠️",
        "label": "Deprecated",
        "style": "bold yellow"
    }
}

# HTTP method colors
METHOD_COLORS = {
    "GET": "green",
    "POST": "blue",
    "PUT": "yellow",
    "DELETE": "red",
    "PATCH": "magenta"
}

# Table display settings
TABLE_SETTINGS = {
    "key_column_style": "bold cyan",
    "key_column_width": 15,
    "value_column_width": 80,
    "padding": (0, 1),
    "box": None
}

# LLM prompts
QUERY_ANALYSIS_PROMPT = """You are an API search expert. Analyze the search query and context to extract structured information.

Your task is to:
1. Understand the user's intent
2. Extract key search parameters
3. Identify ranking priorities
4. Expand and enhance the query

The input will be a JSON object containing:
- query: The original search query
- expanded_versions: List of version variations
- expanded_queries: List of query variations
- analysis: Basic query analysis

Respond with a JSON object containing:
{
    "enhanced_query": "Improved search query",
    "search_parameters": {
        "version": "Extracted version",
        "method": "HTTP method",
        "features": {
            "has_auth": bool,
            "has_examples": bool,
            "supports_pagination": bool,
            "deprecated": bool
        },
        "path_terms": ["relevant", "path", "terms"],
        "tags": ["relevant", "tags"],
        "metadata_filters": {},
        "version_variations": ["version", "variations"],
        "query_variations": ["query", "variations"]
    },
    "ranking_priorities": [
        {"factor": "version", "weight": 0.3},
        {"factor": "method", "weight": 0.2},
        ...
    ]
}"""

RELEVANCE_EXPLANATION_PROMPT = """You are an API documentation expert. Explain why a search result is relevant to the user's query.

The input will be a JSON object containing:
- query: The original search query
- metadata: The API endpoint metadata
- search_parameters: The search parameters used

Generate a concise, natural language explanation focusing on:
1. Version compatibility
2. Feature matches
3. Path relevance
4. Method appropriateness
5. Tag matches

Keep the explanation clear and user-friendly, highlighting the most important matching factors first."""

SEARCH_SUMMARY_PROMPT = """You are an API search expert. Summarize the search results in a clear, informative way.

The input will be a JSON object containing:
- total_results: Number of results found
- top_results: List of top 3 results with basic info
- metrics: Detailed metrics about the results

Generate a concise summary that includes:
1. Overview of results found
2. Distribution of versions and methods
3. Common features and tags
4. Notable patterns or clusters
5. Recommendations for the user

Keep the summary user-friendly and highlight the most relevant information first."""

# Error messages
ERROR_MESSAGES = {
    "api_key_missing": "Missing required API key: {key}",
    "query_analysis_failed": "Failed to analyze query: {error}",
    "search_failed": "Search operation failed: {error}",
    "no_results": "[yellow]No matching endpoints found.[/]",
    "relevance_failed": "Failed to generate relevance explanation: {error}",
    "summary_failed": "Failed to generate search summary: {error}"
}
