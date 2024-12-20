"""
Semantic Search Package for Plexure API Search

This package provides semantic search functionality for the Plexure API Search system.
It enables natural language querying of API endpoints using advanced NLP techniques
and vector similarity search.

Key Components:
1. Searcher Module:
   - Query processing
   - Vector similarity search
   - Result ranking
   - Metadata enrichment

2. Query Processing:
   - Query cleaning
   - Query expansion
   - Query vectorization
   - Query optimization

3. Result Processing:
   - Score calculation
   - Result filtering
   - Metadata formatting
   - Cache management

The package supports:
- Natural language queries
- Semantic understanding
- Fuzzy matching
- Result ranking
- Score thresholding
- Cache optimization

Example Usage:
    from plexure_api_search.search import searcher

    # Perform search
    results = searcher.search(
        query="find authentication endpoints",
        limit=10
    )

    # Process results
    for result in results:
        print(f"Score: {result.score}")
        print(f"Endpoint: {result.endpoint}")

Package Structure:
- searcher.py: Core search functionality
- __init__.py: Package initialization
"""

from .searcher import searcher

__all__ = ["searcher"]
