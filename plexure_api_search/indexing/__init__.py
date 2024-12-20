"""
API Contract Indexing Package for Plexure API Search

This package provides functionality for indexing API contracts in the Plexure API Search system.
It includes modules for parsing, processing, and indexing API contracts to enable efficient
semantic search over API endpoints.

Key Components:
1. Parser Module:
   - Contract parsing
   - Format detection
   - Schema validation
   - Reference resolution

2. Indexer Module:
   - Vector generation
   - Index management
   - Batch processing
   - Cache control

3. Utilities:
   - File handling
   - Error handling
   - Progress tracking
   - Resource management

The package supports:
- Multiple API contract formats
- Efficient vector storage
- Metadata management
- Cache optimization
- Error recovery
- Progress monitoring

Example Usage:
    from plexure_api_search.indexing import indexer, parser

    # Parse contract
    endpoints = parser.parse_contract("api.yaml")

    # Index endpoints
    indexer.index_endpoints(endpoints)

Package Structure:
- indexer.py: Core indexing functionality
- parser.py: Contract parsing and validation
- __init__.py: Package initialization
"""

from .indexer import indexer
from .parser import APIParser

__all__ = ["indexer", "APIParser"]
