"""
Plexure API Search - Semantic Search Engine for API Contracts

This package provides a powerful semantic search engine for API contracts using advanced NLP techniques
and vector embeddings. It enables natural language querying of API endpoints, making it easier to
find and understand API functionality.

Key Features:
- Semantic search over API endpoints using state-of-the-art language models
- High-performance vector similarity search with FAISS
- Efficient vector storage and retrieval
- Automatic metadata association and retrieval
- Configurable similarity thresholds and search parameters
- Built-in monitoring and metrics collection
- Efficient caching system for improved performance
- Support for multiple API contract formats (OpenAPI/Swagger)

Author: Your Team
Version: 1.0.0
License: MIT
"""

import logging
from importlib.metadata import version

__version__ = version("plexure-api-search")

logger = logging.getLogger(__name__)

__all__ = ["__version__"]
