"""Plexure API Search package."""

from plexure_api_search.searcher import SearchEngine
from plexure_api_search.indexer import APIIndexer
from plexure_api_search.config import Config, ConfigManager
from plexure_api_search.cache import SearchCache, EmbeddingCache
from plexure_api_search.monitoring import Logger, MetricsCollector
from plexure_api_search.validation import DataValidator, QualityMetrics
from plexure_api_search.metrics import MetricsCalculator, QualityMonitor

__version__ = "0.1.0"

__all__ = [
    "SearchEngine",
    "APIIndexer",
    "Config",
    "ConfigManager",
    "SearchCache",
    "EmbeddingCache",
    "Logger",
    "MetricsCollector",
    "DataValidator",
    "QualityMetrics",
    "MetricsCalculator",
    "QualityMonitor",
]
