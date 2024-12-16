"""Base classes for search strategy composition."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Protocol, Type, Union

import numpy as np
from pydantic import BaseModel

from ...config import config_instance
from ...embedding.embeddings import ModernEmbeddings
from ..storage import endpoint_store

logger = logging.getLogger(__name__)


class SearchResult(BaseModel):
    """Search result with metadata."""

    id: str
    score: float
    method: str
    path: str
    description: str
    api_name: str
    api_version: str
    metadata: Dict[str, Any]
    strategy: str


class SearchStrategy(Protocol):
    """Protocol for search strategies."""

    def search(
        self,
        query: str,
        top_k: int = 10,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[SearchResult]:
        """Execute search strategy.

        Args:
            query: Search query
            top_k: Number of results to return
            filters: Optional filters to apply

        Returns:
            List of search results
        """
        ...


@dataclass
class SearchConfig:
    """Configuration for search strategies."""

    # Core settings
    strategy_name: str
    enabled: bool = True
    weight: float = 1.0
    
    # RAG settings
    rag_enabled: bool = config_instance.rag_enabled
    rag_chunk_size: int = config_instance.rag_chunk_size
    rag_chunk_overlap: int = config_instance.rag_chunk_overlap
    rag_top_k: int = config_instance.rag_top_k
    
    # Hybrid settings
    hybrid_enabled: bool = config_instance.hybrid_search_enabled
    bm25_weight: float = config_instance.bm25_weight
    vector_weight: float = config_instance.vector_weight
    
    # Advanced features
    semantic_cache_enabled: bool = config_instance.enable_semantic_caching
    query_expansion_enabled: bool = config_instance.enable_query_expansion
    max_query_expansions: int = config_instance.max_query_expansions
    
    # Model settings
    cross_encoder_enabled: bool = True
    sparse_dense_enabled: bool = True
    adaptive_retrieval_enabled: bool = True
    
    # Query processing
    query_decomposition_enabled: bool = True
    contextual_compression_enabled: bool = True
    dynamic_expansion_enabled: bool = True
    
    # Indexing features
    incremental_indexing_enabled: bool = True
    semantic_chunking_enabled: bool = True
    hierarchical_indexing_enabled: bool = True
    
    # Advanced features
    api_chaining_enabled: bool = True
    usage_pattern_enabled: bool = True
    semantic_versioning_enabled: bool = True


class BaseSearchStrategy(ABC):
    """Base class for search strategies."""

    def __init__(self, config: Optional[SearchConfig] = None):
        """Initialize strategy.

        Args:
            config: Optional strategy configuration
        """
        self.config = config or SearchConfig(strategy_name=self.__class__.__name__)
        self.embeddings = ModernEmbeddings()
        self.store = endpoint_store

    @abstractmethod
    def search(
        self,
        query: str,
        top_k: int = 10,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[SearchResult]:
        """Execute search strategy.

        Args:
            query: Search query
            top_k: Number of results to return
            filters: Optional filters to apply

        Returns:
            List of search results
        """
        pass

    def _normalize_scores(self, scores: List[float]) -> List[float]:
        """Normalize scores to 0-1 range.

        Args:
            scores: List of scores

        Returns:
            Normalized scores
        """
        if not scores:
            return scores
        min_score = min(scores)
        max_score = max(scores)
        if max_score == min_score:
            return [1.0] * len(scores)
        return [(s - min_score) / (max_score - min_score) for s in scores]

    def _convert_to_search_result(
        self,
        endpoint: "EndpointData",
        score: float,
        strategy: str,
        extra_metadata: Optional[Dict[str, Any]] = None,
    ) -> SearchResult:
        """Convert endpoint data to search result.

        Args:
            endpoint: Endpoint data
            score: Search score
            strategy: Strategy name
            extra_metadata: Optional additional metadata

        Returns:
            Search result
        """
        metadata = {
            **endpoint.metadata,
            **(extra_metadata or {}),
        }
        
        return SearchResult(
            id=endpoint.id,
            score=float(score),
            method=endpoint.method,
            path=endpoint.path,
            description=endpoint.description,
            api_name=endpoint.api_name,
            api_version=endpoint.api_version,
            metadata=metadata,
            strategy=strategy,
        )


class CompositeSearchStrategy(BaseSearchStrategy):
    """Composite search strategy combining multiple strategies."""

    def __init__(
        self,
        strategies: List[BaseSearchStrategy],
        config: Optional[SearchConfig] = None,
    ):
        """Initialize composite strategy.

        Args:
            strategies: List of search strategies
            config: Optional strategy configuration
        """
        super().__init__(config)
        self.strategies = strategies

    def search(
        self,
        query: str,
        top_k: int = 10,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[SearchResult]:
        """Execute all enabled strategies and combine results.

        Args:
            query: Search query
            top_k: Number of results to return
            filters: Optional filters to apply

        Returns:
            Combined search results
        """
        all_results: Dict[str, SearchResult] = {}
        
        # Execute each enabled strategy
        for strategy in self.strategies:
            if not strategy.config.enabled:
                continue
                
            try:
                results = strategy.search(query, top_k=top_k, filters=filters)
                weight = strategy.config.weight
                
                # Combine results with weighting
                for result in results:
                    if result.id not in all_results:
                        result.score *= weight
                        all_results[result.id] = result
                    else:
                        # Update score if new score is higher
                        weighted_score = result.score * weight
                        if weighted_score > all_results[result.id].score:
                            result.score = weighted_score
                            all_results[result.id] = result
                            
            except Exception as e:
                logger.error(f"Strategy {strategy.__class__.__name__} failed: {e}")
                continue

        # Sort by score and return top_k
        results = list(all_results.values())
        results.sort(key=lambda x: x.score, reverse=True)
        return results[:top_k]


class StrategyFactory:
    """Factory for creating search strategies."""

    _strategies: Dict[str, Type[BaseSearchStrategy]] = {}

    @classmethod
    def register(cls, name: str) -> callable:
        """Register strategy class.

        Args:
            name: Strategy name

        Returns:
            Decorator function
        """
        def decorator(strategy_class: Type[BaseSearchStrategy]) -> Type[BaseSearchStrategy]:
            cls._strategies[name] = strategy_class
            return strategy_class
        return decorator

    @classmethod
    def create(
        cls,
        name: str,
        config: Optional[SearchConfig] = None,
    ) -> BaseSearchStrategy:
        """Create strategy instance.

        Args:
            name: Strategy name
            config: Optional strategy configuration

        Returns:
            Strategy instance
        """
        if name not in cls._strategies:
            raise ValueError(f"Unknown strategy: {name}")
            
        strategy_class = cls._strategies[name]
        return strategy_class(config)

    @classmethod
    def create_composite(
        cls,
        strategies: List[str],
        configs: Optional[Dict[str, SearchConfig]] = None,
    ) -> CompositeSearchStrategy:
        """Create composite strategy.

        Args:
            strategies: List of strategy names
            configs: Optional configurations by strategy name

        Returns:
            Composite strategy instance
        """
        configs = configs or {}
        instances = []
        
        for name in strategies:
            config = configs.get(name)
            instances.append(cls.create(name, config))
            
        return CompositeSearchStrategy(instances) 