"""Semantic filtering for search results."""

import logging
from typing import Dict, List, Any, Optional, Set, Union
from datetime import datetime

import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sentence_transformers import SentenceTransformer, util

from ..config import Config
from ..monitoring.events import Event, EventType
from ..monitoring.metrics import MetricsManager
from ..services.base import BaseService
from ..services.events import PublisherService
from .search_models import SearchResult
from .vectorizer import Triple

logger = logging.getLogger(__name__)


class FilterType:
    """Types of semantic filters."""

    SEMANTIC = "semantic"  # Filter by semantic similarity
    CATEGORY = "category"  # Filter by category match
    ATTRIBUTE = "attribute"  # Filter by attribute value
    CONSTRAINT = "constraint"  # Filter by business constraint
    COMPOSITE = "composite"  # Combine multiple filters


class FilterOperator:
    """Operators for filter conditions."""

    AND = "and"
    OR = "or"
    NOT = "not"
    GT = "gt"  # Greater than
    LT = "lt"  # Less than
    EQ = "eq"  # Equal
    NE = "ne"  # Not equal
    CONTAINS = "contains"
    SIMILAR = "similar"


class FilterCondition:
    """Condition for semantic filtering."""

    def __init__(
        self,
        field: str,
        operator: str,
        value: Any,
        threshold: float = 0.7,
        weight: float = 1.0,
    ) -> None:
        """Initialize filter condition.

        Args:
            field: Field to filter on
            operator: Filter operator
            value: Filter value
            threshold: Similarity threshold
            weight: Condition weight
        """
        self.field = field
        self.operator = operator
        self.value = value
        self.threshold = threshold
        self.weight = weight


class SemanticFilter:
    """Filter for semantic search results."""

    def __init__(
        self,
        type: str,
        conditions: List[FilterCondition],
        operator: str = FilterOperator.AND,
        threshold: float = 0.7,
        weight: float = 1.0,
    ) -> None:
        """Initialize semantic filter.

        Args:
            type: Filter type
            conditions: Filter conditions
            operator: Condition operator
            threshold: Filter threshold
            weight: Filter weight
        """
        self.type = type
        self.conditions = conditions
        self.operator = operator
        self.threshold = threshold
        self.weight = weight


class FilterConfig:
    """Configuration for semantic filtering."""

    def __init__(
        self,
        semantic_weight: float = 0.4,
        category_weight: float = 0.3,
        attribute_weight: float = 0.2,
        constraint_weight: float = 0.1,
        min_threshold: float = 0.5,
        max_threshold: float = 0.9,
        model_name: str = "all-MiniLM-L6-v2",
        cache_embeddings: bool = True,
        max_cache_size: int = 10000,
    ) -> None:
        """Initialize filter config.

        Args:
            semantic_weight: Weight for semantic filters
            category_weight: Weight for category filters
            attribute_weight: Weight for attribute filters
            constraint_weight: Weight for constraint filters
            min_threshold: Minimum similarity threshold
            max_threshold: Maximum similarity threshold
            model_name: Name of the sentence transformer model
            cache_embeddings: Whether to cache embeddings
            max_cache_size: Maximum cache size
        """
        self.semantic_weight = semantic_weight
        self.category_weight = category_weight
        self.attribute_weight = attribute_weight
        self.constraint_weight = constraint_weight
        self.min_threshold = min_threshold
        self.max_threshold = max_threshold
        self.model_name = model_name
        self.cache_embeddings = cache_embeddings
        self.max_cache_size = max_cache_size


class FilterManager(BaseService[Dict[str, Any]]):
    """Filter management service for semantic filtering."""

    def __init__(
        self,
        config: Config,
        publisher: PublisherService,
        metrics_manager: MetricsManager,
        filter_config: Optional[FilterConfig] = None,
    ) -> None:
        """Initialize filter manager.

        Args:
            config: Application configuration
            publisher: Event publisher
            metrics_manager: Metrics manager
            filter_config: Filter configuration
        """
        super().__init__(config, publisher, metrics_manager)
        self.filter_config = filter_config or FilterConfig()
        self._initialized = False
        self._model: Optional[SentenceTransformer] = None
        self._embedding_cache: Dict[str, np.ndarray] = {}
        self._scaler = MinMaxScaler()

    async def initialize(self) -> None:
        """Initialize filter resources."""
        if self._initialized:
            return

        try:
            # Load sentence transformer model
            self._model = SentenceTransformer(self.filter_config.model_name)

            self._initialized = True
            self.publisher.publish(
                Event(
                    type=EventType.SERVICE_STARTED,
                    timestamp=datetime.now(),
                    component="filter_manager",
                    description="Filter manager initialized",
                    metadata={
                        "model": self.filter_config.model_name,
                        "cache_size": len(self._embedding_cache),
                    },
                )
            )

        except Exception as e:
            logger.error(f"Failed to initialize filter manager: {e}")
            raise

    async def cleanup(self) -> None:
        """Clean up filter resources."""
        self._initialized = False
        self._embedding_cache.clear()

        self.publisher.publish(
            Event(
                type=EventType.SERVICE_STOPPED,
                timestamp=datetime.now(),
                component="filter_manager",
                description="Filter manager stopped",
            )
        )

    async def health_check(self) -> Dict[str, Any]:
        """Check filter manager health.

        Returns:
            Health check results
        """
        return {
            "service": "FilterManager",
            "initialized": self._initialized,
            "model": self.filter_config.model_name,
            "cache_size": len(self._embedding_cache),
        }

    async def apply_filters(
        self,
        results: List[SearchResult],
        filters: List[SemanticFilter],
        min_score: float = 0.0,
    ) -> List[SearchResult]:
        """Apply semantic filters to search results.

        Args:
            results: Search results
            filters: Semantic filters
            min_score: Minimum score threshold

        Returns:
            Filtered search results
        """
        if not self._initialized:
            await self.initialize()

        try:
            # Emit filtering started event
            self.publisher.publish(
                Event(
                    type=EventType.FILTERING_STARTED,
                    timestamp=datetime.now(),
                    component="filter_manager",
                    description="Applying semantic filters",
                    metadata={
                        "num_results": len(results),
                        "num_filters": len(filters),
                    },
                )
            )

            # Apply filters
            filtered_results = []
            for result in results:
                # Skip results below threshold
                if result.score < min_score:
                    continue

                # Calculate filter scores
                filter_scores = []
                for filter in filters:
                    score = await self._apply_filter(result, filter)
                    if score is not None:
                        filter_scores.append(score * filter.weight)

                # Skip if any filter fails
                if not filter_scores or any(score == 0.0 for score in filter_scores):
                    continue

                # Calculate final score
                filter_score = np.mean(filter_scores)
                result.score = (result.score + filter_score) / 2
                filtered_results.append(result)

            # Sort by filtered score
            filtered_results.sort(key=lambda x: x.score, reverse=True)

            # Emit success event
            self.publisher.publish(
                Event(
                    type=EventType.FILTERING_COMPLETED,
                    timestamp=datetime.now(),
                    component="filter_manager",
                    description="Filtering completed",
                    metadata={
                        "input_results": len(results),
                        "output_results": len(filtered_results),
                    },
                )
            )

            return filtered_results

        except Exception as e:
            logger.error(f"Filtering failed: {e}")
            self.publisher.publish(
                Event(
                    type=EventType.FILTERING_FAILED,
                    timestamp=datetime.now(),
                    component="filter_manager",
                    description="Filtering failed",
                    error=str(e),
                )
            )
            return results

    async def _apply_filter(
        self,
        result: SearchResult,
        filter: SemanticFilter,
    ) -> Optional[float]:
        """Apply semantic filter to result.

        Args:
            result: Search result
            filter: Semantic filter

        Returns:
            Filter score (0-1) or None if filter fails
        """
        if filter.type == FilterType.SEMANTIC:
            return await self._apply_semantic_filter(result, filter)
        elif filter.type == FilterType.CATEGORY:
            return await self._apply_category_filter(result, filter)
        elif filter.type == FilterType.ATTRIBUTE:
            return await self._apply_attribute_filter(result, filter)
        elif filter.type == FilterType.CONSTRAINT:
            return await self._apply_constraint_filter(result, filter)
        elif filter.type == FilterType.COMPOSITE:
            return await self._apply_composite_filter(result, filter)
        else:
            logger.warning(f"Unknown filter type: {filter.type}")
            return None

    async def _apply_semantic_filter(
        self,
        result: SearchResult,
        filter: SemanticFilter,
    ) -> Optional[float]:
        """Apply semantic similarity filter.

        Args:
            result: Search result
            filter: Semantic filter

        Returns:
            Similarity score (0-1)
        """
        scores = []
        for condition in filter.conditions:
            # Get field value
            if condition.field == "endpoint":
                text = result.endpoint
            elif condition.field == "method":
                text = result.method
            elif condition.field == "description":
                text = result.description
            else:
                continue

            # Calculate similarity
            similarity = await self._calculate_similarity(text, condition.value)
            if similarity < condition.threshold:
                if filter.operator == FilterOperator.AND:
                    return 0.0
                continue

            scores.append(similarity * condition.weight)

        if not scores:
            return None

        if filter.operator == FilterOperator.AND:
            return np.mean(scores)
        elif filter.operator == FilterOperator.OR:
            return np.max(scores)
        else:
            return None

    async def _apply_category_filter(
        self,
        result: SearchResult,
        filter: SemanticFilter,
    ) -> Optional[float]:
        """Apply category filter.

        Args:
            result: Search result
            filter: Category filter

        Returns:
            Category score (0-1)
        """
        scores = []
        for condition in filter.conditions:
            if not result.categories:
                continue

            # Calculate category match
            if condition.operator == FilterOperator.CONTAINS:
                matches = [
                    await self._calculate_similarity(category, condition.value)
                    for category in result.categories
                ]
                score = max(matches) if matches else 0.0
            elif condition.operator == FilterOperator.SIMILAR:
                matches = [
                    await self._calculate_similarity(category, condition.value)
                    for category in result.categories
                ]
                score = np.mean(matches) if matches else 0.0
            else:
                continue

            if score < condition.threshold:
                if filter.operator == FilterOperator.AND:
                    return 0.0
                continue

            scores.append(score * condition.weight)

        if not scores:
            return None

        if filter.operator == FilterOperator.AND:
            return np.mean(scores)
        elif filter.operator == FilterOperator.OR:
            return np.max(scores)
        else:
            return None

    async def _apply_attribute_filter(
        self,
        result: SearchResult,
        filter: SemanticFilter,
    ) -> Optional[float]:
        """Apply attribute filter.

        Args:
            result: Search result
            filter: Attribute filter

        Returns:
            Attribute score (0-1)
        """
        scores = []
        for condition in filter.conditions:
            # Get attribute value
            value = None
            if condition.field == "method":
                value = result.method
            elif condition.field == "endpoint":
                value = result.endpoint
            elif condition.field == "description":
                value = result.description
            elif condition.field in result.metadata:
                value = result.metadata[condition.field]
            else:
                continue

            # Apply operator
            if condition.operator == FilterOperator.EQ:
                score = 1.0 if value == condition.value else 0.0
            elif condition.operator == FilterOperator.NE:
                score = 1.0 if value != condition.value else 0.0
            elif condition.operator == FilterOperator.GT:
                score = 1.0 if value > condition.value else 0.0
            elif condition.operator == FilterOperator.LT:
                score = 1.0 if value < condition.value else 0.0
            elif condition.operator == FilterOperator.CONTAINS:
                score = 1.0 if condition.value in str(value) else 0.0
            elif condition.operator == FilterOperator.SIMILAR:
                score = await self._calculate_similarity(str(value), str(condition.value))
            else:
                continue

            if score < condition.threshold:
                if filter.operator == FilterOperator.AND:
                    return 0.0
                continue

            scores.append(score * condition.weight)

        if not scores:
            return None

        if filter.operator == FilterOperator.AND:
            return np.mean(scores)
        elif filter.operator == FilterOperator.OR:
            return np.max(scores)
        else:
            return None

    async def _apply_constraint_filter(
        self,
        result: SearchResult,
        filter: SemanticFilter,
    ) -> Optional[float]:
        """Apply constraint filter.

        Args:
            result: Search result
            filter: Constraint filter

        Returns:
            Constraint score (0-1)
        """
        scores = []
        for condition in filter.conditions:
            # Get constraint value
            value = None
            if condition.field in result.metadata:
                value = result.metadata[condition.field]
            else:
                continue

            # Apply operator
            if condition.operator == FilterOperator.GT:
                score = 1.0 if value > condition.value else 0.0
            elif condition.operator == FilterOperator.LT:
                score = 1.0 if value < condition.value else 0.0
            elif condition.operator == FilterOperator.EQ:
                score = 1.0 if value == condition.value else 0.0
            elif condition.operator == FilterOperator.NE:
                score = 1.0 if value != condition.value else 0.0
            else:
                continue

            if score < condition.threshold:
                if filter.operator == FilterOperator.AND:
                    return 0.0
                continue

            scores.append(score * condition.weight)

        if not scores:
            return None

        if filter.operator == FilterOperator.AND:
            return np.mean(scores)
        elif filter.operator == FilterOperator.OR:
            return np.max(scores)
        else:
            return None

    async def _apply_composite_filter(
        self,
        result: SearchResult,
        filter: SemanticFilter,
    ) -> Optional[float]:
        """Apply composite filter.

        Args:
            result: Search result
            filter: Composite filter

        Returns:
            Composite score (0-1)
        """
        scores = []
        for condition in filter.conditions:
            # Create sub-filter
            sub_filter = SemanticFilter(
                type=condition.field,
                conditions=[condition],
                operator=filter.operator,
                threshold=condition.threshold,
                weight=condition.weight,
            )

            # Apply sub-filter
            score = await self._apply_filter(result, sub_filter)
            if score is None:
                continue

            if score < condition.threshold:
                if filter.operator == FilterOperator.AND:
                    return 0.0
                continue

            scores.append(score * condition.weight)

        if not scores:
            return None

        if filter.operator == FilterOperator.AND:
            return np.mean(scores)
        elif filter.operator == FilterOperator.OR:
            return np.max(scores)
        else:
            return None

    async def _calculate_similarity(self, text1: str, text2: str) -> float:
        """Calculate semantic similarity between texts.

        Args:
            text1: First text
            text2: Second text

        Returns:
            Similarity score (0-1)
        """
        # Get embeddings
        embedding1 = await self._get_embedding(text1)
        embedding2 = await self._get_embedding(text2)

        # Calculate cosine similarity
        similarity = util.pytorch_cos_sim(embedding1, embedding2).item()
        return float(similarity)

    async def _get_embedding(self, text: str) -> np.ndarray:
        """Get text embedding from cache or model.

        Args:
            text: Input text

        Returns:
            Text embedding
        """
        if self.filter_config.cache_embeddings:
            # Check cache
            if text in self._embedding_cache:
                return self._embedding_cache[text]

            # Calculate embedding
            embedding = self._model.encode(text, convert_to_tensor=True)

            # Update cache
            if len(self._embedding_cache) < self.filter_config.max_cache_size:
                self._embedding_cache[text] = embedding

            return embedding
        else:
            # Calculate embedding without caching
            return self._model.encode(text, convert_to_tensor=True)


# Create service instance
filter_manager = FilterManager(
    Config(),
    PublisherService(Config(), MetricsManager()),
    MetricsManager(),
)

__all__ = [
    "FilterType",
    "FilterOperator",
    "FilterCondition",
    "SemanticFilter",
    "FilterConfig",
    "FilterManager",
    "filter_manager",
] 