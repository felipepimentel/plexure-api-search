"""Advanced API search with triple vector embeddings and contextual boosting."""

import logging
import traceback
import time
from typing import Dict, List, Any, Optional, Set
from datetime import datetime

import numpy as np

from ..config import config_instance
from ..embedding.embeddings import EmbeddingManager
from ..integrations import pinecone_instance
from ..utils.cache import DiskCache
from .boosting import ContextualBooster, BusinessContext, BusinessValue
from .quality import QualityMetrics
from .search_models import SearchResult
from .understanding import ZeroShotUnderstanding
from ..monitoring.metrics_manager import metrics_manager
from ..monitoring.events import Event, EventType, publisher
from .vectorizer import TripleVectorizer, Triple
from .expansion import query_expander
from .intent import intent_detector
from .reranking import reranker
from .fuzzy import fuzzy_matcher
from .templates import template_manager
from .spellcheck import spell_checker
from .clustering import cluster_manager
from .diversity import diversity_manager
from .personalization import personalization_manager
from .context import context_manager
from .feedback import feedback_manager
from .filtering import filter_manager, SemanticFilter, FilterCondition, FilterType, FilterOperator
from .reformulation import reformulation_manager, ReformulationType, ReformulationResult
from .distributed import distributed_manager
from .sharding import sharding_manager
from .optimization import query_optimizer

logger = logging.getLogger(__name__)

search_cache = DiskCache[Dict[str, Any]](
    namespace="search",
    ttl=config_instance.cache_ttl,  # 1 hour
)


class APISearcher:
    """Advanced API search engine with multiple strategies."""

    def __init__(self, top_k: int = 10, use_cache: bool = True):
        """Initialize searcher."""
        self.client = pinecone_instance
        self.embedding_manager = EmbeddingManager()
        self.vectorizer = TripleVectorizer(self.embedding_manager)
        self.booster = ContextualBooster()
        self.understanding = ZeroShotUnderstanding()
        self.metrics = QualityMetrics()
        self.top_k = top_k
        self.use_cache = use_cache
        self.metrics = metrics_manager
        self.quality_metrics = QualityMetrics()
        self.query_expander = query_expander
        self.intent_detector = intent_detector
        self.reranker = reranker
        self.fuzzy_matcher = fuzzy_matcher
        self.template_manager = template_manager
        self.spell_checker = spell_checker
        self.cluster_manager = cluster_manager
        self.diversity_manager = diversity_manager
        self.personalization_manager = personalization_manager
        self.context_manager = context_manager
        self.feedback_manager = feedback_manager
        self.filter_manager = filter_manager
        self.reformulation_manager = reformulation_manager
        self.distributed_manager = distributed_manager
        self.sharding_manager = sharding_manager
        self.query_optimizer = query_optimizer

    async def _base_search(
        self,
        query: str,
        filters: Optional[Dict] = None,
        include_metadata: bool = True,
        use_cache: bool = True,
        enhance_results: bool = True,
        user_id: Optional[str] = None,
        domain: Optional[str] = None,
        semantic_filters: Optional[List[SemanticFilter]] = None,
        reformulation_types: Optional[List[str]] = None,
        min_confidence: Optional[float] = None,
        use_distributed: bool = True,
        use_sharding: bool = True,
    ) -> List[Dict[str, Any]]:
        """Enhanced search with caching and reranking."""
        start_time = time.time()
        
        # Emit search started event
        publisher.emit(Event(
            type=EventType.SEARCH_STARTED,
            timestamp=datetime.now(),
            component="search",
            description=f"Starting search for query: {query}",
            metadata={
                "filters": filters,
                "user_id": user_id,
                "domain": domain,
                "semantic_filters": bool(semantic_filters),
                "reformulation_types": reformulation_types,
                "use_distributed": use_distributed,
                "use_sharding": use_sharding,
            }
        ))
        
        try:
            # Optimize query
            optimized_query = await self.query_optimizer.optimize_query(
                query=query,
                context={
                    "user_id": user_id,
                    "domain": domain,
                    "filters": filters,
                },
            )

            # Use optimized query for search
            query = optimized_query

            # Use sharding if enabled
            if use_sharding:
                return await self.sharding_manager.search(
                    query=query,  # Use optimized query
                    filters=filters,
                    include_metadata=include_metadata,
                    use_cache=use_cache,
                    enhance_results=enhance_results,
                    user_id=user_id,
                    domain=domain,
                    semantic_filters=semantic_filters,
                    reformulation_types=reformulation_types,
                    min_confidence=min_confidence,
                )

            # Use distributed search if enabled
            if use_distributed:
                return await self.distributed_manager.search(
                    query=query,  # Use optimized query
                    filters=filters,
                    include_metadata=include_metadata,
                    use_cache=use_cache,
                    enhance_results=enhance_results,
                    user_id=user_id,
                    domain=domain,
                    semantic_filters=semantic_filters,
                    reformulation_types=reformulation_types,
                    min_confidence=min_confidence,
                )

            # Reformulate query if needed
            reformulated_queries = []
            if enhance_results:
                reformulations = await self.reformulation_manager.reformulate_query(
                    query=query,
                    types=reformulation_types,
                    min_confidence=min_confidence,
                )
                reformulated_queries = [r.reformulated_query for r in reformulations]

            # Combine original and reformulated queries
            all_queries = [query] + reformulated_queries

            # Search with all queries
            all_results = []
            for q in all_queries:
                # ... existing search code ...

                # Apply semantic filters if provided
                if semantic_filters:
                    publisher.emit(Event(
                        type=EventType.FILTERING_STARTED,
                        timestamp=datetime.now(),
                        component="search",
                        description="Applying semantic filters",
                        metadata={
                            "num_filters": len(semantic_filters),
                        }
                    ))
                    
                    final_results = await self.filter_manager.apply_filters(
                        results=final_results,
                        filters=semantic_filters,
                        min_score=0.0,
                    )

                # Apply feedback if available
                final_results = await self.feedback_manager.apply_feedback(
                    results=final_results,
                    query=q,
                )

                all_results.extend(final_results)

            # Deduplicate results
            seen_ids = set()
            unique_results = []
            for result in all_results:
                result_id = f"{result.method}_{result.endpoint}"
                if result_id not in seen_ids:
                    seen_ids.add(result_id)
                    unique_results.append(result)

            # Sort by score
            unique_results.sort(key=lambda x: x.score, reverse=True)

            # Take top-k
            final_results = unique_results[:self.top_k]

            # ... rest of the existing code ...

    def search(
        self,
        query: str,
        filters: Optional[Dict] = None,
        include_metadata: bool = True,
        enhance_results: bool = True,
        user_id: Optional[str] = None,
        domain: Optional[str] = None,
        semantic_filters: Optional[List[SemanticFilter]] = None,
        reformulation_types: Optional[List[str]] = None,
        min_confidence: Optional[float] = None,
        use_distributed: bool = True,
        use_sharding: bool = True,
    ) -> List[Dict[str, Any]]:
        """Search for API endpoints."""
        return self._base_search(
            query=query,
            filters=filters,
            include_metadata=include_metadata,
            use_cache=self.use_cache,
            enhance_results=enhance_results,
            user_id=user_id,
            domain=domain,
            semantic_filters=semantic_filters,
            reformulation_types=reformulation_types,
            min_confidence=min_confidence,
            use_distributed=use_distributed,
            use_sharding=use_sharding,
        )

    async def add_vectors(
        self,
        vectors: List[np.ndarray],
        metadata: List[Dict[str, Any]],
        use_sharding: bool = True,
    ) -> bool:
        """Add vectors to index.

        Args:
            vectors: List of vectors to add
            metadata: List of metadata for vectors
            use_sharding: Whether to use sharding

        Returns:
            True if successful, False otherwise
        """
        try:
            # Use sharding if enabled
            if use_sharding:
                return await self.sharding_manager.add_vectors(vectors, metadata)

            # Add vectors directly to index
            # ... existing code ...

        except Exception as e:
            logger.error(f"Add vectors failed: {e}")
            return False

    def create_semantic_filter(
        self,
        type: str,
        conditions: List[Dict[str, Any]],
        operator: str = FilterOperator.AND,
        threshold: float = 0.7,
        weight: float = 1.0,
    ) -> SemanticFilter:
        """Create semantic filter for search results.

        Args:
            type: Filter type (semantic, category, attribute, constraint, composite)
            conditions: List of filter conditions
            operator: Condition operator (and, or, not)
            threshold: Filter threshold
            weight: Filter weight

        Returns:
            Semantic filter
        """
        filter_conditions = []
        for condition in conditions:
            filter_condition = FilterCondition(
                field=condition["field"],
                operator=condition["operator"],
                value=condition["value"],
                threshold=condition.get("threshold", 0.7),
                weight=condition.get("weight", 1.0),
            )
            filter_conditions.append(filter_condition)

        return SemanticFilter(
            type=type,
            conditions=filter_conditions,
            operator=operator,
            threshold=threshold,
            weight=weight,
        ) 