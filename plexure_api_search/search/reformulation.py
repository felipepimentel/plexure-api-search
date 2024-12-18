"""Query reformulation for improved search results."""

import logging
from typing import Dict, List, Any, Optional, Set
from datetime import datetime

import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sentence_transformers import SentenceTransformer, util

from ..config import Config
from ..monitoring.events import Event, EventType
from ..monitoring.metrics import MetricsManager
from ..services.base import BaseService
from ..services.events import PublisherService
from .expansion import query_expander
from .intent import intent_detector
from .templates import template_manager
from .spellcheck import spell_checker

logger = logging.getLogger(__name__)


class ReformulationType:
    """Types of query reformulation."""

    EXPANSION = "expansion"  # Expand query with similar terms
    TEMPLATE = "template"    # Match and apply query template
    INTENT = "intent"       # Reformulate based on intent
    SPELLING = "spelling"   # Fix spelling errors
    COMPOSITE = "composite" # Combine multiple reformulations


class ReformulationConfig:
    """Configuration for query reformulation."""

    def __init__(
        self,
        expansion_weight: float = 0.4,
        template_weight: float = 0.3,
        intent_weight: float = 0.2,
        spelling_weight: float = 0.1,
        min_confidence: float = 0.5,
        max_expansions: int = 3,
        max_templates: int = 2,
        max_intents: int = 2,
        model_name: str = "all-MiniLM-L6-v2",
        cache_embeddings: bool = True,
        max_cache_size: int = 10000,
    ) -> None:
        """Initialize reformulation config.

        Args:
            expansion_weight: Weight for query expansion
            template_weight: Weight for template matching
            intent_weight: Weight for intent detection
            spelling_weight: Weight for spell checking
            min_confidence: Minimum confidence threshold
            max_expansions: Maximum number of expansions
            max_templates: Maximum number of templates
            max_intents: Maximum number of intents
            model_name: Name of the sentence transformer model
            cache_embeddings: Whether to cache embeddings
            max_cache_size: Maximum cache size
        """
        self.expansion_weight = expansion_weight
        self.template_weight = template_weight
        self.intent_weight = intent_weight
        self.spelling_weight = spelling_weight
        self.min_confidence = min_confidence
        self.max_expansions = max_expansions
        self.max_templates = max_templates
        self.max_intents = max_intents
        self.model_name = model_name
        self.cache_embeddings = cache_embeddings
        self.max_cache_size = max_cache_size


class ReformulationResult:
    """Result of query reformulation."""

    def __init__(
        self,
        original_query: str,
        reformulated_query: str,
        confidence: float,
        type: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Initialize reformulation result.

        Args:
            original_query: Original query
            reformulated_query: Reformulated query
            confidence: Confidence score
            type: Reformulation type
            metadata: Additional metadata
        """
        self.original_query = original_query
        self.reformulated_query = reformulated_query
        self.confidence = confidence
        self.type = type
        self.metadata = metadata or {}


class ReformulationManager(BaseService[Dict[str, Any]]):
    """Query reformulation service."""

    def __init__(
        self,
        config: Config,
        publisher: PublisherService,
        metrics_manager: MetricsManager,
        reformulation_config: Optional[ReformulationConfig] = None,
    ) -> None:
        """Initialize reformulation manager.

        Args:
            config: Application configuration
            publisher: Event publisher
            metrics_manager: Metrics manager
            reformulation_config: Reformulation configuration
        """
        super().__init__(config, publisher, metrics_manager)
        self.reformulation_config = reformulation_config or ReformulationConfig()
        self._initialized = False
        self._model: Optional[SentenceTransformer] = None
        self._embedding_cache: Dict[str, np.ndarray] = {}
        self._scaler = MinMaxScaler()
        self.query_expander = query_expander
        self.intent_detector = intent_detector
        self.template_manager = template_manager
        self.spell_checker = spell_checker

    async def initialize(self) -> None:
        """Initialize reformulation resources."""
        if self._initialized:
            return

        try:
            # Load sentence transformer model
            self._model = SentenceTransformer(self.reformulation_config.model_name)

            self._initialized = True
            self.publisher.publish(
                Event(
                    type=EventType.SERVICE_STARTED,
                    timestamp=datetime.now(),
                    component="reformulation_manager",
                    description="Reformulation manager initialized",
                    metadata={
                        "model": self.reformulation_config.model_name,
                        "cache_size": len(self._embedding_cache),
                    },
                )
            )

        except Exception as e:
            logger.error(f"Failed to initialize reformulation manager: {e}")
            raise

    async def cleanup(self) -> None:
        """Clean up reformulation resources."""
        self._initialized = False
        self._embedding_cache.clear()

        self.publisher.publish(
            Event(
                type=EventType.SERVICE_STOPPED,
                timestamp=datetime.now(),
                component="reformulation_manager",
                description="Reformulation manager stopped",
            )
        )

    async def health_check(self) -> Dict[str, Any]:
        """Check reformulation manager health.

        Returns:
            Health check results
        """
        return {
            "service": "ReformulationManager",
            "initialized": self._initialized,
            "model": self.reformulation_config.model_name,
            "cache_size": len(self._embedding_cache),
        }

    async def reformulate_query(
        self,
        query: str,
        types: Optional[List[str]] = None,
        min_confidence: Optional[float] = None,
    ) -> List[ReformulationResult]:
        """Reformulate search query.

        Args:
            query: Search query
            types: Reformulation types to apply
            min_confidence: Minimum confidence threshold

        Returns:
            List of reformulation results
        """
        if not self._initialized:
            await self.initialize()

        try:
            # Use default types if not specified
            if types is None:
                types = [
                    ReformulationType.SPELLING,
                    ReformulationType.TEMPLATE,
                    ReformulationType.INTENT,
                    ReformulationType.EXPANSION,
                ]

            # Use default confidence if not specified
            if min_confidence is None:
                min_confidence = self.reformulation_config.min_confidence

            # Emit reformulation started event
            self.publisher.publish(
                Event(
                    type=EventType.REFORMULATION_STARTED,
                    timestamp=datetime.now(),
                    component="reformulation_manager",
                    description=f"Reformulating query: {query}",
                    metadata={
                        "types": types,
                        "min_confidence": min_confidence,
                    },
                )
            )

            # Apply reformulations
            results = []

            # Fix spelling errors
            if ReformulationType.SPELLING in types:
                spelling_results = await self._apply_spelling(query)
                results.extend(spelling_results)

            # Match templates
            if ReformulationType.TEMPLATE in types:
                template_results = await self._apply_templates(query)
                results.extend(template_results)

            # Detect intent
            if ReformulationType.INTENT in types:
                intent_results = await self._apply_intent(query)
                results.extend(intent_results)

            # Expand query
            if ReformulationType.EXPANSION in types:
                expansion_results = await self._apply_expansion(query)
                results.extend(expansion_results)

            # Filter by confidence
            results = [r for r in results if r.confidence >= min_confidence]

            # Sort by confidence
            results.sort(key=lambda x: x.confidence, reverse=True)

            # Emit success event
            self.publisher.publish(
                Event(
                    type=EventType.REFORMULATION_COMPLETED,
                    timestamp=datetime.now(),
                    component="reformulation_manager",
                    description="Reformulation completed",
                    metadata={
                        "num_results": len(results),
                    },
                )
            )

            return results

        except Exception as e:
            logger.error(f"Reformulation failed: {e}")
            self.publisher.publish(
                Event(
                    type=EventType.REFORMULATION_FAILED,
                    timestamp=datetime.now(),
                    component="reformulation_manager",
                    description="Reformulation failed",
                    error=str(e),
                )
            )
            return []

    async def _apply_spelling(self, query: str) -> List[ReformulationResult]:
        """Apply spell checking.

        Args:
            query: Search query

        Returns:
            Spelling reformulation results
        """
        results = []

        # Check spelling
        corrections = await self.spell_checker.check(query)
        if corrections:
            for correction in corrections:
                results.append(
                    ReformulationResult(
                        original_query=query,
                        reformulated_query=correction.corrected_text,
                        confidence=correction.confidence * self.reformulation_config.spelling_weight,
                        type=ReformulationType.SPELLING,
                        metadata={
                            "corrections": correction.corrections,
                        },
                    )
                )

        return results

    async def _apply_templates(self, query: str) -> List[ReformulationResult]:
        """Apply query templates.

        Args:
            query: Search query

        Returns:
            Template reformulation results
        """
        results = []

        # Match templates
        matches = await self.template_manager.match(
            query,
            max_matches=self.reformulation_config.max_templates,
        )

        for match in matches:
            results.append(
                ReformulationResult(
                    original_query=query,
                    reformulated_query=match.expanded_query,
                    confidence=match.confidence * self.reformulation_config.template_weight,
                    type=ReformulationType.TEMPLATE,
                    metadata={
                        "template": match.template,
                        "parameters": match.parameters,
                    },
                )
            )

        return results

    async def _apply_intent(self, query: str) -> List[ReformulationResult]:
        """Apply intent detection.

        Args:
            query: Search query

        Returns:
            Intent reformulation results
        """
        results = []

        # Detect intent
        intents = await self.intent_detector.detect(
            query,
            max_intents=self.reformulation_config.max_intents,
        )

        for intent in intents:
            results.append(
                ReformulationResult(
                    original_query=query,
                    reformulated_query=intent.reformulated_query,
                    confidence=intent.confidence * self.reformulation_config.intent_weight,
                    type=ReformulationType.INTENT,
                    metadata={
                        "intent": intent.intent_type,
                        "slots": intent.slots,
                    },
                )
            )

        return results

    async def _apply_expansion(self, query: str) -> List[ReformulationResult]:
        """Apply query expansion.

        Args:
            query: Search query

        Returns:
            Expansion reformulation results
        """
        results = []

        # Expand query
        expansions = await self.query_expander.expand(
            query,
            max_expansions=self.reformulation_config.max_expansions,
        )

        for expansion in expansions:
            results.append(
                ReformulationResult(
                    original_query=query,
                    reformulated_query=expansion.expanded_query,
                    confidence=expansion.confidence * self.reformulation_config.expansion_weight,
                    type=ReformulationType.EXPANSION,
                    metadata={
                        "terms": expansion.terms,
                        "strategy": expansion.strategy,
                    },
                )
            )

        return results


# Create service instance
reformulation_manager = ReformulationManager(
    Config(),
    PublisherService(Config(), MetricsManager()),
    MetricsManager(),
)

__all__ = [
    "ReformulationType",
    "ReformulationConfig",
    "ReformulationResult",
    "ReformulationManager",
    "reformulation_manager",
] 