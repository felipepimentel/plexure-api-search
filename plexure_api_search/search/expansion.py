"""Query expansion module for semantic search enhancement."""

import logging
from typing import List, Dict, Any, Optional
from datetime import datetime

import numpy as np
from sentence_transformers import SentenceTransformer

from ..config import Config
from ..monitoring.events import Event, EventType
from ..monitoring.metrics import MetricsManager
from ..services.base import BaseService
from ..services.events import PublisherService
from ..services.models import ModelService

logger = logging.getLogger(__name__)


class QueryExpander(BaseService[Dict[str, Any]]):
    """Query expansion service for semantic search enhancement."""

    def __init__(
        self,
        config: Config,
        publisher: PublisherService,
        metrics_manager: MetricsManager,
        model_service: ModelService,
    ) -> None:
        """Initialize query expander.

        Args:
            config: Application configuration
            publisher: Event publisher
            metrics_manager: Metrics manager
            model_service: Model service for embeddings
        """
        super().__init__(config, publisher, metrics_manager)
        self.model_service = model_service
        self._initialized = False
        self._model: Optional[SentenceTransformer] = None

    async def initialize(self) -> None:
        """Initialize expander resources."""
        if self._initialized:
            return

        try:
            # Load model
            self._model = await self.model_service.get_model("bi_encoder")
            self._initialized = True

            self.publisher.publish(
                Event(
                    type=EventType.SERVICE_STARTED,
                    timestamp=datetime.now(),
                    component="query_expander",
                    description="Query expander initialized",
                )
            )

        except Exception as e:
            logger.error(f"Failed to initialize query expander: {e}")
            raise

    async def cleanup(self) -> None:
        """Clean up expander resources."""
        self._initialized = False
        self._model = None

        self.publisher.publish(
            Event(
                type=EventType.SERVICE_STOPPED,
                timestamp=datetime.now(),
                component="query_expander",
                description="Query expander stopped",
            )
        )

    async def health_check(self) -> Dict[str, Any]:
        """Check expander health.

        Returns:
            Health check results
        """
        return {
            "service": "QueryExpander",
            "initialized": self._initialized,
            "model_loaded": self._model is not None,
        }

    async def expand_query(
        self,
        query: str,
        num_expansions: int = 3,
        min_similarity: float = 0.7,
    ) -> List[str]:
        """Expand query with semantically related terms.

        Args:
            query: Original search query
            num_expansions: Number of expansions to generate
            min_similarity: Minimum similarity threshold

        Returns:
            List of expanded queries
        """
        if not self._initialized:
            await self.initialize()

        try:
            # Emit expansion started event
            self.publisher.publish(
                Event(
                    type=EventType.EXPANSION_STARTED,
                    timestamp=datetime.now(),
                    component="query_expander",
                    description=f"Expanding query: {query}",
                )
            )

            # Generate candidate expansions
            candidates = await self._generate_candidates(query)

            # Get embeddings for query and candidates
            query_embedding = self._model.encode(query, convert_to_tensor=True)
            candidate_embeddings = self._model.encode(
                candidates,
                convert_to_tensor=True,
                batch_size=32,
            )

            # Calculate similarities
            similarities = []
            for i, candidate in enumerate(candidates):
                similarity = np.dot(query_embedding, candidate_embeddings[i]) / (
                    np.linalg.norm(query_embedding) * np.linalg.norm(candidate_embeddings[i])
                )
                if similarity >= min_similarity:
                    similarities.append((candidate, float(similarity)))

            # Sort by similarity and take top expansions
            similarities.sort(key=lambda x: x[1], reverse=True)
            expansions = [exp for exp, _ in similarities[:num_expansions]]

            # Emit success event
            self.publisher.publish(
                Event(
                    type=EventType.EXPANSION_COMPLETED,
                    timestamp=datetime.now(),
                    component="query_expander",
                    description="Query expansion completed",
                    metadata={
                        "original_query": query,
                        "num_expansions": len(expansions),
                    },
                )
            )

            return expansions

        except Exception as e:
            logger.error(f"Query expansion failed: {e}")
            self.publisher.publish(
                Event(
                    type=EventType.EXPANSION_FAILED,
                    timestamp=datetime.now(),
                    component="query_expander",
                    description="Query expansion failed",
                    error=str(e),
                )
            )
            return [query]  # Return original query on failure

    async def _generate_candidates(self, query: str) -> List[str]:
        """Generate candidate expansions for a query.

        This method uses various techniques to generate candidates:
        1. Synonym expansion
        2. Contextual variations
        3. API-specific terminology

        Args:
            query: Original search query

        Returns:
            List of candidate expansions
        """
        candidates = []

        # Add original query
        candidates.append(query)

        # Add common API variations
        if "get" in query.lower():
            candidates.append(query.lower().replace("get", "retrieve"))
            candidates.append(query.lower().replace("get", "fetch"))
        if "create" in query.lower():
            candidates.append(query.lower().replace("create", "add"))
            candidates.append(query.lower().replace("create", "insert"))
        if "update" in query.lower():
            candidates.append(query.lower().replace("update", "modify"))
            candidates.append(query.lower().replace("update", "edit"))
        if "delete" in query.lower():
            candidates.append(query.lower().replace("delete", "remove"))
            candidates.append(query.lower().replace("delete", "erase"))

        # Add HTTP method variations
        http_methods = ["GET", "POST", "PUT", "DELETE", "PATCH"]
        for method in http_methods:
            if method.lower() in query.lower():
                candidates.append(query.lower().replace(method.lower(), "endpoint"))
                candidates.append(query.lower().replace(method.lower(), "api"))

        # Add common API resource variations
        if "list" in query.lower():
            candidates.append(query.lower().replace("list", "all"))
            candidates.append(query.lower().replace("list", "search"))
        if "find" in query.lower():
            candidates.append(query.lower().replace("find", "search"))
            candidates.append(query.lower().replace("find", "get"))

        # Remove duplicates and empty strings
        candidates = list(set(filter(None, candidates)))

        return candidates


# Create service instance
query_expander = QueryExpander(
    Config(),
    PublisherService(Config(), MetricsManager()),
    MetricsManager(),
    ModelService(Config(), PublisherService(Config(), MetricsManager()), MetricsManager()),
)

__all__ = ["QueryExpander", "query_expander"]
