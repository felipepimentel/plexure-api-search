"""Query intent detection for semantic search enhancement."""

import logging
from typing import Dict, List, Any, Optional, Tuple
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


class IntentType:
    """Intent types for API queries."""

    # CRUD operations
    CREATE = "create"
    READ = "read"
    UPDATE = "update"
    DELETE = "delete"
    LIST = "list"

    # Special operations
    SEARCH = "search"
    FILTER = "filter"
    SORT = "sort"
    AGGREGATE = "aggregate"
    EXPORT = "export"
    IMPORT = "import"
    VALIDATE = "validate"
    ANALYZE = "analyze"

    # Authentication/Authorization
    AUTH = "auth"
    LOGIN = "login"
    LOGOUT = "logout"
    REGISTER = "register"

    # Status operations
    STATUS = "status"
    HEALTH = "health"
    METRICS = "metrics"
    LOGS = "logs"

    # Batch operations
    BATCH = "batch"
    BULK = "bulk"
    SYNC = "sync"

    @classmethod
    def all(cls) -> List[str]:
        """Get all intent types.

        Returns:
            List of all intent types
        """
        return [
            attr for attr in dir(cls)
            if not attr.startswith("_") and isinstance(getattr(cls, attr), str)
        ]


class QueryIntent:
    """Query intent with confidence score."""

    def __init__(
        self,
        intent_type: str,
        confidence: float,
        parameters: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Initialize query intent.

        Args:
            intent_type: Type of intent
            confidence: Confidence score (0-1)
            parameters: Optional parameters extracted from query
        """
        self.intent_type = intent_type
        self.confidence = confidence
        self.parameters = parameters or {}


class IntentDetector(BaseService[Dict[str, Any]]):
    """Intent detection service for semantic search enhancement."""

    def __init__(
        self,
        config: Config,
        publisher: PublisherService,
        metrics_manager: MetricsManager,
        model_service: ModelService,
    ) -> None:
        """Initialize intent detector.

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
        self._intent_embeddings: Dict[str, np.ndarray] = {}

    async def initialize(self) -> None:
        """Initialize detector resources."""
        if self._initialized:
            return

        try:
            # Load model
            self._model = await self.model_service.get_model("bi_encoder")

            # Create intent embeddings
            await self._create_intent_embeddings()

            self._initialized = True
            self.publisher.publish(
                Event(
                    type=EventType.SERVICE_STARTED,
                    timestamp=datetime.now(),
                    component="intent_detector",
                    description="Intent detector initialized",
                )
            )

        except Exception as e:
            logger.error(f"Failed to initialize intent detector: {e}")
            raise

    async def cleanup(self) -> None:
        """Clean up detector resources."""
        self._initialized = False
        self._model = None
        self._intent_embeddings.clear()

        self.publisher.publish(
            Event(
                type=EventType.SERVICE_STOPPED,
                timestamp=datetime.now(),
                component="intent_detector",
                description="Intent detector stopped",
            )
        )

    async def health_check(self) -> Dict[str, Any]:
        """Check detector health.

        Returns:
            Health check results
        """
        return {
            "service": "IntentDetector",
            "initialized": self._initialized,
            "model_loaded": self._model is not None,
            "num_intents": len(self._intent_embeddings),
        }

    async def detect_intent(
        self,
        query: str,
        min_confidence: float = 0.7,
    ) -> QueryIntent:
        """Detect intent from query.

        Args:
            query: Search query
            min_confidence: Minimum confidence threshold

        Returns:
            Detected query intent
        """
        if not self._initialized:
            await self.initialize()

        try:
            # Emit detection started event
            self.publisher.publish(
                Event(
                    type=EventType.INTENT_DETECTION_STARTED,
                    timestamp=datetime.now(),
                    component="intent_detector",
                    description=f"Detecting intent for query: {query}",
                )
            )

            # Get query embedding
            query_embedding = self._model.encode(
                query,
                convert_to_tensor=True,
            )

            # Find most similar intent
            best_intent = None
            best_score = -1.0

            for intent_type, intent_embedding in self._intent_embeddings.items():
                similarity = np.dot(query_embedding, intent_embedding) / (
                    np.linalg.norm(query_embedding) * np.linalg.norm(intent_embedding)
                )
                if similarity > best_score:
                    best_score = float(similarity)
                    best_intent = intent_type

            # Extract parameters if confidence is high enough
            parameters = None
            if best_score >= min_confidence:
                parameters = await self._extract_parameters(query, best_intent)

            # Create intent object
            intent = QueryIntent(
                intent_type=best_intent or IntentType.READ,
                confidence=max(0.0, best_score),
                parameters=parameters,
            )

            # Emit success event
            self.publisher.publish(
                Event(
                    type=EventType.INTENT_DETECTION_COMPLETED,
                    timestamp=datetime.now(),
                    component="intent_detector",
                    description="Intent detection completed",
                    metadata={
                        "query": query,
                        "intent": intent.intent_type,
                        "confidence": intent.confidence,
                    },
                )
            )

            return intent

        except Exception as e:
            logger.error(f"Intent detection failed: {e}")
            self.publisher.publish(
                Event(
                    type=EventType.INTENT_DETECTION_FAILED,
                    timestamp=datetime.now(),
                    component="intent_detector",
                    description="Intent detection failed",
                    error=str(e),
                )
            )
            return QueryIntent(IntentType.READ, 0.0)

    async def _create_intent_embeddings(self) -> None:
        """Create embeddings for all intent types."""
        intent_examples = {
            IntentType.CREATE: [
                "create new resource",
                "add new item",
                "insert record",
                "post new data",
            ],
            IntentType.READ: [
                "get resource",
                "fetch item",
                "retrieve data",
                "read record",
            ],
            IntentType.UPDATE: [
                "update resource",
                "modify item",
                "edit record",
                "patch data",
            ],
            IntentType.DELETE: [
                "delete resource",
                "remove item",
                "erase record",
                "destroy data",
            ],
            IntentType.LIST: [
                "list all resources",
                "get all items",
                "fetch all records",
                "retrieve all",
            ],
            IntentType.SEARCH: [
                "search for resources",
                "find items",
                "query records",
                "lookup data",
            ],
            IntentType.FILTER: [
                "filter resources",
                "filter by condition",
                "get filtered items",
                "search with filters",
            ],
            IntentType.SORT: [
                "sort resources",
                "order items",
                "get sorted data",
                "arrange records",
            ],
            IntentType.AGGREGATE: [
                "aggregate data",
                "calculate summary",
                "get statistics",
                "compute metrics",
            ],
            IntentType.EXPORT: [
                "export data",
                "download resources",
                "get file",
                "save records",
            ],
            IntentType.IMPORT: [
                "import data",
                "upload resources",
                "post file",
                "load records",
            ],
            IntentType.VALIDATE: [
                "validate data",
                "check resource",
                "verify record",
                "test input",
            ],
            IntentType.ANALYZE: [
                "analyze data",
                "process resource",
                "examine record",
                "inspect item",
            ],
            IntentType.AUTH: [
                "authenticate user",
                "get token",
                "verify credentials",
                "check auth",
            ],
            IntentType.LOGIN: [
                "login user",
                "sign in",
                "authenticate",
                "start session",
            ],
            IntentType.LOGOUT: [
                "logout user",
                "sign out",
                "end session",
                "clear auth",
            ],
            IntentType.REGISTER: [
                "register user",
                "sign up",
                "create account",
                "new user",
            ],
            IntentType.STATUS: [
                "get status",
                "check health",
                "system state",
                "service status",
            ],
            IntentType.HEALTH: [
                "health check",
                "system health",
                "service health",
                "check status",
            ],
            IntentType.METRICS: [
                "get metrics",
                "system metrics",
                "service stats",
                "performance data",
            ],
            IntentType.LOGS: [
                "get logs",
                "system logs",
                "service logs",
                "view logs",
            ],
            IntentType.BATCH: [
                "batch process",
                "bulk operation",
                "process multiple",
                "batch job",
            ],
            IntentType.BULK: [
                "bulk update",
                "mass operation",
                "multiple items",
                "bulk job",
            ],
            IntentType.SYNC: [
                "sync data",
                "synchronize",
                "refresh data",
                "update sync",
            ],
        }

        # Create embeddings for each intent
        for intent_type, examples in intent_examples.items():
            # Get embeddings for all examples
            embeddings = self._model.encode(
                examples,
                convert_to_tensor=True,
                batch_size=32,
            )

            # Use mean embedding as intent embedding
            intent_embedding = np.mean(embeddings, axis=0)
            self._intent_embeddings[intent_type] = intent_embedding

    async def _extract_parameters(
        self,
        query: str,
        intent_type: str,
    ) -> Dict[str, Any]:
        """Extract parameters from query based on intent type.

        Args:
            query: Search query
            intent_type: Detected intent type

        Returns:
            Extracted parameters
        """
        parameters = {}

        # Extract common parameters
        words = query.lower().split()

        # Extract resource type
        resource_indicators = ["for", "from", "in", "of"]
        for i, word in enumerate(words):
            if word in resource_indicators and i + 1 < len(words):
                parameters["resource_type"] = words[i + 1]
                break

        # Extract filters
        if "where" in words:
            idx = words.index("where")
            if idx + 2 < len(words):
                parameters["filter"] = {
                    "field": words[idx + 1],
                    "value": words[idx + 2],
                }

        # Extract sorting
        if "sort" in words or "order" in words:
            sort_idx = words.index("sort") if "sort" in words else words.index("order")
            if sort_idx + 2 < len(words):
                parameters["sort"] = {
                    "field": words[sort_idx + 2],
                    "direction": "desc" if "desc" in words else "asc",
                }

        # Extract limit
        for i, word in enumerate(words):
            if word in ["limit", "top", "first"] and i + 1 < len(words):
                try:
                    parameters["limit"] = int(words[i + 1])
                except ValueError:
                    pass

        return parameters


# Create service instance
intent_detector = IntentDetector(
    Config(),
    PublisherService(Config(), MetricsManager()),
    MetricsManager(),
    ModelService(Config(), PublisherService(Config(), MetricsManager()), MetricsManager()),
)

__all__ = ["IntentType", "QueryIntent", "IntentDetector", "intent_detector"] 