"""Event data structure."""

from dataclasses import asdict, dataclass, field
from datetime import datetime
from enum import Enum, auto
from typing import Any, Dict, Optional


class EventType(Enum):
    """Types of events that can be emitted."""

    # Model events
    MODEL_LOADING_STARTED = auto()
    MODEL_LOADING_COMPLETED = auto()
    MODEL_LOADING_FAILED = auto()

    # Indexing events
    INDEXING_STARTED = auto()
    INDEXING_FILE_STARTED = auto()
    INDEXING_FILE_COMPLETED = auto()
    INDEXING_FILE_FAILED = auto()
    INDEXING_COMPLETED = auto()

    # Search events
    SEARCH_STARTED = auto()
    SEARCH_QUERY_PROCESSED = auto()
    SEARCH_RESULTS_FOUND = auto()
    SEARCH_COMPLETED = auto()
    SEARCH_FAILED = auto()

    # Cache events
    CACHE_HIT = auto()
    CACHE_MISS = auto()
    CACHE_UPDATE = auto()

    # Embedding events
    EMBEDDING_STARTED = auto()
    EMBEDDING_COMPLETED = auto()
    EMBEDDING_FAILED = auto()

    # Validation events
    VALIDATION_STARTED = auto()
    VALIDATION_COMPLETED = auto()
    VALIDATION_FAILED = auto()

    # Monitoring events
    MONITORING_STARTED = auto()
    MONITORING_UPDATED = auto()
    MONITORING_STOPPED = auto()


@dataclass
class Event:
    """Event data structure."""

    type: EventType
    timestamp: datetime
    component: str
    description: Optional[str] = None
    duration_ms: Optional[float] = None
    success: bool = True
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    algorithm: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert event to dictionary."""
        data = asdict(self)
        data["timestamp"] = self.timestamp.isoformat()
        data["type"] = self.type.name
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Event":
        """Create event from dictionary."""
        data = data.copy()
        data["timestamp"] = datetime.fromisoformat(data["timestamp"])
        data["type"] = EventType[data["type"]]
        return cls(**data)
