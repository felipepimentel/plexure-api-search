"""Event system for tracking processing steps."""

from dataclasses import dataclass
from datetime import datetime
from enum import Enum, auto
from typing import Any, Callable, Dict, List, Optional


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
    metadata: Optional[Dict[str, Any]] = None
    algorithm: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert event to dictionary."""
        return {
            "type": self.type.name,
            "timestamp": self.timestamp.isoformat(),
            "component": self.component,
            "description": self.description,
            "duration_ms": self.duration_ms,
            "success": self.success,
            "error": self.error,
            "metadata": self.metadata or {},
            "algorithm": self.algorithm,
        }


class EventManager:
    """Manages event subscriptions and progress tracking."""
    
    def __init__(self):
        """Initialize event manager."""
        self._subscribers: List[Callable[[Event], None]] = []
        self._progress: Dict[str, Dict[str, Any]] = {}
    
    def subscribe(self, callback: Callable[[Event], None]) -> None:
        """Subscribe to events."""
        if callback not in self._subscribers:
            self._subscribers.append(callback)
    
    def unsubscribe(self, callback: Callable[[Event], None]) -> None:
        """Unsubscribe from events."""
        if callback in self._subscribers:
            self._subscribers.remove(callback)
    
    def emit(self, event: Event) -> None:
        """Emit an event to all subscribers."""
        # Update progress tracking
        self._update_progress(event)
        
        # Notify subscribers
        for subscriber in self._subscribers:
            try:
                subscriber(event)
            except Exception as e:
                print(f"Error in event subscriber: {e}")
    
    def _update_progress(self, event: Event) -> None:
        """Update progress tracking based on event."""
        component = event.component
        
        # Initialize component status if needed
        if component not in self._progress:
            self._progress[component] = {
                "status": "unknown",
                "last_event": None,
                "error": None,
                "progress": 0.0,
            }
        
        # Update status based on event type
        if event.type.name.endswith("_STARTED"):
            self._progress[component]["status"] = "in_progress"
        elif event.type.name.endswith("_COMPLETED"):
            self._progress[component]["status"] = "completed"
            self._progress[component]["progress"] = 100.0
        elif event.type.name.endswith("_FAILED"):
            self._progress[component]["status"] = "error"
            self._progress[component]["error"] = event.error
        
        # Update progress if provided in metadata
        if event.metadata and "progress" in event.metadata:
            self._progress[component]["progress"] = float(event.metadata["progress"])
        
        # Update last event
        self._progress[component]["last_event"] = event.timestamp.isoformat()
    
    def get_progress(self) -> Dict[str, Dict[str, Any]]:
        """Get current progress status for all components."""
        return self._progress.copy()


# Global event manager instance
event_manager = EventManager() 