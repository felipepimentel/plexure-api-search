"""Event system for tracking processing steps."""

from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum, auto
from typing import Any, Callable, Dict, List, Optional
import statistics


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
        self._events: List[Event] = []  # Store recent events
        self._max_events = 1000  # Maximum number of events to store
        self._monitoring_start_time = None
    
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
        # Store event
        self._events.append(event)
        
        # Trim old events if needed
        if len(self._events) > self._max_events:
            self._events = self._events[-self._max_events:]
        
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
    
    def get_recent_events(self, minutes: int = 5) -> List[Event]:
        """Get events from the last N minutes.
        
        Args:
            minutes: Number of minutes to look back
            
        Returns:
            List of recent events
        """
        cutoff = datetime.now() - timedelta(minutes=minutes)
        return [e for e in self._events if e.timestamp >= cutoff]
    
    def get_cache_hit_rate(self) -> float:
        """Calculate cache hit rate from recent events.
        
        Returns:
            Cache hit rate as a float between 0 and 1
        """
        recent = self.get_recent_events()
        hits = len([e for e in recent if e.type == EventType.CACHE_HIT])
        misses = len([e for e in recent if e.type == EventType.CACHE_MISS])
        total = hits + misses
        return hits / total if total > 0 else 0.0
    
    def get_average_search_time(self) -> float:
        """Calculate average search time from recent events.
        
        Returns:
            Average search time in seconds
        """
        search_events = [
            e for e in self.get_recent_events() 
            if e.type == EventType.SEARCH_COMPLETED and e.duration_ms is not None
        ]
        if not search_events:
            return 0.0
        durations = [e.duration_ms / 1000.0 for e in search_events]  # Convert to seconds
        return statistics.mean(durations) if durations else 0.0
    
    def get_active_model_count(self) -> int:
        """Get count of currently active models.
        
        Returns:
            Number of active models
        """
        recent = self.get_recent_events()
        loaded = set()
        unloaded = set()
        
        for event in recent:
            if event.type == EventType.MODEL_LOADING_COMPLETED and event.algorithm:
                loaded.add(event.algorithm)
            elif event.type == EventType.MODEL_LOADING_FAILED and event.algorithm:
                unloaded.add(event.algorithm)
                
        return len(loaded - unloaded)
    
    def start_monitoring(self) -> None:
        """Start monitoring session."""
        self._monitoring_start_time = datetime.now()
        self.emit(Event(
            type=EventType.MONITORING_STARTED,
            timestamp=datetime.now(),
            component="monitoring",
            description="Started monitoring session"
        ))
    
    def stop_monitoring(self) -> None:
        """Stop monitoring session."""
        if self._monitoring_start_time:
            duration = (datetime.now() - self._monitoring_start_time).total_seconds()
            self.emit(Event(
                type=EventType.MONITORING_STOPPED,
                timestamp=datetime.now(),
                component="monitoring",
                description="Stopped monitoring session",
                duration_ms=duration * 1000,
                metadata={"session_duration": duration}
            ))
            self._monitoring_start_time = None


# Global event manager instance
event_manager = EventManager()

__all__ = ["event_manager", "Event", "EventType"] 