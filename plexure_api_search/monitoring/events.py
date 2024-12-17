"""Event system for tracking processing steps."""

import logging
import threading
import weakref
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum, auto
from typing import Any, Callable, Dict, List, Optional, Set
import statistics
import queue

logger = logging.getLogger(__name__)

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
    """Manages event subscriptions and progress tracking with thread safety."""
    
    def __init__(self, max_events: int = 1000, max_queue_size: int = 1000):
        """Initialize event manager.
        
        Args:
            max_events: Maximum number of events to store in history
            max_queue_size: Maximum size of event queue
        """
        self._subscribers: Set[int] = set()  # Store subscriber IDs
        self._subscriber_callbacks: Dict[int, Callable[[Event], None]] = {}
        self._events: List[Event] = []
        self._max_events = max_events
        self._monitoring_start_time: Optional[datetime] = None
        
        # Thread safety
        self._lock = threading.RLock()
        self._event_queue = queue.Queue(maxsize=max_queue_size)
        self._worker_thread: Optional[threading.Thread] = None
        self._running = False
        
        # Start event processing thread
        self._start_worker()
    
    def _start_worker(self) -> None:
        """Start the event processing worker thread."""
        if self._worker_thread is not None:
            return
            
        self._running = True
        self._worker_thread = threading.Thread(
            target=self._process_events,
            name="EventProcessor",
            daemon=True
        )
        self._worker_thread.start()
    
    def _process_events(self) -> None:
        """Process events from queue and notify subscribers."""
        while self._running:
            try:
                event = self._event_queue.get(timeout=0.1)
                self._handle_event(event)
                self._event_queue.task_done()
            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"Error processing event: {e}")
    
    def _handle_event(self, event: Event) -> None:
        """Handle a single event."""
        with self._lock:
            # Store event
            self._events.append(event)
            if len(self._events) > self._max_events:
                self._events = self._events[-self._max_events:]
            
            # Notify subscribers
            dead_subscribers = set()
            for subscriber_id, callback in self._subscriber_callbacks.items():
                try:
                    callback(event)
                except Exception as e:
                    logger.error(f"Error in subscriber {subscriber_id}: {e}")
                    dead_subscribers.add(subscriber_id)
            
            # Clean up dead subscribers
            for subscriber_id in dead_subscribers:
                self._unsubscribe_by_id(subscriber_id)
    
    def subscribe(self, callback: Callable[[Event], None]) -> int:
        """Subscribe to events.
        
        Args:
            callback: Function to call when events occur
            
        Returns:
            Subscriber ID for unsubscribing
        """
        with self._lock:
            subscriber_id = id(callback)
            self._subscribers.add(subscriber_id)
            self._subscriber_callbacks[subscriber_id] = callback
            return subscriber_id
    
    def unsubscribe(self, callback: Callable[[Event], None]) -> None:
        """Unsubscribe from events.
        
        Args:
            callback: Previously subscribed callback function
        """
        subscriber_id = id(callback)
        self._unsubscribe_by_id(subscriber_id)
    
    def _unsubscribe_by_id(self, subscriber_id: int) -> None:
        """Unsubscribe using subscriber ID."""
        with self._lock:
            self._subscribers.discard(subscriber_id)
            self._subscriber_callbacks.pop(subscriber_id, None)
    
    def emit(self, event: Event) -> None:
        """Emit an event to all subscribers.
        
        Args:
            event: Event to emit
        """
        try:
            self._event_queue.put(event, timeout=1.0)
        except queue.Full:
            logger.error("Event queue full, dropping event")
    
    def get_recent_events(self, minutes: int = 5) -> List[Event]:
        """Get events from the last N minutes.
        
        Args:
            minutes: Number of minutes to look back
            
        Returns:
            List of recent events
        """
        with self._lock:
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
        with self._lock:
            self._monitoring_start_time = datetime.now()
            self.emit(Event(
                type=EventType.MONITORING_STARTED,
                timestamp=datetime.now(),
                component="monitoring",
                description="Started monitoring session"
            ))
    
    def stop_monitoring(self) -> None:
        """Stop monitoring session."""
        with self._lock:
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
            
            # Stop worker thread
            self._running = False
            if self._worker_thread:
                self._worker_thread.join(timeout=1.0)
                self._worker_thread = None


# Global event manager instance
event_manager = EventManager()

__all__ = ["event_manager", "Event", "EventType"] 