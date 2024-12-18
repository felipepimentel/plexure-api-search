"""Event system for tracking processing steps using ZeroMQ pub/sub."""

from .event import Event, EventType
from .publisher import Publisher
from .subscriber import Subscriber

# Global instances
publisher = Publisher()
subscriber = Subscriber()

__all__ = ["Event", "EventType", "publisher", "subscriber"]
