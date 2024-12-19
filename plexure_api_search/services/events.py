"""Event service for publishing and subscribing to events."""

import json
import logging
import time
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional, Union

from dependency_injector.wiring import inject
from dependency_injector.providers import Provider

import zmq
from zmq.error import ZMQError

from ..config import config_instance

logger = logging.getLogger(__name__)

@dataclass
class Event:
    """Event data class."""
    type: str
    timestamp: datetime
    component: str
    description: str
    error: Optional[str] = None
    duration_ms: Optional[float] = None
    success: bool = True
    metadata: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert event to dictionary."""
        return {
            "type": self.type,
            "timestamp": self.timestamp.isoformat(),
            "component": self.component,
            "description": self.description,
            "error": self.error,
            "duration_ms": self.duration_ms,
            "success": self.success,
            "metadata": self.metadata or {},
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Event":
        """Create event from dictionary."""
        data["timestamp"] = datetime.fromisoformat(data["timestamp"])
        return cls(**data)

class PublisherService:
    """Event publisher service."""

    def __init__(self):
        """Initialize publisher service."""
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.PUB)
        self.started = False
        try:
            self.socket.bind(f"tcp://*:{config_instance.publisher_port}")
        except ZMQError as e:
            if e.errno == 98:  # Address already in use
                logger.warning(f"Publisher port {config_instance.publisher_port} already in use")
            else:
                logger.error(f"Failed to bind publisher socket: {e}")

    def start(self) -> None:
        """Start publisher service."""
        if not self.started:
            logger.info("Starting publisher service")
            self.started = True

    def stop(self) -> None:
        """Stop publisher service."""
        if self.started:
            logger.info("Stopping publisher service")
            self.socket.close()
            self.context.term()
            self.started = False

    def emit(self, event: Event) -> None:
        """Emit event.
        
        Args:
            event: Event to emit.
        """
        if not self.started:
            logger.warning("Publisher service not started")
            return

        try:
            # Convert event to JSON
            data = json.dumps(event.to_dict())
            
            # Send event
            self.socket.send_string(data)
            
            logger.debug(f"Emitted event: {event.type}")
            
        except Exception as e:
            logger.error(f"Failed to emit event: {e}")

class SubscriberService:
    """Event subscriber service."""

    def __init__(self):
        """Initialize subscriber service."""
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.SUB)
        try:
            self.socket.connect(f"tcp://localhost:{config_instance.publisher_port}")
            self.socket.setsockopt_string(zmq.SUBSCRIBE, "")
        except ZMQError as e:
            logger.warning(f"Failed to connect subscriber socket: {e}")
        self.started = False
        self.handlers: Dict[str, List[callable]] = {}

    def start(self) -> None:
        """Start subscriber service."""
        if not self.started:
            logger.info("Starting subscriber service")
            self.started = True

    def stop(self) -> None:
        """Stop subscriber service."""
        if self.started:
            logger.info("Stopping subscriber service")
            self.socket.close()
            self.context.term()
            self.started = False

    def subscribe(self, event_type: str, handler: callable) -> None:
        """Subscribe to event type.
        
        Args:
            event_type: Event type to subscribe to.
            handler: Handler function to call when event is received.
        """
        if event_type not in self.handlers:
            self.handlers[event_type] = []
        self.handlers[event_type].append(handler)
        logger.debug(f"Subscribed to event type: {event_type}")

    def unsubscribe(self, event_type: str, handler: callable) -> None:
        """Unsubscribe from event type.
        
        Args:
            event_type: Event type to unsubscribe from.
            handler: Handler function to remove.
        """
        if event_type in self.handlers:
            self.handlers[event_type].remove(handler)
            logger.debug(f"Unsubscribed from event type: {event_type}")

    def poll(self, timeout: int = 1000) -> Optional[Event]:
        """Poll for events.
        
        Args:
            timeout: Poll timeout in milliseconds.
            
        Returns:
            Event if received, None if timeout.
        """
        if not self.started:
            logger.warning("Subscriber service not started")
            return None

        try:
            # Check for message
            if self.socket.poll(timeout=timeout):
                # Receive message
                data = self.socket.recv_string()
                event_dict = json.loads(data)
                event = Event.from_dict(event_dict)
                
                # Call handlers
                if event.type in self.handlers:
                    for handler in self.handlers[event.type]:
                        try:
                            handler(event)
                        except Exception as e:
                            logger.error(f"Handler failed: {e}")
                
                return event
                
        except Exception as e:
            logger.error(f"Failed to poll events: {e}")
            
        return None

# Global instances
publisher = PublisherService()
subscriber = SubscriberService() 