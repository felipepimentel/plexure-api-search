import json
import logging
import threading
import time
from datetime import datetime, timedelta
from typing import Callable, Dict, List, Optional

import zmq

from .event import Event, EventType

logger = logging.getLogger(__name__)


class Subscriber:
    """Event subscriber for receiving events."""

    def __init__(self):
        """Initialize subscriber."""
        self._context = zmq.Context()
        self._socket = None
        self._running = False
        self._thread = None
        self._callbacks: Dict[str, List[Callable]] = {}
        self._events: List[Event] = []
        self._lock = threading.RLock()

    def start(self, address: str = "tcp://127.0.0.1:5555") -> None:
        """Start the subscriber.

        Args:
            address: ZMQ address to bind to
        """
        if self._running:
            return

        try:
            self._socket = self._context.socket(zmq.SUB)
            self._socket.setsockopt(zmq.LINGER, 0)
            self._socket.setsockopt_string(zmq.SUBSCRIBE, "")
            # Subscriber binds to address
            self._socket.bind(address)

            # Allow time for binding
            time.sleep(0.1)

            self._running = True
            self._thread = threading.Thread(
                target=self._receive_loop, daemon=True, name="EventSubscriber"
            )
            self._thread.start()
            logger.debug(f"Subscriber bound to {address}")
        except Exception as e:
            logger.error(f"Failed to start subscriber: {e}", exc_info=True)
            self.stop()
            raise

    def _receive_loop(self) -> None:
        """Main receive loop."""
        while self._running:
            try:
                if self._socket.poll(100):  # 100ms timeout
                    data = self._socket.recv_string()
                    event = Event.from_dict(json.loads(data))

                    with self._lock:
                        # Store event
                        self._events.append(event)
                        while len(self._events) > 1000:  # Keep last 1000 events
                            self._events.pop(0)

                        # Notify callbacks
                        self._notify_callbacks(event)
            except zmq.ZMQError as e:
                if self._running:
                    logger.error(f"ZMQ error: {e}", exc_info=True)
            except Exception as e:
                if self._running:
                    logger.error(f"Error in receive loop: {e}", exc_info=True)

    def _notify_callbacks(self, event: Event) -> None:
        """Notify callbacks about an event."""
        # Notify type-specific callbacks
        for callback in self._callbacks.get(event.type.name, []):
            try:
                callback(event)
            except Exception as e:
                logger.error(f"Error in callback: {e}", exc_info=True)

        # Notify ALL_EVENTS callbacks
        for callback in self._callbacks.get("ALL_EVENTS", []):
            try:
                callback(event)
            except Exception as e:
                logger.error(f"Error in callback: {e}", exc_info=True)

    def subscribe(
        self, callback: Callable[[Event], None], event_type: Optional[EventType] = None
    ) -> None:
        """Subscribe to events.

        Args:
            callback: Function to call for events
            event_type: Optional specific event type to subscribe to
        """
        with self._lock:
            key = event_type.name if event_type else "ALL_EVENTS"
            if key not in self._callbacks:
                self._callbacks[key] = []
            if callback not in self._callbacks[key]:
                self._callbacks[key].append(callback)
                logger.debug(f"Subscribed to {key}")

    def unsubscribe(
        self, callback: Callable[[Event], None], event_type: Optional[EventType] = None
    ) -> None:
        """Unsubscribe from events.

        Args:
            callback: Previously registered callback
            event_type: Optional specific event type
        """
        with self._lock:
            key = event_type.name if event_type else "ALL_EVENTS"
            if key in self._callbacks and callback in self._callbacks[key]:
                self._callbacks[key].remove(callback)
                logger.debug(f"Unsubscribed from {key}")

    def get_recent_events(self, minutes: int = 5) -> List[Event]:
        """Get recent events.

        Args:
            minutes: Number of minutes to look back

        Returns:
            List of recent events
        """
        with self._lock:
            cutoff = datetime.now() - timedelta(minutes=minutes)
            return [e for e in self._events if e.timestamp >= cutoff]

    def stop(self) -> None:
        """Stop the subscriber."""
        self._running = False

        if self._thread:
            self._thread.join(timeout=1.0)
            self._thread = None

        if self._socket:
            self._socket.close()
            self._socket = None

        if self._context:
            self._context.term()
            self._context = None

        with self._lock:
            self._callbacks.clear()
            self._events.clear()

        logger.debug("Subscriber stopped")
