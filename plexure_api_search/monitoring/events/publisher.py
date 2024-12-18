import json
import logging
import os
import sys
import time

import zmq

from .event import Event

logger = logging.getLogger(__name__)


class Publisher:
    """Event publisher for sending events."""

    def __init__(self):
        """Initialize publisher."""
        self._context = zmq.Context()
        self._socket = None
        self._pid = os.getpid()
        self._process_name = os.path.basename(sys.argv[0])

    def start(self, address: str = "tcp://127.0.0.1:5555") -> None:
        """Start the publisher.

        Args:
            address: ZMQ address to connect to
        """
        if self._socket:
            return

        try:
            self._socket = self._context.socket(zmq.PUB)
            self._socket.setsockopt(zmq.LINGER, 0)
            # Publisher connects to subscriber
            self._socket.connect(address)
            # Allow time for connection
            time.sleep(0.1)
            logger.debug(f"Publisher connected to {address}")
        except Exception as e:
            logger.error(f"Failed to start publisher: {e}", exc_info=True)
            self.stop()
            raise

    def emit(self, event: Event) -> None:
        """Emit an event.

        Args:
            event: Event to emit
        """
        if not isinstance(event, Event):
            logger.error(f"Invalid event type: {type(event)}")
            return

        if not self._socket:
            logger.error("Cannot emit event: publisher not started")
            return

        try:
            # Add process info
            event.metadata["pid"] = self._pid
            event.metadata["process_name"] = self._process_name

            # Send event
            data = json.dumps(event.to_dict())
            self._socket.send_string(data, zmq.NOBLOCK)
            logger.debug(f"Event emitted: {event.type.name}")
        except zmq.ZMQError as e:
            if e.errno == zmq.EAGAIN:
                logger.warning("Event dropped: no subscribers")
            else:
                logger.error(f"Failed to send event: {e}", exc_info=True)
        except Exception as e:
            logger.error(f"Error emitting event: {e}", exc_info=True)

    def stop(self) -> None:
        """Stop the publisher."""
        if self._socket:
            self._socket.close()
            self._socket = None
        if self._context:
            self._context.term()
            self._context = None
        logger.debug("Publisher stopped")
