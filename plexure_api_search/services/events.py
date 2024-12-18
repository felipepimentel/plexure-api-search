"""Event service implementation."""

import asyncio
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional, Set

import zmq
import zmq.asyncio
from dependency_injector.wiring import inject, provider

from ..config import Config
from ..monitoring.events import Event, EventType
from ..monitoring.metrics import MetricsManager
from .base import BaseService, ServiceException

logger = logging.getLogger(__name__)


class EventService(BaseService[Event]):
    """Base event service with common ZMQ functionality."""

    def __init__(
        self,
        config: Config,
        metrics_manager: MetricsManager,
        service_name: str,
    ) -> None:
        """Initialize event service.

        Args:
            config: Application configuration
            metrics_manager: Metrics manager
            service_name: Service name for identification
        """
        super().__init__(config, None, metrics_manager)
        self.service_name = service_name
        self.context = zmq.asyncio.Context()
        self.socket = None
        self._running = False
        self._initialized = False

    async def initialize(self) -> None:
        """Initialize ZMQ resources."""
        if self._initialized:
            return

        try:
            # Configure socket in derived classes
            await self._configure_socket()
            self._initialized = True
            logger.info(f"{self.service_name} initialized")

        except Exception as e:
            logger.error(f"Failed to initialize {self.service_name}: {e}")
            raise ServiceException(
                message=f"Failed to initialize {self.service_name}",
                service_name=self.service_name,
                error_code="INIT_FAILED",
                details={"error": str(e)},
            )

    async def cleanup(self) -> None:
        """Clean up ZMQ resources."""
        if self.socket:
            self.socket.close()
        if self.context:
            self.context.term()
        self._initialized = False
        logger.info(f"{self.service_name} cleaned up")

    async def health_check(self) -> Dict[str, Any]:
        """Check service health.

        Returns:
            Health check results
        """
        return {
            "service": self.service_name,
            "initialized": self._initialized,
            "running": self._running,
            "socket_bound": bool(self.socket),
        }

    async def _configure_socket(self) -> None:
        """Configure ZMQ socket. Implemented by derived classes."""
        raise NotImplementedError


class PublisherService(EventService):
    """Event publisher service."""

    def __init__(
        self,
        config: Config,
        metrics_manager: MetricsManager,
    ) -> None:
        """Initialize publisher service.

        Args:
            config: Application configuration
            metrics_manager: Metrics manager
        """
        super().__init__(config, metrics_manager, "EventPublisher")
        self.topic = b"events"

    async def _configure_socket(self) -> None:
        """Configure publisher socket."""
        self.socket = self.context.socket(zmq.PUB)
        self.socket.bind(f"tcp://*:{self.config.zmq_pub_port}")
        await asyncio.sleep(0.1)  # Allow time for binding

    async def publish(self, event: Event) -> None:
        """Publish an event.

        Args:
            event: Event to publish
        """
        if not self._initialized:
            await self.initialize()

        try:
            # Add publisher info
            event.publisher = self.service_name
            event.timestamp = datetime.now()

            # Serialize and send
            message = event.json().encode("utf-8")
            await self.socket.send_multipart([self.topic, message])

            # Update metrics
            self.metrics.increment("events_published", 1, {"type": event.type.value})

        except Exception as e:
            logger.error(f"Failed to publish event: {e}")
            self.metrics.increment("events_failed", 1, {"type": event.type.value})
            raise ServiceException(
                message="Failed to publish event",
                service_name=self.service_name,
                error_code="PUBLISH_FAILED",
                details={"error": str(e), "event": event.dict()},
            )


class SubscriberService(EventService):
    """Event subscriber service."""

    def __init__(
        self,
        config: Config,
        metrics_manager: MetricsManager,
    ) -> None:
        """Initialize subscriber service.

        Args:
            config: Application configuration
            metrics_manager: Metrics manager
        """
        super().__init__(config, metrics_manager, "EventSubscriber")
        self.subscribed_types: Set[EventType] = set()
        self.handlers: Dict[EventType, List[callable]] = {}
        self._queue: asyncio.Queue = asyncio.Queue()

    async def _configure_socket(self) -> None:
        """Configure subscriber socket."""
        self.socket = self.context.socket(zmq.SUB)
        self.socket.connect(f"tcp://localhost:{self.config.zmq_pub_port}")
        self.socket.setsockopt(zmq.SUBSCRIBE, b"events")
        await asyncio.sleep(0.1)  # Allow time for connection

    def subscribe(self, event_type: EventType, handler: callable) -> None:
        """Subscribe to an event type.

        Args:
            event_type: Event type to subscribe to
            handler: Handler function for the event
        """
        if event_type not in self.handlers:
            self.handlers[event_type] = []
        self.handlers[event_type].append(handler)
        self.subscribed_types.add(event_type)

    async def start_listening(self) -> None:
        """Start listening for events."""
        if not self._initialized:
            await self.initialize()

        self._running = True
        try:
            while self._running:
                if self.socket is None:
                    await asyncio.sleep(0.1)
                    continue

                try:
                    [topic, message] = await self.socket.recv_multipart()
                    event = Event.parse_raw(message.decode("utf-8"))

                    # Process event if we have handlers
                    if event.type in self.handlers:
                        for handler in self.handlers[event.type]:
                            try:
                                await handler(event)
                                self.metrics.increment(
                                    "events_processed",
                                    1,
                                    {"type": event.type.value},
                                )
                            except Exception as e:
                                logger.error(f"Event handler failed: {e}")
                                self.metrics.increment(
                                    "handler_errors",
                                    1,
                                    {"type": event.type.value},
                                )

                except Exception as e:
                    logger.error(f"Failed to process event: {e}")
                    self.metrics.increment("events_failed", 1)
                    await asyncio.sleep(0.1)

        except asyncio.CancelledError:
            logger.info("Event listener cancelled")
            self._running = False

        except Exception as e:
            logger.error(f"Event listener failed: {e}")
            self._running = False
            raise ServiceException(
                message="Event listener failed",
                service_name=self.service_name,
                error_code="LISTENER_FAILED",
                details={"error": str(e)},
            )

    async def stop_listening(self) -> None:
        """Stop listening for events."""
        self._running = False
        await self.cleanup()


# Create service instances
publisher_service = PublisherService(Config(), MetricsManager())
subscriber_service = SubscriberService(Config(), MetricsManager())

__all__ = [
    "EventService",
    "PublisherService",
    "SubscriberService",
    "publisher_service",
    "subscriber_service",
] 