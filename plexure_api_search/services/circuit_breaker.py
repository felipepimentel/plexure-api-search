"""Circuit breaker pattern implementation for service resilience."""

import logging
import asyncio
from typing import Dict, List, Any, Optional, Set, Callable, Awaitable
from datetime import datetime
from enum import Enum
from dataclasses import dataclass

from ..config import Config
from ..monitoring.events import Event, EventType
from ..monitoring.metrics import MetricsManager
from .base import BaseService
from .events import PublisherService

logger = logging.getLogger(__name__)


class CircuitState(Enum):
    """Circuit breaker states."""

    CLOSED = "closed"  # Normal operation
    OPEN = "open"  # Service unavailable
    HALF_OPEN = "half_open"  # Testing recovery


@dataclass
class CircuitStats:
    """Circuit breaker statistics."""

    total_requests: int = 0
    failed_requests: int = 0
    success_requests: int = 0
    last_failure: Optional[datetime] = None
    last_success: Optional[datetime] = None
    consecutive_failures: int = 0
    consecutive_successes: int = 0
    total_timeouts: int = 0
    total_errors: int = 0
    average_response_time: float = 0.0


class CircuitConfig:
    """Configuration for circuit breaker."""

    def __init__(
        self,
        failure_threshold: int = 5,  # Number of failures before opening
        success_threshold: int = 3,  # Number of successes before closing
        timeout: float = 5.0,  # Request timeout in seconds
        reset_timeout: float = 60.0,  # Time before attempting reset
        half_open_timeout: float = 30.0,  # Time in half-open state
        error_threshold: float = 0.5,  # Error rate threshold
        min_requests: int = 10,  # Minimum requests before checking error rate
        exclude_exceptions: Optional[List[type]] = None,  # Exceptions to ignore
    ) -> None:
        """Initialize circuit config.

        Args:
            failure_threshold: Number of failures before opening circuit
            success_threshold: Number of successes before closing circuit
            timeout: Request timeout in seconds
            reset_timeout: Time before attempting reset in seconds
            half_open_timeout: Time in half-open state in seconds
            error_threshold: Error rate threshold (0.0 to 1.0)
            min_requests: Minimum requests before checking error rate
            exclude_exceptions: List of exception types to ignore
        """
        self.failure_threshold = failure_threshold
        self.success_threshold = success_threshold
        self.timeout = timeout
        self.reset_timeout = reset_timeout
        self.half_open_timeout = half_open_timeout
        self.error_threshold = error_threshold
        self.min_requests = min_requests
        self.exclude_exceptions = exclude_exceptions or []


class CircuitBreaker(BaseService[Dict[str, Any]]):
    """Circuit breaker implementation."""

    def __init__(
        self,
        config: Config,
        publisher: PublisherService,
        metrics_manager: MetricsManager,
        circuit_config: Optional[CircuitConfig] = None,
    ) -> None:
        """Initialize circuit breaker.

        Args:
            config: Application configuration
            publisher: Event publisher
            metrics_manager: Metrics manager
            circuit_config: Circuit breaker configuration
        """
        super().__init__(config, publisher, metrics_manager)
        self.circuit_config = circuit_config or CircuitConfig()
        self._initialized = False
        self._state = CircuitState.CLOSED
        self._stats = CircuitStats()
        self._last_state_change = datetime.now()
        self._lock = asyncio.Lock()

    async def initialize(self) -> None:
        """Initialize circuit breaker resources."""
        if self._initialized:
            return

        try:
            self._initialized = True
            self.publisher.publish(
                Event(
                    type=EventType.SERVICE_STARTED,
                    timestamp=datetime.now(),
                    component="circuit_breaker",
                    description="Circuit breaker initialized",
                    metadata={
                        "state": self._state.value,
                        "failure_threshold": self.circuit_config.failure_threshold,
                        "success_threshold": self.circuit_config.success_threshold,
                    },
                )
            )

        except Exception as e:
            logger.error(f"Failed to initialize circuit breaker: {e}")
            raise

    async def cleanup(self) -> None:
        """Clean up circuit breaker resources."""
        self._initialized = False
        self._state = CircuitState.CLOSED
        self._stats = CircuitStats()

        self.publisher.publish(
            Event(
                type=EventType.SERVICE_STOPPED,
                timestamp=datetime.now(),
                component="circuit_breaker",
                description="Circuit breaker stopped",
            )
        )

    async def health_check(self) -> Dict[str, Any]:
        """Check circuit breaker health.

        Returns:
            Health check results
        """
        error_rate = (
            self._stats.failed_requests / self._stats.total_requests
            if self._stats.total_requests > 0
            else 0.0
        )

        return {
            "service": "CircuitBreaker",
            "initialized": self._initialized,
            "state": self._state.value,
            "total_requests": self._stats.total_requests,
            "failed_requests": self._stats.failed_requests,
            "success_requests": self._stats.success_requests,
            "error_rate": error_rate,
            "consecutive_failures": self._stats.consecutive_failures,
            "consecutive_successes": self._stats.consecutive_successes,
            "average_response_time": self._stats.average_response_time,
            "status": "healthy" if self._initialized else "unhealthy",
        }

    async def execute(
        self,
        command: Callable[..., Awaitable[Any]],
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        """Execute command with circuit breaker protection.

        Args:
            command: Async function to execute
            *args: Positional arguments
            **kwargs: Keyword arguments

        Returns:
            Command result

        Raises:
            Exception: If circuit is open or command fails
        """
        if not self._initialized:
            await self.initialize()

        async with self._lock:
            # Check if circuit is open
            if self._state == CircuitState.OPEN:
                if self._should_reset():
                    await self._transition_to_half_open()
                else:
                    raise Exception("Circuit breaker is open")

            # Track request
            start_time = datetime.now()
            self._stats.total_requests += 1

            try:
                # Execute with timeout
                result = await asyncio.wait_for(
                    command(*args, **kwargs),
                    timeout=self.circuit_config.timeout,
                )

                # Track success
                await self._on_success(start_time)
                return result

            except asyncio.TimeoutError:
                # Track timeout
                self._stats.total_timeouts += 1
                await self._on_failure(start_time, "timeout")
                raise

            except Exception as e:
                # Check if exception should be ignored
                if not any(
                    isinstance(e, exc_type)
                    for exc_type in self.circuit_config.exclude_exceptions
                ):
                    # Track failure
                    self._stats.total_errors += 1
                    await self._on_failure(start_time, str(e))
                raise

    async def _on_success(self, start_time: datetime) -> None:
        """Handle successful request.

        Args:
            start_time: Request start time
        """
        # Update stats
        self._stats.success_requests += 1
        self._stats.consecutive_successes += 1
        self._stats.consecutive_failures = 0
        self._stats.last_success = datetime.now()

        # Update response time
        duration = (datetime.now() - start_time).total_seconds()
        self._stats.average_response_time = (
            (self._stats.average_response_time * (self._stats.total_requests - 1) + duration)
            / self._stats.total_requests
        )

        # Check if we should close circuit
        if (
            self._state == CircuitState.HALF_OPEN
            and self._stats.consecutive_successes >= self.circuit_config.success_threshold
        ):
            await self._transition_to_closed()

        # Emit event
        self.publisher.publish(
            Event(
                type=EventType.CIRCUIT_SUCCESS,
                timestamp=datetime.now(),
                component="circuit_breaker",
                description="Request succeeded",
                metadata={
                    "state": self._state.value,
                    "consecutive_successes": self._stats.consecutive_successes,
                    "response_time": duration,
                },
            )
        )

    async def _on_failure(self, start_time: datetime, error: str) -> None:
        """Handle failed request.

        Args:
            start_time: Request start time
            error: Error message
        """
        # Update stats
        self._stats.failed_requests += 1
        self._stats.consecutive_failures += 1
        self._stats.consecutive_successes = 0
        self._stats.last_failure = datetime.now()

        # Update response time
        duration = (datetime.now() - start_time).total_seconds()
        self._stats.average_response_time = (
            (self._stats.average_response_time * (self._stats.total_requests - 1) + duration)
            / self._stats.total_requests
        )

        # Check if we should open circuit
        if self._state == CircuitState.CLOSED:
            error_rate = self._stats.failed_requests / self._stats.total_requests
            if (
                self._stats.total_requests >= self.circuit_config.min_requests
                and error_rate >= self.circuit_config.error_threshold
            ) or (
                self._stats.consecutive_failures >= self.circuit_config.failure_threshold
            ):
                await self._transition_to_open()

        # Check if we should reopen circuit
        elif self._state == CircuitState.HALF_OPEN:
            await self._transition_to_open()

        # Emit event
        self.publisher.publish(
            Event(
                type=EventType.CIRCUIT_FAILURE,
                timestamp=datetime.now(),
                component="circuit_breaker",
                description="Request failed",
                metadata={
                    "state": self._state.value,
                    "consecutive_failures": self._stats.consecutive_failures,
                    "response_time": duration,
                    "error": error,
                },
            )
        )

    def _should_reset(self) -> bool:
        """Check if circuit should be reset.

        Returns:
            True if circuit should be reset
        """
        if self._state != CircuitState.OPEN:
            return False

        elapsed = (datetime.now() - self._last_state_change).total_seconds()
        return elapsed >= self.circuit_config.reset_timeout

    async def _transition_to_open(self) -> None:
        """Transition to open state."""
        self._state = CircuitState.OPEN
        self._last_state_change = datetime.now()

        self.publisher.publish(
            Event(
                type=EventType.CIRCUIT_OPENED,
                timestamp=datetime.now(),
                component="circuit_breaker",
                description="Circuit opened",
                metadata={
                    "error_rate": self._stats.failed_requests / self._stats.total_requests,
                    "consecutive_failures": self._stats.consecutive_failures,
                },
            )
        )

    async def _transition_to_half_open(self) -> None:
        """Transition to half-open state."""
        self._state = CircuitState.HALF_OPEN
        self._last_state_change = datetime.now()
        self._stats.consecutive_successes = 0
        self._stats.consecutive_failures = 0

        self.publisher.publish(
            Event(
                type=EventType.CIRCUIT_HALF_OPENED,
                timestamp=datetime.now(),
                component="circuit_breaker",
                description="Circuit half-opened",
            )
        )

    async def _transition_to_closed(self) -> None:
        """Transition to closed state."""
        self._state = CircuitState.CLOSED
        self._last_state_change = datetime.now()
        self._stats = CircuitStats()

        self.publisher.publish(
            Event(
                type=EventType.CIRCUIT_CLOSED,
                timestamp=datetime.now(),
                component="circuit_breaker",
                description="Circuit closed",
            )
        )


# Create service instance
circuit_breaker = CircuitBreaker(
    Config(),
    PublisherService(Config(), MetricsManager()),
    MetricsManager(),
)

__all__ = [
    "CircuitState",
    "CircuitStats",
    "CircuitConfig",
    "CircuitBreaker",
    "circuit_breaker",
] 