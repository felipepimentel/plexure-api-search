"""Base service class."""

from abc import ABC, abstractmethod
from typing import Any, Dict, Generic, Optional, TypeVar

from ..config import Config
from ..monitoring.events import Publisher
from ..monitoring.metrics import MetricsManager

T = TypeVar("T")


class BaseService(Generic[T], ABC):
    """Base service class with common functionality."""

    def __init__(
        self,
        config: Config,
        publisher: Publisher,
        metrics_manager: MetricsManager,
    ) -> None:
        """Initialize base service.

        Args:
            config: Application configuration
            publisher: Event publisher
            metrics_manager: Metrics manager
        """
        self.config = config
        self.publisher = publisher
        self.metrics = metrics_manager

    @abstractmethod
    async def initialize(self) -> None:
        """Initialize service resources."""
        pass

    @abstractmethod
    async def cleanup(self) -> None:
        """Clean up service resources."""
        pass

    @abstractmethod
    async def health_check(self) -> Dict[str, Any]:
        """Check service health.

        Returns:
            Health check results
        """
        pass

    async def start(self) -> None:
        """Start the service."""
        await self.initialize()

    async def stop(self) -> None:
        """Stop the service."""
        await self.cleanup()

    def get_status(self) -> Dict[str, Any]:
        """Get service status.

        Returns:
            Service status information
        """
        return {
            "healthy": True,
            "initialized": True,
            "config": self.config.__dict__,
        }


class ServiceException(Exception):
    """Base exception for service errors."""

    def __init__(
        self,
        message: str,
        service_name: str,
        error_code: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Initialize service exception.

        Args:
            message: Error message
            service_name: Name of the service that raised the exception
            error_code: Optional error code
            details: Optional error details
        """
        super().__init__(message)
        self.service_name = service_name
        self.error_code = error_code
        self.details = details or {}


class ServiceValidationError(ServiceException):
    """Validation error in service operations."""

    pass


class ServiceNotFoundError(ServiceException):
    """Resource not found in service operations."""

    pass


class ServiceUnavailableError(ServiceException):
    """Service is temporarily unavailable."""

    pass


__all__ = [
    "BaseService",
    "ServiceException",
    "ServiceValidationError",
    "ServiceNotFoundError",
    "ServiceUnavailableError",
] 