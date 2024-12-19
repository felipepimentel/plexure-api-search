"""Base service class for all services."""

import logging
from typing import Any, Dict, Generic, Optional, TypeVar

from ..config import config_instance
from ..monitoring.metrics import MetricsManager

logger = logging.getLogger(__name__)

T = TypeVar("T")


class ServiceException(Exception):
    """Base service exception."""

    def __init__(
        self,
        message: str,
        service_name: str,
        error_code: str,
        details: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Initialize service exception.

        Args:
            message: Error message
            service_name: Service name
            error_code: Error code
            details: Additional error details
        """
        super().__init__(message)
        self.service_name = service_name
        self.error_code = error_code
        self.details = details or {}


class BaseService(Generic[T]):
    """Base service class."""

    def __init__(
        self,
        config: Optional[Any] = None,
        publisher: Optional[Any] = None,
        metrics_manager: Optional[MetricsManager] = None,
    ) -> None:
        """Initialize base service.

        Args:
            config: Configuration instance
            publisher: Event publisher
            metrics_manager: Metrics manager
        """
        self.config = config or config_instance
        self.publisher = publisher
        self.metrics = metrics_manager or MetricsManager()

    def initialize(self) -> None:
        """Initialize service resources."""
        raise NotImplementedError

    def cleanup(self) -> None:
        """Clean up service resources."""
        raise NotImplementedError

    def health_check(self) -> Dict[str, Any]:
        """Check service health.

        Returns:
            Health check results
        """
        raise NotImplementedError

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