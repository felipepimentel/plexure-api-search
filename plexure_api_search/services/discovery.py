"""Service discovery for distributed search nodes."""

import logging
import asyncio
from typing import Dict, List, Any, Optional, Set
from datetime import datetime
import json
import aiohttp
import socket
from dataclasses import dataclass, asdict

from ..config import Config
from ..monitoring.events import Event, EventType
from ..monitoring.metrics import MetricsManager
from .base import BaseService
from .events import PublisherService

logger = logging.getLogger(__name__)


@dataclass
class ServiceInfo:
    """Service information."""

    service_id: str
    service_type: str
    host: str
    port: int
    metadata: Dict[str, Any]
    health_check_url: str
    last_heartbeat: datetime
    status: str = "unknown"  # unknown, healthy, unhealthy
    tags: List[str] = None
    version: str = "1.0.0"
    weight: float = 1.0


class DiscoveryConfig:
    """Configuration for service discovery."""

    def __init__(
        self,
        discovery_host: str = "localhost",
        discovery_port: int = 8500,  # Default Consul port
        heartbeat_interval: float = 10.0,
        deregister_after: float = 30.0,
        health_check_interval: float = 5.0,
        health_check_timeout: float = 3.0,
        enable_auto_discovery: bool = True,
        enable_health_checks: bool = True,
        enable_load_balancing: bool = True,
        enable_failover: bool = True,
        service_prefix: str = "plexure",
    ) -> None:
        """Initialize discovery config.

        Args:
            discovery_host: Discovery service hostname
            discovery_port: Discovery service port
            heartbeat_interval: Heartbeat interval in seconds
            deregister_after: Time after which to deregister service
            health_check_interval: Health check interval in seconds
            health_check_timeout: Health check timeout in seconds
            enable_auto_discovery: Whether to enable auto-discovery
            enable_health_checks: Whether to enable health checks
            enable_load_balancing: Whether to enable load balancing
            enable_failover: Whether to enable failover
            service_prefix: Prefix for service names
        """
        self.discovery_host = discovery_host
        self.discovery_port = discovery_port
        self.heartbeat_interval = heartbeat_interval
        self.deregister_after = deregister_after
        self.health_check_interval = health_check_interval
        self.health_check_timeout = health_check_timeout
        self.enable_auto_discovery = enable_auto_discovery
        self.enable_health_checks = enable_health_checks
        self.enable_load_balancing = enable_load_balancing
        self.enable_failover = enable_failover
        self.service_prefix = service_prefix


class ServiceDiscovery(BaseService[Dict[str, Any]]):
    """Service discovery manager."""

    def __init__(
        self,
        config: Config,
        publisher: PublisherService,
        metrics_manager: MetricsManager,
        discovery_config: Optional[DiscoveryConfig] = None,
    ) -> None:
        """Initialize service discovery.

        Args:
            config: Application configuration
            publisher: Event publisher
            metrics_manager: Metrics manager
            discovery_config: Discovery configuration
        """
        super().__init__(config, publisher, metrics_manager)
        self.discovery_config = discovery_config or DiscoveryConfig()
        self._initialized = False
        self._services: Dict[str, ServiceInfo] = {}
        self._local_services: Set[str] = set()
        self._heartbeat_task: Optional[asyncio.Task] = None
        self._discovery_task: Optional[asyncio.Task] = None
        self._session: Optional[aiohttp.ClientSession] = None

    async def initialize(self) -> None:
        """Initialize discovery resources."""
        if self._initialized:
            return

        try:
            # Create HTTP session
            self._session = aiohttp.ClientSession()

            # Start background tasks
            if self.discovery_config.enable_auto_discovery:
                self._discovery_task = asyncio.create_task(self._discovery_loop())
            if self.discovery_config.enable_health_checks:
                self._heartbeat_task = asyncio.create_task(self._heartbeat_loop())

            self._initialized = True
            self.publisher.publish(
                Event(
                    type=EventType.SERVICE_STARTED,
                    timestamp=datetime.now(),
                    component="service_discovery",
                    description="Service discovery initialized",
                    metadata={
                        "auto_discovery": self.discovery_config.enable_auto_discovery,
                        "health_checks": self.discovery_config.enable_health_checks,
                    },
                )
            )

        except Exception as e:
            logger.error(f"Failed to initialize service discovery: {e}")
            raise

    async def cleanup(self) -> None:
        """Clean up discovery resources."""
        # Cancel background tasks
        if self._discovery_task:
            self._discovery_task.cancel()
            try:
                await self._discovery_task
            except asyncio.CancelledError:
                pass

        if self._heartbeat_task:
            self._heartbeat_task.cancel()
            try:
                await self._heartbeat_task
            except asyncio.CancelledError:
                pass

        # Deregister local services
        for service_id in self._local_services:
            await self.deregister_service(service_id)

        # Close HTTP session
        if self._session:
            await self._session.close()

        self._initialized = False
        self._services.clear()
        self._local_services.clear()

        self.publisher.publish(
            Event(
                type=EventType.SERVICE_STOPPED,
                timestamp=datetime.now(),
                component="service_discovery",
                description="Service discovery stopped",
            )
        )

    async def health_check(self) -> Dict[str, Any]:
        """Check discovery health.

        Returns:
            Health check results
        """
        # Count healthy services
        healthy_services = sum(
            1 for service in self._services.values()
            if service.status == "healthy"
        )

        return {
            "service": "ServiceDiscovery",
            "initialized": self._initialized,
            "total_services": len(self._services),
            "healthy_services": healthy_services,
            "local_services": len(self._local_services),
            "status": "healthy" if self._initialized else "unhealthy",
        }

    async def register_service(
        self,
        service_type: str,
        host: str,
        port: int,
        metadata: Optional[Dict[str, Any]] = None,
        tags: Optional[List[str]] = None,
        version: str = "1.0.0",
        weight: float = 1.0,
    ) -> str:
        """Register service with discovery.

        Args:
            service_type: Type of service
            host: Service hostname
            port: Service port
            metadata: Service metadata
            tags: Service tags
            version: Service version
            weight: Service weight

        Returns:
            Service ID
        """
        if not self._initialized:
            await self.initialize()

        try:
            # Generate service ID
            service_id = f"{self.discovery_config.service_prefix}_{service_type}_{host}_{port}"

            # Create service info
            service_info = ServiceInfo(
                service_id=service_id,
                service_type=service_type,
                host=host,
                port=port,
                metadata=metadata or {},
                health_check_url=f"http://{host}:{port}/health",
                last_heartbeat=datetime.now(),
                status="unknown",
                tags=tags or [],
                version=version,
                weight=weight,
            )

            # Register with discovery service
            async with self._session.put(
                f"http://{self.discovery_config.discovery_host}:"
                f"{self.discovery_config.discovery_port}/v1/agent/service/register",
                json={
                    "ID": service_id,
                    "Name": service_type,
                    "Tags": tags or [],
                    "Address": host,
                    "Port": port,
                    "Meta": metadata or {},
                    "Check": {
                        "HTTP": service_info.health_check_url,
                        "Interval": f"{self.discovery_config.health_check_interval}s",
                        "Timeout": f"{self.discovery_config.health_check_timeout}s",
                        "DeregisterCriticalServiceAfter": f"{self.discovery_config.deregister_after}s",
                    },
                },
            ) as response:
                if response.status != 200:
                    raise Exception(f"Failed to register service: {await response.text()}")

            # Add to local registry
            self._services[service_id] = service_info
            self._local_services.add(service_id)

            # Emit event
            self.publisher.publish(
                Event(
                    type=EventType.SERVICE_REGISTERED,
                    timestamp=datetime.now(),
                    component="service_discovery",
                    description=f"Service registered: {service_id}",
                    metadata=asdict(service_info),
                )
            )

            return service_id

        except Exception as e:
            logger.error(f"Service registration failed: {e}")
            raise

    async def deregister_service(self, service_id: str) -> None:
        """Deregister service from discovery.

        Args:
            service_id: Service identifier
        """
        if not self._initialized:
            return

        try:
            # Deregister from discovery service
            async with self._session.put(
                f"http://{self.discovery_config.discovery_host}:"
                f"{self.discovery_config.discovery_port}/v1/agent/service/deregister/{service_id}",
            ) as response:
                if response.status != 200:
                    raise Exception(f"Failed to deregister service: {await response.text()}")

            # Remove from local registry
            if service_id in self._services:
                del self._services[service_id]
            self._local_services.discard(service_id)

            # Emit event
            self.publisher.publish(
                Event(
                    type=EventType.SERVICE_DEREGISTERED,
                    timestamp=datetime.now(),
                    component="service_discovery",
                    description=f"Service deregistered: {service_id}",
                    metadata={"service_id": service_id},
                )
            )

        except Exception as e:
            logger.error(f"Service deregistration failed: {e}")
            raise

    async def get_services(
        self,
        service_type: Optional[str] = None,
        tag: Optional[str] = None,
        healthy_only: bool = True,
    ) -> List[ServiceInfo]:
        """Get registered services.

        Args:
            service_type: Filter by service type
            tag: Filter by tag
            healthy_only: Whether to return only healthy services

        Returns:
            List of service information
        """
        if not self._initialized:
            await self.initialize()

        try:
            # Get services from discovery service
            async with self._session.get(
                f"http://{self.discovery_config.discovery_host}:"
                f"{self.discovery_config.discovery_port}/v1/agent/services",
            ) as response:
                if response.status != 200:
                    raise Exception(f"Failed to get services: {await response.text()}")
                services = await response.json()

            # Filter and convert services
            result = []
            for service_data in services.values():
                # Skip if wrong type
                if service_type and service_data["Service"] != service_type:
                    continue

                # Skip if wrong tag
                if tag and tag not in service_data.get("Tags", []):
                    continue

                # Create service info
                service_info = ServiceInfo(
                    service_id=service_data["ID"],
                    service_type=service_data["Service"],
                    host=service_data["Address"],
                    port=service_data["Port"],
                    metadata=service_data.get("Meta", {}),
                    health_check_url=service_data.get("Check", {}).get("HTTP", ""),
                    last_heartbeat=datetime.now(),  # TODO: Get from health check
                    status=service_data.get("Status", "unknown"),
                    tags=service_data.get("Tags", []),
                    version=service_data.get("Meta", {}).get("version", "1.0.0"),
                    weight=float(service_data.get("Meta", {}).get("weight", 1.0)),
                )

                # Skip if unhealthy
                if healthy_only and service_info.status != "healthy":
                    continue

                result.append(service_info)

            return result

        except Exception as e:
            logger.error(f"Get services failed: {e}")
            return []

    async def _discovery_loop(self) -> None:
        """Background task for service discovery."""
        while True:
            try:
                # Sleep for discovery interval
                await asyncio.sleep(self.discovery_config.health_check_interval)

                # Get all services
                services = await self.get_services(healthy_only=False)

                # Update local registry
                for service in services:
                    self._services[service.service_id] = service

                # Remove stale services
                now = datetime.now()
                stale_services = []
                for service_id, service in self._services.items():
                    if service_id not in self._local_services:
                        age = (now - service.last_heartbeat).total_seconds()
                        if age > self.discovery_config.deregister_after:
                            stale_services.append(service_id)

                for service_id in stale_services:
                    del self._services[service_id]

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Service discovery failed: {e}")

    async def _heartbeat_loop(self) -> None:
        """Background task for service heartbeat."""
        while True:
            try:
                # Sleep for heartbeat interval
                await asyncio.sleep(self.discovery_config.heartbeat_interval)

                # Send heartbeat for local services
                for service_id in self._local_services:
                    service = self._services.get(service_id)
                    if not service:
                        continue

                    # Update health check
                    async with self._session.put(
                        f"http://{self.discovery_config.discovery_host}:"
                        f"{self.discovery_config.discovery_port}/v1/agent/check/pass/service:{service_id}",
                    ) as response:
                        if response.status != 200:
                            logger.error(f"Heartbeat failed for {service_id}: {await response.text()}")
                            continue

                    # Update last heartbeat
                    service.last_heartbeat = datetime.now()
                    service.status = "healthy"

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Service heartbeat failed: {e}")


# Create service instance
service_discovery = ServiceDiscovery(
    Config(),
    PublisherService(Config(), MetricsManager()),
    MetricsManager(),
)

__all__ = [
    "ServiceInfo",
    "DiscoveryConfig",
    "ServiceDiscovery",
    "service_discovery",
] 