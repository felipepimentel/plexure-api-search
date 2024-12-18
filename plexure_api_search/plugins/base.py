"""Plugin system for extending application functionality."""

import logging
import asyncio
from typing import Dict, List, Any, Optional, Set, Type
from abc import ABC, abstractmethod
from dataclasses import dataclass
import importlib
import os
import sys
import pkg_resources

from ..config import Config
from ..monitoring.events import Event, EventType
from ..monitoring.metrics import MetricsManager
from ..services.base import BaseService
from ..services.events import PublisherService

logger = logging.getLogger(__name__)


@dataclass
class PluginInfo:
    """Plugin information."""

    name: str  # Plugin name
    version: str  # Plugin version
    description: str  # Plugin description
    author: str  # Plugin author
    dependencies: List[str]  # Required dependencies
    entry_point: str  # Plugin entry point
    enabled: bool = True  # Whether plugin is enabled


class PluginInterface(ABC):
    """Base interface for plugins."""

    @abstractmethod
    async def initialize(self) -> None:
        """Initialize plugin resources."""
        pass

    @abstractmethod
    async def cleanup(self) -> None:
        """Clean up plugin resources."""
        pass

    @abstractmethod
    async def health_check(self) -> Dict[str, Any]:
        """Check plugin health.

        Returns:
            Health check results
        """
        pass


class SearchPlugin(PluginInterface):
    """Base class for search plugins."""

    @abstractmethod
    async def preprocess_query(self, query: str) -> str:
        """Preprocess search query.

        Args:
            query: Original query

        Returns:
            Processed query
        """
        return query

    @abstractmethod
    async def postprocess_results(
        self,
        results: List[Dict[str, Any]],
        query: str,
    ) -> List[Dict[str, Any]]:
        """Postprocess search results.

        Args:
            results: Search results
            query: Original query

        Returns:
            Processed results
        """
        return results


class IndexPlugin(PluginInterface):
    """Base class for indexing plugins."""

    @abstractmethod
    async def preprocess_document(
        self,
        document: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Preprocess document before indexing.

        Args:
            document: Original document

        Returns:
            Processed document
        """
        return document

    @abstractmethod
    async def postprocess_document(
        self,
        document: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Postprocess document after indexing.

        Args:
            document: Indexed document

        Returns:
            Processed document
        """
        return document


class PluginManager(BaseService[Dict[str, Any]]):
    """Plugin manager implementation."""

    def __init__(
        self,
        config: Config,
        publisher: PublisherService,
        metrics_manager: MetricsManager,
    ) -> None:
        """Initialize plugin manager.

        Args:
            config: Application configuration
            publisher: Event publisher
            metrics_manager: Metrics manager
        """
        super().__init__(config, publisher, metrics_manager)
        self._initialized = False
        self._plugins: Dict[str, PluginInterface] = {}
        self._plugin_info: Dict[str, PluginInfo] = {}
        self._search_plugins: List[SearchPlugin] = []
        self._index_plugins: List[IndexPlugin] = []

    async def initialize(self) -> None:
        """Initialize plugin resources."""
        if self._initialized:
            return

        try:
            # Discover plugins
            await self._discover_plugins()

            # Initialize plugins
            for plugin in self._plugins.values():
                await plugin.initialize()

            self._initialized = True
            self.publisher.publish(
                Event(
                    type=EventType.SERVICE_STARTED,
                    timestamp=datetime.now(),
                    component="plugin_manager",
                    description="Plugin manager initialized",
                    metadata={
                        "total_plugins": len(self._plugins),
                        "search_plugins": len(self._search_plugins),
                        "index_plugins": len(self._index_plugins),
                    },
                )
            )

        except Exception as e:
            logger.error(f"Failed to initialize plugin manager: {e}")
            raise

    async def cleanup(self) -> None:
        """Clean up plugin resources."""
        # Clean up plugins in reverse order
        for plugin in reversed(list(self._plugins.values())):
            try:
                await plugin.cleanup()
            except Exception as e:
                logger.error(f"Failed to clean up plugin {plugin.__class__.__name__}: {e}")

        self._initialized = False
        self._plugins.clear()
        self._plugin_info.clear()
        self._search_plugins.clear()
        self._index_plugins.clear()

        self.publisher.publish(
            Event(
                type=EventType.SERVICE_STOPPED,
                timestamp=datetime.now(),
                component="plugin_manager",
                description="Plugin manager stopped",
            )
        )

    async def health_check(self) -> Dict[str, Any]:
        """Check plugin manager health.

        Returns:
            Health check results
        """
        # Check plugin health
        plugin_health = {}
        for name, plugin in self._plugins.items():
            try:
                health = await plugin.health_check()
                plugin_health[name] = health
            except Exception as e:
                plugin_health[name] = {
                    "status": "unhealthy",
                    "error": str(e),
                }

        return {
            "service": "PluginManager",
            "initialized": self._initialized,
            "total_plugins": len(self._plugins),
            "search_plugins": len(self._search_plugins),
            "index_plugins": len(self._index_plugins),
            "plugin_health": plugin_health,
            "status": "healthy" if self._initialized else "unhealthy",
        }

    async def _discover_plugins(self) -> None:
        """Discover available plugins."""
        # Check plugin directories
        plugin_dirs = [
            os.path.join(os.path.dirname(__file__), "plugins"),  # Built-in plugins
            os.path.expanduser("~/.plexure/plugins"),  # User plugins
            "/etc/plexure/plugins",  # System plugins
        ]

        for plugin_dir in plugin_dirs:
            if os.path.exists(plugin_dir):
                sys.path.append(plugin_dir)
                for file in os.listdir(plugin_dir):
                    if file.endswith(".py") and not file.startswith("_"):
                        module_name = file[:-3]
                        try:
                            module = importlib.import_module(module_name)
                            await self._load_plugin(module)
                        except Exception as e:
                            logger.error(f"Failed to load plugin {module_name}: {e}")

        # Check entry points
        for entry_point in pkg_resources.iter_entry_points("plexure_plugins"):
            try:
                plugin_class = entry_point.load()
                plugin = plugin_class(self.config, self.publisher, self.metrics)
                await self._register_plugin(plugin, entry_point.name)
            except Exception as e:
                logger.error(f"Failed to load plugin {entry_point.name}: {e}")

    async def _load_plugin(self, module: Any) -> None:
        """Load plugin from module.

        Args:
            module: Plugin module
        """
        for attr_name in dir(module):
            attr = getattr(module, attr_name)
            if (
                isinstance(attr, type)
                and issubclass(attr, PluginInterface)
                and attr not in {PluginInterface, SearchPlugin, IndexPlugin}
            ):
                try:
                    plugin = attr(self.config, self.publisher, self.metrics)
                    await self._register_plugin(plugin, attr_name)
                except Exception as e:
                    logger.error(f"Failed to initialize plugin {attr_name}: {e}")

    async def _register_plugin(
        self,
        plugin: PluginInterface,
        name: str,
    ) -> None:
        """Register plugin.

        Args:
            plugin: Plugin instance
            name: Plugin name
        """
        # Get plugin info
        info = getattr(plugin, "plugin_info", None)
        if not info:
            info = PluginInfo(
                name=name,
                version="0.1.0",
                description="No description available",
                author="Unknown",
                dependencies=[],
                entry_point=f"{plugin.__class__.__module__}.{plugin.__class__.__name__}",
            )

        # Check dependencies
        for dependency in info.dependencies:
            try:
                importlib.import_module(dependency)
            except ImportError:
                logger.error(f"Plugin {name} requires {dependency}")
                return

        # Register plugin
        self._plugins[name] = plugin
        self._plugin_info[name] = info

        # Register by type
        if isinstance(plugin, SearchPlugin):
            self._search_plugins.append(plugin)
        elif isinstance(plugin, IndexPlugin):
            self._index_plugins.append(plugin)

        # Emit event
        self.publisher.publish(
            Event(
                type=EventType.PLUGIN_REGISTERED,
                timestamp=datetime.now(),
                component="plugin_manager",
                description=f"Plugin registered: {name}",
                metadata={
                    "name": name,
                    "version": info.version,
                    "description": info.description,
                    "author": info.author,
                    "type": plugin.__class__.__name__,
                },
            )
        )

    async def preprocess_query(self, query: str) -> str:
        """Preprocess search query through plugins.

        Args:
            query: Original query

        Returns:
            Processed query
        """
        if not self._initialized:
            return query

        try:
            # Process through each search plugin
            current_query = query
            for plugin in self._search_plugins:
                try:
                    current_query = await plugin.preprocess_query(current_query)
                except Exception as e:
                    logger.error(f"Plugin {plugin.__class__.__name__} query preprocessing failed: {e}")

            return current_query

        except Exception as e:
            logger.error(f"Query preprocessing failed: {e}")
            return query

    async def postprocess_results(
        self,
        results: List[Dict[str, Any]],
        query: str,
    ) -> List[Dict[str, Any]]:
        """Postprocess search results through plugins.

        Args:
            results: Search results
            query: Original query

        Returns:
            Processed results
        """
        if not self._initialized:
            return results

        try:
            # Process through each search plugin
            current_results = results
            for plugin in self._search_plugins:
                try:
                    current_results = await plugin.postprocess_results(current_results, query)
                except Exception as e:
                    logger.error(f"Plugin {plugin.__class__.__name__} result postprocessing failed: {e}")

            return current_results

        except Exception as e:
            logger.error(f"Result postprocessing failed: {e}")
            return results

    async def preprocess_document(
        self,
        document: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Preprocess document through plugins.

        Args:
            document: Original document

        Returns:
            Processed document
        """
        if not self._initialized:
            return document

        try:
            # Process through each index plugin
            current_document = document
            for plugin in self._index_plugins:
                try:
                    current_document = await plugin.preprocess_document(current_document)
                except Exception as e:
                    logger.error(f"Plugin {plugin.__class__.__name__} document preprocessing failed: {e}")

            return current_document

        except Exception as e:
            logger.error(f"Document preprocessing failed: {e}")
            return document

    async def postprocess_document(
        self,
        document: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Postprocess document through plugins.

        Args:
            document: Indexed document

        Returns:
            Processed document
        """
        if not self._initialized:
            return document

        try:
            # Process through each index plugin
            current_document = document
            for plugin in self._index_plugins:
                try:
                    current_document = await plugin.postprocess_document(current_document)
                except Exception as e:
                    logger.error(f"Plugin {plugin.__class__.__name__} document postprocessing failed: {e}")

            return current_document

        except Exception as e:
            logger.error(f"Document postprocessing failed: {e}")
            return document


# Create manager instance
plugin_manager = PluginManager(
    Config(),
    PublisherService(Config(), MetricsManager()),
    MetricsManager(),
)

__all__ = [
    "PluginInfo",
    "PluginInterface",
    "SearchPlugin",
    "IndexPlugin",
    "PluginManager",
    "plugin_manager",
] 