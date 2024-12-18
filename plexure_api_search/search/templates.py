"""Query templates for common API search patterns."""

import logging
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime

from ..config import Config
from ..monitoring.events import Event, EventType
from ..monitoring.metrics import MetricsManager
from ..services.base import BaseService
from ..services.events import PublisherService

logger = logging.getLogger(__name__)


class QueryTemplate:
    """Template for common API search patterns."""

    def __init__(
        self,
        name: str,
        pattern: str,
        description: str,
        examples: List[str],
        parameters: List[str],
        category: str,
    ) -> None:
        """Initialize query template.

        Args:
            name: Template name
            pattern: Query pattern with placeholders
            description: Template description
            examples: Example queries
            parameters: Required parameters
            category: Template category
        """
        self.name = name
        self.pattern = pattern
        self.description = description
        self.examples = examples
        self.parameters = parameters
        self.category = category


class TemplateManager(BaseService[Dict[str, Any]]):
    """Template management service for query enhancement."""

    def __init__(
        self,
        config: Config,
        publisher: PublisherService,
        metrics_manager: MetricsManager,
    ) -> None:
        """Initialize template manager.

        Args:
            config: Application configuration
            publisher: Event publisher
            metrics_manager: Metrics manager
        """
        super().__init__(config, publisher, metrics_manager)
        self._initialized = False
        self._templates: Dict[str, QueryTemplate] = {}
        self._load_default_templates()

    def _load_default_templates(self) -> None:
        """Load default query templates."""
        # CRUD templates
        self._templates["create"] = QueryTemplate(
            name="create",
            pattern="create {resource} with {data}",
            description="Create a new resource",
            examples=[
                "create user with name and email",
                "create order with items",
                "create product with details",
            ],
            parameters=["resource", "data"],
            category="crud",
        )

        self._templates["read"] = QueryTemplate(
            name="read",
            pattern="get {resource} by {id}",
            description="Get a resource by ID",
            examples=[
                "get user by id",
                "get order by number",
                "get product by sku",
            ],
            parameters=["resource", "id"],
            category="crud",
        )

        self._templates["update"] = QueryTemplate(
            name="update",
            pattern="update {resource} {id} with {data}",
            description="Update an existing resource",
            examples=[
                "update user 123 with new email",
                "update order status",
                "update product price",
            ],
            parameters=["resource", "id", "data"],
            category="crud",
        )

        self._templates["delete"] = QueryTemplate(
            name="delete",
            pattern="delete {resource} {id}",
            description="Delete a resource",
            examples=[
                "delete user 123",
                "delete order",
                "delete product",
            ],
            parameters=["resource", "id"],
            category="crud",
        )

        # List templates
        self._templates["list"] = QueryTemplate(
            name="list",
            pattern="list {resource} {filter}",
            description="List resources with optional filter",
            examples=[
                "list users by role",
                "list orders by status",
                "list products by category",
            ],
            parameters=["resource", "filter"],
            category="list",
        )

        # Search templates
        self._templates["search"] = QueryTemplate(
            name="search",
            pattern="search {resource} with {criteria}",
            description="Search resources by criteria",
            examples=[
                "search users by name",
                "search orders by date",
                "search products by price",
            ],
            parameters=["resource", "criteria"],
            category="search",
        )

        # Filter templates
        self._templates["filter"] = QueryTemplate(
            name="filter",
            pattern="filter {resource} where {condition}",
            description="Filter resources by condition",
            examples=[
                "filter users where active",
                "filter orders where pending",
                "filter products where in stock",
            ],
            parameters=["resource", "condition"],
            category="filter",
        )

        # Sort templates
        self._templates["sort"] = QueryTemplate(
            name="sort",
            pattern="sort {resource} by {field} {direction}",
            description="Sort resources by field",
            examples=[
                "sort users by name asc",
                "sort orders by date desc",
                "sort products by price",
            ],
            parameters=["resource", "field", "direction"],
            category="sort",
        )

        # Aggregate templates
        self._templates["aggregate"] = QueryTemplate(
            name="aggregate",
            pattern="aggregate {resource} by {field}",
            description="Aggregate resources by field",
            examples=[
                "aggregate users by role",
                "aggregate orders by status",
                "aggregate products by category",
            ],
            parameters=["resource", "field"],
            category="aggregate",
        )

        # Export templates
        self._templates["export"] = QueryTemplate(
            name="export",
            pattern="export {resource} as {format}",
            description="Export resources in format",
            examples=[
                "export users as csv",
                "export orders as json",
                "export products as excel",
            ],
            parameters=["resource", "format"],
            category="export",
        )

        # Import templates
        self._templates["import"] = QueryTemplate(
            name="import",
            pattern="import {resource} from {source}",
            description="Import resources from source",
            examples=[
                "import users from csv",
                "import orders from json",
                "import products from excel",
            ],
            parameters=["resource", "source"],
            category="import",
        )

        # Validate templates
        self._templates["validate"] = QueryTemplate(
            name="validate",
            pattern="validate {resource} {data}",
            description="Validate resource data",
            examples=[
                "validate user email",
                "validate order items",
                "validate product price",
            ],
            parameters=["resource", "data"],
            category="validate",
        )

        # Status templates
        self._templates["status"] = QueryTemplate(
            name="status",
            pattern="get {resource} status",
            description="Get resource status",
            examples=[
                "get system status",
                "get service health",
                "get api status",
            ],
            parameters=["resource"],
            category="status",
        )

    async def initialize(self) -> None:
        """Initialize template manager."""
        if self._initialized:
            return

        try:
            self._initialized = True
            self.publisher.publish(
                Event(
                    type=EventType.SERVICE_STARTED,
                    timestamp=datetime.now(),
                    component="template_manager",
                    description="Template manager initialized",
                    metadata={
                        "num_templates": len(self._templates),
                    },
                )
            )

        except Exception as e:
            logger.error(f"Failed to initialize template manager: {e}")
            raise

    async def cleanup(self) -> None:
        """Clean up template manager resources."""
        self._initialized = False
        self._templates.clear()

        self.publisher.publish(
            Event(
                type=EventType.SERVICE_STOPPED,
                timestamp=datetime.now(),
                component="template_manager",
                description="Template manager stopped",
            )
        )

    async def health_check(self) -> Dict[str, Any]:
        """Check template manager health.

        Returns:
            Health check results
        """
        return {
            "service": "TemplateManager",
            "initialized": self._initialized,
            "num_templates": len(self._templates),
        }

    def get_template(self, name: str) -> Optional[QueryTemplate]:
        """Get template by name.

        Args:
            name: Template name

        Returns:
            Query template if found, None otherwise
        """
        return self._templates.get(name)

    def get_templates_by_category(self, category: str) -> List[QueryTemplate]:
        """Get templates by category.

        Args:
            category: Template category

        Returns:
            List of matching templates
        """
        return [
            template for template in self._templates.values()
            if template.category == category
        ]

    def find_matching_template(self, query: str) -> Optional[Tuple[QueryTemplate, Dict[str, str]]]:
        """Find template matching query and extract parameters.

        Args:
            query: Search query

        Returns:
            Tuple of (template, parameters) if found, None otherwise
        """
        try:
            # Emit template matching started event
            self.publisher.publish(
                Event(
                    type=EventType.TEMPLATE_MATCHING_STARTED,
                    timestamp=datetime.now(),
                    component="template_manager",
                    description=f"Finding template for query: {query}",
                )
            )

            # Try each template
            for template in self._templates.values():
                params = self._extract_parameters(query, template)
                if params:
                    # Emit success event
                    self.publisher.publish(
                        Event(
                            type=EventType.TEMPLATE_MATCHING_COMPLETED,
                            timestamp=datetime.now(),
                            component="template_manager",
                            description="Template found",
                            metadata={
                                "template": template.name,
                                "parameters": params,
                            },
                        )
                    )
                    return template, params

            # No matching template found
            self.publisher.publish(
                Event(
                    type=EventType.TEMPLATE_MATCHING_COMPLETED,
                    timestamp=datetime.now(),
                    component="template_manager",
                    description="No matching template found",
                )
            )
            return None

        except Exception as e:
            logger.error(f"Template matching failed: {e}")
            self.publisher.publish(
                Event(
                    type=EventType.TEMPLATE_MATCHING_FAILED,
                    timestamp=datetime.now(),
                    component="template_manager",
                    description="Template matching failed",
                    error=str(e),
                )
            )
            return None

    def _extract_parameters(self, query: str, template: QueryTemplate) -> Optional[Dict[str, str]]:
        """Extract parameters from query using template pattern.

        Args:
            query: Search query
            template: Query template

        Returns:
            Dictionary of extracted parameters if query matches template
        """
        # Convert template pattern to regex
        import re
        pattern = template.pattern
        for param in template.parameters:
            pattern = pattern.replace(f"{{{param}}}", f"(?P<{param}>[^\\s]+)")
        pattern = f"^{pattern}$"

        # Try to match query
        match = re.match(pattern, query, re.IGNORECASE)
        if match:
            return match.groupdict()
        return None


# Create service instance
template_manager = TemplateManager(
    Config(),
    PublisherService(Config(), MetricsManager()),
    MetricsManager(),
)

__all__ = ["QueryTemplate", "TemplateManager", "template_manager"] 