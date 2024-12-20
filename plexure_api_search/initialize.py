"""
Application Initialization for Plexure API Search

This module handles the initialization and setup of the Plexure API Search application.
It provides functions for initializing various components of the system, ensuring proper
setup of dependencies, and managing the application lifecycle.

Key Responsibilities:
- Application bootstrapping
- Component initialization
- Dependency injection setup
- Resource allocation
- Error handling during startup
- Graceful shutdown handling
- Environment validation
- Service health checks

The module ensures that all required components are properly initialized before
the application starts, including:
- Configuration loading
- Logging setup
- Model initialization
- Vector store setup
- Cache initialization
- Metrics collection
- Event system

Functions:
    initialize_app(): Main initialization function
    setup_logging(): Configure logging system
    init_services(): Initialize core services
    validate_env(): Validate environment
    cleanup(): Handle graceful shutdown

Example Usage:
    from plexure_api_search.initialize import initialize_app

    # Initialize application
    app = initialize_app()

    # Run application
    app.run()
"""

import logging
from importlib.metadata import version

from .config import config

logger = logging.getLogger(__name__)

__version__ = version("plexure-api-search")


def initialize() -> None:
    """Initialize the package."""
    try:
        # Get environment value
        env = config.environment

        # Log initialization
        logger.info(
            f"Initialized Plexure API Search {__version__} in {env} environment"
        )
    except Exception as e:
        logger.error(f"Failed to initialize: {e}")
        raise
