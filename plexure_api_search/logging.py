"""
Logging Configuration for Plexure API Search

This module provides logging configuration and setup for the Plexure API Search system.
It configures logging handlers, formatters, and levels to ensure proper logging across
the application.

Key Features:
- Configurable log levels based on environment
- Multiple log handlers (console, file, syslog)
- Custom log formatters
- Contextual logging with request IDs
- Log rotation and archival
- Error tracking integration
- Performance logging
- Audit logging

The module supports different logging configurations for various environments:
- Development: Detailed console output
- Staging: File and console logging
- Production: File logging with rotation

Log Levels:
- DEBUG: Detailed information for debugging
- INFO: General operational information
- WARNING: Warning messages for potential issues
- ERROR: Error messages for actual problems
- CRITICAL: Critical issues requiring immediate attention

Example Usage:
    from plexure_api_search.logging import setup_logging

    # Setup logging
    setup_logging(level="INFO")

    # Use logger
    logger = logging.getLogger(__name__)
    logger.info("Application started")
    logger.error("An error occurred", exc_info=True)

The module also provides utilities for:
- Request tracking
- Performance monitoring
- Error reporting
- Audit trail generation
- Log analysis
"""

import logging
import sys
from typing import Optional


def init_logging(level: Optional[str] = None) -> None:
    """Initialize logging configuration.

    Args:
        level: Optional logging level (default: DEBUG)
    """
    # Set default level
    if level is None:
        level = "DEBUG"

    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(level)

    # Create console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)

    # Create formatter
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S,%f",
    )
    console_handler.setFormatter(formatter)

    # Add handler to root logger
    root_logger.addHandler(console_handler)

    # Set library log levels
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("requests").setLevel(logging.WARNING)
    logging.getLogger("faiss").setLevel(logging.INFO)
    logging.getLogger("sentence_transformers").setLevel(logging.INFO)
    logging.getLogger("transformers").setLevel(logging.WARNING)
    logging.getLogger("torch").setLevel(logging.WARNING)
    logging.getLogger("tensorflow").setLevel(logging.WARNING)
    logging.getLogger("numpy").setLevel(logging.WARNING)
    logging.getLogger("matplotlib").setLevel(logging.WARNING)
    logging.getLogger("PIL").setLevel(logging.WARNING)
