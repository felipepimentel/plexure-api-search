"""Logging configuration for the CLI."""

import logging
import sys
from typing import Optional

from rich.console import Console
from rich.logging import RichHandler

console = Console()


def setup_logging(verbosity: int = 0, is_monitor: bool = False) -> None:
    """Setup logging configuration.

    Args:
        verbosity: Verbosity level (0-3)
        is_monitor: Whether this is being setup for the monitor
    """
    # Map verbosity to log level
    log_level = {
        0: logging.WARNING,
        1: logging.INFO,
        2: logging.DEBUG,
        3: logging.DEBUG,
    }.get(verbosity, logging.DEBUG)

    # Remove all existing handlers
    root_logger = logging.getLogger()
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)

    # Configure root logger
    root_logger.setLevel(log_level)

    # Create console handler with rich formatting
    console_handler = RichHandler(
        console=console,
        show_time=True,
        show_path=False,
        rich_tracebacks=True,
        tracebacks_show_locals=verbosity > 2,
    )
    console_handler.setLevel(log_level)

    # Create formatter
    formatter = logging.Formatter(
        "%(message)s",
        datefmt="[%X]"
    )
    console_handler.setFormatter(formatter)

    # Add handlers
    root_logger.addHandler(console_handler)

    # Set specific module levels
    if verbosity > 2:
        # More verbose logging for debugging
        logging.getLogger("plexure_api_search").setLevel(logging.DEBUG)
    else:
        # Less verbose for normal operation
        logging.getLogger("plexure_api_search").setLevel(log_level)

    # Suppress some noisy loggers
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("asyncio").setLevel(logging.WARNING)

    # Log initial setup
    logger = logging.getLogger(__name__)
    if is_monitor:
        logger.debug(f"Monitor logging configured (level: {logging.getLevelName(log_level)})")
    else:
        logger.debug(f"Logging configured (level: {logging.getLevelName(log_level)})")