"""Logging configuration."""

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