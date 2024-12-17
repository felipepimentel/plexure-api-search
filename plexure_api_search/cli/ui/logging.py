"""Logging configuration for CLI."""

import logging
from rich.logging import RichHandler

def setup_logging(verbosity: int) -> None:
    """Configure logging based on verbosity level.
    
    Args:
        verbosity: Verbosity level (0-3)
            0: ERROR only
            1: WARNING and above
            2: INFO and above
            3: DEBUG and all
    """
    if verbosity == 0:
        logging.basicConfig(
            level=logging.ERROR,
            format="%(message)s",
            handlers=[RichHandler(rich_tracebacks=True, markup=True)]
        )
    elif verbosity == 1:
        logging.basicConfig(
            level=logging.WARNING,
            format="%(message)s",
            handlers=[RichHandler(rich_tracebacks=True, markup=True)]
        )
    elif verbosity == 2:
        logging.basicConfig(
            level=logging.INFO,
            format="%(message)s",
            handlers=[RichHandler(rich_tracebacks=True, markup=True)]
        )
    else:
        logging.basicConfig(
            level=logging.DEBUG,
            format="%(message)s",
            handlers=[RichHandler(rich_tracebacks=True, markup=True)]
        )

    # Suppress specific loggers unless in debug mode
    if verbosity < 3:
        logging.getLogger("pinecone").setLevel(logging.WARNING)
        logging.getLogger("sentence_transformers").setLevel(logging.WARNING)
        logging.getLogger("pinecone_plugin_interface").setLevel(logging.WARNING) 