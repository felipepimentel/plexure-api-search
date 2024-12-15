"""Logging utilities."""

import logging
import os
from typing import Optional

from rich.console import Console
from rich.logging import RichHandler


class Logger:
    """Logger class with rich formatting."""

    def __init__(self, log_file: Optional[str] = None, log_level: str = "INFO"):
        """Initialize logger.

        Args:
            log_file: Optional path to log file.
            log_level: Logging level.
        """
        self.console = Console()

        # Create logger
        self.logger = logging.getLogger("plexure_api_search")
        self.logger.setLevel(getattr(logging, log_level.upper()))

        # Remove existing handlers
        self.logger.handlers = []

        # Add console handler with rich formatting
        console_handler = RichHandler(
            console=self.console, show_time=True, show_path=False
        )
        console_handler.setFormatter(logging.Formatter("%(message)s"))
        self.logger.addHandler(console_handler)

        # Add file handler if specified
        if log_file:
            log_dir = os.path.dirname(log_file)
            if log_dir:
                os.makedirs(log_dir, exist_ok=True)

            file_handler = logging.FileHandler(log_file)
            file_handler.setFormatter(
                logging.Formatter(
                    "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
                )
            )
            self.logger.addHandler(file_handler)

    def debug(self, message: str) -> None:
        """Log debug message.

        Args:
            message: Message to log.
        """
        self.logger.debug(message)

    def info(self, message: str) -> None:
        """Log info message.

        Args:
            message: Message to log.
        """
        self.logger.info(message)

    def warning(self, message: str) -> None:
        """Log warning message.

        Args:
            message: Message to log.
        """
        self.logger.warning(message)

    def error(self, message: str) -> None:
        """Log error message.

        Args:
            message: Message to log.
        """
        self.logger.error(message)

    def critical(self, message: str) -> None:
        """Log critical message.

        Args:
            message: Message to log.
        """
        self.logger.critical(message)

    def exception(self, message: str) -> None:
        """Log exception message with traceback.

        Args:
            message: Message to log.
        """
        self.logger.exception(message)
