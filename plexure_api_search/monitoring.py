"""Monitoring and logging utilities."""

import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

from rich.console import Console
from rich.logging import RichHandler


class Logger:
    """Logger class with rich formatting."""
    
    def __init__(
        self,
        log_file: Optional[str] = None,
        log_level: str = "INFO"
    ):
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
            console=self.console,
            show_time=True,
            show_path=False
        )
        console_handler.setFormatter(
            logging.Formatter("%(message)s")
        )
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


class MetricsCollector:
    """Collects and records metrics."""
    
    def __init__(self, metrics_dir: str = ".metrics"):
        """Initialize metrics collector.
        
        Args:
            metrics_dir: Directory to store metrics.
        """
        self.metrics_dir = Path(metrics_dir)
        self.metrics_dir.mkdir(parents=True, exist_ok=True)
        self.console = Console()
        
    def log_search_metrics(self, query: str, num_results: int) -> None:
        """Log search-related metrics.
        
        Args:
            query: Search query.
            num_results: Number of results found.
        """
        timestamp = datetime.now().isoformat()
        metrics = {
            "timestamp": timestamp,
            "query": query,
            "num_results": num_results
        }
        
        metrics_file = self.metrics_dir / "search_metrics.jsonl"
        with open(metrics_file, "a") as f:
            f.write(f"{metrics}\n")
            
    def log_indexing_metrics(
        self,
        num_files: int,
        num_endpoints: int,
        duration: float
    ) -> None:
        """Log indexing-related metrics.
        
        Args:
            num_files: Number of files processed.
            num_endpoints: Number of endpoints indexed.
            duration: Time taken in seconds.
        """
        timestamp = datetime.now().isoformat()
        metrics = {
            "timestamp": timestamp,
            "num_files": num_files,
            "num_endpoints": num_endpoints,
            "duration": duration
        }
        
        metrics_file = self.metrics_dir / "indexing_metrics.jsonl"
        with open(metrics_file, "a") as f:
            f.write(f"{metrics}\n")