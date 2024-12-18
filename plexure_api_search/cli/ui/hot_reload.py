"""Hot reload functionality for development."""

import logging
import os
from typing import Callable

from watchdog.events import FileSystemEventHandler
from watchdog.observers import Observer

logger = logging.getLogger(__name__)


class ReloadHandler(FileSystemEventHandler):
    """Handler for file system events that triggers reload."""

    def __init__(self, callback: Callable[[], None]):
        """Initialize the handler.
        
        Args:
            callback: Function to call when files change
        """
        self.callback = callback
        self._last_reload = 0

    def on_modified(self, event):
        """Handle file modification events."""
        if event.is_directory:
            return

        if event.src_path.endswith('.py'):
            try:
                self.callback()
            except Exception as e:
                logger.error(f"Error in reload callback: {e}", exc_info=True)


def start_hot_reload(base_path: str, patterns: list[str], callback: Callable[[], None]) -> Observer:
    """Start watching for file changes.
    
    Args:
        base_path: Base directory to watch
        patterns: List of file patterns to watch
        callback: Function to call when files change
    
    Returns:
        Observer instance that can be used to stop watching
    """
    observer = Observer()
    handler = ReloadHandler(callback)
    
    # Normalize path
    base_path = os.path.abspath(base_path)
    
    # Start watching
    observer.schedule(handler, base_path, recursive=True)
    observer.start()
    
    logger.debug(f"Hot reload watching {base_path} for changes")
    return observer 