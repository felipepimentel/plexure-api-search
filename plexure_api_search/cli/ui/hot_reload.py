"""Hot reload functionality for monitor UI."""

import importlib
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from watchdog.events import FileSystemEvent, FileSystemEventHandler
from watchdog.observers import Observer

logger = logging.getLogger(__name__)

class ModuleReloader(FileSystemEventHandler):
    """File system event handler for module reloading."""
    
    def __init__(
        self,
        base_path: str,
        callback: Optional[Callable] = None,
        patterns: Optional[List[str]] = None
    ):
        """Initialize module reloader.
        
        Args:
            base_path: Base path to monitor for changes
            callback: Optional callback to run after reload
            patterns: Optional list of file patterns to watch
        """
        self.base_path = Path(base_path)
        self.callback = callback
        self.patterns = patterns or ["*.py"]
        self.last_reload = time.time()
        self.reload_delay = 1.0  # Minimum seconds between reloads
        
        # Track loaded modules
        self.tracked_modules = {
            name: module for name, module in sys.modules.items()
            if self._should_track(name, module)
        }
        
    def _should_track(self, name: str, module: Any) -> bool:
        """Check if module should be tracked for reloading."""
        if not hasattr(module, "__file__") or not module.__file__:
            return False
            
        module_path = Path(module.__file__)
        return (
            name.startswith("plexure_api_search") and
            self.base_path in Path(module_path).parents and
            any(module_path.match(pattern) for pattern in self.patterns)
        )
    
    def on_modified(self, event: FileSystemEvent) -> None:
        """Handle file modification event."""
        if not event.is_directory and any(Path(event.src_path).match(p) for p in self.patterns):
            # Throttle reloads
            now = time.time()
            if now - self.last_reload < self.reload_delay:
                return
                
            self.last_reload = now
            self._reload_modules()
            
            if self.callback:
                try:
                    self.callback()
                except Exception as e:
                    logger.error(f"Error in reload callback: {e}")
    
    def _reload_modules(self) -> None:
        """Reload all tracked modules."""
        reloaded = set()
        
        def reload_module(name: str) -> None:
            if name in reloaded:
                return
                
            module = sys.modules.get(name)
            if not module or not self._should_track(name, module):
                return
                
            try:
                logger.debug(f"Reloading module: {name}")
                reloaded.add(name)
                importlib.reload(module)
            except Exception as e:
                logger.error(f"Failed to reload {name}: {e}")
        
        # Reload all tracked modules
        for name in self.tracked_modules:
            reload_module(name)
            
        logger.info(f"Reloaded {len(reloaded)} modules")


def start_hot_reload(
    base_path: str,
    callback: Optional[Callable] = None,
    patterns: Optional[List[str]] = None
) -> Observer:
    """Start hot reload observer.
    
    Args:
        base_path: Base path to monitor for changes
        callback: Optional callback to run after reload
        patterns: Optional list of file patterns to watch
        
    Returns:
        Running observer instance
    """
    event_handler = ModuleReloader(base_path, callback, patterns)
    observer = Observer()
    observer.schedule(event_handler, base_path, recursive=True)
    observer.start()
    return observer 