"""Rate limiting implementation using file-based storage."""

import json
import logging
import os
import time
from dataclasses import dataclass
from pathlib import Path
from threading import Lock
from typing import Dict, Optional, Tuple

from ..config import config_instance

logger = logging.getLogger(__name__)

@dataclass
class RateLimitInfo:
    """Rate limit information."""
    
    key: str
    window_start: float
    request_count: int
    window_size: int
    max_requests: int


class RateLimiter:
    """File-based rate limiter implementation."""
    
    def __init__(
        self,
        max_requests: int = 100,
        window_size: int = 60,
        storage_dir: Optional[str] = None,
    ):
        """Initialize rate limiter.
        
        Args:
            max_requests: Maximum requests per window
            window_size: Window size in seconds
            storage_dir: Optional storage directory
        """
        self.max_requests = max_requests
        self.window_size = window_size
        
        # Set up storage directory
        storage_dir = storage_dir or getattr(config_instance, "rate_limit_dir", None)
        if not storage_dir:
            storage_dir = os.path.join(os.path.expanduser("~"), ".cache", "plexure_api_search", "rate_limits")
            
        self.storage_dir = storage_dir
        os.makedirs(self.storage_dir, exist_ok=True)
        logger.info(f"Using rate limit storage at {self.storage_dir}")
        
        # Thread safety
        self._lock = Lock()
        
        # Clean up expired entries
        self._cleanup()
    
    def _get_path(self, key: str) -> str:
        """Get storage file path for key.
        
        Args:
            key: Rate limit key
            
        Returns:
            File path
        """
        # Use hash of key to avoid filesystem issues
        hashed_key = str(hash(key))
        return os.path.join(self.storage_dir, f"{hashed_key}.json")
    
    def _load_info(self, key: str) -> Optional[Dict]:
        """Load rate limit info from file.
        
        Args:
            key: Rate limit key
            
        Returns:
            Rate limit info or None if not found
        """
        try:
            path = self._get_path(key)
            if not os.path.exists(path):
                return None
                
            with open(path, "r") as f:
                return json.load(f)
                
        except Exception as e:
            logger.error(f"Error loading rate limit info: {e}")
            return None
    
    def _save_info(self, key: str, info: Dict) -> None:
        """Save rate limit info to file.
        
        Args:
            key: Rate limit key
            info: Rate limit info
        """
        try:
            path = self._get_path(key)
            with open(path, "w") as f:
                json.dump(info, f)
                
        except Exception as e:
            logger.error(f"Error saving rate limit info: {e}")
    
    def _cleanup(self) -> None:
        """Clean up expired rate limit entries."""
        try:
            now = time.time()
            for filename in os.listdir(self.storage_dir):
                try:
                    path = os.path.join(self.storage_dir, filename)
                    if not os.path.isfile(path):
                        continue
                        
                    # Load info
                    with open(path, "r") as f:
                        info = json.load(f)
                        
                    # Check if expired
                    if now - info["window_start"] > self.window_size:
                        os.remove(path)
                        
                except Exception as e:
                    logger.error(f"Error cleaning up rate limit file {filename}: {e}")
                    
        except Exception as e:
            logger.error(f"Error cleaning up rate limits: {e}")
    
    def check(self, key: str) -> Tuple[bool, Optional[RateLimitInfo]]:
        """Check if request is allowed.
        
        Args:
            key: Rate limit key
            
        Returns:
            Tuple of (allowed, limit_info)
        """
        with self._lock:
            try:
                now = time.time()
                
                # Load existing info
                info = self._load_info(key)
                
                if info is None:
                    # First request
                    info = {
                        "window_start": now,
                        "request_count": 1
                    }
                    self._save_info(key, info)
                    return True, RateLimitInfo(
                        key=key,
                        window_start=now,
                        request_count=1,
                        window_size=self.window_size,
                        max_requests=self.max_requests,
                    )
                
                # Check if window expired
                if now - info["window_start"] > self.window_size:
                    # Start new window
                    info = {
                        "window_start": now,
                        "request_count": 1
                    }
                    self._save_info(key, info)
                    return True, RateLimitInfo(
                        key=key,
                        window_start=now,
                        request_count=1,
                        window_size=self.window_size,
                        max_requests=self.max_requests,
                    )
                
                # Check limit
                if info["request_count"] >= self.max_requests:
                    return False, RateLimitInfo(
                        key=key,
                        window_start=info["window_start"],
                        request_count=info["request_count"],
                        window_size=self.window_size,
                        max_requests=self.max_requests,
                    )
                
                # Increment counter
                info["request_count"] += 1
                self._save_info(key, info)
                
                return True, RateLimitInfo(
                    key=key,
                    window_start=info["window_start"],
                    request_count=info["request_count"],
                    window_size=self.window_size,
                    max_requests=self.max_requests,
                )
                
            except Exception as e:
                logger.error(f"Error checking rate limit: {e}")
                return True, None  # Allow request on error
    
    def reset(self, key: str) -> None:
        """Reset rate limit for key.
        
        Args:
            key: Rate limit key
        """
        try:
            path = self._get_path(key)
            if os.path.exists(path):
                os.remove(path)
                
        except Exception as e:
            logger.error(f"Error resetting rate limit: {e}")
    
    def clear(self) -> None:
        """Clear all rate limits."""
        try:
            for filename in os.listdir(self.storage_dir):
                path = os.path.join(self.storage_dir, filename)
                try:
                    if os.path.isfile(path):
                        os.remove(path)
                except Exception as e:
                    logger.error(f"Error removing rate limit file {path}: {e}")
                    
        except Exception as e:
            logger.error(f"Error clearing rate limits: {e}")


# Default rate limiter instance
rate_limiter = RateLimiter()

__all__ = ["RateLimiter", "RateLimitInfo", "rate_limiter"] 