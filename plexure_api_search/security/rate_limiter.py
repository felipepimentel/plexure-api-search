"""Rate limiting and security system."""

import logging
import time
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional, Set, Tuple

import redis

from ..config import config_instance
from ..utils.cache import DiskCache

logger = logging.getLogger(__name__)

# Cache for rate limiting
rate_cache = DiskCache[Dict[str, Any]](
    namespace="rate_limiting",
    ttl=config_instance.cache_ttl,
)


@dataclass
class RateLimitInfo:
    """Rate limit information."""

    key: str
    window_start: float
    request_count: int
    window_size: int
    max_requests: int

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format."""
        return {
            "key": self.key,
            "window_start": self.window_start,
            "request_count": self.request_count,
            "window_size": self.window_size,
            "max_requests": self.max_requests,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "RateLimitInfo":
        """Create from dictionary format."""
        return cls(**data)


class RateLimiter:
    """Advanced rate limiting with multiple strategies."""

    def __init__(self):
        """Initialize rate limiter."""
        self.enabled = config_instance.rate_limit_enabled
        self.max_requests = config_instance.rate_limit_requests
        self.window_size = config_instance.rate_limit_window
        
        # Initialize Redis if available
        self.redis = None
        if config_instance.redis_url:
            try:
                self.redis = redis.from_url(config_instance.redis_url)
                logger.info("Using Redis for rate limiting")
            except Exception as e:
                logger.error(f"Failed to connect to Redis: {e}")

    def _get_redis_key(self, key: str) -> str:
        """Get Redis key with namespace.

        Args:
            key: Original key

        Returns:
            Namespaced Redis key
        """
        return f"rate_limit:{key}"

    def _check_redis(self, key: str) -> Tuple[bool, Optional[RateLimitInfo]]:
        """Check rate limit using Redis.

        Args:
            key: Rate limit key

        Returns:
            Tuple of (allowed, limit_info)
        """
        try:
            redis_key = self._get_redis_key(key)
            pipe = self.redis.pipeline()

            # Get current count and window start
            pipe.get(redis_key)
            pipe.ttl(redis_key)
            current, ttl = pipe.execute()

            now = time.time()

            if current is None:
                # First request in window
                self.redis.setex(redis_key, self.window_size, 1)
                return True, RateLimitInfo(
                    key=key,
                    window_start=now,
                    request_count=1,
                    window_size=self.window_size,
                    max_requests=self.max_requests,
                )

            current = int(current)
            if current >= self.max_requests:
                # Rate limit exceeded
                return False, RateLimitInfo(
                    key=key,
                    window_start=now - (self.window_size - ttl),
                    request_count=current,
                    window_size=self.window_size,
                    max_requests=self.max_requests,
                )

            # Increment counter
            self.redis.incr(redis_key)
            return True, RateLimitInfo(
                key=key,
                window_start=now - (self.window_size - ttl),
                request_count=current + 1,
                window_size=self.window_size,
                max_requests=self.max_requests,
            )

        except Exception as e:
            logger.error(f"Redis rate limit check failed: {e}")
            return True, None

    def _check_disk(self, key: str) -> Tuple[bool, Optional[RateLimitInfo]]:
        """Check rate limit using disk cache.

        Args:
            key: Rate limit key

        Returns:
            Tuple of (allowed, limit_info)
        """
        try:
            now = time.time()
            info = rate_cache.get(key)

            if info is None:
                # First request in window
                info = RateLimitInfo(
                    key=key,
                    window_start=now,
                    request_count=1,
                    window_size=self.window_size,
                    max_requests=self.max_requests,
                )
                rate_cache.set(key, info.to_dict())
                return True, info

            info = RateLimitInfo.from_dict(info)

            # Check if window has expired
            if now - info.window_start >= self.window_size:
                # Start new window
                info = RateLimitInfo(
                    key=key,
                    window_start=now,
                    request_count=1,
                    window_size=self.window_size,
                    max_requests=self.max_requests,
                )
                rate_cache.set(key, info.to_dict())
                return True, info

            if info.request_count >= self.max_requests:
                # Rate limit exceeded
                return False, info

            # Increment counter
            info.request_count += 1
            rate_cache.set(key, info.to_dict())
            return True, info

        except Exception as e:
            logger.error(f"Disk rate limit check failed: {e}")
            return True, None

    def check_rate_limit(self, key: str) -> Tuple[bool, Optional[RateLimitInfo]]:
        """Check if request is allowed under rate limit.

        Args:
            key: Rate limit key (e.g. IP address, API key)

        Returns:
            Tuple of (allowed, limit_info)
        """
        if not self.enabled:
            return True, None

        try:
            # Try Redis first if available
            if self.redis:
                return self._check_redis(key)
            
            # Fall back to disk cache
            return self._check_disk(key)

        except Exception as e:
            logger.error(f"Rate limit check failed: {e}")
            return True, None

    def get_limit_info(self, key: str) -> Optional[RateLimitInfo]:
        """Get current rate limit information.

        Args:
            key: Rate limit key

        Returns:
            Rate limit information if available
        """
        try:
            if self.redis:
                redis_key = self._get_redis_key(key)
                pipe = self.redis.pipeline()
                pipe.get(redis_key)
                pipe.ttl(redis_key)
                current, ttl = pipe.execute()

                if current is None:
                    return None

                return RateLimitInfo(
                    key=key,
                    window_start=time.time() - (self.window_size - ttl),
                    request_count=int(current),
                    window_size=self.window_size,
                    max_requests=self.max_requests,
                )
            else:
                info = rate_cache.get(key)
                if info:
                    return RateLimitInfo.from_dict(info)
                return None

        except Exception as e:
            logger.error(f"Failed to get limit info: {e}")
            return None

    def reset_limit(self, key: str) -> None:
        """Reset rate limit for key.

        Args:
            key: Rate limit key
        """
        try:
            if self.redis:
                self.redis.delete(self._get_redis_key(key))
            else:
                rate_cache.delete(key)

        except Exception as e:
            logger.error(f"Failed to reset limit: {e}")


class SecurityManager:
    """Security manager for API search."""

    def __init__(self):
        """Initialize security manager."""
        self.rate_limiter = RateLimiter()
        self.blocked_ips: Set[str] = set()
        self.suspicious_patterns = [
            r"(?i)(union|select|insert|update|delete|drop)\s+",  # SQL injection
            r"(?i)<script",  # XSS
            r"(?i)../../",  # Path traversal
            r"(?i)\{\{.*\}\}",  # Template injection
        ]

    def check_request(
        self,
        ip: str,
        query: str,
        api_key: Optional[str] = None,
    ) -> Tuple[bool, str]:
        """Check if request should be allowed.

        Args:
            ip: Client IP address
            query: Search query
            api_key: Optional API key

        Returns:
            Tuple of (allowed, reason)
        """
        try:
            # Check blocked IPs
            if ip in self.blocked_ips:
                return False, "IP address is blocked"

            # Check rate limit
            allowed, info = self.rate_limiter.check_rate_limit(ip)
            if not allowed:
                return False, "Rate limit exceeded"

            # Check for suspicious patterns
            for pattern in self.suspicious_patterns:
                if re.search(pattern, query):
                    self.blocked_ips.add(ip)
                    return False, "Suspicious query pattern detected"

            # Validate API key if required
            if config_instance.require_api_key and not api_key:
                return False, "API key required"

            return True, "Request allowed"

        except Exception as e:
            logger.error(f"Security check failed: {e}")
            return False, "Security check failed"

    def unblock_ip(self, ip: str) -> None:
        """Unblock an IP address.

        Args:
            ip: IP address to unblock
        """
        try:
            self.blocked_ips.discard(ip)
            self.rate_limiter.reset_limit(ip)
        except Exception as e:
            logger.error(f"Failed to unblock IP: {e}")


# Global instances
rate_limiter = RateLimiter()
security_manager = SecurityManager()

__all__ = ["rate_limiter", "security_manager", "RateLimitInfo"] 