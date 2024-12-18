"""Rate limiting for API endpoints."""

import logging
import asyncio
from typing import Dict, List, Any, Optional, Set
from datetime import datetime
import time
from dataclasses import dataclass
from collections import defaultdict

from ..config import Config
from ..monitoring.events import Event, EventType
from ..monitoring.metrics import MetricsManager
from .base import BaseService
from .events import PublisherService

logger = logging.getLogger(__name__)


@dataclass
class RateLimitRule:
    """Rate limit rule configuration."""

    requests: int  # Number of requests allowed
    window: int  # Time window in seconds
    burst: Optional[int] = None  # Burst size (if different from requests)
    cost: float = 1.0  # Request cost


class RateLimitConfig:
    """Configuration for rate limiting."""

    def __init__(
        self,
        default_rule: Optional[RateLimitRule] = None,
        endpoint_rules: Optional[Dict[str, RateLimitRule]] = None,
        user_rules: Optional[Dict[str, RateLimitRule]] = None,
        ip_rules: Optional[Dict[str, RateLimitRule]] = None,
        enable_endpoint_limits: bool = True,
        enable_user_limits: bool = True,
        enable_ip_limits: bool = True,
        enable_global_limits: bool = True,
        enable_burst: bool = True,
        cleanup_interval: float = 60.0,
    ) -> None:
        """Initialize rate limit config.

        Args:
            default_rule: Default rate limit rule
            endpoint_rules: Rate limit rules by endpoint
            user_rules: Rate limit rules by user
            ip_rules: Rate limit rules by IP address
            enable_endpoint_limits: Whether to enable endpoint limits
            enable_user_limits: Whether to enable user limits
            enable_ip_limits: Whether to enable IP limits
            enable_global_limits: Whether to enable global limits
            enable_burst: Whether to enable burst limits
            cleanup_interval: Cleanup interval in seconds
        """
        self.default_rule = default_rule or RateLimitRule(
            requests=100,
            window=60,
            burst=200,
        )
        self.endpoint_rules = endpoint_rules or {}
        self.user_rules = user_rules or {}
        self.ip_rules = ip_rules or {}
        self.enable_endpoint_limits = enable_endpoint_limits
        self.enable_user_limits = enable_user_limits
        self.enable_ip_limits = enable_ip_limits
        self.enable_global_limits = enable_global_limits
        self.enable_burst = enable_burst
        self.cleanup_interval = cleanup_interval


class TokenBucket:
    """Token bucket rate limiter."""

    def __init__(
        self,
        rule: RateLimitRule,
    ) -> None:
        """Initialize token bucket.

        Args:
            rule: Rate limit rule
        """
        self.rule = rule
        self.tokens = rule.burst or rule.requests
        self.last_update = time.time()

    def update(self) -> None:
        """Update token count."""
        now = time.time()
        elapsed = now - self.last_update
        self.tokens = min(
            self.rule.burst or self.rule.requests,
            self.tokens + elapsed * (self.rule.requests / self.rule.window),
        )
        self.last_update = now

    def consume(self, tokens: float = 1.0) -> bool:
        """Consume tokens.

        Args:
            tokens: Number of tokens to consume

        Returns:
            True if tokens were consumed
        """
        self.update()
        if self.tokens >= tokens:
            self.tokens -= tokens
            return True
        return False


class RateLimiter(BaseService[Dict[str, Any]]):
    """Rate limiter implementation."""

    def __init__(
        self,
        config: Config,
        publisher: PublisherService,
        metrics_manager: MetricsManager,
        rate_limit_config: Optional[RateLimitConfig] = None,
    ) -> None:
        """Initialize rate limiter.

        Args:
            config: Application configuration
            publisher: Event publisher
            metrics_manager: Metrics manager
            rate_limit_config: Rate limit configuration
        """
        super().__init__(config, publisher, metrics_manager)
        self.rate_limit_config = rate_limit_config or RateLimitConfig()
        self._initialized = False
        self._endpoint_buckets: Dict[str, TokenBucket] = {}
        self._user_buckets: Dict[str, TokenBucket] = {}
        self._ip_buckets: Dict[str, TokenBucket] = {}
        self._global_bucket: Optional[TokenBucket] = None
        self._cleanup_task: Optional[asyncio.Task] = None
        self._request_counts: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))

    async def initialize(self) -> None:
        """Initialize rate limiter resources."""
        if self._initialized:
            return

        try:
            # Create global bucket
            if self.rate_limit_config.enable_global_limits:
                self._global_bucket = TokenBucket(self.rate_limit_config.default_rule)

            # Start cleanup task
            self._cleanup_task = asyncio.create_task(self._cleanup_loop())

            self._initialized = True
            self.publisher.publish(
                Event(
                    type=EventType.SERVICE_STARTED,
                    timestamp=datetime.now(),
                    component="rate_limiter",
                    description="Rate limiter initialized",
                    metadata={
                        "endpoint_limits": self.rate_limit_config.enable_endpoint_limits,
                        "user_limits": self.rate_limit_config.enable_user_limits,
                        "ip_limits": self.rate_limit_config.enable_ip_limits,
                        "global_limits": self.rate_limit_config.enable_global_limits,
                    },
                )
            )

        except Exception as e:
            logger.error(f"Failed to initialize rate limiter: {e}")
            raise

    async def cleanup(self) -> None:
        """Clean up rate limiter resources."""
        if self._cleanup_task:
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass

        self._initialized = False
        self._endpoint_buckets.clear()
        self._user_buckets.clear()
        self._ip_buckets.clear()
        self._global_bucket = None
        self._request_counts.clear()

        self.publisher.publish(
            Event(
                type=EventType.SERVICE_STOPPED,
                timestamp=datetime.now(),
                component="rate_limiter",
                description="Rate limiter stopped",
            )
        )

    async def health_check(self) -> Dict[str, Any]:
        """Check rate limiter health.

        Returns:
            Health check results
        """
        return {
            "service": "RateLimiter",
            "initialized": self._initialized,
            "endpoint_buckets": len(self._endpoint_buckets),
            "user_buckets": len(self._user_buckets),
            "ip_buckets": len(self._ip_buckets),
            "global_bucket": bool(self._global_bucket),
            "status": "healthy" if self._initialized else "unhealthy",
        }

    async def check_rate_limit(
        self,
        endpoint: str,
        user_id: Optional[str] = None,
        ip_address: Optional[str] = None,
        cost: float = 1.0,
    ) -> bool:
        """Check if request is allowed.

        Args:
            endpoint: API endpoint
            user_id: Optional user identifier
            ip_address: Optional IP address
            cost: Request cost

        Returns:
            True if request is allowed
        """
        if not self._initialized:
            await self.initialize()

        try:
            # Check global limit
            if (
                self.rate_limit_config.enable_global_limits
                and self._global_bucket
                and not self._global_bucket.consume(cost)
            ):
                self._record_rejection("global", endpoint)
                return False

            # Check endpoint limit
            if self.rate_limit_config.enable_endpoint_limits:
                if not await self._check_endpoint_limit(endpoint, cost):
                    self._record_rejection("endpoint", endpoint)
                    return False

            # Check user limit
            if (
                self.rate_limit_config.enable_user_limits
                and user_id
                and not await self._check_user_limit(user_id, cost)
            ):
                self._record_rejection("user", user_id)
                return False

            # Check IP limit
            if (
                self.rate_limit_config.enable_ip_limits
                and ip_address
                and not await self._check_ip_limit(ip_address, cost)
            ):
                self._record_rejection("ip", ip_address)
                return False

            # Record success
            self._record_success(endpoint, user_id, ip_address)
            return True

        except Exception as e:
            logger.error(f"Rate limit check failed: {e}")
            return True  # Allow request on error

    async def _check_endpoint_limit(
        self,
        endpoint: str,
        cost: float,
    ) -> bool:
        """Check endpoint rate limit.

        Args:
            endpoint: API endpoint
            cost: Request cost

        Returns:
            True if request is allowed
        """
        # Get or create bucket
        if endpoint not in self._endpoint_buckets:
            rule = self.rate_limit_config.endpoint_rules.get(
                endpoint,
                self.rate_limit_config.default_rule,
            )
            self._endpoint_buckets[endpoint] = TokenBucket(rule)

        # Check limit
        return self._endpoint_buckets[endpoint].consume(cost)

    async def _check_user_limit(
        self,
        user_id: str,
        cost: float,
    ) -> bool:
        """Check user rate limit.

        Args:
            user_id: User identifier
            cost: Request cost

        Returns:
            True if request is allowed
        """
        # Get or create bucket
        if user_id not in self._user_buckets:
            rule = self.rate_limit_config.user_rules.get(
                user_id,
                self.rate_limit_config.default_rule,
            )
            self._user_buckets[user_id] = TokenBucket(rule)

        # Check limit
        return self._user_buckets[user_id].consume(cost)

    async def _check_ip_limit(
        self,
        ip_address: str,
        cost: float,
    ) -> bool:
        """Check IP rate limit.

        Args:
            ip_address: IP address
            cost: Request cost

        Returns:
            True if request is allowed
        """
        # Get or create bucket
        if ip_address not in self._ip_buckets:
            rule = self.rate_limit_config.ip_rules.get(
                ip_address,
                self.rate_limit_config.default_rule,
            )
            self._ip_buckets[ip_address] = TokenBucket(rule)

        # Check limit
        return self._ip_buckets[ip_address].consume(cost)

    def _record_success(
        self,
        endpoint: str,
        user_id: Optional[str] = None,
        ip_address: Optional[str] = None,
    ) -> None:
        """Record successful request.

        Args:
            endpoint: API endpoint
            user_id: Optional user identifier
            ip_address: Optional IP address
        """
        self._request_counts["endpoint"][endpoint] += 1
        if user_id:
            self._request_counts["user"][user_id] += 1
        if ip_address:
            self._request_counts["ip"][ip_address] += 1

        # Update metrics
        self.metrics.increment(
            "rate_limit_allowed",
            1,
            {
                "endpoint": endpoint,
                "user_id": user_id or "anonymous",
                "ip_address": ip_address or "unknown",
            },
        )

    def _record_rejection(
        self,
        limit_type: str,
        identifier: str,
    ) -> None:
        """Record rate limit rejection.

        Args:
            limit_type: Type of limit (endpoint, user, ip, global)
            identifier: Identifier that was limited
        """
        # Update metrics
        self.metrics.increment(
            "rate_limit_rejected",
            1,
            {
                "type": limit_type,
                "identifier": identifier,
            },
        )

        # Emit event
        self.publisher.publish(
            Event(
                type=EventType.RATE_LIMIT_EXCEEDED,
                timestamp=datetime.now(),
                component="rate_limiter",
                description=f"Rate limit exceeded for {limit_type}: {identifier}",
                metadata={
                    "type": limit_type,
                    "identifier": identifier,
                },
            )
        )

    async def _cleanup_loop(self) -> None:
        """Background task for cleanup."""
        while True:
            try:
                # Sleep for cleanup interval
                await asyncio.sleep(self.rate_limit_config.cleanup_interval)

                # Clear old buckets
                now = time.time()
                for buckets in [
                    self._endpoint_buckets,
                    self._user_buckets,
                    self._ip_buckets,
                ]:
                    expired = []
                    for key, bucket in buckets.items():
                        if now - bucket.last_update > bucket.rule.window:
                            expired.append(key)
                    for key in expired:
                        del buckets[key]

                # Clear old request counts
                for counter in self._request_counts.values():
                    counter.clear()

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Rate limiter cleanup failed: {e}")


# Create service instance
rate_limiter = RateLimiter(
    Config(),
    PublisherService(Config(), MetricsManager()),
    MetricsManager(),
)

__all__ = [
    "RateLimitRule",
    "RateLimitConfig",
    "TokenBucket",
    "RateLimiter",
    "rate_limiter",
] 