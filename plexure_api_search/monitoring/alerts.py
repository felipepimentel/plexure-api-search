"""Alerting system for monitoring notifications."""

import logging
import asyncio
from typing import Dict, List, Any, Optional, Set, Union
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum
import smtplib
import json
import aiohttp
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

from ..config import Config
from .events import Event, EventType
from .metrics import MetricsManager
from ..services.base import BaseService
from ..services.events import PublisherService

logger = logging.getLogger(__name__)


class AlertSeverity(Enum):
    """Alert severity levels."""

    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class AlertChannel(Enum):
    """Alert notification channels."""

    EMAIL = "email"
    SLACK = "slack"
    WEBHOOK = "webhook"
    SMS = "sms"


@dataclass
class AlertRule:
    """Alert rule configuration."""

    name: str  # Rule name
    description: str  # Rule description
    severity: AlertSeverity  # Alert severity
    channels: List[AlertChannel]  # Notification channels
    cooldown: int = 300  # Cooldown period in seconds
    enabled: bool = True  # Whether rule is enabled
    metadata: Dict[str, Any] = None  # Additional metadata


@dataclass
class AlertEvent:
    """Alert event information."""

    rule: AlertRule  # Applied rule
    message: str  # Alert message
    timestamp: datetime  # Event timestamp
    metadata: Dict[str, Any]  # Additional metadata
    acknowledged: bool = False  # Whether alert is acknowledged


class AlertConfig:
    """Configuration for alerting system."""

    def __init__(
        self,
        email_config: Optional[Dict[str, Any]] = None,
        slack_config: Optional[Dict[str, Any]] = None,
        webhook_config: Optional[Dict[str, Any]] = None,
        sms_config: Optional[Dict[str, Any]] = None,
        enable_email: bool = True,
        enable_slack: bool = True,
        enable_webhook: bool = True,
        enable_sms: bool = False,
        default_cooldown: int = 300,
        cleanup_interval: float = 3600.0,
    ) -> None:
        """Initialize alert config.

        Args:
            email_config: Email configuration
            slack_config: Slack configuration
            webhook_config: Webhook configuration
            sms_config: SMS configuration
            enable_email: Whether to enable email alerts
            enable_slack: Whether to enable Slack alerts
            enable_webhook: Whether to enable webhook alerts
            enable_sms: Whether to enable SMS alerts
            default_cooldown: Default alert cooldown in seconds
            cleanup_interval: Alert cleanup interval in seconds
        """
        self.email_config = email_config or {}
        self.slack_config = slack_config or {}
        self.webhook_config = webhook_config or {}
        self.sms_config = sms_config or {}
        self.enable_email = enable_email
        self.enable_slack = enable_slack
        self.enable_webhook = enable_webhook
        self.enable_sms = enable_sms
        self.default_cooldown = default_cooldown
        self.cleanup_interval = cleanup_interval


class AlertManager(BaseService[Dict[str, Any]]):
    """Alert manager implementation."""

    def __init__(
        self,
        config: Config,
        publisher: PublisherService,
        metrics_manager: MetricsManager,
        alert_config: Optional[AlertConfig] = None,
    ) -> None:
        """Initialize alert manager.

        Args:
            config: Application configuration
            publisher: Event publisher
            metrics_manager: Metrics manager
            alert_config: Alert configuration
        """
        super().__init__(config, publisher, metrics_manager)
        self.alert_config = alert_config or AlertConfig()
        self._initialized = False
        self._rules: Dict[str, AlertRule] = {}
        self._events: List[AlertEvent] = []
        self._last_alert: Dict[str, datetime] = {}
        self._cleanup_task: Optional[asyncio.Task] = None
        self._session: Optional[aiohttp.ClientSession] = None

        # Register default rules
        self._register_default_rules()

    def _register_default_rules(self) -> None:
        """Register default alert rules."""
        # System health rules
        self.add_rule(
            AlertRule(
                name="system_error",
                description="System error detected",
                severity=AlertSeverity.ERROR,
                channels=[AlertChannel.EMAIL, AlertChannel.SLACK],
            )
        )
        self.add_rule(
            AlertRule(
                name="service_down",
                description="Service is down",
                severity=AlertSeverity.CRITICAL,
                channels=[AlertChannel.EMAIL, AlertChannel.SLACK, AlertChannel.SMS],
            )
        )

        # Performance rules
        self.add_rule(
            AlertRule(
                name="high_latency",
                description="High latency detected",
                severity=AlertSeverity.WARNING,
                channels=[AlertChannel.SLACK],
            )
        )
        self.add_rule(
            AlertRule(
                name="error_rate",
                description="High error rate detected",
                severity=AlertSeverity.ERROR,
                channels=[AlertChannel.EMAIL, AlertChannel.SLACK],
            )
        )

        # Resource rules
        self.add_rule(
            AlertRule(
                name="memory_usage",
                description="High memory usage",
                severity=AlertSeverity.WARNING,
                channels=[AlertChannel.SLACK],
            )
        )
        self.add_rule(
            AlertRule(
                name="disk_space",
                description="Low disk space",
                severity=AlertSeverity.WARNING,
                channels=[AlertChannel.SLACK],
            )
        )

        # Quality rules
        self.add_rule(
            AlertRule(
                name="quality_drop",
                description="Search quality degradation",
                severity=AlertSeverity.WARNING,
                channels=[AlertChannel.EMAIL, AlertChannel.SLACK],
            )
        )
        self.add_rule(
            AlertRule(
                name="relevance_drop",
                description="Relevance score degradation",
                severity=AlertSeverity.WARNING,
                channels=[AlertChannel.EMAIL, AlertChannel.SLACK],
            )
        )

    async def initialize(self) -> None:
        """Initialize alert manager resources."""
        if self._initialized:
            return

        try:
            # Create HTTP session
            self._session = aiohttp.ClientSession()

            # Start cleanup task
            self._cleanup_task = asyncio.create_task(self._cleanup_loop())

            self._initialized = True
            self.publisher.publish(
                Event(
                    type=EventType.SERVICE_STARTED,
                    timestamp=datetime.now(),
                    component="alert_manager",
                    description="Alert manager initialized",
                    metadata={
                        "num_rules": len(self._rules),
                        "channels": {
                            "email": self.alert_config.enable_email,
                            "slack": self.alert_config.enable_slack,
                            "webhook": self.alert_config.enable_webhook,
                            "sms": self.alert_config.enable_sms,
                        },
                    },
                )
            )

        except Exception as e:
            logger.error(f"Failed to initialize alert manager: {e}")
            raise

    async def cleanup(self) -> None:
        """Clean up alert manager resources."""
        if self._cleanup_task:
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass

        if self._session:
            await self._session.close()

        self._initialized = False
        self._rules.clear()
        self._events.clear()
        self._last_alert.clear()

        self.publisher.publish(
            Event(
                type=EventType.SERVICE_STOPPED,
                timestamp=datetime.now(),
                component="alert_manager",
                description="Alert manager stopped",
            )
        )

    async def health_check(self) -> Dict[str, Any]:
        """Check alert manager health.

        Returns:
            Health check results
        """
        return {
            "service": "AlertManager",
            "initialized": self._initialized,
            "num_rules": len(self._rules),
            "num_events": len(self._events),
            "channels": {
                "email": self.alert_config.enable_email,
                "slack": self.alert_config.enable_slack,
                "webhook": self.alert_config.enable_webhook,
                "sms": self.alert_config.enable_sms,
            },
            "status": "healthy" if self._initialized else "unhealthy",
        }

    def add_rule(self, rule: AlertRule) -> None:
        """Add alert rule.

        Args:
            rule: Alert rule
        """
        self._rules[rule.name] = rule

    async def trigger_alert(
        self,
        rule_name: str,
        message: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Optional[AlertEvent]:
        """Trigger alert for rule.

        Args:
            rule_name: Rule name
            message: Alert message
            metadata: Additional metadata

        Returns:
            Alert event if triggered
        """
        if not self._initialized:
            return None

        try:
            # Get rule
            rule = self._rules.get(rule_name)
            if not rule or not rule.enabled:
                return None

            # Check cooldown
            now = datetime.now()
            if rule_name in self._last_alert:
                elapsed = (now - self._last_alert[rule_name]).total_seconds()
                if elapsed < rule.cooldown:
                    return None

            # Create event
            event = AlertEvent(
                rule=rule,
                message=message,
                timestamp=now,
                metadata=metadata or {},
            )

            # Send notifications
            await self._send_notifications(event)

            # Update state
            self._events.append(event)
            self._last_alert[rule_name] = now

            # Emit event
            self.publisher.publish(
                Event(
                    type=EventType.ALERT_TRIGGERED,
                    timestamp=now,
                    component="alert_manager",
                    description=f"Alert triggered: {rule_name}",
                    metadata={
                        "rule": rule_name,
                        "severity": rule.severity.value,
                        "message": message,
                        "metadata": metadata,
                    },
                )
            )

            return event

        except Exception as e:
            logger.error(f"Failed to trigger alert: {e}")
            return None

    async def acknowledge_alert(
        self,
        event_id: int,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """Acknowledge alert event.

        Args:
            event_id: Event identifier
            metadata: Additional metadata

        Returns:
            True if acknowledged
        """
        if not self._initialized or event_id >= len(self._events):
            return False

        try:
            # Update event
            event = self._events[event_id]
            event.acknowledged = True
            if metadata:
                event.metadata.update(metadata)

            # Emit event
            self.publisher.publish(
                Event(
                    type=EventType.ALERT_ACKNOWLEDGED,
                    timestamp=datetime.now(),
                    component="alert_manager",
                    description=f"Alert acknowledged: {event.rule.name}",
                    metadata={
                        "rule": event.rule.name,
                        "severity": event.rule.severity.value,
                        "message": event.message,
                        "metadata": event.metadata,
                    },
                )
            )

            return True

        except Exception as e:
            logger.error(f"Failed to acknowledge alert: {e}")
            return False

    async def _send_notifications(self, event: AlertEvent) -> None:
        """Send alert notifications.

        Args:
            event: Alert event
        """
        for channel in event.rule.channels:
            try:
                if channel == AlertChannel.EMAIL and self.alert_config.enable_email:
                    await self._send_email(event)
                elif channel == AlertChannel.SLACK and self.alert_config.enable_slack:
                    await self._send_slack(event)
                elif channel == AlertChannel.WEBHOOK and self.alert_config.enable_webhook:
                    await self._send_webhook(event)
                elif channel == AlertChannel.SMS and self.alert_config.enable_sms:
                    await self._send_sms(event)
            except Exception as e:
                logger.error(f"Failed to send {channel.value} notification: {e}")

    async def _send_email(self, event: AlertEvent) -> None:
        """Send email notification.

        Args:
            event: Alert event
        """
        config = self.alert_config.email_config
        if not config:
            return

        try:
            # Create message
            msg = MIMEMultipart()
            msg["From"] = config["from"]
            msg["To"] = config["to"]
            msg["Subject"] = f"[{event.rule.severity.value.upper()}] {event.rule.name}"

            # Add body
            body = f"""
            Alert: {event.rule.name}
            Severity: {event.rule.severity.value}
            Time: {event.timestamp}
            Message: {event.message}
            """
            if event.metadata:
                body += f"\nMetadata: {json.dumps(event.metadata, indent=2)}"
            msg.attach(MIMEText(body, "plain"))

            # Send email
            with smtplib.SMTP(config["host"], config["port"]) as server:
                if config.get("use_tls"):
                    server.starttls()
                if config.get("username") and config.get("password"):
                    server.login(config["username"], config["password"])
                server.send_message(msg)

        except Exception as e:
            logger.error(f"Failed to send email: {e}")
            raise

    async def _send_slack(self, event: AlertEvent) -> None:
        """Send Slack notification.

        Args:
            event: Alert event
        """
        config = self.alert_config.slack_config
        if not config or not self._session:
            return

        try:
            # Create message
            message = {
                "text": f"[{event.rule.severity.value.upper()}] {event.rule.name}",
                "blocks": [
                    {
                        "type": "section",
                        "text": {
                            "type": "mrkdwn",
                            "text": f"*Alert:* {event.rule.name}\n"
                            f"*Severity:* {event.rule.severity.value}\n"
                            f"*Time:* {event.timestamp}\n"
                            f"*Message:* {event.message}",
                        },
                    },
                ],
            }
            if event.metadata:
                message["blocks"].append({
                    "type": "section",
                    "text": {
                        "type": "mrkdwn",
                        "text": f"*Metadata:*\n```{json.dumps(event.metadata, indent=2)}```",
                    },
                })

            # Send message
            async with self._session.post(
                config["webhook_url"],
                json=message,
            ) as response:
                if response.status != 200:
                    raise Exception(f"Slack API error: {await response.text()}")

        except Exception as e:
            logger.error(f"Failed to send Slack message: {e}")
            raise

    async def _send_webhook(self, event: AlertEvent) -> None:
        """Send webhook notification.

        Args:
            event: Alert event
        """
        config = self.alert_config.webhook_config
        if not config or not self._session:
            return

        try:
            # Create payload
            payload = {
                "rule": event.rule.name,
                "severity": event.rule.severity.value,
                "timestamp": event.timestamp.isoformat(),
                "message": event.message,
                "metadata": event.metadata,
            }

            # Send request
            async with self._session.post(
                config["url"],
                json=payload,
                headers=config.get("headers", {}),
            ) as response:
                if response.status not in {200, 201, 202, 204}:
                    raise Exception(f"Webhook error: {await response.text()}")

        except Exception as e:
            logger.error(f"Failed to send webhook: {e}")
            raise

    async def _send_sms(self, event: AlertEvent) -> None:
        """Send SMS notification.

        Args:
            event: Alert event
        """
        # TODO: Implement SMS notifications
        pass

    async def _cleanup_loop(self) -> None:
        """Background task for alert cleanup."""
        while True:
            try:
                # Sleep for cleanup interval
                await asyncio.sleep(self.alert_config.cleanup_interval)

                # Remove old events
                now = datetime.now()
                max_age = max(rule.cooldown for rule in self._rules.values())
                cutoff = now - timedelta(seconds=max_age)

                self._events = [
                    event for event in self._events
                    if event.timestamp >= cutoff or not event.acknowledged
                ]

                # Clear old alerts
                for rule_name in list(self._last_alert.keys()):
                    if now - self._last_alert[rule_name] > timedelta(seconds=max_age):
                        del self._last_alert[rule_name]

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Alert cleanup failed: {e}")


# Create manager instance
alert_manager = AlertManager(
    Config(),
    PublisherService(Config(), MetricsManager()),
    MetricsManager(),
)

__all__ = [
    "AlertSeverity",
    "AlertChannel",
    "AlertRule",
    "AlertEvent",
    "AlertConfig",
    "AlertManager",
    "alert_manager",
] 