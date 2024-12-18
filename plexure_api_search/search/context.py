"""Context-aware boosting based on business context and user behavior."""

import logging
from typing import Dict, List, Any, Optional, Set
from datetime import datetime

import numpy as np
from sklearn.preprocessing import MinMaxScaler

from ..config import Config
from ..monitoring.events import Event, EventType
from ..monitoring.metrics import MetricsManager
from ..services.base import BaseService
from ..services.events import PublisherService
from .search_models import SearchResult

logger = logging.getLogger(__name__)


class BusinessContext:
    """Business context for boosting."""

    def __init__(
        self,
        domain: str,
        priority: float = 1.0,
        rules: Dict[str, float] = None,
        features: Dict[str, Any] = None,
        constraints: Dict[str, Any] = None,
    ) -> None:
        """Initialize business context.

        Args:
            domain: Business domain
            priority: Domain priority (0-1)
            rules: Business rules with weights
            features: Feature flags and values
            constraints: Business constraints
        """
        self.domain = domain
        self.priority = priority
        self.rules = rules or {}
        self.features = features or {}
        self.constraints = constraints or {}
        self.last_updated = datetime.now()


class ContextConfig:
    """Configuration for context-aware boosting."""

    def __init__(
        self,
        business_weight: float = 0.4,
        feature_weight: float = 0.3,
        constraint_weight: float = 0.3,
        min_confidence: float = 0.5,
        max_boost: float = 2.0,
        decay_factor: float = 0.1,
    ) -> None:
        """Initialize context config.

        Args:
            business_weight: Weight for business rules
            feature_weight: Weight for feature flags
            constraint_weight: Weight for constraints
            min_confidence: Minimum confidence threshold
            max_boost: Maximum boost factor
            decay_factor: Time decay factor
        """
        self.business_weight = business_weight
        self.feature_weight = feature_weight
        self.constraint_weight = constraint_weight
        self.min_confidence = min_confidence
        self.max_boost = max_boost
        self.decay_factor = decay_factor


class ContextManager(BaseService[Dict[str, Any]]):
    """Context management service for search boosting."""

    def __init__(
        self,
        config: Config,
        publisher: PublisherService,
        metrics_manager: MetricsManager,
        context_config: Optional[ContextConfig] = None,
    ) -> None:
        """Initialize context manager.

        Args:
            config: Application configuration
            publisher: Event publisher
            metrics_manager: Metrics manager
            context_config: Context configuration
        """
        super().__init__(config, publisher, metrics_manager)
        self.context_config = context_config or ContextConfig()
        self._initialized = False
        self._contexts: Dict[str, BusinessContext] = {}
        self._scaler = MinMaxScaler()

    async def initialize(self) -> None:
        """Initialize context resources."""
        if self._initialized:
            return

        try:
            self._initialized = True
            self.publisher.publish(
                Event(
                    type=EventType.SERVICE_STARTED,
                    timestamp=datetime.now(),
                    component="context_manager",
                    description="Context manager initialized",
                    metadata={
                        "num_contexts": len(self._contexts),
                    },
                )
            )

        except Exception as e:
            logger.error(f"Failed to initialize context manager: {e}")
            raise

    async def cleanup(self) -> None:
        """Clean up context resources."""
        self._initialized = False
        self._contexts.clear()

        self.publisher.publish(
            Event(
                type=EventType.SERVICE_STOPPED,
                timestamp=datetime.now(),
                component="context_manager",
                description="Context manager stopped",
            )
        )

    async def health_check(self) -> Dict[str, Any]:
        """Check context manager health.

        Returns:
            Health check results
        """
        return {
            "service": "ContextManager",
            "initialized": self._initialized,
            "num_contexts": len(self._contexts),
            "business_weight": self.context_config.business_weight,
            "feature_weight": self.context_config.feature_weight,
        }

    def get_business_context(self, domain: str) -> Optional[BusinessContext]:
        """Get business context by domain.

        Args:
            domain: Business domain

        Returns:
            Business context if found
        """
        return self._contexts.get(domain)

    def update_business_context(
        self,
        domain: str,
        priority: Optional[float] = None,
        rules: Optional[Dict[str, float]] = None,
        features: Optional[Dict[str, Any]] = None,
        constraints: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Update business context.

        Args:
            domain: Business domain
            priority: Updated priority
            rules: Updated business rules
            features: Updated features
            constraints: Updated constraints
        """
        context = self._contexts.get(domain)
        if not context:
            context = BusinessContext(domain)
            self._contexts[domain] = context

        if priority is not None:
            context.priority = priority

        if rules:
            context.rules.update(rules)

        if features:
            context.features.update(features)

        if constraints:
            context.constraints.update(constraints)

        context.last_updated = datetime.now()

        # Emit context updated event
        self.publisher.publish(
            Event(
                type=EventType.CONTEXT_UPDATED,
                timestamp=datetime.now(),
                component="context_manager",
                description=f"Updated context for domain {domain}",
                metadata={
                    "domain": domain,
                    "priority": context.priority,
                    "num_rules": len(context.rules),
                    "num_features": len(context.features),
                },
            )
        )

    async def boost_results(
        self,
        domain: str,
        results: List[SearchResult],
        min_score: float = 0.0,
    ) -> List[SearchResult]:
        """Boost search results based on business context.

        Args:
            domain: Business domain
            results: Search results to boost
            min_score: Minimum score threshold

        Returns:
            Boosted search results
        """
        if not self._initialized:
            await self.initialize()

        context = self.get_business_context(domain)
        if not context:
            return results

        try:
            # Emit boosting started event
            self.publisher.publish(
                Event(
                    type=EventType.BOOSTING_STARTED,
                    timestamp=datetime.now(),
                    component="context_manager",
                    description=f"Boosting results for domain {domain}",
                    metadata={
                        "num_results": len(results),
                    },
                )
            )

            # Calculate boost scores
            boosted = []
            for result in results:
                # Skip results below threshold
                if result.score < min_score:
                    continue

                # Calculate business rule score
                rule_score = self._calculate_rule_score(result, context)

                # Calculate feature score
                feature_score = self._calculate_feature_score(result, context)

                # Calculate constraint score
                constraint_score = self._calculate_constraint_score(result, context)

                # Combine scores
                boost_score = (
                    self.context_config.business_weight * rule_score +
                    self.context_config.feature_weight * feature_score +
                    self.context_config.constraint_weight * constraint_score
                )

                # Apply domain priority
                boost_score *= context.priority

                # Apply boost factor
                boost_factor = 1.0 + min(boost_score, self.context_config.max_boost - 1.0)
                result.score *= boost_factor

                boosted.append(result)

            # Sort by boosted score
            boosted.sort(key=lambda x: x.score, reverse=True)

            # Emit success event
            self.publisher.publish(
                Event(
                    type=EventType.BOOSTING_COMPLETED,
                    timestamp=datetime.now(),
                    component="context_manager",
                    description="Boosting completed",
                    metadata={
                        "domain": domain,
                        "input_results": len(results),
                        "output_results": len(boosted),
                    },
                )
            )

            return boosted

        except Exception as e:
            logger.error(f"Boosting failed: {e}")
            self.publisher.publish(
                Event(
                    type=EventType.BOOSTING_FAILED,
                    timestamp=datetime.now(),
                    component="context_manager",
                    description="Boosting failed",
                    error=str(e),
                )
            )
            return results

    def _calculate_rule_score(self, result: SearchResult, context: BusinessContext) -> float:
        """Calculate business rule score for result.

        Args:
            result: Search result
            context: Business context

        Returns:
            Rule score (0-1)
        """
        if not context.rules:
            return 0.0

        # Match result with business rules
        matches = 0
        total = 0
        for rule, weight in context.rules.items():
            if (
                rule in result.endpoint.lower() or
                rule in result.description.lower() or
                any(rule in tag.lower() for tag in result.tags)
            ):
                matches += weight
            total += weight

        return matches / total if total > 0 else 0.0

    def _calculate_feature_score(self, result: SearchResult, context: BusinessContext) -> float:
        """Calculate feature score for result.

        Args:
            result: Search result
            context: Business context

        Returns:
            Feature score (0-1)
        """
        if not context.features:
            return 0.0

        # Check feature flags and values
        matches = 0
        total = len(context.features)
        for feature, value in context.features.items():
            if isinstance(value, bool):
                # Boolean feature flag
                if value and feature in result.tags:
                    matches += 1
            elif isinstance(value, (int, float)):
                # Numeric feature value
                if any(feature in param.get("name", "") for param in result.parameters):
                    matches += 1
            elif isinstance(value, str):
                # String feature value
                if value.lower() in result.description.lower():
                    matches += 1

        return matches / total if total > 0 else 0.0

    def _calculate_constraint_score(self, result: SearchResult, context: BusinessContext) -> float:
        """Calculate constraint score for result.

        Args:
            result: Search result
            context: Business context

        Returns:
            Constraint score (0-1)
        """
        if not context.constraints:
            return 1.0  # No constraints means full score

        # Check business constraints
        violations = 0
        total = len(context.constraints)
        for constraint, value in context.constraints.items():
            if isinstance(value, bool):
                # Boolean constraint
                if value and constraint in result.tags:
                    violations += 1
            elif isinstance(value, (int, float)):
                # Numeric constraint
                for param in result.parameters:
                    if constraint in param.get("name", ""):
                        param_value = param.get("value")
                        if param_value and float(param_value) > value:
                            violations += 1
            elif isinstance(value, str):
                # String constraint
                if value.lower() in result.description.lower():
                    violations += 1

        return 1.0 - (violations / total if total > 0 else 0.0)


# Create service instance
context_manager = ContextManager(
    Config(),
    PublisherService(Config(), MetricsManager()),
    MetricsManager(),
)

__all__ = [
    "BusinessContext",
    "ContextConfig",
    "ContextManager",
    "context_manager",
] 