"""Personalized ranking based on user preferences and history."""

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


class UserProfile:
    """User profile for personalization."""

    def __init__(
        self,
        user_id: str,
        preferences: Dict[str, float] = None,
        history: List[Dict[str, Any]] = None,
        tags: Set[str] = None,
    ) -> None:
        """Initialize user profile.

        Args:
            user_id: Unique user identifier
            preferences: Category preferences (0-1)
            history: Search and interaction history
            tags: Preferred API tags
        """
        self.user_id = user_id
        self.preferences = preferences or {}
        self.history = history or []
        self.tags = tags or set()
        self.last_updated = datetime.now()


class PersonalizationConfig:
    """Configuration for personalized ranking."""

    def __init__(
        self,
        history_weight: float = 0.3,
        preference_weight: float = 0.3,
        tag_weight: float = 0.2,
        recency_weight: float = 0.2,
        history_window: int = 100,
        min_interactions: int = 5,
        decay_factor: float = 0.1,
    ) -> None:
        """Initialize personalization config.

        Args:
            history_weight: Weight for historical interactions
            preference_weight: Weight for category preferences
            tag_weight: Weight for tag preferences
            recency_weight: Weight for recency bias
            history_window: Maximum history size
            min_interactions: Minimum interactions for personalization
            decay_factor: Time decay factor
        """
        self.history_weight = history_weight
        self.preference_weight = preference_weight
        self.tag_weight = tag_weight
        self.recency_weight = recency_weight
        self.history_window = history_window
        self.min_interactions = min_interactions
        self.decay_factor = decay_factor


class PersonalizationManager(BaseService[Dict[str, Any]]):
    """Personalization service for search ranking."""

    def __init__(
        self,
        config: Config,
        publisher: PublisherService,
        metrics_manager: MetricsManager,
        personalization_config: Optional[PersonalizationConfig] = None,
    ) -> None:
        """Initialize personalization manager.

        Args:
            config: Application configuration
            publisher: Event publisher
            metrics_manager: Metrics manager
            personalization_config: Personalization configuration
        """
        super().__init__(config, publisher, metrics_manager)
        self.personalization_config = personalization_config or PersonalizationConfig()
        self._initialized = False
        self._profiles: Dict[str, UserProfile] = {}
        self._scaler = MinMaxScaler()

    async def initialize(self) -> None:
        """Initialize personalization resources."""
        if self._initialized:
            return

        try:
            self._initialized = True
            self.publisher.publish(
                Event(
                    type=EventType.SERVICE_STARTED,
                    timestamp=datetime.now(),
                    component="personalization_manager",
                    description="Personalization manager initialized",
                    metadata={
                        "num_profiles": len(self._profiles),
                    },
                )
            )

        except Exception as e:
            logger.error(f"Failed to initialize personalization manager: {e}")
            raise

    async def cleanup(self) -> None:
        """Clean up personalization resources."""
        self._initialized = False
        self._profiles.clear()

        self.publisher.publish(
            Event(
                type=EventType.SERVICE_STOPPED,
                timestamp=datetime.now(),
                component="personalization_manager",
                description="Personalization manager stopped",
            )
        )

    async def health_check(self) -> Dict[str, Any]:
        """Check personalization manager health.

        Returns:
            Health check results
        """
        return {
            "service": "PersonalizationManager",
            "initialized": self._initialized,
            "num_profiles": len(self._profiles),
            "history_weight": self.personalization_config.history_weight,
            "preference_weight": self.personalization_config.preference_weight,
        }

    def get_user_profile(self, user_id: str) -> Optional[UserProfile]:
        """Get user profile by ID.

        Args:
            user_id: User identifier

        Returns:
            User profile if found
        """
        return self._profiles.get(user_id)

    def update_user_profile(
        self,
        user_id: str,
        preferences: Optional[Dict[str, float]] = None,
        interaction: Optional[Dict[str, Any]] = None,
        tags: Optional[Set[str]] = None,
    ) -> None:
        """Update user profile with new information.

        Args:
            user_id: User identifier
            preferences: Updated preferences
            interaction: New interaction
            tags: Updated tags
        """
        profile = self._profiles.get(user_id)
        if not profile:
            profile = UserProfile(user_id)
            self._profiles[user_id] = profile

        if preferences:
            profile.preferences.update(preferences)

        if interaction:
            profile.history.append({
                **interaction,
                "timestamp": datetime.now(),
            })
            # Trim history if needed
            if len(profile.history) > self.personalization_config.history_window:
                profile.history = profile.history[-self.personalization_config.history_window:]

        if tags:
            profile.tags.update(tags)

        profile.last_updated = datetime.now()

        # Emit profile updated event
        self.publisher.publish(
            Event(
                type=EventType.PROFILE_UPDATED,
                timestamp=datetime.now(),
                component="personalization_manager",
                description=f"Updated profile for user {user_id}",
                metadata={
                    "user_id": user_id,
                    "num_preferences": len(profile.preferences),
                    "num_interactions": len(profile.history),
                    "num_tags": len(profile.tags),
                },
            )
        )

    async def personalize_results(
        self,
        user_id: str,
        results: List[SearchResult],
        min_score: float = 0.0,
    ) -> List[SearchResult]:
        """Personalize search results for user.

        Args:
            user_id: User identifier
            results: Search results to personalize
            min_score: Minimum score threshold

        Returns:
            Personalized search results
        """
        if not self._initialized:
            await self.initialize()

        profile = self.get_user_profile(user_id)
        if not profile:
            return results

        try:
            # Emit personalization started event
            self.publisher.publish(
                Event(
                    type=EventType.PERSONALIZATION_STARTED,
                    timestamp=datetime.now(),
                    component="personalization_manager",
                    description=f"Personalizing results for user {user_id}",
                    metadata={
                        "num_results": len(results),
                    },
                )
            )

            # Check if we have enough interactions
            if len(profile.history) < self.personalization_config.min_interactions:
                self.publisher.publish(
                    Event(
                        type=EventType.PERSONALIZATION_SKIPPED,
                        timestamp=datetime.now(),
                        component="personalization_manager",
                        description="Insufficient interactions for personalization",
                        metadata={
                            "user_id": user_id,
                            "num_interactions": len(profile.history),
                            "min_interactions": self.personalization_config.min_interactions,
                        },
                    )
                )
                return results

            # Calculate personalization scores
            personalized = []
            for result in results:
                # Skip results below threshold
                if result.score < min_score:
                    continue

                # Calculate history score
                history_score = self._calculate_history_score(result, profile)

                # Calculate preference score
                preference_score = self._calculate_preference_score(result, profile)

                # Calculate tag score
                tag_score = self._calculate_tag_score(result, profile)

                # Calculate recency score
                recency_score = self._calculate_recency_score(result, profile)

                # Combine scores
                personalization_score = (
                    self.personalization_config.history_weight * history_score +
                    self.personalization_config.preference_weight * preference_score +
                    self.personalization_config.tag_weight * tag_score +
                    self.personalization_config.recency_weight * recency_score
                )

                # Update result score
                result.score = (result.score + personalization_score) / 2
                personalized.append(result)

            # Sort by personalized score
            personalized.sort(key=lambda x: x.score, reverse=True)

            # Emit success event
            self.publisher.publish(
                Event(
                    type=EventType.PERSONALIZATION_COMPLETED,
                    timestamp=datetime.now(),
                    component="personalization_manager",
                    description="Personalization completed",
                    metadata={
                        "user_id": user_id,
                        "input_results": len(results),
                        "output_results": len(personalized),
                    },
                )
            )

            return personalized

        except Exception as e:
            logger.error(f"Personalization failed: {e}")
            self.publisher.publish(
                Event(
                    type=EventType.PERSONALIZATION_FAILED,
                    timestamp=datetime.now(),
                    component="personalization_manager",
                    description="Personalization failed",
                    error=str(e),
                )
            )
            return results

    def _calculate_history_score(self, result: SearchResult, profile: UserProfile) -> float:
        """Calculate history-based score for result.

        Args:
            result: Search result
            profile: User profile

        Returns:
            History score (0-1)
        """
        if not profile.history:
            return 0.0

        # Count interactions with similar endpoints
        interactions = 0
        for interaction in profile.history:
            if (
                interaction["endpoint"] == result.endpoint or
                interaction["method"] == result.method or
                any(t in result.tags for t in interaction.get("tags", []))
            ):
                interactions += 1

        return min(interactions / len(profile.history), 1.0)

    def _calculate_preference_score(self, result: SearchResult, profile: UserProfile) -> float:
        """Calculate preference-based score for result.

        Args:
            result: Search result
            profile: User profile

        Returns:
            Preference score (0-1)
        """
        if not profile.preferences:
            return 0.0

        # Match result categories with preferences
        matches = 0
        total = 0
        for category, weight in profile.preferences.items():
            if category in result.categories:
                matches += weight
            total += weight

        return matches / total if total > 0 else 0.0

    def _calculate_tag_score(self, result: SearchResult, profile: UserProfile) -> float:
        """Calculate tag-based score for result.

        Args:
            result: Search result
            profile: User profile

        Returns:
            Tag score (0-1)
        """
        if not profile.tags or not result.tags:
            return 0.0

        # Calculate tag overlap
        overlap = len(set(result.tags) & profile.tags)
        return overlap / max(len(result.tags), len(profile.tags))

    def _calculate_recency_score(self, result: SearchResult, profile: UserProfile) -> float:
        """Calculate recency-based score for result.

        Args:
            result: Search result
            profile: User profile

        Returns:
            Recency score (0-1)
        """
        if not profile.history:
            return 0.0

        # Find most recent interaction with similar endpoint
        now = datetime.now()
        most_recent = None
        for interaction in reversed(profile.history):
            if (
                interaction["endpoint"] == result.endpoint or
                interaction["method"] == result.method or
                any(t in result.tags for t in interaction.get("tags", []))
            ):
                most_recent = interaction["timestamp"]
                break

        if not most_recent:
            return 0.0

        # Calculate time-based decay
        time_diff = (now - most_recent).total_seconds()
        return np.exp(-self.personalization_config.decay_factor * time_diff)


# Create service instance
personalization_manager = PersonalizationManager(
    Config(),
    PublisherService(Config(), MetricsManager()),
    MetricsManager(),
)

__all__ = [
    "UserProfile",
    "PersonalizationConfig",
    "PersonalizationManager",
    "personalization_manager",
] 