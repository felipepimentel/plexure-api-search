"""Feedback loop for search result improvement."""

import logging
from typing import Dict, List, Any, Optional, Set
from datetime import datetime
import json
from pathlib import Path

import numpy as np
from sklearn.preprocessing import MinMaxScaler

from ..config import Config
from ..monitoring.events import Event, EventType
from ..monitoring.metrics import MetricsManager
from ..services.base import BaseService
from ..services.events import PublisherService
from .search_models import SearchResult

logger = logging.getLogger(__name__)


class FeedbackType:
    """Types of search result feedback."""

    CLICK = "click"
    RELEVANT = "relevant"
    IRRELEVANT = "irrelevant"
    BOOKMARK = "bookmark"
    REPORT = "report"


class FeedbackEntry:
    """User feedback for a search result."""

    def __init__(
        self,
        user_id: str,
        query: str,
        result_id: str,
        feedback_type: str,
        timestamp: datetime,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Initialize feedback entry.

        Args:
            user_id: User identifier
            query: Search query
            result_id: Result identifier
            feedback_type: Type of feedback
            timestamp: Feedback timestamp
            metadata: Additional metadata
        """
        self.user_id = user_id
        self.query = query
        self.result_id = result_id
        self.feedback_type = feedback_type
        self.timestamp = timestamp
        self.metadata = metadata or {}


class FeedbackConfig:
    """Configuration for feedback loop."""

    def __init__(
        self,
        click_weight: float = 0.3,
        relevance_weight: float = 0.4,
        bookmark_weight: float = 0.2,
        report_weight: float = 0.1,
        min_feedback: int = 5,
        max_feedback: int = 1000,
        learning_rate: float = 0.1,
        decay_factor: float = 0.01,
        feedback_dir: str = "data/feedback",
    ) -> None:
        """Initialize feedback config.

        Args:
            click_weight: Weight for click feedback
            relevance_weight: Weight for relevance feedback
            bookmark_weight: Weight for bookmark feedback
            report_weight: Weight for report feedback
            min_feedback: Minimum feedback entries
            max_feedback: Maximum feedback entries
            learning_rate: Learning rate for updates
            decay_factor: Time decay factor
            feedback_dir: Directory for feedback storage
        """
        self.click_weight = click_weight
        self.relevance_weight = relevance_weight
        self.bookmark_weight = bookmark_weight
        self.report_weight = report_weight
        self.min_feedback = min_feedback
        self.max_feedback = max_feedback
        self.learning_rate = learning_rate
        self.decay_factor = decay_factor
        self.feedback_dir = feedback_dir


class FeedbackManager(BaseService[Dict[str, Any]]):
    """Feedback management service for search improvement."""

    def __init__(
        self,
        config: Config,
        publisher: PublisherService,
        metrics_manager: MetricsManager,
        feedback_config: Optional[FeedbackConfig] = None,
    ) -> None:
        """Initialize feedback manager.

        Args:
            config: Application configuration
            publisher: Event publisher
            metrics_manager: Metrics manager
            feedback_config: Feedback configuration
        """
        super().__init__(config, publisher, metrics_manager)
        self.feedback_config = feedback_config or FeedbackConfig()
        self._initialized = False
        self._feedback: Dict[str, List[FeedbackEntry]] = {}
        self._relevance_scores: Dict[str, float] = {}
        self._scaler = MinMaxScaler()

    async def initialize(self) -> None:
        """Initialize feedback resources."""
        if self._initialized:
            return

        try:
            # Create feedback directory if needed
            Path(self.feedback_config.feedback_dir).mkdir(parents=True, exist_ok=True)

            # Load existing feedback
            await self._load_feedback()

            self._initialized = True
            self.publisher.publish(
                Event(
                    type=EventType.SERVICE_STARTED,
                    timestamp=datetime.now(),
                    component="feedback_manager",
                    description="Feedback manager initialized",
                    metadata={
                        "num_entries": sum(len(entries) for entries in self._feedback.values()),
                    },
                )
            )

        except Exception as e:
            logger.error(f"Failed to initialize feedback manager: {e}")
            raise

    async def cleanup(self) -> None:
        """Clean up feedback resources."""
        if self._initialized:
            # Save feedback before cleanup
            await self._save_feedback()

        self._initialized = False
        self._feedback.clear()
        self._relevance_scores.clear()

        self.publisher.publish(
            Event(
                type=EventType.SERVICE_STOPPED,
                timestamp=datetime.now(),
                component="feedback_manager",
                description="Feedback manager stopped",
            )
        )

    async def health_check(self) -> Dict[str, Any]:
        """Check feedback manager health.

        Returns:
            Health check results
        """
        return {
            "service": "FeedbackManager",
            "initialized": self._initialized,
            "num_entries": sum(len(entries) for entries in self._feedback.values()),
            "num_scores": len(self._relevance_scores),
        }

    async def add_feedback(
        self,
        user_id: str,
        query: str,
        result_id: str,
        feedback_type: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Add user feedback for a search result.

        Args:
            user_id: User identifier
            query: Search query
            result_id: Result identifier
            feedback_type: Type of feedback
            metadata: Additional metadata
        """
        if not self._initialized:
            await self.initialize()

        # Create feedback entry
        entry = FeedbackEntry(
            user_id=user_id,
            query=query,
            result_id=result_id,
            feedback_type=feedback_type,
            timestamp=datetime.now(),
            metadata=metadata,
        )

        # Add to feedback store
        if result_id not in self._feedback:
            self._feedback[result_id] = []
        self._feedback[result_id].append(entry)

        # Trim if needed
        if len(self._feedback[result_id]) > self.feedback_config.max_feedback:
            self._feedback[result_id] = sorted(
                self._feedback[result_id],
                key=lambda x: x.timestamp,
                reverse=True,
            )[:self.feedback_config.max_feedback]

        # Update relevance score
        await self._update_relevance_score(result_id)

        # Save feedback
        await self._save_feedback()

        # Emit feedback added event
        self.publisher.publish(
            Event(
                type=EventType.FEEDBACK_ADDED,
                timestamp=datetime.now(),
                component="feedback_manager",
                description=f"Added {feedback_type} feedback for result {result_id}",
                metadata={
                    "user_id": user_id,
                    "query": query,
                    "result_id": result_id,
                    "feedback_type": feedback_type,
                },
            )
        )

    async def get_relevance_score(self, result_id: str) -> float:
        """Get relevance score for result.

        Args:
            result_id: Result identifier

        Returns:
            Relevance score (0-1)
        """
        if not self._initialized:
            await self.initialize()

        return self._relevance_scores.get(result_id, 0.0)

    async def apply_feedback(
        self,
        results: List[SearchResult],
        query: str,
    ) -> List[SearchResult]:
        """Apply feedback to search results.

        Args:
            results: Search results
            query: Search query

        Returns:
            Updated search results
        """
        if not self._initialized:
            await self.initialize()

        try:
            # Emit feedback application started event
            self.publisher.publish(
                Event(
                    type=EventType.FEEDBACK_STARTED,
                    timestamp=datetime.now(),
                    component="feedback_manager",
                    description="Applying feedback to results",
                    metadata={
                        "num_results": len(results),
                        "query": query,
                    },
                )
            )

            # Apply feedback scores
            updated_results = []
            for result in results:
                result_id = f"{result.method}_{result.endpoint}"
                relevance_score = await self.get_relevance_score(result_id)

                # Combine original score with relevance
                result.score = (result.score + relevance_score) / 2
                updated_results.append(result)

            # Sort by updated scores
            updated_results.sort(key=lambda x: x.score, reverse=True)

            # Emit success event
            self.publisher.publish(
                Event(
                    type=EventType.FEEDBACK_COMPLETED,
                    timestamp=datetime.now(),
                    component="feedback_manager",
                    description="Feedback application completed",
                    metadata={
                        "input_results": len(results),
                        "output_results": len(updated_results),
                    },
                )
            )

            return updated_results

        except Exception as e:
            logger.error(f"Feedback application failed: {e}")
            self.publisher.publish(
                Event(
                    type=EventType.FEEDBACK_FAILED,
                    timestamp=datetime.now(),
                    component="feedback_manager",
                    description="Feedback application failed",
                    error=str(e),
                )
            )
            return results

    async def _update_relevance_score(self, result_id: str) -> None:
        """Update relevance score for result.

        Args:
            result_id: Result identifier
        """
        entries = self._feedback.get(result_id, [])
        if len(entries) < self.feedback_config.min_feedback:
            return

        # Calculate weighted scores
        total_score = 0.0
        total_weight = 0.0
        now = datetime.now()

        for entry in entries:
            # Get base weight by type
            if entry.feedback_type == FeedbackType.CLICK:
                weight = self.feedback_config.click_weight
            elif entry.feedback_type == FeedbackType.RELEVANT:
                weight = self.feedback_config.relevance_weight
            elif entry.feedback_type == FeedbackType.BOOKMARK:
                weight = self.feedback_config.bookmark_weight
            elif entry.feedback_type == FeedbackType.REPORT:
                weight = -self.feedback_config.report_weight
            else:
                continue

            # Apply time decay
            time_diff = (now - entry.timestamp).total_seconds()
            decay = np.exp(-self.feedback_config.decay_factor * time_diff)
            weight *= decay

            # Add to totals
            if entry.feedback_type == FeedbackType.IRRELEVANT:
                total_score += 0.0
            else:
                total_score += weight
            total_weight += abs(weight)

        # Calculate final score
        if total_weight > 0:
            score = total_score / total_weight
            self._relevance_scores[result_id] = max(0.0, min(1.0, score))

    async def _load_feedback(self) -> None:
        """Load feedback from storage."""
        feedback_file = Path(self.feedback_config.feedback_dir) / "feedback.json"
        if not feedback_file.exists():
            return

        try:
            with open(feedback_file, "r") as f:
                data = json.load(f)

            # Convert stored data to FeedbackEntry objects
            for result_id, entries in data.items():
                self._feedback[result_id] = [
                    FeedbackEntry(
                        user_id=entry["user_id"],
                        query=entry["query"],
                        result_id=entry["result_id"],
                        feedback_type=entry["feedback_type"],
                        timestamp=datetime.fromisoformat(entry["timestamp"]),
                        metadata=entry.get("metadata"),
                    )
                    for entry in entries
                ]

            # Update relevance scores
            for result_id in self._feedback:
                await self._update_relevance_score(result_id)

        except Exception as e:
            logger.error(f"Failed to load feedback: {e}")
            self._feedback.clear()
            self._relevance_scores.clear()

    async def _save_feedback(self) -> None:
        """Save feedback to storage."""
        feedback_file = Path(self.feedback_config.feedback_dir) / "feedback.json"

        try:
            # Convert FeedbackEntry objects to serializable data
            data = {
                result_id: [
                    {
                        "user_id": entry.user_id,
                        "query": entry.query,
                        "result_id": entry.result_id,
                        "feedback_type": entry.feedback_type,
                        "timestamp": entry.timestamp.isoformat(),
                        "metadata": entry.metadata,
                    }
                    for entry in entries
                ]
                for result_id, entries in self._feedback.items()
            }

            with open(feedback_file, "w") as f:
                json.dump(data, f, indent=2)

        except Exception as e:
            logger.error(f"Failed to save feedback: {e}")


# Create service instance
feedback_manager = FeedbackManager(
    Config(),
    PublisherService(Config(), MetricsManager()),
    MetricsManager(),
)

__all__ = [
    "FeedbackType",
    "FeedbackEntry",
    "FeedbackConfig",
    "FeedbackManager",
    "feedback_manager",
] 