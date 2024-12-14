"""Contextual boosting and dynamic weight adjustment."""

import json
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from .config import CACHE_DIR


@dataclass
class SearchContext:
    """Context information for a search query."""

    query: str
    query_type: str  # 'semantic', 'structural', 'parameter'
    timestamp: datetime
    user_feedback: Optional[float] = None  # 0 to 1 rating
    success_rate: Optional[float] = None


@dataclass
class WeightProfile:
    """Weight profile for different vector components."""

    semantic_weight: float
    structure_weight: float
    parameter_weight: float

    def to_dict(self) -> Dict[str, float]:
        """Convert to dictionary format."""
        return {
            "semantic": self.semantic_weight,
            "structure": self.structure_weight,
            "parameter": self.parameter_weight,
        }


class ContextualBooster:
    """Handles dynamic weight adjustment based on context."""

    def __init__(self, history_file: str = f"{CACHE_DIR}/search_history.json"):
        """Initialize booster.

        Args:
            history_file: Path to search history file.
        """
        self.history_file = Path(history_file)
        self.history_file.parent.mkdir(parents=True, exist_ok=True)
        self.search_history: List[SearchContext] = []
        self._load_history()

        # Default weights
        self.default_weights = WeightProfile(
            semantic_weight=0.4, structure_weight=0.3, parameter_weight=0.3
        )

    def _load_history(self) -> None:
        """Load search history from file."""
        if self.history_file.exists():
            try:
                with open(self.history_file, "r") as f:
                    data = json.load(f)
                    self.search_history = [
                        SearchContext(
                            query=item["query"],
                            query_type=item["query_type"],
                            timestamp=datetime.fromisoformat(item["timestamp"]),
                            user_feedback=item.get("user_feedback"),
                            success_rate=item.get("success_rate"),
                        )
                        for item in data
                    ]
            except Exception as e:
                print(f"Error loading search history: {e}")
                self.search_history = []

    def _save_history(self) -> None:
        """Save search history to file."""
        try:
            with open(self.history_file, "w") as f:
                json.dump(
                    [
                        {
                            "query": ctx.query,
                            "query_type": ctx.query_type,
                            "timestamp": ctx.timestamp.isoformat(),
                            "user_feedback": ctx.user_feedback,
                            "success_rate": ctx.success_rate,
                        }
                        for ctx in self.search_history
                    ],
                    f,
                )
        except Exception as e:
            print(f"Error saving search history: {e}")

    def detect_query_type(self, query: str) -> str:
        """Detect the type of query based on its content.

        Args:
            query: Search query string.

        Returns:
            Query type ('semantic', 'structural', or 'parameter').
        """
        query = query.lower()

        # Check for structural patterns
        structural_patterns = [
            "get",
            "post",
            "put",
            "delete",
            "endpoint",
            "path",
            "route",
            "version",
            "v1",
            "v2",
        ]

        # Check for parameter patterns
        parameter_patterns = [
            "parameter",
            "input",
            "body",
            "query",
            "type",
            "string",
            "integer",
            "boolean",
            "required",
        ]

        # Count pattern matches
        structural_count = sum(1 for pattern in structural_patterns if pattern in query)
        parameter_count = sum(1 for pattern in parameter_patterns if pattern in query)

        # Determine type based on highest count
        if structural_count > parameter_count:
            return "structural"
        elif parameter_count > structural_count:
            return "parameter"
        else:
            return "semantic"

    def calculate_success_rate(self, contexts: List[SearchContext]) -> float:
        """Calculate success rate from a list of contexts.

        Args:
            contexts: List of search contexts.

        Returns:
            Success rate between 0 and 1.
        """
        if not contexts:
            return 0.5

        feedbacks = [
            ctx.user_feedback for ctx in contexts if ctx.user_feedback is not None
        ]
        return np.mean(feedbacks) if feedbacks else 0.5

    def get_recent_contexts(
        self, query_type: str, window: timedelta = timedelta(days=7)
    ) -> List[SearchContext]:
        """Get recent search contexts of a specific type.

        Args:
            query_type: Type of query to filter for.
            window: Time window to consider.

        Returns:
            List of recent search contexts.
        """
        cutoff = datetime.now() - window
        return [
            ctx
            for ctx in self.search_history
            if ctx.query_type == query_type and ctx.timestamp > cutoff
        ]

    def adjust_weights(
        self, query: str, base_weights: Optional[WeightProfile] = None
    ) -> WeightProfile:
        """Adjust weights based on query context and history.

        Args:
            query: Search query string.
            base_weights: Optional base weights to adjust from.

        Returns:
            Adjusted weight profile.
        """
        if base_weights is None:
            base_weights = self.default_weights

        # Detect query type
        query_type = self.detect_query_type(query)

        # Get recent performance for each type
        semantic_rate = self.calculate_success_rate(
            self.get_recent_contexts("semantic")
        )
        structural_rate = self.calculate_success_rate(
            self.get_recent_contexts("structural")
        )
        parameter_rate = self.calculate_success_rate(
            self.get_recent_contexts("parameter")
        )

        # Calculate adjustment factors
        total_rate = semantic_rate + structural_rate + parameter_rate
        if total_rate == 0:
            total_rate = 1  # Prevent division by zero

        semantic_factor = semantic_rate / total_rate
        structural_factor = structural_rate / total_rate
        parameter_factor = parameter_rate / total_rate

        # Apply adjustments
        return WeightProfile(
            semantic_weight=base_weights.semantic_weight * semantic_factor,
            structure_weight=base_weights.structure_weight * structural_factor,
            parameter_weight=base_weights.parameter_weight * parameter_factor,
        )

    def record_search(self, query: str, user_feedback: Optional[float] = None) -> None:
        """Record a search event.

        Args:
            query: Search query string.
            user_feedback: Optional user feedback score (0 to 1).
        """
        context = SearchContext(
            query=query,
            query_type=self.detect_query_type(query),
            timestamp=datetime.now(),
            user_feedback=user_feedback,
        )

        self.search_history.append(context)
        self._save_history()

    def update_feedback(self, query: str, feedback_score: float) -> None:
        """Update feedback for a recent search.

        Args:
            query: Search query string.
            feedback_score: User feedback score (0 to 1).
        """
        # Find most recent matching query
        for ctx in reversed(self.search_history):
            if ctx.query == query:
                ctx.user_feedback = feedback_score
                break

        self._save_history()
