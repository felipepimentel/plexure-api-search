"""Search quality metrics and evaluation."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import TYPE_CHECKING, Dict, List, Optional

import numpy as np
from sklearn.metrics import ndcg_score

if TYPE_CHECKING:
    from .search_models import SearchResult


@dataclass
class SearchEvaluation:
    """Evaluation metrics for a search query."""

    query: str
    mrr: float
    ndcg: float
    precision_at_1: float
    precision_at_5: float
    recall: float
    latency_ms: float
    timestamp: datetime
    # Business metrics
    time_to_first_success: Optional[float] = None  # Time until user finds first relevant result
    session_duration: Optional[float] = None  # Total session duration
    click_through_rate: Optional[float] = None  # Ratio of clicked results
    conversion_rate: Optional[float] = None  # Ratio of queries leading to API usage
    user_satisfaction: Optional[float] = None  # User satisfaction score (0-1)
    api_adoption_rate: Optional[float] = None  # Rate of API adoption after search
    endpoint_coverage: Optional[float] = None  # Coverage of searched endpoints
    business_value_score: Optional[float] = None  # Composite business value score


@dataclass
class SearchMetrics:
    precision: float
    recall: float
    f1_score: float
    ndcg: float
    latency: float


class QualityMetrics:
    """Handles search quality metrics and evaluation."""

    def __init__(self, metrics_file: str = ".metrics/search_quality.json"):
        """Initialize quality metrics.

        Args:
            metrics_file: Path to metrics storage file.
        """
        self.metrics_file = Path(metrics_file)
        self.metrics_file.parent.mkdir(parents=True, exist_ok=True)
        self.evaluations: List[SearchEvaluation] = []
        self.metrics_history: List[SearchMetrics] = []
        self._load_metrics()

    def _load_metrics(self) -> None:
        """Load metrics from file."""
        if self.metrics_file.exists():
            try:
                with open(self.metrics_file, "r") as f:
                    data = json.load(f)
                    self.evaluations = [
                        SearchEvaluation(
                            query=item["query"],
                            mrr=item["mrr"],
                            ndcg=item["ndcg"],
                            precision_at_1=item["precision_at_1"],
                            precision_at_5=item["precision_at_5"],
                            recall=item["recall"],
                            latency_ms=item["latency_ms"],
                            timestamp=datetime.fromisoformat(item["timestamp"]),
                        )
                        for item in data
                    ]
            except Exception as e:
                print(f"Error loading metrics: {e}")
                self.evaluations = []

    def _save_metrics(self) -> None:
        """Save metrics to file."""
        try:
            with open(self.metrics_file, "w") as f:
                json.dump(
                    [
                        {
                            "query": eval.query,
                            "mrr": eval.mrr,
                            "ndcg": eval.ndcg,
                            "precision_at_1": eval.precision_at_1,
                            "precision_at_5": eval.precision_at_5,
                            "recall": eval.recall,
                            "latency_ms": eval.latency_ms,
                            "timestamp": eval.timestamp.isoformat(),
                        }
                        for eval in self.evaluations
                    ],
                    f,
                )
        except Exception as e:
            print(f"Error saving metrics: {e}")

    def calculate_mrr(self, results: List[SearchResult]) -> float:
        """Calculate Mean Reciprocal Rank.

        Args:
            results: List of search results.

        Returns:
            MRR score.
        """
        for i, result in enumerate(results, 1):
            if result.is_relevant:
                return 1.0 / i
        return 0.0

    def calculate_ndcg(
        self, relevance_scores: List[float], ideal_scores: Optional[List[float]] = None
    ) -> float:
        """Calculate Normalized Discounted Cumulative Gain."""
        if ideal_scores is None:
            ideal_scores = sorted(relevance_scores, reverse=True)

        return ndcg_score([relevance_scores], [ideal_scores])

    def calculate_precision(
        self, results: List[SearchResult], k: Optional[int] = None
    ) -> float:
        """Calculate Precision@k.

        Args:
            results: List of search results.
            k: Optional cutoff rank.

        Returns:
            Precision score.
        """
        if k is not None:
            results = results[:k]

        if not results:
            return 0.0

        return sum(1 for r in results if r.is_relevant) / len(results)

    def calculate_recall(
        self, results: List[SearchResult], total_relevant: int
    ) -> float:
        """Calculate Recall.

        Args:
            results: List of search results.
            total_relevant: Total number of relevant items.

        Returns:
            Recall score.
        """
        if total_relevant == 0:
            return 0.0

        return sum(1 for r in results if r.is_relevant) / total_relevant

    def calculate_business_value(
        self,
        results: List[SearchResult],
        user_metrics: Dict[str, float],
    ) -> float:
        """Calculate business value score for search results.

        Args:
            results: List of search results
            user_metrics: Dictionary of user interaction metrics

        Returns:
            Business value score between 0 and 1
        """
        # Weight factors for different components
        weights = {
            "api_adoption": 0.3,
            "user_satisfaction": 0.2,
            "endpoint_coverage": 0.2,
            "conversion": 0.2,
            "performance": 0.1
        }

        # Calculate component scores
        api_adoption = user_metrics.get("api_adoption_rate", 0.0)
        user_satisfaction = user_metrics.get("user_satisfaction", 0.0)
        endpoint_coverage = len(set(r.endpoint_id for r in results)) / self.total_endpoints
        conversion = user_metrics.get("conversion_rate", 0.0)
        performance = 1.0 - min(user_metrics.get("latency_ms", 0.0) / 1000.0, 1.0)

        # Calculate weighted score
        business_value = (
            weights["api_adoption"] * api_adoption +
            weights["user_satisfaction"] * user_satisfaction +
            weights["endpoint_coverage"] * endpoint_coverage +
            weights["conversion"] * conversion +
            weights["performance"] * performance
        )

        return min(max(business_value, 0.0), 1.0)

    def evaluate_search(
        self,
        query: str,
        results: List[SearchResult],
        total_relevant: int,
        latency_ms: float,
    ) -> SearchEvaluation:
        """Evaluate search results.

        Args:
            query: Search query.
            results: List of search results.
            total_relevant: Total number of relevant items.
            latency_ms: Search latency in milliseconds.

        Returns:
            SearchEvaluation with metrics.
        """
        evaluation = SearchEvaluation(
            query=query,
            mrr=self.calculate_mrr(results),
            ndcg=self.calculate_ndcg(results),
            precision_at_1=self.calculate_precision(results, k=1),
            precision_at_5=self.calculate_precision(results, k=5),
            recall=self.calculate_recall(results, total_relevant),
            latency_ms=latency_ms,
            timestamp=datetime.now(),
        )

        self.evaluations.append(evaluation)
        self._save_metrics()

        return evaluation

    def get_average_metrics(
        self, window: timedelta = timedelta(days=7)
    ) -> Dict[str, float]:
        """Get average metrics over a time window.

        Args:
            window: Time window for averaging.

        Returns:
            Dictionary of average metrics.
        """
        cutoff = datetime.now() - window
        recent_evals = [eval for eval in self.evaluations if eval.timestamp > cutoff]

        if not recent_evals:
            return {
                "mrr": 0.0,
                "ndcg": 0.0,
                "precision_at_1": 0.0,
                "precision_at_5": 0.0,
                "recall": 0.0,
                "latency_ms": 0.0,
            }

        return {
            "mrr": np.mean([eval.mrr for eval in recent_evals]),
            "ndcg": np.mean([eval.ndcg for eval in recent_evals]),
            "precision_at_1": np.mean([eval.precision_at_1 for eval in recent_evals]),
            "precision_at_5": np.mean([eval.precision_at_5 for eval in recent_evals]),
            "recall": np.mean([eval.recall for eval in recent_evals]),
            "latency_ms": np.mean([eval.latency_ms for eval in recent_evals]),
        }

    def get_metric_trends(
        self,
        window: timedelta = timedelta(days=30),
        interval: timedelta = timedelta(days=1),
    ) -> Dict[str, List[float]]:
        """Get metric trends over time.

        Args:
            window: Time window for analysis.
            interval: Time interval for aggregation.

        Returns:
            Dictionary of metric trends.
        """
        cutoff = datetime.now() - window
        recent_evals = [eval for eval in self.evaluations if eval.timestamp > cutoff]

        if not recent_evals:
            return {
                "mrr": [],
                "ndcg": [],
                "precision_at_1": [],
                "precision_at_5": [],
                "recall": [],
                "latency_ms": [],
            }

        # Sort by timestamp
        recent_evals.sort(key=lambda x: x.timestamp)

        # Initialize bins
        num_bins = int(window / interval)
        bins = [[] for _ in range(num_bins)]

        # Distribute evaluations into bins
        for eval in recent_evals:
            bin_index = int((eval.timestamp - cutoff) / interval)
            if 0 <= bin_index < num_bins:
                bins[bin_index].append(eval)

        # Calculate averages for each bin
        trends = {
            "mrr": [],
            "ndcg": [],
            "precision_at_1": [],
            "precision_at_5": [],
            "recall": [],
            "latency_ms": [],
        }

        for bin_evals in bins:
            if bin_evals:
                trends["mrr"].append(np.mean([e.mrr for e in bin_evals]))
                trends["ndcg"].append(np.mean([e.ndcg for e in bin_evals]))
                trends["precision_at_1"].append(
                    np.mean([e.precision_at_1 for e in bin_evals])
                )
                trends["precision_at_5"].append(
                    np.mean([e.precision_at_5 for e in bin_evals])
                )
                trends["recall"].append(np.mean([e.recall for e in bin_evals]))
                trends["latency_ms"].append(np.mean([e.latency_ms for e in bin_evals]))
            else:
                # Fill gaps with None
                for metric in trends:
                    trends[metric].append(None)

        return trends

    def get_business_metrics(
        self,
        window: timedelta = timedelta(days=30)
    ) -> Dict[str, float]:
        """Get business-focused metrics over a time window.

        Args:
            window: Time window for analysis

        Returns:
            Dictionary of business metrics
        """
        cutoff = datetime.now() - window
        recent_evals = [eval for eval in self.evaluations if eval.timestamp > cutoff]

        if not recent_evals:
            return {
                "avg_time_to_success": 0.0,
                "avg_session_duration": 0.0,
                "avg_click_through_rate": 0.0,
                "avg_conversion_rate": 0.0,
                "avg_user_satisfaction": 0.0,
                "avg_api_adoption_rate": 0.0,
                "avg_endpoint_coverage": 0.0,
                "avg_business_value": 0.0,
                "roi_score": 0.0,
            }

        metrics = {
            "avg_time_to_success": np.mean([e.time_to_first_success for e in recent_evals if e.time_to_first_success is not None]),
            "avg_session_duration": np.mean([e.session_duration for e in recent_evals if e.session_duration is not None]),
            "avg_click_through_rate": np.mean([e.click_through_rate for e in recent_evals if e.click_through_rate is not None]),
            "avg_conversion_rate": np.mean([e.conversion_rate for e in recent_evals if e.conversion_rate is not None]),
            "avg_user_satisfaction": np.mean([e.user_satisfaction for e in recent_evals if e.user_satisfaction is not None]),
            "avg_api_adoption_rate": np.mean([e.api_adoption_rate for e in recent_evals if e.api_adoption_rate is not None]),
            "avg_endpoint_coverage": np.mean([e.endpoint_coverage for e in recent_evals if e.endpoint_coverage is not None]),
            "avg_business_value": np.mean([e.business_value_score for e in recent_evals if e.business_value_score is not None]),
        }

        # Calculate ROI score based on conversion and adoption metrics
        metrics["roi_score"] = (
            metrics["avg_conversion_rate"] * 0.6 +
            metrics["avg_api_adoption_rate"] * 0.4
        )

        return metrics
