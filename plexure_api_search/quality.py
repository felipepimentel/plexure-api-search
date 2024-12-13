"""Search quality metrics and evaluation."""

from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import numpy as np
from datetime import datetime, timedelta
import json
from pathlib import Path


@dataclass
class SearchResult:
    """Single search result with relevance information."""
    endpoint_id: str
    score: float
    rank: int
    is_relevant: Optional[bool] = None


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
        self._load_metrics()
        
    def _load_metrics(self) -> None:
        """Load metrics from file."""
        if self.metrics_file.exists():
            try:
                with open(self.metrics_file, 'r') as f:
                    data = json.load(f)
                    self.evaluations = [
                        SearchEvaluation(
                            query=item['query'],
                            mrr=item['mrr'],
                            ndcg=item['ndcg'],
                            precision_at_1=item['precision_at_1'],
                            precision_at_5=item['precision_at_5'],
                            recall=item['recall'],
                            latency_ms=item['latency_ms'],
                            timestamp=datetime.fromisoformat(item['timestamp'])
                        )
                        for item in data
                    ]
            except Exception as e:
                print(f"Error loading metrics: {e}")
                self.evaluations = []
                
    def _save_metrics(self) -> None:
        """Save metrics to file."""
        try:
            with open(self.metrics_file, 'w') as f:
                json.dump([
                    {
                        'query': eval.query,
                        'mrr': eval.mrr,
                        'ndcg': eval.ndcg,
                        'precision_at_1': eval.precision_at_1,
                        'precision_at_5': eval.precision_at_5,
                        'recall': eval.recall,
                        'latency_ms': eval.latency_ms,
                        'timestamp': eval.timestamp.isoformat()
                    }
                    for eval in self.evaluations
                ], f)
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
        self,
        results: List[SearchResult],
        k: Optional[int] = None
    ) -> float:
        """Calculate Normalized Discounted Cumulative Gain.
        
        Args:
            results: List of search results.
            k: Optional cutoff rank.
            
        Returns:
            NDCG score.
        """
        if k is not None:
            results = results[:k]
            
        dcg = 0.0
        idcg = 0.0
        
        # Calculate DCG
        for i, result in enumerate(results, 1):
            if result.is_relevant:
                dcg += 1.0 / np.log2(i + 1)
                
        # Calculate IDCG
        relevant_count = sum(1 for r in results if r.is_relevant)
        for i in range(relevant_count):
            idcg += 1.0 / np.log2(i + 2)
            
        return dcg / idcg if idcg > 0 else 0.0
        
    def calculate_precision(
        self,
        results: List[SearchResult],
        k: Optional[int] = None
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
        self,
        results: List[SearchResult],
        total_relevant: int
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
        
    def evaluate_search(
        self,
        query: str,
        results: List[SearchResult],
        total_relevant: int,
        latency_ms: float
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
            timestamp=datetime.now()
        )
        
        self.evaluations.append(evaluation)
        self._save_metrics()
        
        return evaluation
        
    def get_average_metrics(
        self,
        window: timedelta = timedelta(days=7)
    ) -> Dict[str, float]:
        """Get average metrics over a time window.
        
        Args:
            window: Time window for averaging.
            
        Returns:
            Dictionary of average metrics.
        """
        cutoff = datetime.now() - window
        recent_evals = [
            eval for eval in self.evaluations
            if eval.timestamp > cutoff
        ]
        
        if not recent_evals:
            return {
                'mrr': 0.0,
                'ndcg': 0.0,
                'precision_at_1': 0.0,
                'precision_at_5': 0.0,
                'recall': 0.0,
                'latency_ms': 0.0
            }
            
        return {
            'mrr': np.mean([eval.mrr for eval in recent_evals]),
            'ndcg': np.mean([eval.ndcg for eval in recent_evals]),
            'precision_at_1': np.mean([eval.precision_at_1 for eval in recent_evals]),
            'precision_at_5': np.mean([eval.precision_at_5 for eval in recent_evals]),
            'recall': np.mean([eval.recall for eval in recent_evals]),
            'latency_ms': np.mean([eval.latency_ms for eval in recent_evals])
        }
        
    def get_metric_trends(
        self,
        window: timedelta = timedelta(days=30),
        interval: timedelta = timedelta(days=1)
    ) -> Dict[str, List[float]]:
        """Get metric trends over time.
        
        Args:
            window: Time window for analysis.
            interval: Time interval for aggregation.
            
        Returns:
            Dictionary of metric trends.
        """
        cutoff = datetime.now() - window
        recent_evals = [
            eval for eval in self.evaluations
            if eval.timestamp > cutoff
        ]
        
        if not recent_evals:
            return {
                'mrr': [],
                'ndcg': [],
                'precision_at_1': [],
                'precision_at_5': [],
                'recall': [],
                'latency_ms': []
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
            'mrr': [],
            'ndcg': [],
            'precision_at_1': [],
            'precision_at_5': [],
            'recall': [],
            'latency_ms': []
        }
        
        for bin_evals in bins:
            if bin_evals:
                trends['mrr'].append(np.mean([e.mrr for e in bin_evals]))
                trends['ndcg'].append(np.mean([e.ndcg for e in bin_evals]))
                trends['precision_at_1'].append(np.mean([e.precision_at_1 for e in bin_evals]))
                trends['precision_at_5'].append(np.mean([e.precision_at_5 for e in bin_evals]))
                trends['recall'].append(np.mean([e.recall for e in bin_evals]))
                trends['latency_ms'].append(np.mean([e.latency_ms for e in bin_evals]))
            else:
                # Fill gaps with None
                for metric in trends:
                    trends[metric].append(None)
                    
        return trends 