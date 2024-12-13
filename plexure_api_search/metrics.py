"""Metrics collection and quality monitoring."""

import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
from rich.console import Console


class MetricsCalculator:
    """Calculates and records search metrics."""
    
    def __init__(self, metrics_dir: str = ".metrics"):
        """Initialize metrics calculator.
        
        Args:
            metrics_dir: Directory to store metrics.
        """
        self.metrics_dir = Path(metrics_dir)
        self.metrics_dir.mkdir(parents=True, exist_ok=True)
        self.console = Console()
        
    def log_search_metrics(self, query: str, num_results: int) -> None:
        """Log search-related metrics.
        
        Args:
            query: Search query.
            num_results: Number of results found.
        """
        metrics = {
            "timestamp": datetime.now().isoformat(),
            "query": query,
            "num_results": num_results
        }
        
        self._append_metrics("search_metrics.jsonl", metrics)
        
    def log_indexing_metrics(
        self,
        num_files: int,
        num_endpoints: int,
        duration: float
    ) -> None:
        """Log indexing-related metrics.
        
        Args:
            num_files: Number of files processed.
            num_endpoints: Number of endpoints indexed.
            duration: Time taken in seconds.
        """
        metrics = {
            "timestamp": datetime.now().isoformat(),
            "num_files": num_files,
            "num_endpoints": num_endpoints,
            "duration": duration
        }
        
        self._append_metrics("indexing_metrics.jsonl", metrics)
        
    def calculate_search_stats(self) -> Dict[str, float]:
        """Calculate search statistics.
        
        Returns:
            Dictionary of search statistics.
        """
        metrics = self._read_metrics("search_metrics.jsonl")
        
        if not metrics:
            return {}
            
        num_results = [m["num_results"] for m in metrics]
        
        return {
            "total_searches": len(metrics),
            "avg_results": np.mean(num_results),
            "p95_results": np.percentile(num_results, 95),
            "zero_results_rate": sum(1 for n in num_results if n == 0) / len(num_results)
        }
        
    def calculate_indexing_stats(self) -> Dict[str, float]:
        """Calculate indexing statistics.
        
        Returns:
            Dictionary of indexing statistics.
        """
        metrics = self._read_metrics("indexing_metrics.jsonl")
        
        if not metrics:
            return {}
            
        durations = [m["duration"] for m in metrics]
        endpoints = [m["num_endpoints"] for m in metrics]
        
        return {
            "total_indexing_runs": len(metrics),
            "avg_duration": np.mean(durations),
            "p95_duration": np.percentile(durations, 95),
            "total_endpoints": sum(endpoints),
            "avg_endpoints_per_run": np.mean(endpoints)
        }
        
    def _append_metrics(self, filename: str, metrics: Dict) -> None:
        """Append metrics to file.
        
        Args:
            filename: Name of metrics file.
            metrics: Metrics to append.
        """
        file_path = self.metrics_dir / filename
        with open(file_path, "a") as f:
            f.write(json.dumps(metrics) + "\n")
            
    def _read_metrics(self, filename: str) -> List[Dict]:
        """Read metrics from file.
        
        Args:
            filename: Name of metrics file.
            
        Returns:
            List of metrics dictionaries.
        """
        file_path = self.metrics_dir / filename
        metrics = []
        
        if not file_path.exists():
            return metrics
            
        with open(file_path) as f:
            for line in f:
                try:
                    metrics.append(json.loads(line.strip()))
                except Exception as e:
                    self.console.print(f"[yellow]Warning: Failed to parse metrics: {e}[/]")
                    
        return metrics


class QualityMonitor:
    """Monitors search quality metrics."""
    
    def __init__(self, metrics_dir: str = ".metrics"):
        """Initialize quality monitor.
        
        Args:
            metrics_dir: Directory to store metrics.
        """
        self.metrics_dir = Path(metrics_dir)
        self.metrics_dir.mkdir(parents=True, exist_ok=True)
        self.console = Console()
        
    def log_quality_metrics(
        self,
        query: str,
        results: List[Dict],
        relevance_scores: Optional[List[float]] = None
    ) -> None:
        """Log quality-related metrics.
        
        Args:
            query: Search query.
            results: Search results.
            relevance_scores: Optional relevance scores.
        """
        metrics = {
            "timestamp": datetime.now().isoformat(),
            "query": query,
            "num_results": len(results),
            "has_results": len(results) > 0
        }
        
        if relevance_scores:
            metrics.update({
                "avg_relevance": np.mean(relevance_scores),
                "min_relevance": min(relevance_scores),
                "max_relevance": max(relevance_scores)
            })
            
        self._append_metrics("quality_metrics.jsonl", metrics)
        
    def calculate_quality_stats(self) -> Dict[str, float]:
        """Calculate quality statistics.
        
        Returns:
            Dictionary of quality statistics.
        """
        metrics = self._read_metrics("quality_metrics.jsonl")
        
        if not metrics:
            return {}
            
        has_results = [m["has_results"] for m in metrics]
        
        stats = {
            "total_queries": len(metrics),
            "success_rate": sum(has_results) / len(has_results)
        }
        
        # Add relevance stats if available
        relevance_metrics = [
            m for m in metrics
            if "avg_relevance" in m
        ]
        
        if relevance_metrics:
            avg_relevance = [m["avg_relevance"] for m in relevance_metrics]
            min_relevance = [m["min_relevance"] for m in relevance_metrics]
            
            stats.update({
                "avg_relevance": np.mean(avg_relevance),
                "p95_relevance": np.percentile(avg_relevance, 95),
                "min_relevance": min(min_relevance)
            })
            
        return stats
        
    def _append_metrics(self, filename: str, metrics: Dict) -> None:
        """Append metrics to file.
        
        Args:
            filename: Name of metrics file.
            metrics: Metrics to append.
        """
        file_path = self.metrics_dir / filename
        with open(file_path, "a") as f:
            f.write(json.dumps(metrics) + "\n")
            
    def _read_metrics(self, filename: str) -> List[Dict]:
        """Read metrics from file.
        
        Args:
            filename: Name of metrics file.
            
        Returns:
            List of metrics dictionaries.
        """
        file_path = self.metrics_dir / filename
        metrics = []
        
        if not file_path.exists():
            return metrics
            
        with open(file_path) as f:
            for line in f:
                try:
                    metrics.append(json.loads(line.strip()))
                except Exception as e:
                    self.console.print(f"[yellow]Warning: Failed to parse metrics: {e}[/]")
                    
        return metrics 