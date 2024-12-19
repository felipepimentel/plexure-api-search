"""Consistency checking for search results."""

import json
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from ..config import config_instance

class Consistency:
    """Consistency checking for search results."""

    def __init__(
        self,
        cache_file: Optional[Path] = None,
        max_age: int = 86400,  # 24 hours
        min_confidence: float = 0.8,
    ):
        """Initialize consistency checker.
        
        Args:
            cache_file: Path to cache file.
                Defaults to health_dir/consistency.json.
            max_age: Maximum age of consistency data in seconds.
                Defaults to 24 hours.
            min_confidence: Minimum confidence score for consistent results.
                Defaults to 0.8.
        """
        self.cache_file = cache_file or (config_instance.health_dir / "consistency.json")
        self.max_age = max_age
        self.min_confidence = min_confidence
        self._cache: Dict[str, Dict] = {}
        self._load_cache()

    def _load_cache(self) -> None:
        """Load consistency cache from file."""
        if self.cache_file.exists():
            try:
                with open(self.cache_file) as f:
                    self._cache = json.load(f)
            except Exception:
                self._cache = {}

    def _save_cache(self) -> None:
        """Save consistency cache to file."""
        try:
            self.cache_file.parent.mkdir(parents=True, exist_ok=True)
            with open(self.cache_file, "w") as f:
                json.dump(self._cache, f, indent=2)
        except Exception:
            pass  # Ignore save errors

    def check_result(
        self,
        query: str,
        result_id: str,
        score: float,
        timestamp: Optional[float] = None,
    ) -> bool:
        """Check if search result is consistent.
        
        Args:
            query: Search query.
            result_id: Result identifier.
            score: Result confidence score.
            timestamp: Result timestamp.
                Defaults to current time.
                
        Returns:
            True if result is consistent, False otherwise.
        """
        timestamp = timestamp or time.time()
        
        # Get or create query entry
        if query not in self._cache:
            self._cache[query] = {
                "results": {},
                "last_update": timestamp
            }
        
        # Update result
        self._cache[query]["results"][result_id] = {
            "score": score,
            "timestamp": timestamp
        }
        
        # Save cache
        self._save_cache()
        
        # Check consistency
        return self._check_consistency(query, result_id)

    def _check_consistency(self, query: str, result_id: str) -> bool:
        """Check consistency of result.
        
        Args:
            query: Search query.
            result_id: Result identifier.
            
        Returns:
            True if result is consistent, False otherwise.
        """
        if query not in self._cache:
            return True
            
        query_data = self._cache[query]
        results = query_data["results"]
        
        if result_id not in results:
            return True
            
        result = results[result_id]
        
        # Check age
        age = time.time() - result["timestamp"]
        if age > self.max_age:
            return True
            
        # Check confidence
        return result["score"] >= self.min_confidence

    def get_inconsistent_results(
        self,
        min_occurrences: int = 2,
    ) -> List[Tuple[str, str, float]]:
        """Get list of inconsistent results.
        
        Args:
            min_occurrences: Minimum number of occurrences for inconsistency.
                Defaults to 2.
                
        Returns:
            List of (query, result_id, score) tuples.
        """
        inconsistent = []
        
        for query, query_data in self._cache.items():
            results = query_data["results"]
            
            # Count occurrences
            occurrences: Dict[str, int] = {}
            for result_id in results:
                if not self._check_consistency(query, result_id):
                    occurrences[result_id] = occurrences.get(result_id, 0) + 1
            
            # Add inconsistent results
            for result_id, count in occurrences.items():
                if count >= min_occurrences:
                    score = results[result_id]["score"]
                    inconsistent.append((query, result_id, score))
        
        return inconsistent

    def clear_old_entries(self) -> None:
        """Clear entries older than max_age."""
        now = time.time()
        
        for query in list(self._cache.keys()):
            query_data = self._cache[query]
            results = query_data["results"]
            
            # Remove old results
            for result_id in list(results.keys()):
                result = results[result_id]
                age = now - result["timestamp"]
                if age > self.max_age:
                    del results[result_id]
            
            # Remove empty queries
            if not results:
                del self._cache[query]
        
        # Save cache
        self._save_cache()

    def get_stats(self) -> Dict:
        """Get consistency statistics.
        
        Returns:
            Dictionary with statistics:
            - total_queries: Total number of queries
            - total_results: Total number of results
            - inconsistent_results: Number of inconsistent results
            - avg_score: Average confidence score
            - min_score: Minimum confidence score
            - max_score: Maximum confidence score
            - avg_age: Average age in seconds
            - cache_size: Cache size in bytes
        """
        total_queries = len(self._cache)
        total_results = 0
        inconsistent_results = 0
        scores = []
        ages = []
        
        now = time.time()
        
        for query_data in self._cache.values():
            results = query_data["results"]
            total_results += len(results)
            
            for result_id, result in results.items():
                scores.append(result["score"])
                ages.append(now - result["timestamp"])
                
                if not self._check_consistency(query, result_id):
                    inconsistent_results += 1
        
        return {
            "total_queries": total_queries,
            "total_results": total_results,
            "inconsistent_results": inconsistent_results,
            "avg_score": sum(scores) / len(scores) if scores else 0,
            "min_score": min(scores) if scores else 0,
            "max_score": max(scores) if scores else 0,
            "avg_age": sum(ages) / len(ages) if ages else 0,
            "cache_size": len(json.dumps(self._cache))
        }
