"""Project consistency and health checks."""

import importlib
import inspect
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

from .config import CACHE_DIR, HEALTH_DIR, METRICS_DIR


class ProjectHealth:
    """Monitors and validates project consistency."""

    def __init__(self, cache_file: str = f"{HEALTH_DIR}/consistency.json"):
        """Initialize health checker.

        Args:
            cache_file: Path to cache file.
        """
        self.cache_file = Path(cache_file)
        self.cache_file.parent.mkdir(parents=True, exist_ok=True)

    def check_dependencies(self) -> Dict[str, bool]:
        """Check if all required dependencies are installed.

        Returns:
            Dictionary of dependency status.
        """
        required = [
            "pinecone",
            "sentence_transformers",
            "numpy",
            "spacy",
            "networkx",
            "fastapi",
            "uvicorn",
            "python-dotenv",
            "pydantic",
            "openapi-spec-validator",
            "requests",
            "tqdm",
            "scikit-learn",
        ]

        status = {}
        for dep in required:
            try:
                importlib.import_module(dep.replace("-", "_"))
                status[dep] = True
            except ImportError:
                status[dep] = False

        return status

    def check_model_files(self) -> Dict[str, bool]:
        """Check if required model files exist.

        Returns:
            Dictionary of model file status.
        """
        import spacy.util

        models = {
            "en_core_web_sm": spacy.util.is_package("en_core_web_sm"),
            "sentence_transformer": Path("models/all-MiniLM-L6-v2").exists(),
        }

        return models

    def check_cache_consistency(self) -> Dict[str, Any]:
        """Check cache consistency across components.

        Returns:
            Dictionary of cache status.
        """
        cache_paths = [
            f"{CACHE_DIR}/search_history.json",
            f"{METRICS_DIR}/search_quality.json",
            f"{HEALTH_DIR}/consistency.json",
        ]

        status = {}
        for path in cache_paths:
            cache_file = Path(path)
            if cache_file.exists():
                try:
                    with open(cache_file, "r") as f:
                        data = json.load(f)
                    status[path] = {
                        "exists": True,
                        "valid_json": True,
                        "last_modified": datetime.fromtimestamp(
                            cache_file.stat().st_mtime
                        ).isoformat(),
                    }
                except json.JSONDecodeError:
                    status[path] = {"exists": True, "valid_json": False}
            else:
                status[path] = {"exists": False}

        return status

    def check_api_consistency(self) -> Dict[str, Any]:
        """Check API component consistency.

        Returns:
            Dictionary of API consistency status.
        """
        from .boosting import ContextualBooster
        from .embeddings import TripleVectorizer
        from .expansion import QueryExpander
        from .quality import QualityMetrics
        from .searcher import APISearcher
        from .understanding import ZeroShotUnderstanding

        components = {
            "APISearcher": APISearcher,
            "TripleVectorizer": TripleVectorizer,
            "ContextualBooster": ContextualBooster,
            "ZeroShotUnderstanding": ZeroShotUnderstanding,
            "QueryExpander": QueryExpander,
            "QualityMetrics": QualityMetrics,
        }

        status = {}
        for name, component in components.items():
            methods = inspect.getmembers(component, predicate=inspect.isfunction)
            status[name] = {
                "method_count": len(methods),
                "has_init": hasattr(component, "__init__"),
                "has_docstring": bool(component.__doc__),
            }

        return status

    def run_full_check(self) -> Dict[str, Any]:
        """Run all consistency checks.

        Returns:
            Dictionary with all check results.
        """
        results = {
            "timestamp": datetime.now().isoformat(),
            "dependencies": self.check_dependencies(),
            "models": self.check_model_files(),
            "cache": self.check_cache_consistency(),
            "api": self.check_api_consistency(),
        }

        # Cache results
        self.cache_file.parent.mkdir(parents=True, exist_ok=True)
        with open(self.cache_file, "w") as f:
            json.dump(results, f, indent=2)

        return results

    def get_health_summary(self) -> Dict[str, str]:
        """Get a summary of project health.

        Returns:
            Dictionary with health summary.
        """
        results = self.run_full_check()

        # Check dependencies
        deps_ok = all(results["dependencies"].values())

        # Check models
        models_ok = all(results["models"].values())

        # Check cache
        cache_ok = all(
            status.get("valid_json", False)
            for status in results["cache"].values()
            if status["exists"]
        )

        # Check API
        api_ok = all(
            status["has_init"] and status["has_docstring"]
            for status in results["api"].values()
        )

        return {
            "status": "healthy"
            if all([deps_ok, models_ok, cache_ok, api_ok])
            else "unhealthy",
            "dependencies": "ok" if deps_ok else "missing dependencies",
            "models": "ok" if models_ok else "missing models",
            "cache": "ok" if cache_ok else "cache inconsistency",
            "api": "ok" if api_ok else "api inconsistency",
        }
