"""Result clustering for improved search organization."""

import logging
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime

import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

from ..config import Config
from ..monitoring.events import Event, EventType
from ..monitoring.metrics import MetricsManager
from ..services.base import BaseService
from ..services.events import PublisherService
from .search_models import SearchResult

logger = logging.getLogger(__name__)


class ClusterConfig:
    """Configuration for result clustering."""

    def __init__(
        self,
        min_clusters: int = 2,
        max_clusters: int = 5,
        min_cluster_size: int = 2,
        min_silhouette_score: float = 0.1,
        max_iterations: int = 100,
        random_state: int = 42,
    ) -> None:
        """Initialize cluster config.

        Args:
            min_clusters: Minimum number of clusters
            max_clusters: Maximum number of clusters
            min_cluster_size: Minimum size per cluster
            min_silhouette_score: Minimum silhouette score
            max_iterations: Maximum clustering iterations
            random_state: Random state for reproducibility
        """
        self.min_clusters = min_clusters
        self.max_clusters = max_clusters
        self.min_cluster_size = min_cluster_size
        self.min_silhouette_score = min_silhouette_score
        self.max_iterations = max_iterations
        self.random_state = random_state


class ResultCluster:
    """Cluster of search results."""

    def __init__(
        self,
        label: str,
        description: str,
        results: List[SearchResult],
        centroid: np.ndarray,
        score: float,
    ) -> None:
        """Initialize result cluster.

        Args:
            label: Cluster label
            description: Cluster description
            results: Clustered search results
            centroid: Cluster centroid vector
            score: Cluster quality score
        """
        self.label = label
        self.description = description
        self.results = results
        self.centroid = centroid
        self.score = score


class ClusterManager(BaseService[Dict[str, Any]]):
    """Clustering service for search results."""

    def __init__(
        self,
        config: Config,
        publisher: PublisherService,
        metrics_manager: MetricsManager,
        cluster_config: Optional[ClusterConfig] = None,
    ) -> None:
        """Initialize cluster manager.

        Args:
            config: Application configuration
            publisher: Event publisher
            metrics_manager: Metrics manager
            cluster_config: Clustering configuration
        """
        super().__init__(config, publisher, metrics_manager)
        self.cluster_config = cluster_config or ClusterConfig()
        self._initialized = False

    async def initialize(self) -> None:
        """Initialize clustering resources."""
        if self._initialized:
            return

        try:
            self._initialized = True
            self.publisher.publish(
                Event(
                    type=EventType.SERVICE_STARTED,
                    timestamp=datetime.now(),
                    component="cluster_manager",
                    description="Cluster manager initialized",
                    metadata={
                        "min_clusters": self.cluster_config.min_clusters,
                        "max_clusters": self.cluster_config.max_clusters,
                    },
                )
            )

        except Exception as e:
            logger.error(f"Failed to initialize cluster manager: {e}")
            raise

    async def cleanup(self) -> None:
        """Clean up clustering resources."""
        self._initialized = False

        self.publisher.publish(
            Event(
                type=EventType.SERVICE_STOPPED,
                timestamp=datetime.now(),
                component="cluster_manager",
                description="Cluster manager stopped",
            )
        )

    async def health_check(self) -> Dict[str, Any]:
        """Check cluster manager health.

        Returns:
            Health check results
        """
        return {
            "service": "ClusterManager",
            "initialized": self._initialized,
            "min_clusters": self.cluster_config.min_clusters,
            "max_clusters": self.cluster_config.max_clusters,
        }

    async def cluster_results(
        self,
        results: List[SearchResult],
        vectors: List[np.ndarray],
        min_clusters: Optional[int] = None,
        max_clusters: Optional[int] = None,
    ) -> List[ResultCluster]:
        """Cluster search results based on vectors.

        Args:
            results: Search results to cluster
            vectors: Result vectors for clustering
            min_clusters: Optional minimum clusters
            max_clusters: Optional maximum clusters

        Returns:
            List of result clusters
        """
        if not self._initialized:
            await self.initialize()

        if not results or not vectors:
            return []

        try:
            # Emit clustering started event
            self.publisher.publish(
                Event(
                    type=EventType.CLUSTERING_STARTED,
                    timestamp=datetime.now(),
                    component="cluster_manager",
                    description=f"Clustering {len(results)} results",
                    metadata={
                        "num_results": len(results),
                    },
                )
            )

            # Convert vectors to numpy array
            X = np.array(vectors)

            # Get cluster range
            min_k = min_clusters or self.cluster_config.min_clusters
            max_k = min(
                max_clusters or self.cluster_config.max_clusters,
                len(results) // self.cluster_config.min_cluster_size,
            )

            # Find optimal number of clusters
            best_score = -1
            best_k = min_k
            best_labels = None
            best_centroids = None

            for k in range(min_k, max_k + 1):
                # Fit KMeans
                kmeans = KMeans(
                    n_clusters=k,
                    max_iter=self.cluster_config.max_iterations,
                    random_state=self.cluster_config.random_state,
                )
                labels = kmeans.fit_predict(X)

                # Calculate silhouette score
                score = silhouette_score(X, labels)
                if score > best_score:
                    best_score = score
                    best_k = k
                    best_labels = labels
                    best_centroids = kmeans.cluster_centers_

            # Skip clustering if quality is too low
            if best_score < self.cluster_config.min_silhouette_score:
                self.publisher.publish(
                    Event(
                        type=EventType.CLUSTERING_COMPLETED,
                        timestamp=datetime.now(),
                        component="cluster_manager",
                        description="Clustering skipped (low quality)",
                        metadata={
                            "silhouette_score": best_score,
                        },
                    )
                )
                return []

            # Group results by cluster
            clusters = []
            for i in range(best_k):
                cluster_results = [
                    result for j, result in enumerate(results)
                    if best_labels[j] == i
                ]

                if len(cluster_results) >= self.cluster_config.min_cluster_size:
                    # Generate cluster description
                    description = self._generate_cluster_description(cluster_results)

                    # Create cluster
                    cluster = ResultCluster(
                        label=f"Cluster {i + 1}",
                        description=description,
                        results=cluster_results,
                        centroid=best_centroids[i],
                        score=best_score,
                    )
                    clusters.append(cluster)

            # Sort clusters by size
            clusters.sort(key=lambda x: len(x.results), reverse=True)

            # Emit success event
            self.publisher.publish(
                Event(
                    type=EventType.CLUSTERING_COMPLETED,
                    timestamp=datetime.now(),
                    component="cluster_manager",
                    description="Clustering completed",
                    metadata={
                        "num_clusters": len(clusters),
                        "silhouette_score": best_score,
                    },
                )
            )

            return clusters

        except Exception as e:
            logger.error(f"Clustering failed: {e}")
            self.publisher.publish(
                Event(
                    type=EventType.CLUSTERING_FAILED,
                    timestamp=datetime.now(),
                    component="cluster_manager",
                    description="Clustering failed",
                    error=str(e),
                )
            )
            return []

    def _generate_cluster_description(self, results: List[SearchResult]) -> str:
        """Generate description for result cluster.

        Args:
            results: Clustered search results

        Returns:
            Cluster description
        """
        # Get common HTTP methods
        methods = set(result.method for result in results)
        method_str = ", ".join(sorted(methods))

        # Get common path segments
        paths = [result.endpoint.strip("/").split("/") for result in results]
        common_segments = []
        if paths:
            min_len = min(len(p) for p in paths)
            for i in range(min_len):
                segment = paths[0][i]
                if all(p[i] == segment for p in paths):
                    common_segments.append(segment)

        # Build description
        if common_segments:
            path_str = "/".join(common_segments)
            return f"{method_str} endpoints under /{path_str}"
        else:
            return f"{method_str} endpoints"


# Create service instance
cluster_manager = ClusterManager(
    Config(),
    PublisherService(Config(), MetricsManager()),
    MetricsManager(),
)

__all__ = ["ClusterConfig", "ResultCluster", "ClusterManager", "cluster_manager"] 