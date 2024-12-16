"""Hybrid search combining BM25 and vector similarity."""

import logging
import math
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from rank_bm25 import BM25Okapi

from ..config import config_instance
from ..embedding.embeddings import embeddings
from ..utils.cache import DiskCache
from .semantic import semantic_search

logger = logging.getLogger(__name__)

# Cache for hybrid search results
hybrid_cache = DiskCache[Dict[str, Any]](
    namespace="hybrid_search",
    ttl=config_instance.cache_ttl,
)


class HybridSearch:
    """Hybrid search combining BM25 and vector similarity."""

    def __init__(self, use_cache: bool = True):
        """Initialize hybrid search.

        Args:
            use_cache: Whether to use caching
        """
        self.use_cache = use_cache
        self.bm25_weight = config_instance.bm25_weight
        self.vector_weight = config_instance.vector_weight
        self.bm25_index = None
        self.corpus = []
        self.doc_mapping = {}

    def _tokenize(self, text: str) -> List[str]:
        """Tokenize text for BM25.

        Args:
            text: Input text

        Returns:
            List of tokens
        """
        # Simple tokenization - can be enhanced with better tokenizers
        return text.lower().split()

    def _prepare_document(self, endpoint: Dict[str, Any]) -> str:
        """Prepare endpoint document for indexing.

        Args:
            endpoint: Endpoint data

        Returns:
            Formatted document text
        """
        parts = [
            endpoint.get("method", ""),
            endpoint.get("path", ""),
            endpoint.get("description", ""),
            endpoint.get("summary", ""),
        ]

        # Add parameters
        params = endpoint.get("parameters", [])
        if params:
            for param in params:
                if isinstance(param, dict):
                    parts.extend([
                        param.get("name", ""),
                        param.get("description", ""),
                        str(param.get("type", "")),
                    ])

        # Add responses
        responses = endpoint.get("responses", [])
        if responses:
            for response in responses:
                if isinstance(response, dict):
                    parts.extend([
                        str(response.get("status", "")),
                        response.get("description", ""),
                    ])

        # Add tags
        tags = endpoint.get("tags", [])
        if tags:
            parts.extend(tags)

        return " ".join(str(part) for part in parts if part)

    def index_endpoints(self, endpoints: List[Dict[str, Any]]) -> None:
        """Index endpoints for hybrid search.

        Args:
            endpoints: List of endpoint data
        """
        try:
            # Prepare corpus and mapping
            self.corpus = []
            self.doc_mapping = {}

            for i, endpoint in enumerate(endpoints):
                # Create document text
                doc = self._prepare_document(endpoint)
                
                # Tokenize
                tokenized_doc = self._tokenize(doc)
                
                # Add to corpus and mapping
                self.corpus.append(tokenized_doc)
                self.doc_mapping[i] = endpoint

            # Create BM25 index
            self.bm25_index = BM25Okapi(self.corpus)
            
            logger.info(f"Indexed {len(endpoints)} endpoints for hybrid search")

        except Exception as e:
            logger.error(f"Failed to index endpoints: {e}")
            raise

    def _get_bm25_scores(self, query: str, top_k: int = 10) -> List[Tuple[Dict[str, Any], float]]:
        """Get BM25 scores for query.

        Args:
            query: Search query
            top_k: Number of results to return

        Returns:
            List of (endpoint, score) tuples
        """
        try:
            # Tokenize query
            tokenized_query = self._tokenize(query)
            
            # Get BM25 scores
            scores = self.bm25_index.get_scores(tokenized_query)
            
            # Get top k results
            top_indices = np.argsort(scores)[-top_k:][::-1]
            
            results = []
            for idx in top_indices:
                endpoint = self.doc_mapping[idx]
                score = float(scores[idx])
                results.append((endpoint, score))
            
            return results

        except Exception as e:
            logger.error(f"BM25 scoring failed: {e}")
            return []

    def _normalize_scores(self, scores: List[float]) -> List[float]:
        """Normalize scores to 0-1 range.

        Args:
            scores: List of scores

        Returns:
            Normalized scores
        """
        if not scores:
            return scores
        min_score = min(scores)
        max_score = max(scores)
        if max_score == min_score:
            return [1.0] * len(scores)
        return [(s - min_score) / (max_score - min_score) for s in scores]

    def search(
        self,
        query: str,
        top_k: int = 10,
        filters: Optional[Dict] = None,
    ) -> List[Dict[str, Any]]:
        """Perform hybrid search combining BM25 and vector similarity.

        Args:
            query: Search query
            top_k: Number of results to return
            filters: Optional filters to apply

        Returns:
            List of search results
        """
        try:
            # Check cache
            if self.use_cache:
                cache_key = f"hybrid:{query}:{top_k}:{filters}"
                cached = hybrid_cache.get(cache_key)
                if cached is not None:
                    return cached

            # Get BM25 results
            bm25_results = self._get_bm25_scores(query, top_k=top_k)
            
            # Get vector search results
            vector_results = semantic_search.search(
                query=query,
                top_k=top_k,
                filters=filters,
            )

            # Combine results
            combined_results = {}
            
            # Process BM25 results
            bm25_scores = []
            for endpoint, score in bm25_results:
                endpoint_id = endpoint.get("id")
                if endpoint_id:
                    combined_results[endpoint_id] = {
                        "endpoint": endpoint,
                        "bm25_score": score,
                        "vector_score": 0.0,
                    }
                    bm25_scores.append(score)

            # Normalize BM25 scores
            normalized_bm25 = self._normalize_scores(bm25_scores)
            for endpoint_id, scores in zip(combined_results.keys(), normalized_bm25):
                combined_results[endpoint_id]["bm25_score"] = scores

            # Process vector results
            vector_scores = []
            for result in vector_results:
                endpoint_id = result.get("id")
                if endpoint_id:
                    if endpoint_id not in combined_results:
                        combined_results[endpoint_id] = {
                            "endpoint": result,
                            "bm25_score": 0.0,
                            "vector_score": result.get("score", 0.0),
                        }
                    else:
                        combined_results[endpoint_id]["vector_score"] = result.get("score", 0.0)
                    vector_scores.append(result.get("score", 0.0))

            # Normalize vector scores
            normalized_vector = self._normalize_scores(vector_scores)
            vector_idx = 0
            for result in combined_results.values():
                if result["vector_score"] > 0:
                    result["vector_score"] = normalized_vector[vector_idx]
                    vector_idx += 1

            # Calculate final scores
            final_results = []
            for result in combined_results.values():
                final_score = (
                    self.bm25_weight * result["bm25_score"]
                    + self.vector_weight * result["vector_score"]
                )
                
                endpoint = result["endpoint"]
                endpoint["score"] = final_score
                endpoint["bm25_score"] = result["bm25_score"]
                endpoint["vector_score"] = result["vector_score"]
                
                final_results.append(endpoint)

            # Sort by final score
            final_results.sort(key=lambda x: x["score"], reverse=True)
            
            # Limit to top_k
            final_results = final_results[:top_k]

            # Cache results
            if self.use_cache:
                hybrid_cache.set(cache_key, final_results)

            return final_results

        except Exception as e:
            logger.error(f"Hybrid search failed: {e}")
            return []


# Global hybrid search instance
hybrid_search = HybridSearch()

__all__ = ["hybrid_search", "HybridSearch"] 