"""Advanced query expansion and enrichment system."""

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
from nltk.corpus import wordnet
from nltk.tokenize import word_tokenize
from rank_bm25 import BM25Okapi

from ..config import config_instance
from ..embedding.embeddings import embeddings
from ..integrations.llm.openrouter_client import OpenRouterClient
from ..utils.cache import DiskCache

logger = logging.getLogger(__name__)

# Cache for expanded queries
expansion_cache = DiskCache[Dict[str, Any]](
    namespace="query_expansion",
    ttl=config_instance.cache_ttl,
)


@dataclass
class ExpandedQuery:
    """Expanded query with metadata."""

    original_query: str
    expanded_query: str
    expansion_type: str
    confidence: float
    terms_added: List[str]
    terms_removed: List[str]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format."""
        return {
            "original_query": self.original_query,
            "expanded_query": self.expanded_query,
            "expansion_type": self.expansion_type,
            "confidence": self.confidence,
            "terms_added": self.terms_added,
            "terms_removed": self.terms_removed,
        }


class QueryExpander:
    """Advanced query expansion and enrichment."""

    def __init__(self, use_cache: bool = True):
        """Initialize query expander.

        Args:
            use_cache: Whether to use caching
        """
        self.use_cache = use_cache
        self.llm = OpenRouterClient(use_cache=use_cache)
        self.max_expansions = config_instance.max_query_expansions
        self.api_corpus: List[str] = []
        self.bm25: Optional[BM25Okapi] = None

    def _get_wordnet_synonyms(self, term: str) -> Set[str]:
        """Get WordNet synonyms for term.

        Args:
            term: Input term

        Returns:
            Set of synonyms
        """
        synonyms = set()
        for syn in wordnet.synsets(term):
            for lemma in syn.lemmas():
                synonym = lemma.name().lower()
                if synonym != term and "_" not in synonym:
                    synonyms.add(synonym)
        return synonyms

    def _get_technical_terms(self, query: str) -> List[str]:
        """Extract technical terms from query.

        Args:
            query: Input query

        Returns:
            List of technical terms
        """
        # Common API-related terms
        api_terms = {
            "get", "post", "put", "delete", "patch",
            "api", "endpoint", "route", "path",
            "auth", "token", "jwt", "oauth",
            "json", "xml", "http", "https",
            "request", "response", "header", "body",
            "parameter", "query", "path", "form",
            "status", "code", "error", "success",
        }

        # Extract terms that match API terminology
        tokens = word_tokenize(query.lower())
        return [term for term in tokens if term in api_terms]

    def _expand_with_llm(self, query: str) -> List[str]:
        """Expand query using LLM.

        Args:
            query: Input query

        Returns:
            List of expanded queries
        """
        prompt = f"""Given this API search query: "{query}"

Suggest 3 alternative ways to express the same search intent, focusing on API terminology.
Return a JSON array of strings.

Example:
If the query is "user authentication", you might return:
[
    "login and token generation endpoints",
    "oauth and jwt authentication apis",
    "user authorization and identity verification"
]"""

        try:
            response = self.llm.call(
                prompt=prompt,
                temperature=0.3,
                cache_key=f"expand_{query}",
            )

            if "error" in response:
                logger.error(f"LLM expansion failed: {response['error']}")
                return []

            content = response["choices"][0]["message"]["content"]
            expansions = json.loads(content)

            if not isinstance(expansions, list):
                logger.error("Invalid expansions format - not a list")
                return []

            return [str(exp) for exp in expansions]

        except Exception as e:
            logger.error(f"Failed to expand query with LLM: {e}")
            return []

    def _get_similar_queries(self, query: str) -> List[Tuple[str, float]]:
        """Find similar historical queries.

        Args:
            query: Input query

        Returns:
            List of (query, similarity) tuples
        """
        try:
            # Get query embedding
            query_embedding = embeddings.get_embeddings(query)

            # Load historical queries
            historical_queries = []
            for file in expansion_cache.cache_dir.glob("*.json"):
                try:
                    data = json.loads(file.read_text())
                    if "original_query" in data:
                        historical_queries.append(data["original_query"])
                except Exception:
                    continue

            if not historical_queries:
                return []

            # Get embeddings for historical queries
            historical_embeddings = embeddings.get_embeddings(historical_queries)

            # Calculate similarities
            similarities = []
            for hist_query, hist_emb in zip(historical_queries, historical_embeddings):
                similarity = np.dot(query_embedding, hist_emb) / (
                    np.linalg.norm(query_embedding) * np.linalg.norm(hist_emb)
                )
                similarities.append((hist_query, float(similarity)))

            # Sort by similarity
            return sorted(similarities, key=lambda x: x[1], reverse=True)

        except Exception as e:
            logger.error(f"Failed to get similar queries: {e}")
            return []

    def _filter_expansions(
        self,
        original_query: str,
        expansions: List[str],
    ) -> List[ExpandedQuery]:
        """Filter and validate expanded queries.

        Args:
            original_query: Original query
            expansions: List of expanded queries

        Returns:
            List of validated expanded queries
        """
        filtered = []
        original_tokens = set(word_tokenize(original_query.lower()))

        for expansion in expansions:
            try:
                # Tokenize expansion
                expansion_tokens = set(word_tokenize(expansion.lower()))

                # Calculate changes
                added = expansion_tokens - original_tokens
                removed = original_tokens - expansion_tokens

                # Calculate confidence based on token overlap
                overlap = len(original_tokens & expansion_tokens)
                total = len(original_tokens | expansion_tokens)
                confidence = overlap / total if total > 0 else 0.0

                # Create expanded query object
                expanded = ExpandedQuery(
                    original_query=original_query,
                    expanded_query=expansion,
                    expansion_type="semantic",
                    confidence=confidence,
                    terms_added=list(added),
                    terms_removed=list(removed),
                )

                filtered.append(expanded)

            except Exception as e:
                logger.error(f"Failed to filter expansion: {e}")
                continue

        return filtered

    def expand_query(self, query: str) -> List[ExpandedQuery]:
        """Expand search query using multiple strategies.

        Args:
            query: Search query

        Returns:
            List of expanded queries
        """
        try:
            # Check cache
            if self.use_cache:
                cache_key = f"expand:{query}"
                cached = expansion_cache.get(cache_key)
                if cached is not None:
                    return [ExpandedQuery(**exp) for exp in cached]

            expansions = []

            # 1. LLM-based expansion
            llm_expansions = self._expand_with_llm(query)
            if llm_expansions:
                expansions.extend(llm_expansions)

            # 2. Technical term expansion
            tech_terms = self._get_technical_terms(query)
            for term in tech_terms:
                synonyms = self._get_wordnet_synonyms(term)
                for synonym in synonyms:
                    expanded = query.replace(term, synonym)
                    expansions.append(expanded)

            # 3. Similar historical queries
            similar_queries = self._get_similar_queries(query)
            for similar_query, _ in similar_queries[:self.max_expansions]:
                expansions.append(similar_query)

            # Filter and validate expansions
            filtered = self._filter_expansions(query, expansions)

            # Sort by confidence and limit
            filtered.sort(key=lambda x: x.confidence, reverse=True)
            filtered = filtered[:self.max_expansions]

            # Cache results
            if self.use_cache:
                expansion_cache.set(
                    cache_key,
                    [exp.to_dict() for exp in filtered],
                )

            return filtered

        except Exception as e:
            logger.error(f"Query expansion failed: {e}")
            return []


# Global query expander instance
query_expander = QueryExpander()

__all__ = ["query_expander", "ExpandedQuery"]
