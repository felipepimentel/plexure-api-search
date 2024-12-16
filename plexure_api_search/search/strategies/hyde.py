"""HyDE (Hypothetical Document Embeddings) search strategy."""

import logging
from typing import Any, Dict, List, Optional

from ...integrations.llm.openrouter_client import OpenRouterClient
from .base import BaseSearchStrategy, SearchConfig, SearchResult, StrategyFactory

logger = logging.getLogger(__name__)


@StrategyFactory.register("hyde")
class HyDESearchStrategy(BaseSearchStrategy):
    """HyDE search strategy using hypothetical documents."""

    def __init__(self, config: Optional[SearchConfig] = None):
        """Initialize HyDE strategy.

        Args:
            config: Optional strategy configuration
        """
        super().__init__(config)
        self.llm = OpenRouterClient()

    def _generate_hypothetical_doc(self, query: str) -> str:
        """Generate hypothetical API document for query.

        Args:
            query: Search query

        Returns:
            Generated hypothetical document
        """
        prompt = f"""Given this API search query: "{query}"

Generate a hypothetical ideal API endpoint that would perfectly match this query.
Include these components:
- HTTP method
- Path
- Description
- Parameters
- Response format
- Use cases

Format as a concise technical description."""

        try:
            response = self.llm.call(
                prompt=prompt,
                temperature=0.3,
                cache_key=f"hyde_{query}",
            )

            if "error" in response:
                logger.error(f"Failed to generate hypothetical doc: {response['error']}")
                return query

            return response["choices"][0]["message"]["content"]

        except Exception as e:
            logger.error(f"Error generating hypothetical doc: {e}")
            return query

    def search(
        self,
        query: str,
        top_k: int = 10,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[SearchResult]:
        """Execute HyDE search strategy.

        Args:
            query: Search query
            top_k: Number of results to return
            filters: Optional filters to apply

        Returns:
            Search results
        """
        try:
            # Generate hypothetical document
            hyp_doc = self._generate_hypothetical_doc(query)
            
            # Get embeddings for both query and hypothetical doc
            query_embedding = self.embeddings.get_embeddings(query)
            doc_embedding = self.embeddings.get_embeddings(hyp_doc)
            
            # Combine embeddings with weighted average
            combined = (query_embedding * 0.3 + doc_embedding * 0.7)
            
            # Search using combined embedding
            results = []  # TODO: Implement vector search with combined embedding
            
            # Convert to SearchResult objects
            search_results = []
            for result in results:
                search_results.append(
                    SearchResult(
                        id=result["id"],
                        score=float(result["score"]),
                        method=result["method"],
                        path=result["path"],
                        description=result["description"],
                        api_name=result["api_name"],
                        api_version=result["api_version"],
                        metadata=result.get("metadata", {}),
                        strategy="hyde",
                    )
                )

            return search_results

        except Exception as e:
            logger.error(f"HyDE search failed: {e}")
            return [] 