"""Retrieval Augmented Generation (RAG) for enhanced search results."""

import logging
from typing import Any, Dict, List, Optional

import numpy as np

from ..config import config_instance
from ..embedding.embeddings import embeddings
from ..integrations.llm.openrouter_client import OpenRouterClient
from ..utils.cache import DiskCache

logger = logging.getLogger(__name__)

# Cache for RAG results
rag_cache = DiskCache[Dict[str, Any]](
    namespace="rag",
    ttl=config_instance.cache_ttl,
)


class RAGEnhancer:
    """RAG-based search result enhancement."""

    def __init__(self, use_cache: bool = True):
        """Initialize RAG enhancer.

        Args:
            use_cache: Whether to use caching
        """
        self.use_cache = use_cache
        self.llm = OpenRouterClient(use_cache=use_cache)
        self.chunk_size = config_instance.rag_chunk_size
        self.chunk_overlap = config_instance.rag_chunk_overlap
        self.top_k = config_instance.rag_top_k

    def _chunk_text(self, text: str) -> List[str]:
        """Split text into overlapping chunks.

        Args:
            text: Text to split

        Returns:
            List of text chunks
        """
        # Split into sentences (simple approach)
        sentences = text.split(". ")
        chunks = []
        current_chunk = []
        current_size = 0

        for sentence in sentences:
            sentence_size = len(sentence.split())
            if current_size + sentence_size > self.chunk_size:
                # Add current chunk if not empty
                if current_chunk:
                    chunks.append(". ".join(current_chunk) + ".")
                # Start new chunk with overlap
                overlap_start = max(0, len(current_chunk) - self.chunk_overlap)
                current_chunk = current_chunk[overlap_start:] + [sentence]
                current_size = sum(len(s.split()) for s in current_chunk)
            else:
                current_chunk.append(sentence)
                current_size += sentence_size

        # Add final chunk
        if current_chunk:
            chunks.append(". ".join(current_chunk) + ".")

        return chunks

    def _prepare_context(self, results: List[Dict[str, Any]]) -> str:
        """Prepare context from search results.

        Args:
            results: List of search results

        Returns:
            Formatted context string
        """
        context_parts = []
        for result in results[:self.top_k]:
            # Extract relevant information
            method = result.get("method", "")
            path = result.get("path", "")
            description = result.get("description", "")
            parameters = result.get("parameters", [])
            responses = result.get("responses", [])

            # Format endpoint information
            endpoint_info = f"""
Endpoint: {method} {path}
Description: {description}
Parameters: {', '.join(str(p) for p in parameters)}
Responses: {', '.join(str(r) for r in responses)}
"""
            context_parts.append(endpoint_info)

        return "\n\n".join(context_parts)

    def _create_rag_prompt(self, query: str, context: str) -> str:
        """Create RAG prompt with query and context.

        Args:
            query: User query
            context: Retrieved context

        Returns:
            Formatted prompt
        """
        return f"""Given this API search query: "{query}"

And these relevant API endpoints as context:

{context}

Provide a detailed analysis in JSON format with these fields:
- relevance: How relevant the results are to the query (0-1)
- coverage: How well the results cover the query intent (0-1)
- explanation: Detailed explanation of how the endpoints relate to the query
- suggestions: List of suggestions for better search results
- highlights: Key points about the endpoints
- missing_aspects: Any aspects of the query not covered by the results

Example response:
{{
    "relevance": 0.85,
    "coverage": 0.9,
    "explanation": "The endpoints provide direct functionality for the query...",
    "suggestions": ["Consider filtering by HTTP method", "Try more specific terms"],
    "highlights": ["Includes both read and write operations", "Covers all CRUD functions"],
    "missing_aspects": ["No authentication endpoints found", "Missing batch operations"]
}}"""

    def enhance_results(self, query: str, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Enhance search results using RAG.

        Args:
            query: Search query
            results: List of search results

        Returns:
            Enhanced results with RAG analysis
        """
        try:
            # Check cache
            if self.use_cache:
                cache_key = f"rag:{query}"
                cached = rag_cache.get(cache_key)
                if cached is not None:
                    return cached

            # Prepare context from results
            context = self._prepare_context(results)

            # Create chunks for long context
            chunks = self._chunk_text(context)

            # Get embeddings for query and chunks
            query_embedding = embeddings.get_embeddings(query)
            chunk_embeddings = embeddings.get_embeddings(chunks)

            # Find most relevant chunks
            similarities = []
            for chunk_emb in chunk_embeddings:
                similarity = np.dot(query_embedding, chunk_emb) / (
                    np.linalg.norm(query_embedding) * np.linalg.norm(chunk_emb)
                )
                similarities.append(similarity)

            # Select top chunks
            top_indices = np.argsort(similarities)[-self.top_k:]
            selected_chunks = [chunks[i] for i in top_indices]
            selected_context = "\n\n".join(selected_chunks)

            # Generate RAG analysis
            prompt = self._create_rag_prompt(query, selected_context)
            response = self.llm.call(
                prompt=prompt,
                temperature=0.3,
                cache_key=f"rag_llm:{query}",
            )

            # Extract and validate content
            if "error" in response:
                logger.error(f"LLM error in RAG: {response['error']}")
                return {}

            content = response["choices"][0]["message"]["content"]
            enhanced = {
                "rag_analysis": content,
                "relevant_chunks": selected_chunks,
                "chunk_similarities": [float(similarities[i]) for i in top_indices],
            }

            # Cache result
            if self.use_cache:
                rag_cache.set(cache_key, enhanced)

            return enhanced

        except Exception as e:
            logger.error(f"RAG enhancement failed: {e}")
            return {}


# Global RAG enhancer instance
rag_enhancer = RAGEnhancer()

__all__ = ["rag_enhancer", "RAGEnhancer"] 