"""Triple vector embedding module."""

import logging
import re
from typing import Dict, List, Optional, Tuple, Any, NamedTuple
from collections import Counter

import numpy as np
from sentence_transformers import util

from ..config import config_instance
from ..monitoring.metrics import MetricsManager
from ..services.models import ModelService

logger = logging.getLogger(__name__)

class Triple(NamedTuple):
    """API endpoint triple."""
    endpoint: str
    method: str
    description: str

class TripleVectorizer:
    """Triple vector embeddings for API endpoints."""

    def __init__(self):
        """Initialize vectorizer."""
        self.model_service = ModelService()
        self.metrics = MetricsManager()
        self.initialized = False
        self.endpoint_texts = []
        self.embedding_dim = 768  # Default dimension

    def initialize(self) -> None:
        """Initialize vectorizer."""
        if self.initialized:
            return

        try:
            # Initialize model service
            self.model_service.initialize()
            
            # Get embedding dimension from model
            test_text = "test"
            test_embedding = self.model_service.encode(test_text)
            self.embedding_dim = test_embedding.shape[0]
            
            self.initialized = True
            logger.info("Vectorizer initialized")

        except Exception as e:
            logger.error(f"Failed to initialize vectorizer: {e}")
            raise

    def cleanup(self) -> None:
        """Clean up vectorizer."""
        self.model_service.cleanup()
        self.initialized = False
        self.endpoint_texts = []
        logger.info("Vectorizer cleaned up")

    def encode_endpoint(
        self,
        endpoint: Dict[str, Any],
    ) -> Dict[str, np.ndarray]:
        """Encode API endpoint.
        
        Args:
            endpoint: API endpoint data
            
        Returns:
            Dictionary of vector types and embeddings
        """
        if not self.initialized:
            self.initialize()

        try:
            # Start timer
            start_time = self.metrics.start_timer()

            # Create text representation
            text = self._get_endpoint_text(endpoint)
            if not text.strip():
                raise ValueError("Empty endpoint text")

            # Store text for vocabulary building
            self.endpoint_texts.append(text)

            # Get semantic embedding
            semantic = self.model_service.encode(text)
            self.embedding_dim = semantic.shape[0]  # Update dimension based on model output

            # Create triple embeddings with consistent dimensions
            embeddings = {
                "semantic": semantic,
                "lexical": np.zeros(self.embedding_dim),
                "contextual": np.zeros(self.embedding_dim)
            }

            # Try to get lexical embedding
            try:
                lexical = self._get_lexical_embedding(text)
                if lexical is not None and lexical.size > 0:
                    # Ensure lexical embedding has correct dimension
                    if lexical.size != self.embedding_dim:
                        lexical = np.resize(lexical, (self.embedding_dim,))
                    embeddings["lexical"] = lexical
            except Exception as e:
                logger.warning(f"Failed to get lexical embedding: {e}")

            # Try to get contextual embedding
            try:
                contextual = self._get_contextual_embedding(text)
                if contextual is not None and contextual.size > 0:
                    # Ensure contextual embedding has correct dimension
                    if contextual.size != self.embedding_dim:
                        contextual = np.resize(contextual, (self.embedding_dim,))
                    embeddings["contextual"] = contextual
            except Exception as e:
                logger.warning(f"Failed to get contextual embedding: {e}")

            # Verify all embeddings have correct dimensions
            for key, embedding in embeddings.items():
                if embedding.shape[0] != self.embedding_dim:
                    raise ValueError(f"Incorrect dimension for {key} embedding: {embedding.shape[0]} != {self.embedding_dim}")

            # Stop timer
            self.metrics.stop_timer(
                start_time,
                "index",
                {
                    "operation": "encode",
                    "status": "success",
                },
            )

            return embeddings

        except Exception as e:
            logger.error(f"Failed to encode endpoint: {e}")
            self.metrics.increment(
                "index_operations",
                {
                    "operation": "encode",
                    "status": "error",
                    "error_type": type(e).__name__,
                },
            )
            raise

    def _get_endpoint_text(self, endpoint: Dict[str, Any]) -> str:
        """Get text representation of endpoint.
        
        Args:
            endpoint: API endpoint data
            
        Returns:
            Text representation
        """
        parts = []

        # Add method and path
        method = endpoint.get("method", "").upper()
        path = endpoint.get("path", "")
        if method and path:
            parts.append(f"{method} {path}")

        # Add summary and description
        summary = endpoint.get("summary", "")
        if summary:
            parts.append(summary)
            
        description = endpoint.get("description", "")
        if description and description != summary:
            parts.append(description)

        # Add operation ID if available
        operation_id = endpoint.get("operationId", "")
        if operation_id:
            parts.append(f"Operation: {operation_id}")

        # Add tags
        tags = endpoint.get("tags", [])
        if tags:
            parts.append(f"Tags: {', '.join(tags)}")

        # Add parameters
        parameters = endpoint.get("parameters", [])
        if parameters:
            param_parts = []
            for param in parameters:
                if isinstance(param, dict):
                    name = param.get("name", "")
                    schema = param.get("schema", {})
                    param_type = schema.get("type", "") if isinstance(schema, dict) else ""
                    param_in = param.get("in", "")
                    required = "required" if param.get("required") else "optional"
                    description = param.get("description", "")
                    
                    param_text = f"{name}"
                    if param_type:
                        param_text += f" ({param_type})"
                    if param_in:
                        param_text += f" in {param_in}"
                    param_text += f" {required}"
                    if description:
                        param_text += f": {description}"
                        
                    param_parts.append(param_text)
            
            if param_parts:
                parts.append("Parameters:")
                parts.extend(param_parts)

        # Add responses
        responses = endpoint.get("responses", {})
        if responses:
            response_parts = []
            for status, response in responses.items():
                if isinstance(response, dict):
                    description = response.get("description", "")
                    if description:
                        response_parts.append(f"{status}: {description}")
            
            if response_parts:
                parts.append("Responses:")
                parts.extend(response_parts)

        # Add security requirements
        security = endpoint.get("security", [])
        if security:
            security_parts = []
            for requirement in security:
                if isinstance(requirement, dict):
                    for scheme, scopes in requirement.items():
                        security_text = f"Security: {scheme}"
                        if scopes:
                            security_text += f" with scopes: {', '.join(scopes)}"
                        security_parts.append(security_text)
            if security_parts:
                parts.extend(security_parts)

        return " ".join(parts)

    def _get_lexical_embedding(self, text: str) -> np.ndarray:
        """Get lexical embedding using word frequencies.
        
        Args:
            text: Input text
            
        Returns:
            Lexical embedding
        """
        try:
            # Preprocess text
            text = text.lower()
            text = re.sub(r'[^\w\s]', ' ', text)
            words = text.split()
            
            # Calculate word frequencies
            word_freq = Counter(words)
            total_words = len(words)
            
            # Create frequency vector
            vector = np.zeros(self.embedding_dim)
            for i, (word, freq) in enumerate(word_freq.most_common(self.embedding_dim)):
                vector[i] = freq / total_words
                
            return vector
            
        except Exception as e:
            logger.error(f"Failed to get lexical embedding: {e}")
            self.metrics.increment(
                "index_operations",
                {
                    "operation": "lexical",
                    "status": "error",
                    "error_type": type(e).__name__,
                },
            )
            return np.zeros(self.embedding_dim)

    def _get_contextual_embedding(self, text: str) -> np.ndarray:
        """Get contextual embedding using sliding window approach.
        
        Args:
            text: Input text
            
        Returns:
            Contextual embedding
        """
        try:
            # Split into sentences
            sentences = text.split('.')
            if len(sentences) < 2:
                return self.model_service.encode(text)
                
            # Use sliding window to get local context
            window_size = 2
            contextual_vectors = []
            
            for i in range(len(sentences)):
                # Get window of sentences
                start = max(0, i - window_size)
                end = min(len(sentences), i + window_size + 1)
                context = ' '.join(sentences[start:end])
                
                # Get embedding for context
                if context.strip():
                    vector = self.model_service.encode(context)
                    contextual_vectors.append(vector)
            
            # Average contextual vectors
            if contextual_vectors:
                return np.mean(contextual_vectors, axis=0)
            return np.zeros(self.embedding_dim)
            
        except Exception as e:
            logger.error(f"Failed to get contextual embedding: {e}")
            self.metrics.increment(
                "index_operations",
                {
                    "operation": "contextual",
                    "status": "error",
                    "error_type": type(e).__name__,
                },
            )
            return np.zeros(self.embedding_dim)

    def encode_query(self, query: str) -> Dict[str, np.ndarray]:
        """Encode search query.
        
        Args:
            query: Search query
            
        Returns:
            Dictionary of vector types and embeddings
        """
        if not self.initialized:
            self.initialize()

        try:
            # Start timer
            start_time = self.metrics.start_timer()

            # Get semantic embedding
            semantic = self.model_service.encode(query)
            self.embedding_dim = semantic.shape[0]  # Update dimension based on model output

            # Create embeddings with consistent dimensions
            embeddings = {
                "semantic": semantic,
                "lexical": np.zeros(self.embedding_dim),
                "contextual": np.zeros(self.embedding_dim)
            }

            # Try to get lexical embedding
            try:
                lexical = self._get_lexical_embedding(query)
                if lexical is not None and lexical.size > 0:
                    if lexical.size != self.embedding_dim:
                        lexical = np.resize(lexical, (self.embedding_dim,))
                    embeddings["lexical"] = lexical
            except Exception as e:
                logger.warning(f"Failed to get lexical embedding: {e}")

            # Try to get contextual embedding
            try:
                contextual = self._get_contextual_embedding(query)
                if contextual is not None and contextual.size > 0:
                    if contextual.size != self.embedding_dim:
                        contextual = np.resize(contextual, (self.embedding_dim,))
                    embeddings["contextual"] = contextual
            except Exception as e:
                logger.warning(f"Failed to get contextual embedding: {e}")

            # Verify all embeddings have correct dimensions
            for key, embedding in embeddings.items():
                if embedding.shape[0] != self.embedding_dim:
                    raise ValueError(f"Incorrect dimension for {key} embedding: {embedding.shape[0]} != {self.embedding_dim}")

            # Stop timer
            self.metrics.stop_timer(
                start_time,
                "search",
                {
                    "operation": "encode",
                    "status": "success",
                },
            )

            return embeddings

        except Exception as e:
            logger.error(f"Failed to encode query: {e}")
            self.metrics.increment(
                "search_requests",
                {
                    "operation": "encode",
                    "status": "error",
                    "error_type": type(e).__name__,
                },
            )
            raise

# Global instance
vectorizer = TripleVectorizer() 