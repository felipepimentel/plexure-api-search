"""Advanced embedding generation with modern models and caching."""

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union, Tuple

import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.decomposition import PCA
from transformers import AutoTokenizer, AutoModel
import torch
import torch.nn.functional as F

from ..utils.cache import DiskCache
from ..config import config_instance

logger = logging.getLogger(__name__)

@dataclass
class TripleVector:
    """Vector representation of an API triple (endpoint, description, example)."""
    
    endpoint: np.ndarray
    description: np.ndarray
    example: Optional[np.ndarray] = None
    metadata: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format."""
        return {
            "endpoint": self.endpoint.tolist(),
            "description": self.description.tolist(),
            "example": self.example.tolist() if self.example is not None else None,
            "metadata": self.metadata or {},
        }

    def to_combined_vector(self) -> np.ndarray:
        """Combine endpoint and description vectors into a single vector.
        
        Returns:
            Combined vector representation with dimension matching config
        """
        # Normalize vectors before combining
        endpoint_norm = self.endpoint / np.linalg.norm(self.endpoint)
        description_norm = self.description / np.linalg.norm(self.description)
        
        # Average the vectors instead of concatenating
        combined = (endpoint_norm + description_norm) / 2
        
        # Add example vector if available
        if self.example is not None:
            example_norm = self.example / np.linalg.norm(self.example)
            combined = (combined + example_norm) / 2
            
        # Final normalization
        combined = combined / np.linalg.norm(combined)
        return combined

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TripleVector":
        """Create from dictionary format."""
        return cls(
            endpoint=np.array(data["endpoint"]),
            description=np.array(data["description"]),
            example=np.array(data["example"]) if data.get("example") is not None else None,
            metadata=data.get("metadata", {}),
        )

class TripleVectorizer:
    """Vectorize API triples using modern embedding models."""

    def __init__(self):
        """Initialize the vectorizer with embedding models."""
        self.embeddings = ModernEmbeddings()

    def vectorize(
        self, endpoint: str, description: str, example: Optional[str] = None
    ) -> TripleVector:
        """Vectorize an API triple.

        Args:
            endpoint: API endpoint path
            description: API description
            example: Optional example usage

        Returns:
            TripleVector containing embeddings
        """
        # Generate embeddings for endpoint and description
        endpoint_vector = self.embeddings.get_embeddings(endpoint)
        description_vector = self.embeddings.get_embeddings(description)
        
        # Generate example embedding if provided
        example_vector = None
        if example:
            example_vector = self.embeddings.get_embeddings(example)

        return TripleVector(
            endpoint=endpoint_vector,
            description=description_vector,
            example=example_vector,
            metadata={
                "endpoint": endpoint,
                "description": description,
                "example": example,
            },
        )

    def batch_vectorize(
        self, triples: List[Tuple[str, str, Optional[str]]]
    ) -> List[TripleVector]:
        """Vectorize multiple API triples in batch.

        Args:
            triples: List of (endpoint, description, example) tuples

        Returns:
            List of TripleVectors
        """
        results = []
        for endpoint, description, example in triples:
            vector = self.vectorize(endpoint, description, example)
            results.append(vector)
        return results

# Pre-configured cache instances
embedding_cache = DiskCache[np.ndarray](
    namespace="embeddings",
    ttl=config_instance.embedding_cache_ttl,
)


@dataclass
class EmbeddingResult:
    """Result of embedding generation with metadata."""

    vector: np.ndarray
    text: str
    model_name: str
    dimension: int
    strategy: str = "bi_encoder"
    confidence: float = 1.0
    metadata: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format for serialization."""
        return {
            "vector": self.vector.tolist(),
            "text": self.text,
            "model_name": self.model_name,
            "dimension": self.dimension,
            "strategy": self.strategy,
            "confidence": self.confidence,
            "metadata": self.metadata or {},
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "EmbeddingResult":
        """Create from dictionary format."""
        data["vector"] = np.array(data["vector"])
        return cls(**data)


class ModernEmbeddings:
    """Modern embedding generation with advanced features."""

    def __init__(self, use_cache: bool = True):
        """Initialize embedding models.

        Args:
            use_cache: Whether to use caching
        """
        self.use_cache = use_cache
        
        # Initialize main bi-encoder
        self.bi_encoder = SentenceTransformer(config_instance.bi_encoder_model)
        
        # Initialize fallback model
        self.fallback_encoder = SentenceTransformer(config_instance.bi_encoder_fallback)
        
        # Initialize cross-encoder for reranking
        self.cross_encoder = SentenceTransformer(config_instance.cross_encoder_model)
        
        # Initialize multilingual model if needed
        self.multilingual_encoder = None
        if hasattr(config_instance, "multilingual_model"):
            self.multilingual_encoder = SentenceTransformer(config_instance.multilingual_model)
        
        # Initialize PCA if enabled
        self.pca = None
        if config_instance.pca_enabled:
            self.pca = PCA(n_components=config_instance.pca_components)
            
        # Set up pooling strategy
        self.pooling_strategy = config_instance.pooling_strategy
        
        # Initialize tokenizer for advanced processing
        self.tokenizer = AutoTokenizer.from_pretrained(config_instance.bi_encoder_model)
        
        logger.info(f"Initialized embeddings with model: {config_instance.bi_encoder_model}")

    def _mean_pooling(self, model_output: Any, attention_mask: Any) -> torch.Tensor:
        """Perform mean pooling on token embeddings.

        Args:
            model_output: Model's output
            attention_mask: Attention mask for tokens

        Returns:
            Pooled embeddings
        """
        token_embeddings = model_output[0]
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

    def _get_embedding(self, text: str, model: SentenceTransformer) -> np.ndarray:
        """Get embedding from specified model with proper processing.

        Args:
            text: Input text
            model: SentenceTransformer model to use

        Returns:
            Embedding vector
        """
        # Encode text
        encoded = model.encode(
            text,
            convert_to_numpy=True,
            normalize_embeddings=config_instance.normalize_embeddings,
            show_progress_bar=False,
        )
        
        # Ensure vector dimension matches configuration
        if len(encoded) != config_instance.vector_dimension:
            # If vector is too large, truncate it
            if len(encoded) > config_instance.vector_dimension:
                encoded = encoded[:config_instance.vector_dimension]
            # If vector is too small, pad with zeros
            else:
                padding = np.zeros(config_instance.vector_dimension - len(encoded))
                encoded = np.concatenate([encoded, padding])
                
        # Normalize final vector
        if config_instance.normalize_embeddings:
            encoded = encoded / np.linalg.norm(encoded)
            
        return encoded

    def get_embeddings(self, texts: Union[str, List[str]], retry_with_fallback: bool = True) -> Union[np.ndarray, List[np.ndarray]]:
        """Generate embeddings for one or more texts.

        Args:
            texts: Input text or list of texts
            retry_with_fallback: Whether to retry with fallback model on failure

        Returns:
            Single embedding vector or list of vectors
        """
        single_input = isinstance(texts, str)
        texts_list = [texts] if single_input else texts
        
        results = []
        for text in texts_list:
            try:
                # Check cache first
                if self.use_cache:
                    cache_key = f"emb:{text}"
                    cached = embedding_cache.get(cache_key)
                    if cached is not None:
                        results.append(cached)
                        continue
                
                # Generate embedding with main model
                vector = self._get_embedding(text, self.bi_encoder)
                
                # Cache result
                if self.use_cache:
                    embedding_cache.set(cache_key, vector)
                
                results.append(vector)
                
            except Exception as e:
                logger.error(f"Error generating embedding: {e}")
                if retry_with_fallback:
                    try:
                        # Try fallback model
                        logger.info("Retrying with fallback model")
                        vector = self._get_embedding(text, self.fallback_encoder)
                        results.append(vector)
                    except Exception as e2:
                        logger.error(f"Fallback model also failed: {e2}")
                        raise
                else:
                    raise
        
        return results[0] if single_input else results

    def compute_similarity(self, text1: str, text2: str, strategy: str = "cross_encoder") -> float:
        """Compute similarity between two texts.

        Args:
            text1: First text
            text2: Second text
            strategy: Similarity strategy ('cross_encoder' or 'bi_encoder')

        Returns:
            Similarity score between 0 and 1
        """
        try:
            if strategy == "cross_encoder":
                # Use cross-encoder for more accurate similarity
                score = self.cross_encoder.predict([(text1, text2)])
                # Normalize to 0-1 range
                return float((1 + score) / 2)
            else:  # bi_encoder
                # Get embeddings
                vec1, vec2 = self.get_embeddings([text1, text2])
                # Compute cosine similarity
                similarity = np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
                return float(similarity)
        except Exception as e:
            logger.error(f"Failed to compute similarity: {e}")
            raise

    def rerank_results(
        self, query: str, results: List[Dict[str, Any]], top_k: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """Rerank search results using cross-encoder.

        Args:
            query: Search query
            results: List of search results
            top_k: Optional limit on number of results

        Returns:
            Reranked results
        """
        try:
            if not results:
                return results

            # Create text pairs for scoring
            pairs = [(query, result.get("description", "")) for result in results]

            # Get cross-encoder scores
            scores = self.cross_encoder.predict(pairs)

            # Sort by score
            scored_results = list(zip(results, scores))
            reranked = sorted(scored_results, key=lambda x: x[1], reverse=True)

            # Update scores and limit results
            final_results = []
            for i, (result, score) in enumerate(reranked[:top_k]):
                result["score"] = float((1 + score) / 2)  # Normalize to 0-1
                result["rank"] = i + 1
                final_results.append(result)

            return final_results

        except Exception as e:
            logger.error(f"Failed to rerank results: {e}")
            raise

    def get_multilingual_embeddings(self, text: str) -> np.ndarray:
        """Generate multilingual embeddings if supported.

        Args:
            text: Input text

        Returns:
            Multilingual embedding vector
        """
        if not self.multilingual_encoder:
            raise ValueError("Multilingual model not initialized")
            
        return self._get_embedding(text, self.multilingual_encoder)


# Global embeddings instance
embeddings = ModernEmbeddings()

__all__ = ["embeddings", "EmbeddingResult", "ModernEmbeddings"]
