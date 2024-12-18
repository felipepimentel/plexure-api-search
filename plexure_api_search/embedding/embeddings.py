"""Advanced embedding generation with modern models and caching."""

import json
import logging
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union, Tuple
from datetime import datetime

import numpy as np
from sentence_transformers import SentenceTransformer, CrossEncoder
from sklearn.decomposition import PCA
from transformers import AutoTokenizer, AutoModel
import torch
import torch.nn.functional as F

from ..utils.cache import DiskCache
from ..config import config_instance
from ..monitoring.events import Event, EventType, publisher

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
        self.embeddings = EmbeddingManager()

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


class EmbeddingManager:
    """Modern embedding generation with advanced features."""

    def __init__(self, use_cache: bool = True):
        """Initialize embedding models."""
        self.use_cache = use_cache
        
        # Start publisher
        try:
            publisher.start()
            time.sleep(0.1)  # Allow time for connection
            logger.debug("Publisher started for embeddings")
        except Exception as e:
            logger.warning(f"Failed to start publisher: {e}")
        
        # Initialize main bi-encoder
        publisher.emit(Event(
            type=EventType.MODEL_LOADING_STARTED,
            timestamp=datetime.now(),
            component="embeddings",
            description=f"Loading bi-encoder model: {config_instance.bi_encoder_model}",
            algorithm="sentence-transformers"
        ))
        try:
            self.bi_encoder = SentenceTransformer(config_instance.bi_encoder_model)
            # Verify model loaded correctly
            test_embedding = self.bi_encoder.encode("test", convert_to_numpy=True)
            if test_embedding is None or len(test_embedding) == 0:
                raise ValueError("Bi-encoder model failed verification")
            publisher.emit(Event(
                type=EventType.MODEL_LOADING_COMPLETED,
                timestamp=datetime.now(),
                component="embeddings",
                description="Bi-encoder model loaded successfully",
                algorithm="sentence-transformers"
            ))
        except Exception as e:
            logger.error(f"Failed to load primary bi-encoder: {e}")
            self.bi_encoder = None
        
        # Initialize fallback model
        publisher.emit(Event(
            type=EventType.MODEL_LOADING_STARTED,
            timestamp=datetime.now(),
            component="embeddings",
            description=f"Loading fallback model: {config_instance.bi_encoder_fallback}",
            algorithm="sentence-transformers"
        ))
        try:
            self.fallback_encoder = SentenceTransformer(config_instance.bi_encoder_fallback)
            # Verify fallback model
            test_embedding = self.fallback_encoder.encode("test", convert_to_numpy=True)
            if test_embedding is None or len(test_embedding) == 0:
                raise ValueError("Fallback model failed verification")
            publisher.emit(Event(
                type=EventType.MODEL_LOADING_COMPLETED,
                timestamp=datetime.now(),
                component="embeddings",
                description="Fallback model loaded successfully",
                algorithm="sentence-transformers"
            ))
        except Exception as e:
            logger.error(f"Failed to load fallback encoder: {e}")
            self.fallback_encoder = None
        
        # Initialize multilingual model if needed
        self.multilingual_encoder = None
        if hasattr(config_instance, "multilingual_model"):
            publisher.emit(Event(
                type=EventType.MODEL_LOADING_STARTED,
                timestamp=datetime.now(),
                component="embeddings",
                description=f"Loading multilingual model: {config_instance.multilingual_model}",
                algorithm="sentence-transformers"
            ))
            try:
                self.multilingual_encoder = SentenceTransformer(config_instance.multilingual_model)
                # Verify multilingual model
                test_embedding = self.multilingual_encoder.encode("test", convert_to_numpy=True)
                if test_embedding is None or len(test_embedding) == 0:
                    raise ValueError("Multilingual model failed verification")
                publisher.emit(Event(
                    type=EventType.MODEL_LOADING_COMPLETED,
                    timestamp=datetime.now(),
                    component="embeddings",
                    description="Multilingual model loaded successfully",
                    algorithm="sentence-transformers"
                ))
            except Exception as e:
                logger.error(f"Failed to load multilingual model: {e}")
                self.multilingual_encoder = None
        
        # Initialize tokenizer for advanced processing
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(config_instance.bi_encoder_model)
        except Exception as e:
            logger.error(f"Failed to load tokenizer: {e}")
            self.tokenizer = None
        
        # Verify at least one model is available
        if self.bi_encoder is None and self.fallback_encoder is None:
            raise RuntimeError("No embedding models available - both primary and fallback failed to load")
        
        logger.info(f"Initialized embeddings with model: {config_instance.bi_encoder_model}")

    def get_embeddings(self, text: Union[str, List[str]]) -> np.ndarray:
        """Get embeddings for text.

        Args:
            text: Input text or list of texts

        Returns:
            Vector embeddings
        """
        try:
            # Handle empty input
            if not text or (isinstance(text, list) and not text):
                raise ValueError("Empty input text")

            # Get model
            model = self.bi_encoder or self.fallback_encoder
            if not model:
                raise ValueError("No embedding model available")

            # Convert single text to list for consistent handling
            texts = [text] if isinstance(text, str) else text
            
            publisher.emit(Event(
                type=EventType.EMBEDDING_STARTED,
                timestamp=datetime.now(),
                component="embeddings",
                description=f"Generating embeddings for {len(texts)} texts",
                metadata={"text_count": len(texts)}
            ))
            
            start_time = time.time()
            
            # Generate embeddings
            embeddings = model.encode(
                texts,
                convert_to_numpy=True,
                normalize_embeddings=config_instance.normalize_embeddings,
                show_progress_bar=False
            )
            
            duration_ms = (time.time() - start_time) * 1000
            
            # Handle empty results
            if embeddings is None or (isinstance(embeddings, np.ndarray) and embeddings.size == 0):
                raise ValueError("Model returned empty embeddings")

            # Convert back to single vector if input was single text
            if isinstance(text, str):
                embeddings = embeddings[0]
            
            publisher.emit(Event(
                type=EventType.EMBEDDING_COMPLETED,
                timestamp=datetime.now(),
                component="embeddings",
                description=f"Generated embeddings successfully",
                duration_ms=duration_ms,
                metadata={
                    "text_count": len(texts),
                    "embedding_dim": embeddings.shape[-1] if isinstance(embeddings, np.ndarray) else len(embeddings),
                    "normalized": config_instance.normalize_embeddings
                }
            ))

            return embeddings

        except Exception as e:
            logger.error(f"Failed to get embeddings: {e}")
            publisher.emit(Event(
                type=EventType.EMBEDDING_FAILED,
                timestamp=datetime.now(),
                component="embeddings",
                description="Failed to generate embeddings",
                error=str(e),
                success=False
            ))
            raise

    def compute_similarity(self, text1: str, text2: str, strategy: str = "bi_encoder") -> float:
        """Compute similarity between two texts.

        Args:
            text1: First text
            text2: Second text
            strategy: Similarity strategy (only 'bi_encoder' supported)

        Returns:
            Similarity score between 0 and 1
        """
        try:
            publisher.emit(Event(
                type=EventType.EMBEDDING_STARTED,
                timestamp=datetime.now(),
                component="embeddings",
                description="Computing text similarity",
                metadata={"strategy": strategy}
            ))
            
            start_time = time.time()
            
            # Get embeddings using bi-encoder
            model = self.bi_encoder or self.fallback_encoder
            if not model:
                raise ValueError("No embedding model available")
                
            embedding1 = model.encode(text1, convert_to_numpy=True)
            embedding2 = model.encode(text2, convert_to_numpy=True)
            
            # Compute cosine similarity
            similarity = np.dot(embedding1, embedding2) / (
                np.linalg.norm(embedding1) * np.linalg.norm(embedding2)
            )
            
            # Ensure score is between 0 and 1
            score = float(max(0.0, min(1.0, (similarity + 1) / 2)))
            
            duration_ms = (time.time() - start_time) * 1000
            
            publisher.emit(Event(
                type=EventType.EMBEDDING_COMPLETED,
                timestamp=datetime.now(),
                component="embeddings",
                description="Computed similarity successfully",
                duration_ms=duration_ms,
                metadata={
                    "strategy": strategy,
                    "similarity_score": score
                }
            ))
            
            return score
            
        except Exception as e:
            logger.error(f"Failed to compute similarity: {e}")
            publisher.emit(Event(
                type=EventType.EMBEDDING_FAILED,
                timestamp=datetime.now(),
                component="embeddings",
                description="Failed to compute similarity",
                error=str(e),
                success=False,
                metadata={"strategy": strategy}
            ))
            return 0.0

    def rerank_results(
        self, query: str, results: List[Dict[str, Any]], top_k: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """Rerank search results using bi-encoder.

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

            # Get query embedding using bi-encoder
            model = self.bi_encoder or self.fallback_encoder
            if not model:
                return results
                
            query_embedding = model.encode(query, convert_to_numpy=True)

            # Get result embeddings and compute scores
            scored_results = []
            for result in results:
                try:
                    # Get result text
                    result_text = result.get("description", "")
                    if not result_text:
                        continue
                        
                    # Get result embedding
                    result_embedding = model.encode(result_text, convert_to_numpy=True)
                    
                    # Compute similarity
                    score = np.dot(query_embedding, result_embedding) / (
                        np.linalg.norm(query_embedding) * np.linalg.norm(result_embedding)
                    )
                    
                    # Normalize score to 0-1
                    score = float(max(0.0, min(1.0, (score + 1) / 2)))
                    
                    scored_results.append((result, score))
                except Exception as e:
                    logger.error(f"Failed to score result: {e}")
                    continue

            # Sort by score
            reranked = sorted(scored_results, key=lambda x: x[1], reverse=True)

            # Update scores and limit results
            final_results = []
            for i, (result, score) in enumerate(reranked[:top_k]):
                result["score"] = score
                result["rank"] = i + 1
                final_results.append(result)

            return final_results

        except Exception as e:
            logger.error(f"Failed to rerank results: {e}")
            return results


# Global embeddings instance
embeddings = EmbeddingManager()

__all__ = ["embeddings", "EmbeddingResult", "EmbeddingManager"]
