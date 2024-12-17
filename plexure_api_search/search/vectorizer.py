"""Vectorizer module for converting API endpoints into vector representations."""

from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import numpy as np
from sentence_transformers import SentenceTransformer

from ..embedding.embeddings import EmbeddingManager
from ..monitoring.events import Event, EventType, event_manager
from ..config import config_instance


@dataclass
class Triple:
    """Represents a triple of endpoint information."""
    
    endpoint: str
    method: str
    description: str
    path_params: Optional[List[str]] = None
    query_params: Optional[List[str]] = None
    response_codes: Optional[List[str]] = None
    
    def to_text(self) -> str:
        """Convert triple to text representation."""
        text_parts = [
            f"{self.method} {self.endpoint}",
            f"Description: {self.description}",
        ]
        
        if self.path_params:
            text_parts.append(f"Path parameters: {', '.join(self.path_params)}")
        if self.query_params:
            text_parts.append(f"Query parameters: {', '.join(self.query_params)}")
        if self.response_codes:
            text_parts.append(f"Response codes: {', '.join(self.response_codes)}")
            
        return " | ".join(text_parts)


class TripleVectorizer:
    """Converts API endpoint triples into vector representations."""
    
    def __init__(self, embedding_manager: Optional[EmbeddingManager] = None):
        """Initialize vectorizer with embedding manager.
        
        Args:
            embedding_manager: Optional embedding manager instance. If not provided,
                             a new instance will be created.
        """
        self.embedding_manager = embedding_manager or EmbeddingManager()
        
        # Emit initialization event
        event_manager.emit(
            Event(
                type=EventType.MODEL_LOADING_STARTED,
                timestamp=datetime.now(),
                component="vectorizer",
                description="Initializing triple vectorizer",
            )
        )
    
    def vectorize(self, triple: Triple) -> np.ndarray:
        """Convert a triple into its vector representation."""
        # Emit vectorization start event
        event_manager.emit(
            Event(
                type=EventType.EMBEDDING_STARTED,
                timestamp=datetime.now(),
                component="vectorizer",
                description=f"Vectorizing triple: {triple.endpoint}",
            )
        )
        
        try:
            # Convert triple to text representation
            text = triple.to_text()
            
            # Get vector embedding
            vector = self.embedding_manager.get_embeddings(text)
            
            # Emit success event
            event_manager.emit(
                Event(
                    type=EventType.EMBEDDING_COMPLETED,
                    timestamp=datetime.now(),
                    component="vectorizer",
                    description=f"Successfully vectorized triple: {triple.endpoint}",
                )
            )
            
            return vector
            
        except Exception as e:
            # Emit failure event
            event_manager.emit(
                Event(
                    type=EventType.EMBEDDING_FAILED,
                    timestamp=datetime.now(),
                    component="vectorizer",
                    description=f"Failed to vectorize triple: {triple.endpoint}",
                    error=str(e),
                    success=False,
                )
            )
            raise
    
    def vectorize_batch(self, triples: List[Triple]) -> np.ndarray:
        """Convert a batch of triples into vector representations."""
        # Convert triples to text representations
        texts = [triple.to_text() for triple in triples]
        
        # Get vector embeddings in batch
        return self.embedding_manager.get_embeddings(texts) 