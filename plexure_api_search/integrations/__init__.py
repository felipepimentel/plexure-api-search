"""Vector storage and preprocessing management."""

from .faiss_preprocessor import FAISSPreprocessor
from .llm.openrouter_client import OpenRouterClient
from .pinecone_client import PineconeClient
from .vector_manager import VectorManager

# Create the vector manager instance
vector_manager = VectorManager()

# Make the instance available as the default vector store
vector_store = vector_manager

# For backward compatibility
pinecone_instance = vector_store

__all__ = [
    "vector_store",
    "vector_manager",
    "pinecone_instance",
    "PineconeClient",
    "FAISSPreprocessor",
    "VectorManager",
    "OpenRouterClient",
]
