"""Vector storage backends."""

from typing import Optional

from ..config import config_instance
from .faiss_client import FAISSClient
from .pinecone_client import PineconeClient

# Choose backend based on configuration
if config_instance.vector_store == "faiss":
    vector_store = FAISSClient()
else:
    vector_store = PineconeClient()

# Make the instance available
pinecone_instance = vector_store  # For backward compatibility

__all__ = ["vector_store", "pinecone_instance", "PineconeClient", "FAISSClient"]
