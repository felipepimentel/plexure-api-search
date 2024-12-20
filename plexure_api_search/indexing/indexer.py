"""API contract indexing."""

import logging
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from ..config import config
from ..monitoring.metrics import MetricsManager
from ..services.models import model_service
from ..services.vector_store import vector_store
from .parser import APIParser

logger = logging.getLogger(__name__)

class APIIndexer:
    """API contract indexer."""

    def __init__(self):
        """Initialize indexer."""
        self.parser = APIParser()
        self.metrics = MetricsManager()
        self.initialized = False

    def initialize(self) -> None:
        """Initialize indexer."""
        if self.initialized:
            return

        try:
            # Initialize dependencies
            model_service.initialize()
            vector_store.initialize()
            self.initialized = True
            logger.info("Indexer initialized")

        except Exception as e:
            logger.error(f"Failed to initialize indexer: {e}")
            raise

    def cleanup(self) -> None:
        """Clean up indexer."""
        if self.initialized:
            vector_store.cleanup()
            model_service.cleanup()
            self.initialized = False
            logger.info("Indexer cleaned up")

    def clear(self) -> None:
        """Clear the index."""
        if not self.initialized:
            self.initialize()

        try:
            vector_store.clear()
            logger.info("Cleared index")

        except Exception as e:
            logger.error(f"Failed to clear index: {e}")
            raise

    def index_contract(self, contract_path: str) -> None:
        """Index an API contract.
        
        Args:
            contract_path: Path to contract file
        """
        if not self.initialized:
            self.initialize()

        try:
            logger.info(f"Processing contract: {contract_path}")

            # Parse contract
            endpoints = self.parser.parse_contract(contract_path)
            logger.info(f"Found {len(endpoints)} endpoints in {contract_path}")

            if not endpoints:
                logger.warning(f"No endpoints found in {contract_path}")
                return

            # Index endpoints
            self.index_endpoints(endpoints)

        except Exception as e:
            logger.error(f"Failed to index contract {contract_path}: {e}")
            self.metrics.increment_counter("contract_errors")
            raise

    def index_endpoints(self, endpoints: List[Dict]) -> None:
        """Index a list of endpoints.
        
        Args:
            endpoints: List of endpoint dictionaries
        """
        if not endpoints:
            return

        try:
            logger.info(f"Indexing {len(endpoints)} endpoints")

            # Generate vectors
            vectors = []
            metadata = []
            ids = []
            
            for i, endpoint in enumerate(endpoints):
                # Generate text for embedding
                text = self._prepare_endpoint_text(endpoint)
                
                # Get vector
                vector = model_service.get_embeddings(text)
                
                # Handle 3D output (batch_size x 1 x dimension)
                if vector.ndim == 3:
                    vector = vector.squeeze(1)
                
                vectors.append(vector)
                
                # Store metadata
                metadata.append(endpoint)
                ids.append(i)

            # Convert to numpy arrays
            vectors_array = np.vstack(vectors)
            ids_array = np.array(ids)

            # Store vectors with metadata
            vector_store.store_vectors(vectors_array, ids_array, metadata)
            logger.info(f"Stored {len(vectors)} vectors")

        except Exception as e:
            logger.error(f"Failed to index endpoints: {e}")
            self.metrics.increment_counter("contract_errors")
            raise

    def _prepare_endpoint_text(self, endpoint: Dict) -> str:
        """Prepare endpoint text for embedding.
        
        Args:
            endpoint: Endpoint dictionary
            
        Returns:
            Text representation of endpoint
        """
        parts = []
        
        # Add basic info
        parts.extend([
            endpoint.get("method", ""),
            endpoint.get("path", ""),
            endpoint.get("summary", ""),
            endpoint.get("description", ""),
        ])
        
        # Add parameters
        params = endpoint.get("parameters", [])
        for param in params:
            param_parts = [
                param.get("name", ""),
                param.get("in", ""),
                param.get("description", ""),
            ]
            parts.extend(param_parts)
            
        # Add responses
        responses = endpoint.get("responses", {})
        for code, resp in responses.items():
            resp_parts = [
                str(code),
                resp.get("description", ""),
            ]
            parts.extend(resp_parts)
            
        # Add tags
        tags = endpoint.get("tags", [])
        parts.extend(tags)
        
        # Join all parts
        return " ".join(str(part) for part in parts if part)

# Global instance
indexer = APIIndexer()
