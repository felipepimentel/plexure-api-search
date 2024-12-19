"""API contract indexer."""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Any

import numpy as np
from dependency_injector.wiring import inject

from ..config import config_instance
from ..monitoring.metrics import MetricsManager
from ..services.models import model_service
from ..services.vector_store import vector_store
from .parser import APIContractParser

logger = logging.getLogger(__name__)

class APIContractIndexer:
    """API contract indexer."""

    def __init__(self):
        """Initialize indexer."""
        self.metrics = MetricsManager()
        self.parser = APIContractParser()
        self.endpoint_metadata = {}  # type: Dict[int, Dict[str, Any]]
        self.next_id = 0
        self.initialized = False

    def initialize(self) -> None:
        """Initialize indexer."""
        if self.initialized:
            return

        try:
            # Initialize services
            model_service.initialize()
            vector_store.initialize()
            self.initialized = True
            logger.info("Indexer initialized")

        except Exception as e:
            logger.error(f"Failed to initialize indexer: {e}")
            raise

    def cleanup(self) -> None:
        """Clean up indexer."""
        model_service.cleanup()
        vector_store.cleanup()
        self.endpoint_metadata = {}
        self.next_id = 0
        self.initialized = False
        logger.info("Indexer cleaned up")

    def index_contracts(self, contract_paths: List[str]) -> None:
        """Index API contracts.
        
        Args:
            contract_paths: List of paths to API contract files
        """
        if not self.initialized:
            self.initialize()

        try:
            # Start timer
            start_time = self.metrics.start_timer()

            # Process each contract
            for path in contract_paths:
                logger.info(f"Processing contract: {path}")
                try:
                    # Parse contract
                    endpoints = self.parser.parse_contract(path)
                    logger.info(f"Found {len(endpoints)} endpoints in {path}")

                    # Index endpoints
                    self.index_endpoints(endpoints)

                except Exception as e:
                    logger.error(f"Failed to process contract {path}: {e}")
                    self.metrics.increment("contract_errors")
                    continue

            # Stop timer
            self.metrics.stop_timer(
                start_time,
                "index_contracts",
                {"count": len(contract_paths)},
            )

        except Exception as e:
            logger.error(f"Failed to index contracts: {e}")
            self.metrics.increment("indexing_errors")
            raise

    def index_endpoints(self, endpoints: List[Dict[str, Any]]) -> None:
        """Index API endpoints.
        
        Args:
            endpoints: List of endpoint metadata dicts
        """
        if not endpoints:
            logger.warning("No endpoints to index")
            return

        try:
            # Prepare texts and metadata
            texts = []
            ids = []
            for endpoint in endpoints:
                # Build text representation
                text_parts = []
                
                # Add method and path
                method = endpoint.get("method", "").upper()
                path = endpoint.get("path", "")
                if method and path:
                    text_parts.append(f"{method} {path}")
                
                # Add description
                description = endpoint.get("description", "").strip()
                if description:
                    text_parts.append(description)
                
                # Add summary
                summary = endpoint.get("summary", "").strip()
                if summary and summary != description:
                    text_parts.append(summary)
                
                # Add tags
                tags = endpoint.get("tags", [])
                if tags:
                    text_parts.append(" ".join(tags))
                
                # Add parameters
                params = endpoint.get("parameters", [])
                if params:
                    param_texts = []
                    for param in params:
                        param_text = f"{param.get('name', '')} ({param.get('in', '')})"
                        param_desc = param.get("description", "").strip()
                        if param_desc:
                            param_text += f": {param_desc}"
                        param_texts.append(param_text)
                    text_parts.append(" Parameters: " + ", ".join(param_texts))
                
                # Add responses
                responses = endpoint.get("responses", {})
                if responses:
                    response_texts = []
                    for code, details in responses.items():
                        desc = details.get("description", "").strip()
                        if desc:
                            response_texts.append(f"{code}: {desc}")
                    if response_texts:
                        text_parts.append(" Responses: " + ", ".join(response_texts))

                # Combine all parts
                text = " ".join(text_parts).strip()
                if not text:
                    logger.warning(f"Empty text for endpoint: {method} {path}")
                    continue

                # Store text and metadata
                texts.append(text)
                ids.append(self.next_id)
                self.endpoint_metadata[self.next_id] = endpoint
                self.next_id += 1

            # Log indexing info
            logger.info(f"Indexing {len(texts)} endpoints")
            logger.debug("Sample endpoint texts:")
            for i, text in enumerate(texts[:3]):
                logger.debug(f"  {i+1}. {text[:100]}...")

            # Generate embeddings
            embeddings = model_service.encode(texts)
            logger.debug(f"Generated embeddings shape: {embeddings.shape}")

            # Store vectors
            vector_store.store_vectors(embeddings, np.array(ids))
            logger.info(f"Stored {len(embeddings)} vectors")

            # Update metrics
            self.metrics.increment(
                "endpoints_indexed",
                {"count": len(texts)},
            )

        except Exception as e:
            logger.error(f"Failed to index endpoints: {e}")
            self.metrics.increment("indexing_errors")
            raise

    def clear(self) -> None:
        """Clear the index."""
        if not self.initialized:
            self.initialize()

        try:
            # Clear vector store
            vector_store.clear()
            
            # Reset metadata
            self.endpoint_metadata = {}
            self.next_id = 0
            
            logger.info("Cleared index")
            self.metrics.increment("index_cleared")

        except Exception as e:
            logger.error(f"Failed to clear index: {e}")
            self.metrics.increment("indexing_errors")
            raise

# Global instance
api_indexer = APIContractIndexer()
