"""Vectorization of API endpoints for indexing."""

import logging
from typing import Dict, List, Optional

from ..embeddings.transformer import get_embeddings
from ..utils.file import save_data
from .processor import processor

logger = logging.getLogger(__name__)


class APIVectorizer:
    """Vectorizer for API endpoints."""

    def _prepare_text(self, endpoint: Dict) -> str:
        """Prepare endpoint text for vectorization.

        Args:
            endpoint: Endpoint dictionary with metadata

        Returns:
            Concatenated text representation
        """
        parts = [
            endpoint.get("api_name", ""),
            endpoint.get("method", ""),
            endpoint.get("path", ""),
            endpoint.get("description", ""),
            endpoint.get("summary", ""),
        ]

        # Add parameters
        params = endpoint.get("parameters", [])
        if params:
            parts.extend(params)

        # Add responses
        responses = endpoint.get("responses", [])
        if responses:
            parts.extend(responses)

        # Add tags
        tags = endpoint.get("tags", [])
        if tags:
            parts.extend(tags)

        # Join all parts
        return " ".join(str(part) for part in parts if part)

    def vectorize_endpoint(self, endpoint: Dict) -> Dict:
        """Create vector representation of endpoint.

        Args:
            endpoint: Endpoint dictionary with metadata

        Returns:
            Dictionary with vector and metadata
        """
        try:
            # Prepare text
            text = self._prepare_text(endpoint)

            # Generate vector
            vector = get_embeddings(text)

            # Create vector entry
            return {
                "id": endpoint["id"],
                "values": vector,
                "metadata": endpoint
            }

        except Exception as e:
            logger.error(f"Error vectorizing endpoint {endpoint.get('id')}: {e}")
            return {}

    def vectorize_endpoints(
        self,
        endpoints: Optional[List[Dict]] = None,
        directory: Optional[str] = None,
        save_path: Optional[str] = None
    ) -> List[Dict]:
        """Vectorize multiple endpoints.

        Args:
            endpoints: List of endpoints to vectorize (optional)
            directory: Directory to process API files from (optional)
            save_path: Path to save vectors (optional)

        Returns:
            List of vector entries
        """
        try:
            # Get endpoints if not provided
            if endpoints is None:
                endpoints = processor.process_api_files(directory)

            # Vectorize each endpoint
            vectors = []
            for endpoint in endpoints:
                vector = self.vectorize_endpoint(endpoint)
                if vector:
                    vectors.append(vector)
                    logger.debug(f"Vectorized endpoint {endpoint.get('id')}")

            logger.info(f"Vectorized {len(vectors)} endpoints")

            # Save vectors if path provided
            if save_path and vectors:
                if save_data({"vectors": vectors}, save_path):
                    logger.info(f"Saved vectors to {save_path}")

            return vectors

        except Exception as e:
            logger.error(f"Error vectorizing endpoints: {e}")
            return []


# Global vectorizer instance
vectorizer = APIVectorizer()

__all__ = ["vectorizer", "APIVectorizer"] 