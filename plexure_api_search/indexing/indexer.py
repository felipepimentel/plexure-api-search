"""API contract indexer with triple vector embeddings."""

import logging
from typing import Dict, List

from ..model.api_spec import APISpec
from ..utils.config import API_DIR
from ..embedding.embeddings import TripleVectorizer
from ..utils.file import FileUtils
from ..model.metadata import Metadata
from ..storage.pinecone_client import PineconeClient

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class APIIndexer:
    """Indexes API contracts with triple vector embeddings."""

    def __init__(self, pinecone_client: PineconeClient, api_dir: str = API_DIR):
        """Initialize indexer.

        Args:
            pinecone_client: PineconeClient instance
            api_dir: Directory containing API contracts
        """
        self.client = pinecone_client
        self.api_dir = api_dir
        self.vectorizer = TripleVectorizer()

    def index_apis(self, force: bool = False) -> None:
        """Index API contracts."""
        try:
            logger.info("Starting API indexation...")

            if force:
                logger.info("Force reindex requested. Cleaning up...")
                if not self.client.delete_all():
                    logger.warning("Could not clear index")

            # Process API files and create vectors
            vectors = []
            for endpoint in self._get_all_endpoints():
                try:
                    vector = self._create_vector_entry(endpoint)
                    vectors.append(vector)
                except Exception as e:
                    logger.error(f"Error processing endpoint: {e}")
                    continue

            # Upsert vectors
            total_vectors = self.client.upsert_vectors(vectors)

            # Verify final state
            final_count = self.client.get_vector_count()
            logger.info(f"Final vector count: {final_count}")

            if final_count != total_vectors:
                logger.warning(
                    f"Vector count mismatch. Expected: {total_vectors}, Found: {final_count}"
                )

        except Exception as e:
            logger.error(f"Indexing failed: {e}")
            raise

    def _get_all_endpoints(self) -> List[Dict]:
        """Get all endpoints from API files."""
        endpoints = []

        # Find API files
        api_files = FileUtils.find_api_files(self.api_dir)

        # Process each file
        for file_path in api_files:
            spec = FileUtils.load_api_file(file_path)
            if spec:
                endpoints.extend(APISpec.extract_endpoints(spec))

        return endpoints

    def _create_vector_entry(self, endpoint: Dict) -> Dict:
        """Create vector entry for an endpoint."""
        vector = self.vectorizer.vectorize(endpoint)
        combined_vector = vector.to_combined_vector()

        # Create unique ID
        vector_id = Metadata.create_endpoint_id(endpoint)

        # Prepare metadata
        metadata = Metadata.sanitize_metadata(endpoint)

        # Create vector entry
        return {
            "id": vector_id,
            "values": combined_vector.tolist(),
            "metadata": metadata,
        }
