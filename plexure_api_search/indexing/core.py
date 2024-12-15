"""API contract indexer with triple vector embeddings."""

import logging
from typing import Dict, List

from ..model.api_spec import APISpec
from ..utils.config import API_DIR
from ..embedding.embeddings import TripleVectorizer
from ..model.metadata import Metadata
from ..storage.vector_stores.pinecone import vector_store
from ..utils import FileUtils

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ... existing code ... 