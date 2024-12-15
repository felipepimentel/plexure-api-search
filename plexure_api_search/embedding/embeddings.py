"""Advanced embedding generation with triple vector strategies and caching."""

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np
from sentence_transformers import CrossEncoder, SentenceTransformer
from sklearn.decomposition import PCA

from ..utils.cache import DiskCache
from ..config import config_instance

logger = logging.getLogger(__name__)

# Pre-configured cache instances
embedding_cache = DiskCache[np.ndarray](
    namespace="embeddings",
    ttl=config_instance.embedding_cache_ttl,  # 24 hours
)


@dataclass
class EmbeddingResult:
    """Result of embedding generation with metadata."""

    # Core data
    vector: np.ndarray
    text: str
    model_name: str

    # Additional metadata
    dimension: int
    strategy: str = "bi_encoder"  # bi_encoder or cross_encoder
    confidence: float = 1.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format for serialization."""
        return {
            "vector": self.vector.tolist(),
            "text": self.text,
            "model_name": self.model_name,
            "dimension": self.dimension,
            "strategy": self.strategy,
            "confidence": self.confidence,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "EmbeddingResult":
        """Create from dictionary format."""
        data["vector"] = np.array(data["vector"])
        return cls(**data)


@dataclass
class TripleVector:
    """Triple vector representation of an API endpoint."""

    semantic_vector: np.ndarray
    structure_vector: np.ndarray
    parameter_vector: np.ndarray

    def to_combined_vector(
        self, weights: Optional[Dict[str, float]] = None
    ) -> np.ndarray:
        """Combine vectors with optional weights.

        Args:
            weights: Dictionary with weights for each vector type.
                    Defaults to semantic: 0.4, structure: 0.3, parameter: 0.3

        Returns:
            Combined vector representation.
        """
        if weights is None:
            weights = {"semantic": 0.4, "structure": 0.3, "parameter": 0.3}

        return (
            weights["semantic"] * self.semantic_vector
            + weights["structure"] * self.structure_vector
            + weights["parameter"] * self.parameter_vector
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format for serialization."""
        return {
            "semantic_vector": self.semantic_vector.tolist(),
            "structure_vector": self.structure_vector.tolist(),
            "parameter_vector": self.parameter_vector.tolist(),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TripleVector":
        """Create from dictionary format."""
        return cls(
            semantic_vector=np.array(data["semantic_vector"]),
            structure_vector=np.array(data["structure_vector"]),
            parameter_vector=np.array(data["parameter_vector"]),
        )


class TripleVectorizer:
    """Advanced embedding generator with multiple strategies."""

    def __init__(
        self,
        bi_encoder_model: str = config_instance.bi_encoder_model,
        cross_encoder_model: str = config_instance.cross_encoder_model,
        vector_dim: int = config_instance.vector_dimension,
        pca_components: int = config_instance.pca_components,
        use_cache: bool = True,
    ):
        """Initialize vectorizer.

        Args:
            bi_encoder_model: Name/path of bi-encoder model
            cross_encoder_model: Name/path of cross-encoder model
            vector_dim: Dimension of each vector type
            pca_components: Number of PCA components for dimension reduction
            use_cache: Whether to use embedding cache
        """
        self.use_cache = use_cache
        self.vector_dim = vector_dim

        # Initialize models
        try:
            self.bi_encoder = SentenceTransformer(bi_encoder_model)
            self.cross_encoder = CrossEncoder(cross_encoder_model)
            logger.info(
                f"Initialized models: {bi_encoder_model}, {cross_encoder_model}"
            )
        except Exception as e:
            logger.error(f"Failed to initialize models: {e}")
            raise

        # Initialize PCAs for dimension reduction
        self.semantic_pca = PCA(n_components=pca_components)
        self.structure_pca = PCA(n_components=pca_components)
        self.parameter_pca = PCA(n_components=pca_components)

        # Get embedding dimension
        self.embedding_dim = self.bi_encoder.get_sentence_embedding_dimension()

    def vectorize_query(self, query: str) -> np.ndarray:
        """Generate embedding for search query.

        Args:
            query: Search query text

        Returns:
            Query embedding vector
        """
        try:
            # Check cache first
            if self.use_cache:
                cached = embedding_cache.get(f"query:{query}")
                if cached is not None:
                    return cached

            # Generate embedding
            vector = self.bi_encoder.encode(
                query, convert_to_numpy=True, normalize_embeddings=True
            )

            # Cache result
            if self.use_cache:
                embedding_cache.set(f"query:{query}", vector)

            return vector

        except Exception as e:
            logger.error(f"Failed to vectorize query: {e}")
            raise

    def create_semantic_vector(self, endpoint_data: Dict[str, Any]) -> np.ndarray:
        """Create semantic vector from endpoint description and documentation.

        Args:
            endpoint_data: Dictionary containing endpoint information

        Returns:
            Semantic vector representation
        """
        # Combine relevant text fields
        text = f"""
        Path: {endpoint_data.get("path", "")}
        Method: {endpoint_data.get("method", "")}
        Description: {endpoint_data.get("description", "")}
        Summary: {endpoint_data.get("summary", "")}
        Tags: {", ".join(endpoint_data.get("tags", []))}
        """

        # Check cache
        if self.use_cache:
            cached = embedding_cache.get(f"semantic:{text}")
            if cached is not None:
                return cached

        # Generate embedding
        vector = self.bi_encoder.encode(
            text, convert_to_numpy=True, normalize_embeddings=True
        )

        # Apply PCA if needed
        if len(vector) > self.vector_dim:
            vector = self.semantic_pca.fit_transform([vector])[0]

        # Cache result
        if self.use_cache:
            embedding_cache.set(f"semantic:{text}", vector)

        return vector

    def create_structure_vector(self, endpoint_data: Dict[str, Any]) -> np.ndarray:
        """Create structure vector representing API endpoint structure.

        Args:
            endpoint_data: Dictionary containing endpoint information

        Returns:
            Structure vector representation
        """
        # Extract structural features
        path_parts = endpoint_data.get("path", "").split("/")
        method = endpoint_data.get("method", "").upper()
        version = endpoint_data.get("api_version", "")

        # Create structural embedding
        features = []

        # Path depth and components
        features.append(len(path_parts))
        features.extend([hash(part) % 100 for part in path_parts])

        # Method encoding
        method_encoding = {
            "GET": [1, 0, 0, 0],
            "POST": [0, 1, 0, 0],
            "PUT": [0, 0, 1, 0],
            "DELETE": [0, 0, 0, 1],
        }.get(method, [0, 0, 0, 0])
        features.extend(method_encoding)

        # Version encoding
        try:
            version_parts = [float(v) for v in version.split(".")]
            features.extend(version_parts)
        except:
            features.extend([0, 0, 0])

        # Convert to numpy array and pad/truncate
        vector = np.array(features, dtype=np.float32)
        if len(vector) > self.vector_dim:
            vector = self.structure_pca.fit_transform([vector])[0]
        else:
            vector = np.pad(vector, (0, self.vector_dim - len(vector)))

        return vector

    def create_parameter_vector(self, endpoint_data: Dict[str, Any]) -> np.ndarray:
        """Create parameter vector encoding API parameters and types.

        Args:
            endpoint_data: Dictionary containing endpoint information

        Returns:
            Parameter vector representation
        """
        # Extract parameter information
        parameters = endpoint_data.get("parameters", [])

        # Parameter type encoding
        type_encoding = {
            "string": [1, 0, 0, 0],
            "integer": [0, 1, 0, 0],
            "boolean": [0, 0, 1, 0],
            "array": [0, 0, 0, 1],
        }

        features = []

        # Number of parameters
        features.append(len(parameters))

        # Parameter types and locations
        for param in parameters:
            # Skip if param is not a dict
            if not isinstance(param, dict):
                continue

            # Get parameter type and location
            schema = param.get("schema", {})
            if isinstance(schema, dict):
                param_type = schema.get("type", "string")
            else:
                param_type = "string"
                
            param_in = param.get("in", "query")

            # Add type encoding
            features.extend(type_encoding.get(param_type, [0, 0, 0, 0]))

            # Add location encoding
            location_encoding = {
                "query": [1, 0, 0],
                "path": [0, 1, 0],
                "body": [0, 0, 1],
            }.get(param_in, [0, 0, 0])
            features.extend(location_encoding)

        # Convert to numpy array and pad/truncate
        vector = np.array(features, dtype=np.float32)
        if len(vector) > self.vector_dim:
            vector = self.parameter_pca.fit_transform([vector])[0]
        else:
            vector = np.pad(vector, (0, self.vector_dim - len(vector)))

        return vector

    def vectorize(self, endpoint_data: Dict[str, Any]) -> TripleVector:
        """Create triple vector representation for an endpoint.

        Args:
            endpoint_data: Dictionary containing endpoint information

        Returns:
            TripleVector representation
        """
        return TripleVector(
            semantic_vector=self.create_semantic_vector(endpoint_data),
            structure_vector=self.create_structure_vector(endpoint_data),
            parameter_vector=self.create_parameter_vector(endpoint_data),
        )

    def vectorize_batch(
        self, texts: List[str], batch_size: int = 32
    ) -> List[EmbeddingResult]:
        """Generate embeddings for multiple texts.

        Args:
            texts: List of texts to embed
            batch_size: Batch size for processing

        Returns:
            List of embedding results
        """
        results = []

        try:
            # Process in batches
            for i in range(0, len(texts), batch_size):
                batch = texts[i : i + batch_size]

                # Check cache for each text
                batch_vectors = []
                uncached_indices = []
                uncached_texts = []

                for j, text in enumerate(batch):
                    if self.use_cache:
                        cached = embedding_cache.get(f"batch:{text}")
                        if cached is not None:
                            batch_vectors.append(cached)
                            continue

                    uncached_indices.append(j)
                    uncached_texts.append(text)

                # Generate embeddings for uncached texts
                if uncached_texts:
                    vectors = self.bi_encoder.encode(
                        uncached_texts,
                        convert_to_numpy=True,
                        normalize_embeddings=True,
                        batch_size=batch_size,
                    )

                    # Cache new embeddings
                    if self.use_cache:
                        for text, vector in zip(uncached_texts, vectors):
                            embedding_cache.set(f"batch:{text}", vector)

                    # Insert new vectors at correct positions
                    for idx, vector in zip(uncached_indices, vectors):
                        batch_vectors.insert(idx, vector)

                # Create EmbeddingResults
                for text, vector in zip(batch, batch_vectors):
                    result = EmbeddingResult(
                        vector=vector,
                        text=text,
                        model_name=self.bi_encoder.get_model_card(),
                        dimension=self.embedding_dim,
                    )
                    results.append(result)

        except Exception as e:
            logger.error(f"Failed to vectorize batch: {e}")
            raise

        return results

    def compute_similarity(
        self, text1: str, text2: str, strategy: str = "cross_encoder"
    ) -> float:
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
                vec1 = self.vectorize_query(text1)
                vec2 = self.vectorize_query(text2)

                # Compute cosine similarity
                similarity = np.dot(vec1, vec2)
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


def create_embeddings(text):
    logger.info(f"Creating embedding for text: {text[:100]}...")
    # Rest of the code...
