"""Advanced embedding strategies for API contracts."""

from dataclasses import dataclass
from typing import List, Dict, Any, Optional
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.decomposition import PCA
import logging

logger = logging.getLogger(__name__)

@dataclass
class TripleVector:
    """Triple vector representation of an API endpoint."""
    semantic_vector: np.ndarray
    structure_vector: np.ndarray
    parameter_vector: np.ndarray
    
    def to_combined_vector(self, weights: Optional[Dict[str, float]] = None) -> np.ndarray:
        """Combine vectors with optional weights.
        
        Args:
            weights: Dictionary with weights for each vector type.
                    Defaults to equal weights.
        
        Returns:
            Combined vector representation.
        """
        if weights is None:
            weights = {
                "semantic": 0.4,
                "structure": 0.3,
                "parameter": 0.3
            }
            
        return (
            weights["semantic"] * self.semantic_vector +
            weights["structure"] * self.structure_vector +
            weights["parameter"] * self.parameter_vector
        )


class TripleVectorizer:
    """Generates triple vector embeddings for API endpoints."""
    
    def __init__(
        self,
        semantic_model: str = "all-MiniLM-L6-v2",
        vector_dim: int = 384,
        pca_components: int = 128
    ):
        """Initialize vectorizer.
        
        Args:
            semantic_model: Name of the sentence transformer model.
            vector_dim: Dimension of each vector type.
            pca_components: Number of PCA components for dimension reduction.
        """
        self.semantic_model = SentenceTransformer(semantic_model)
        self.vector_dim = vector_dim
        self.pca = PCA(n_components=pca_components)
        
        # Initialize PCAs for each vector type
        self.semantic_pca = PCA(n_components=pca_components)
        self.structure_pca = PCA(n_components=pca_components)
        self.parameter_pca = PCA(n_components=pca_components)
        
    def create_semantic_vector(self, endpoint_data: Dict[str, Any]) -> np.ndarray:
        """Create semantic vector from endpoint description and documentation.
        
        Args:
            endpoint_data: Dictionary containing endpoint information.
            
        Returns:
            Semantic vector representation.
        """
        # Combine relevant text fields
        text = f"""
        Path: {endpoint_data.get('path', '')}
        Method: {endpoint_data.get('method', '')}
        Description: {endpoint_data.get('description', '')}
        Summary: {endpoint_data.get('summary', '')}
        Tags: {', '.join(endpoint_data.get('tags', []))}
        """
        
        # Generate embedding
        vector = self.semantic_model.encode(text)
        
        # Apply PCA if needed
        if len(vector) > self.vector_dim:
            vector = self.semantic_pca.fit_transform([vector])[0]
            
        return vector
        
    def create_structure_vector(self, endpoint_data: Dict[str, Any]) -> np.ndarray:
        """Create structure vector representing API endpoint structure.
        
        Args:
            endpoint_data: Dictionary containing endpoint information.
            
        Returns:
            Structure vector representation.
        """
        # Extract structural features
        path_parts = endpoint_data.get('path', '').split('/')
        method = endpoint_data.get('method', '').upper()
        version = endpoint_data.get('api_version', '')
        
        # Create structural embedding
        features = []
        
        # Path depth and components
        features.append(len(path_parts))
        features.extend([hash(part) % 100 for part in path_parts])
        
        # Method encoding
        method_encoding = {
            'GET': [1, 0, 0, 0],
            'POST': [0, 1, 0, 0],
            'PUT': [0, 0, 1, 0],
            'DELETE': [0, 0, 0, 1]
        }.get(method, [0, 0, 0, 0])
        features.extend(method_encoding)
        
        # Version encoding
        try:
            version_parts = [float(v) for v in version.split('.')]
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
            endpoint_data: Dictionary containing endpoint information.
            
        Returns:
            Parameter vector representation.
        """
        # Extract parameter information
        parameters = endpoint_data.get('parameters', [])
        
        # Parameter type encoding
        type_encoding = {
            'string': [1, 0, 0, 0],
            'integer': [0, 1, 0, 0],
            'boolean': [0, 0, 1, 0],
            'array': [0, 0, 0, 1]
        }
        
        features = []
        
        # Number of parameters
        features.append(len(parameters))
        
        # Parameter types and locations
        for param in parameters:
            param_type = param.get('schema', {}).get('type', 'string')
            param_in = param.get('in', 'query')
            
            # Add type encoding
            features.extend(type_encoding.get(param_type, [0, 0, 0, 0]))
            
            # Add location encoding
            location_encoding = {
                'query': [1, 0, 0],
                'path': [0, 1, 0],
                'body': [0, 0, 1]
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
            endpoint_data: Dictionary containing endpoint information.
            
        Returns:
            TripleVector representation.
        """
        return TripleVector(
            semantic_vector=self.create_semantic_vector(endpoint_data),
            structure_vector=self.create_structure_vector(endpoint_data),
            parameter_vector=self.create_parameter_vector(endpoint_data)
        )
        
    def bulk_vectorize(self, endpoints: List[Dict[str, Any]]) -> List[TripleVector]:
        """Vectorize multiple endpoints.
        
        Args:
            endpoints: List of endpoint dictionaries.
            
        Returns:
            List of TripleVector representations.
        """
        return [self.vectorize(endpoint) for endpoint in endpoints] 

    def vectorize_query(self, query: str) -> List[float]:
        """Vectorize a search query.
        
        Args:
            query: Search query string
            
        Returns:
            Combined vector representation
        """
        # Create embeddings for query
        embeddings = self.semantic_model.encode(query)
        
        # Convert to list if needed
        if hasattr(embeddings, 'tolist'):
            embeddings = embeddings.tolist()
            
        return embeddings

def create_embeddings(text):
    logger.info(f"Creating embedding for text: {text[:100]}...")
    # Rest of the code... 