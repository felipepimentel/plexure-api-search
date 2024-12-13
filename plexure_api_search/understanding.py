"""Zero-shot API understanding and relationship mapping."""

from typing import Dict, List, Set, Tuple, Optional
import networkx as nx
from dataclasses import dataclass
from collections import defaultdict
import re
from sentence_transformers import SentenceTransformer, util


@dataclass
class APICategory:
    """Category information for an API endpoint."""
    name: str
    confidence: float
    subcategories: List[str]
    features: List[str]


@dataclass
class APIRelationship:
    """Relationship between two API endpoints."""
    source: str
    target: str
    relationship_type: str  # 'depends_on', 'similar_to', 'alternative_to'
    confidence: float
    description: str


class ZeroShotUnderstanding:
    """Handles zero-shot understanding of APIs."""
    
    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        similarity_threshold: float = 0.7
    ):
        """Initialize understanding module.
        
        Args:
            model_name: Name of the sentence transformer model.
            similarity_threshold: Threshold for similarity matching.
        """
        self.model = SentenceTransformer(model_name)
        self.similarity_threshold = similarity_threshold
        
        # Pre-defined categories with examples
        self.categories = {
            "Authentication": [
                "login", "logout", "token", "auth", "password",
                "credentials", "session", "oauth"
            ],
            "User Management": [
                "user", "profile", "account", "register",
                "signup", "permissions", "roles"
            ],
            "Data Operations": [
                "create", "read", "update", "delete",
                "list", "search", "filter", "query"
            ],
            "File Operations": [
                "upload", "download", "file", "document",
                "attachment", "media", "image"
            ],
            "Configuration": [
                "settings", "config", "preferences",
                "options", "parameters", "setup"
            ],
            "Analytics": [
                "metrics", "statistics", "reports",
                "analytics", "dashboard", "monitoring"
            ],
            "Communication": [
                "notify", "message", "email", "sms",
                "webhook", "callback", "notification"
            ],
            "Payment": [
                "payment", "transaction", "invoice",
                "billing", "subscription", "price"
            ]
        }
        
        # Pre-defined features
        self.features = {
            "Pagination": [
                "page", "limit", "offset", "size",
                "next", "previous", "total"
            ],
            "Filtering": [
                "filter", "query", "search", "where",
                "condition", "criteria"
            ],
            "Sorting": [
                "sort", "order", "direction", "asc",
                "desc", "ascending", "descending"
            ],
            "Caching": [
                "cache", "etag", "if-none-match",
                "if-modified-since", "expires"
            ],
            "Rate Limiting": [
                "rate", "limit", "throttle", "quota",
                "requests", "window"
            ],
            "Versioning": [
                "version", "v1", "v2", "deprecated",
                "sunset", "legacy"
            ]
        }
        
        # Initialize relationship graph
        self.relationship_graph = nx.DiGraph()
        
    def _calculate_similarity(self, text1: str, text2: str) -> float:
        """Calculate semantic similarity between two texts.
        
        Args:
            text1: First text.
            text2: Second text.
            
        Returns:
            Similarity score between 0 and 1.
        """
        embedding1 = self.model.encode(text1, convert_to_tensor=True)
        embedding2 = self.model.encode(text2, convert_to_tensor=True)
        return float(util.pytorch_cos_sim(embedding1, embedding2))
        
    def categorize_endpoint(
        self,
        endpoint_data: Dict
    ) -> APICategory:
        """Categorize an API endpoint.
        
        Args:
            endpoint_data: Dictionary containing endpoint information.
            
        Returns:
            APICategory with categorization information.
        """
        # Combine relevant text for analysis
        endpoint_text = f"""
        Path: {endpoint_data.get('path', '')}
        Method: {endpoint_data.get('method', '')}
        Description: {endpoint_data.get('description', '')}
        Summary: {endpoint_data.get('summary', '')}
        Tags: {', '.join(endpoint_data.get('tags', []))}
        """
        
        # Calculate similarities with categories
        category_scores = []
        for category, examples in self.categories.items():
            category_text = ' '.join(examples)
            similarity = self._calculate_similarity(endpoint_text, category_text)
            category_scores.append((category, similarity))
        
        # Get best category
        best_category, best_score = max(category_scores, key=lambda x: x[1])
        
        # Find subcategories (other categories with high similarity)
        subcategories = [
            category for category, score in category_scores
            if score > self.similarity_threshold and category != best_category
        ]
        
        # Detect features
        features = []
        for feature, patterns in self.features.items():
            feature_text = ' '.join(patterns)
            if self._calculate_similarity(endpoint_text, feature_text) > self.similarity_threshold:
                features.append(feature)
        
        return APICategory(
            name=best_category,
            confidence=best_score,
            subcategories=subcategories,
            features=features
        )
        
    def find_relationships(
        self,
        endpoints: List[Dict]
    ) -> List[APIRelationship]:
        """Find relationships between API endpoints.
        
        Args:
            endpoints: List of endpoint dictionaries.
            
        Returns:
            List of discovered relationships.
        """
        relationships = []
        
        # Clear existing graph
        self.relationship_graph.clear()
        
        # Add nodes
        for endpoint in endpoints:
            node_id = f"{endpoint['method']}_{endpoint['path']}"
            self.relationship_graph.add_node(
                node_id,
                data=endpoint
            )
        
        # Analyze each pair of endpoints
        for i, endpoint1 in enumerate(endpoints):
            node1 = f"{endpoint1['method']}_{endpoint1['path']}"
            
            for endpoint2 in endpoints[i+1:]:
                node2 = f"{endpoint2['method']}_{endpoint2['path']}"
                
                # Skip self-relationships
                if node1 == node2:
                    continue
                
                # Check for dependencies
                if self._has_dependency(endpoint1, endpoint2):
                    relationship = APIRelationship(
                        source=node1,
                        target=node2,
                        relationship_type='depends_on',
                        confidence=0.9,
                        description=f"{node1} requires {node2}"
                    )
                    relationships.append(relationship)
                    self.relationship_graph.add_edge(
                        node1, node2,
                        relationship=relationship
                    )
                
                # Check for similarities
                similarity = self._calculate_endpoint_similarity(endpoint1, endpoint2)
                if similarity > self.similarity_threshold:
                    relationship = APIRelationship(
                        source=node1,
                        target=node2,
                        relationship_type='similar_to',
                        confidence=similarity,
                        description=f"{node1} is similar to {node2}"
                    )
                    relationships.append(relationship)
                    self.relationship_graph.add_edge(
                        node1, node2,
                        relationship=relationship
                    )
                
                # Check for alternatives
                if self._are_alternatives(endpoint1, endpoint2):
                    relationship = APIRelationship(
                        source=node1,
                        target=node2,
                        relationship_type='alternative_to',
                        confidence=0.8,
                        description=f"{node1} is an alternative to {node2}"
                    )
                    relationships.append(relationship)
                    self.relationship_graph.add_edge(
                        node1, node2,
                        relationship=relationship
                    )
        
        return relationships
        
    def _has_dependency(self, endpoint1: Dict, endpoint2: Dict) -> bool:
        """Check if endpoint1 depends on endpoint2.
        
        Args:
            endpoint1: First endpoint dictionary.
            endpoint2: Second endpoint dictionary.
            
        Returns:
            True if dependency exists, False otherwise.
        """
        # Check if endpoint2's path appears in endpoint1's description
        path2 = endpoint2['path']
        desc1 = endpoint1.get('description', '').lower()
        
        if path2.lower() in desc1:
            return True
            
        # Check for ID parameter patterns
        path1_parts = endpoint1['path'].split('/')
        path2_parts = endpoint2['path'].split('/')
        
        for part1, part2 in zip(path1_parts, path2_parts):
            if part1.startswith('{') and part2.endswith('_id}'):
                return True
                
        return False
        
    def _calculate_endpoint_similarity(self, endpoint1: Dict, endpoint2: Dict) -> float:
        """Calculate similarity between two endpoints.
        
        Args:
            endpoint1: First endpoint dictionary.
            endpoint2: Second endpoint dictionary.
            
        Returns:
            Similarity score between 0 and 1.
        """
        # Combine relevant information for each endpoint
        text1 = f"""
        Path: {endpoint1['path']}
        Method: {endpoint1['method']}
        Description: {endpoint1.get('description', '')}
        Summary: {endpoint1.get('summary', '')}
        """
        
        text2 = f"""
        Path: {endpoint2['path']}
        Method: {endpoint2['method']}
        Description: {endpoint2.get('description', '')}
        Summary: {endpoint2.get('summary', '')}
        """
        
        return self._calculate_similarity(text1, text2)
        
    def _are_alternatives(self, endpoint1: Dict, endpoint2: Dict) -> bool:
        """Check if endpoints are alternatives to each other.
        
        Args:
            endpoint1: First endpoint dictionary.
            endpoint2: Second endpoint dictionary.
            
        Returns:
            True if endpoints are alternatives, False otherwise.
        """
        # Check if endpoints have similar paths but different methods
        path1 = re.sub(r'{[^}]+}', '*', endpoint1['path'])
        path2 = re.sub(r'{[^}]+}', '*', endpoint2['path'])
        
        if path1 == path2 and endpoint1['method'] != endpoint2['method']:
            return True
            
        # Check for bulk vs. single operation patterns
        if (
            endpoint1['path'] + 's' == endpoint2['path'] or
            endpoint2['path'] + 's' == endpoint1['path']
        ):
            return True
            
        return False
        
    def get_api_dependencies(self, endpoint: Dict) -> List[str]:
        """Get list of endpoints that this endpoint depends on.
        
        Args:
            endpoint: Endpoint dictionary.
            
        Returns:
            List of dependent endpoint paths.
        """
        node_id = f"{endpoint['method']}_{endpoint['path']}"
        if node_id not in self.relationship_graph:
            return []
            
        dependencies = []
        for _, target, data in self.relationship_graph.out_edges(node_id, data=True):
            if data['relationship'].relationship_type == 'depends_on':
                dependencies.append(target)
                
        return dependencies
        
    def get_similar_endpoints(self, endpoint: Dict) -> List[str]:
        """Get list of endpoints similar to this one.
        
        Args:
            endpoint: Endpoint dictionary.
            
        Returns:
            List of similar endpoint paths.
        """
        node_id = f"{endpoint['method']}_{endpoint['path']}"
        if node_id not in self.relationship_graph:
            return []
            
        similar = []
        for _, target, data in self.relationship_graph.out_edges(node_id, data=True):
            if data['relationship'].relationship_type == 'similar_to':
                similar.append(target)
                
        return similar
        
    def get_alternative_endpoints(self, endpoint: Dict) -> List[str]:
        """Get list of alternative endpoints.
        
        Args:
            endpoint: Endpoint dictionary.
            
        Returns:
            List of alternative endpoint paths.
        """
        node_id = f"{endpoint['method']}_{endpoint['path']}"
        if node_id not in self.relationship_graph:
            return []
            
        alternatives = []
        for _, target, data in self.relationship_graph.out_edges(node_id, data=True):
            if data['relationship'].relationship_type == 'alternative_to':
                alternatives.append(target)
                
        return alternatives 