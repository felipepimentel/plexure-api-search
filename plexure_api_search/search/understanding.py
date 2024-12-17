"""Natural language understanding for API search."""

import logging
import time
from datetime import datetime
from typing import List, Optional, Dict, Any
import numpy as np

from sentence_transformers import SentenceTransformer
from ..monitoring.events import Event, EventType, event_manager

logger = logging.getLogger(__name__)

class ZeroShotUnderstanding:
    """Zero-shot understanding of API queries and endpoints."""

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
        
        # Pre-defined intents with examples
        self.intents = {
            "search": [
                "find", "search", "look for", "get", "list",
                "show", "display", "retrieve"
            ],
            "create": [
                "create", "add", "new", "register", "insert",
                "make", "generate"
            ],
            "update": [
                "update", "modify", "change", "edit", "patch",
                "alter"
            ],
            "delete": [
                "delete", "remove", "destroy", "clear", "erase"
            ],
            "authenticate": [
                "login", "authenticate", "sign in", "token",
                "credentials", "auth"
            ]
        }

    def get_categories(self, text: str) -> List[str]:
        """Get matching categories for text.
        
        Args:
            text: Input text to categorize
            
        Returns:
            List of matching category names
        """
        try:
            # Get text embedding
            text_embedding = self.model.encode(text, convert_to_numpy=True)
            
            # Get embeddings for all category examples
            matching_categories = set()
            for category, examples in self.categories.items():
                # Get embeddings for examples
                example_embeddings = self.model.encode(examples, convert_to_numpy=True)
                
                # Compute similarities
                similarities = []
                for example_embedding in example_embeddings:
                    similarity = np.dot(text_embedding, example_embedding) / (
                        np.linalg.norm(text_embedding) * np.linalg.norm(example_embedding)
                    )
                    similarities.append(similarity)
                
                # If any example is above threshold, add category
                if max(similarities) > self.similarity_threshold:
                    matching_categories.add(category)
            
            return list(matching_categories)
            
        except Exception as e:
            logger.error(f"Failed to get categories: {e}")
            return []

    def classify_intent(self, query: str) -> str:
        """Classify the intent of a search query.

        Args:
            query: Search query

        Returns:
            Classified intent
        """
        try:
            # Get query embedding
            query_embedding = self.model.encode(query, convert_to_numpy=True)
            
            # Get embeddings for all intent examples
            best_intent = "search"  # Default intent
            best_score = 0.0
            
            for intent, examples in self.intents.items():
                # Get embeddings for examples
                example_embeddings = self.model.encode(examples, convert_to_numpy=True)
                
                # Compute similarities
                similarities = []
                for example_embedding in example_embeddings:
                    similarity = np.dot(query_embedding, example_embedding) / (
                        np.linalg.norm(query_embedding) * np.linalg.norm(example_embedding)
                    )
                    similarities.append(similarity)
                
                # Get max similarity for this intent
                max_similarity = max(similarities)
                
                # Update best intent if this one is better
                if max_similarity > best_score:
                    best_score = max_similarity
                    best_intent = intent
            
            return best_intent
            
        except Exception as e:
            logger.error(f"Failed to classify intent: {e}")
            return "search"  # Default intent on error