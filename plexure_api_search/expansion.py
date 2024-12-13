"""Smart query expansion and semantic variant generation."""

from typing import List, Dict, Set, Optional, Tuple
from dataclasses import dataclass
import spacy
from sentence_transformers import SentenceTransformer
import numpy as np
from collections import defaultdict


@dataclass
class ExpandedQuery:
    """Expanded query with variants and mappings."""

    original_query: str
    semantic_variants: List[str]
    technical_mappings: List[str]
    use_cases: List[str]
    weights: Dict[str, float]


class QueryExpander:
    """Handles smart query expansion and variant generation."""

    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        spacy_model: str = "en_core_web_sm"
    ):
        """Initialize query expander.

        Args:
            model_name: Name of the sentence transformer model.
            spacy_model: Name of the spaCy model for NLP.
        """
        self.model = SentenceTransformer(model_name)
        self.nlp = spacy.load(spacy_model)

        # Technical term mappings
        self.technical_mappings = {
            "get": ["retrieve", "fetch", "read", "list", "query"],
            "create": ["post", "add", "insert", "register", "new"],
            "update": ["put", "modify", "change", "edit", "patch"],
            "delete": ["remove", "destroy", "erase"],
            "auth": ["authentication", "authorization", "login", "token"],
            "user": ["account", "profile", "member", "client"],
            "search": ["find", "query", "filter", "lookup"],
            "list": ["all", "many", "multiple", "batch"],
            "upload": ["send", "submit", "transfer", "push"],
            "download": ["receive", "fetch", "pull", "get"],
        }

        # Common use case patterns
        self.use_case_patterns = {
            "authentication": [
                "login user",
                "get authentication token",
                "verify credentials",
                "refresh token"
            ],
            "user_management": [
                "create new user",
                "update user profile",
                "change password",
                "delete account"
            ],
            "data_operations": [
                "search records",
                "filter results",
                "sort items",
                "paginate list"
            ],
            "file_operations": [
                "upload file",
                "download document",
                "delete image",
                "update attachment"
            ]
        }

    def expand_query(self, query: str) -> ExpandedQuery:
        """Expand a query with variants and mappings.

        Args:
            query: Original search query.

        Returns:
            ExpandedQuery with variants and mappings.
        """
        # Process query with spaCy
        doc = self.nlp(query)

        # Generate semantic variants
        semantic_variants = self._generate_semantic_variants(doc)

        # Generate technical mappings
        technical_mappings = self._generate_technical_mappings(doc)

        # Find relevant use cases
        use_cases = self._find_relevant_use_cases(query)

        # Calculate weights
        weights = self._calculate_variant_weights(
            query,
            semantic_variants,
            technical_mappings,
            use_cases
        )

        return ExpandedQuery(
            original_query=query,
            semantic_variants=semantic_variants,
            technical_mappings=technical_mappings,
            use_cases=use_cases,
            weights=weights
        )

    def _generate_semantic_variants(self, doc: spacy.tokens.Doc) -> List[str]:
        """Generate semantic variants of the query.

        Args:
            doc: spaCy Doc object of the query.

        Returns:
            List of semantic variants.
        """
        variants = set()

        # Add original tokens
        base_tokens = [token.text.lower() for token in doc if not token.is_stop]

        # Add lemmatized versions
        lemmas = [token.lemma_.lower() for token in doc if not token.is_stop]

        # Add noun chunks
        chunks = [chunk.text.lower() for chunk in doc.noun_chunks]

        # Combine variations
        for token in base_tokens:
            # Add singular/plural forms
            if token.endswith('s'):
                variants.add(token[:-1])
            else:
                variants.add(token + 's')

            # Add common prefixes
            variants.add(f"get_{token}")
            variants.add(f"find_{token}")
            variants.add(f"search_{token}")

        # Add all variations
        variants.update(base_tokens)
        variants.update(lemmas)
        variants.update(chunks)

        return list(variants)

    def _generate_technical_mappings(self, doc: spacy.tokens.Doc) -> List[str]:
        """Generate technical mappings for the query.

        Args:
            doc: spaCy Doc object of the query.

        Returns:
            List of technical mappings.
        """
        mappings = set()

        # Look up technical mappings for each token
        for token in doc:
            word = token.text.lower()
            if word in self.technical_mappings:
                mappings.update(self.technical_mappings[word])

        # Generate compound mappings
        for chunk in doc.noun_chunks:
            # Add REST-style paths
            path = chunk.text.lower().replace(" ", "-")
            mappings.add(f"GET /{path}")
            mappings.add(f"POST /{path}")
            mappings.add(f"PUT /{path}")
            mappings.add(f"DELETE /{path}")

            # Add camelCase variations
            camel_case = "".join(word.capitalize() for word in chunk.text.split())
            mappings.add(camel_case)

        return list(mappings)

    def _find_relevant_use_cases(self, query: str) -> List[str]:
        """Find relevant use cases for the query.

        Args:
            query: Original search query.

        Returns:
            List of relevant use cases.
        """
        query_embedding = self.model.encode(query)

        relevant_cases = []
        for category, patterns in self.use_case_patterns.items():
            # Calculate similarity with each pattern
            for pattern in patterns:
                pattern_embedding = self.model.encode(pattern)
                similarity = np.dot(query_embedding, pattern_embedding)

                if similarity > 0.7:  # Similarity threshold
                    relevant_cases.append(pattern)

        return relevant_cases

    def _calculate_variant_weights(
        self,
        original_query: str,
        semantic_variants: List[str],
        technical_mappings: List[str],
        use_cases: List[str]
    ) -> Dict[str, float]:
        """Calculate weights for each query variant.

        Args:
            original_query: Original search query.
            semantic_variants: List of semantic variants.
            technical_mappings: List of technical mappings.
            use_cases: List of relevant use cases.

        Returns:
            Dictionary of variant weights.
        """
        weights = {}
        query_embedding = self.model.encode(original_query)

        # Calculate weights for semantic variants
        for variant in semantic_variants:
            variant_embedding = self.model.encode(variant)
            similarity = float(np.dot(query_embedding, variant_embedding))
            weights[variant] = similarity

        # Calculate weights for technical mappings
        for mapping in technical_mappings:
            mapping_embedding = self.model.encode(mapping)
            similarity = float(np.dot(query_embedding, mapping_embedding))
            weights[mapping] = similarity * 0.8  # Slightly lower weight for technical mappings

        # Calculate weights for use cases
        for use_case in use_cases:
            use_case_embedding = self.model.encode(use_case)
            similarity = float(np.dot(query_embedding, use_case_embedding))
            weights[use_case] = similarity * 0.6  # Lower weight for use cases

        # Normalize weights
        total_weight = sum(weights.values())
        if total_weight > 0:
            weights = {k: v/total_weight for k, v in weights.items()}

        return weights

    def get_expanded_terms(
        self,
        query: str,
        min_weight: float = 0.1
    ) -> List[Tuple[str, float]]:
        """Get expanded search terms with weights.

        Args:
            query: Original search query.
            min_weight: Minimum weight threshold.

        Returns:
            List of (term, weight) tuples.
        """
        expanded = self.expand_query(query)

        # Combine all terms with weights
        terms = []
        for variant in expanded.semantic_variants:
            weight = expanded.weights.get(variant, 0.0)
            if weight >= min_weight:
                terms.append((variant, weight))

        for mapping in expanded.technical_mappings:
            weight = expanded.weights.get(mapping, 0.0)
            if weight >= min_weight:
                terms.append((mapping, weight))

        for use_case in expanded.use_cases:
            weight = expanded.weights.get(use_case, 0.0)
            if weight >= min_weight:
                terms.append((use_case, weight))

        # Sort by weight descending
        return sorted(terms, key=lambda x: x[1], reverse=True)
