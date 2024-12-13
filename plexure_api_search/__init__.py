"""Advanced API search with triple vector embeddings and contextual boosting."""

from .searcher import APISearcher
from .embeddings import TripleVector, TripleVectorizer
from .boosting import ContextualBooster, WeightProfile
from .understanding import ZeroShotUnderstanding, APICategory, APIRelationship
from .expansion import QueryExpander, ExpandedQuery
from .quality import QualityMetrics, SearchResult, SearchEvaluation

__version__ = "2.0.0"

__all__ = [
    "APISearcher",
    "TripleVector",
    "TripleVectorizer",
    "ContextualBooster",
    "WeightProfile",
    "ZeroShotUnderstanding",
    "APICategory",
    "APIRelationship",
    "QueryExpander",
    "ExpandedQuery",
    "QualityMetrics",
    "SearchResult",
    "SearchEvaluation"
]
