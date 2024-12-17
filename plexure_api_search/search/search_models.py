"""Search result data structures."""

from dataclasses import asdict, dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional


@dataclass
class SearchResult:
    """Represents a single search result with metadata and evaluation information."""

    # Core metadata
    endpoint: str
    method: str
    description: str
    score: float
    tags: List[str]
    parameters: List[Dict[str, str]]
    responses: Dict[str, Dict[str, str]]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format for serialization."""
        return {
            "endpoint": self.endpoint,
            "method": self.method,
            "description": self.description,
            "score": self.score,
            "tags": self.tags,
            "parameters": self.parameters,
            "responses": self.responses,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SearchResult":
        """Create from dictionary format."""
        return cls(**data)


@dataclass
class SearchContext:
    """Context information for a search query."""

    query: str
    query_type: str  # 'semantic', 'structural', 'parameter'
    timestamp: datetime
    user_feedback: Optional[float] = None  # 0 to 1 rating
    success_rate: Optional[float] = None


@dataclass
class WeightProfile:
    """Weight profile for different vector components."""

    semantic_weight: float
    structure_weight: float
    parameter_weight: float

    def to_dict(self) -> Dict[str, float]:
        """Convert to dictionary format."""
        return {
            "semantic": self.semantic_weight,
            "structure": self.structure_weight,
            "parameter": self.parameter_weight,
        }
