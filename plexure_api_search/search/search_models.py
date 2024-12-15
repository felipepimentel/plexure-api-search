"""Search result data structures."""

from dataclasses import asdict, dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional


@dataclass
class SearchResult:
    """Represents a single search result with metadata and evaluation information."""

    # Core metadata
    score: float
    method: str
    path: str
    description: str
    api_name: str
    api_version: str
    parameters: List[str]
    responses: List[str]
    tags: List[str]
    requires_auth: bool
    deprecated: bool

    # Additional metadata
    summary: str = ""
    endpoint_id: str = ""

    # Evaluation fields
    rank: int = 0
    is_relevant: Optional[bool] = None
    evaluation_timestamp: Optional[datetime] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format for serialization."""
        data = asdict(self)
        if self.evaluation_timestamp:
            data["evaluation_timestamp"] = self.evaluation_timestamp.isoformat()
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SearchResult":
        """Create from dictionary format."""
        if "evaluation_timestamp" in data and data["evaluation_timestamp"]:
            data["evaluation_timestamp"] = datetime.fromisoformat(
                data["evaluation_timestamp"]
            )
        return cls(**data)

    @classmethod
    def from_pinecone_match(
        cls, match: Dict[str, Any], rank: int = 0
    ) -> "SearchResult":
        """Create from Pinecone match data.

        Args:
            match: Raw match data from Pinecone
            rank: Optional rank position in results

        Returns:
            SearchResult instance
        """
        score = (1 + match.get("score", 0)) / 2  # Normalize to 0-1
        metadata = match.get("metadata", {})

        return cls(
            score=score,
            method=metadata.get("method", "N/A"),
            path=metadata.get("path", "N/A"),
            description=metadata.get("description", ""),
            api_name=metadata.get("api_name", "N/A"),
            api_version=metadata.get("api_version", "N/A"),
            parameters=metadata.get("parameters", []),
            responses=metadata.get("responses", []),
            tags=metadata.get("tags", []),
            requires_auth=metadata.get("requires_auth", "false") == "true",
            deprecated=metadata.get("deprecated", "false") == "true",
            summary=metadata.get("summary", ""),
            endpoint_id=metadata.get("id", ""),
            rank=rank,
            evaluation_timestamp=datetime.now(),
        )


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
