"""Search result models."""

import json
from dataclasses import dataclass, asdict, field
from typing import Any, Dict, List, Optional


@dataclass
class SearchResult:
    """Search result with metadata."""

    path: str
    method: str
    description: str
    summary: str
    api_name: str
    api_version: str
    parameters: List[str]
    responses: List[str]
    tags: List[str]
    requires_auth: bool
    deprecated: bool
    score: float
    rank: int
    enriched: Optional[Dict[str, Any]] = None
    id: str = field(default="")

    @classmethod
    def from_pinecone_match(cls, match: Dict[str, Any], rank: int) -> "SearchResult":
        """Create SearchResult from Pinecone match.

        Args:
            match: Pinecone match dictionary
            rank: Result rank

        Returns:
            SearchResult instance
        """
        metadata = match.get("metadata", {})
        
        # Deserialize enriched data if it exists
        enriched = None
        if "enriched" in metadata:
            try:
                enriched = json.loads(metadata["enriched"])
            except (json.JSONDecodeError, TypeError) as e:
                enriched = metadata.get("enriched")

        return cls(
            id=match.get("id", ""),
            path=metadata.get("path", ""),
            method=metadata.get("method", ""),
            description=metadata.get("description", ""),
            summary=metadata.get("summary", ""),
            api_name=metadata.get("api_name", ""),
            api_version=metadata.get("api_version", ""),
            parameters=metadata.get("parameters", []),
            responses=metadata.get("responses", []),
            tags=metadata.get("tags", []),
            requires_auth=metadata.get("requires_auth", False),
            deprecated=metadata.get("deprecated", False),
            score=match.get("score", 0.0),
            rank=rank,
            enriched=enriched,
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary.

        Returns:
            Dictionary representation
        """
        # Use dataclass asdict for base conversion
        result = asdict(self)
        
        # Ensure all values are JSON serializable
        if self.enriched is not None:
            if isinstance(self.enriched, str):
                try:
                    result["enriched"] = json.loads(self.enriched)
                except json.JSONDecodeError:
                    result["enriched"] = self.enriched
            else:
                result["enriched"] = self.enriched
                
        return result 