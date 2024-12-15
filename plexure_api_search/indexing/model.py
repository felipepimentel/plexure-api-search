"""API data quality metrics."""

from dataclasses import dataclass
from datetime import datetime


@dataclass
class APIDataQualityMetrics:
    """Quality metrics for API data validation."""

    completeness: float
    consistency: float
    accuracy: float
    uniqueness: float
    timestamp: datetime
