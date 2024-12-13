"""Data validation and quality metrics."""

import re
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
from rich.console import Console


@dataclass
class QualityMetrics:
    """Quality metrics for API data."""
    
    completeness: float
    consistency: float
    accuracy: float
    uniqueness: float
    timestamp: datetime


class DataValidator:
    """Validates API data quality."""
    
    def __init__(self):
        """Initialize data validator."""
        self.console = Console()
        self.required_fields = {
            "path",
            "method",
            "description",
            "version"
        }
        self.valid_methods = {
            "GET",
            "POST",
            "PUT",
            "DELETE",
            "PATCH"
        }
        
    def validate_api_data(
        self,
        data: List[Dict],
        strict: bool = False
    ) -> Tuple[bool, List[str]]:
        """Validate API data.
        
        Args:
            data: List of API data dictionaries.
            strict: Whether to enforce strict validation.
            
        Returns:
            Tuple of (is_valid, error_messages).
        """
        if not data:
            return False, ["No data to validate"]
            
        errors = []
        
        for i, item in enumerate(data):
            item_errors = self._validate_item(item, strict)
            if item_errors:
                errors.extend([
                    f"Item {i}: {error}"
                    for error in item_errors
                ])
                
        return len(errors) == 0, errors
        
    def calculate_quality_metrics(self, data: List[Dict]) -> QualityMetrics:
        """Calculate quality metrics for API data.
        
        Args:
            data: List of API data dictionaries.
            
        Returns:
            Quality metrics.
        """
        if not data:
            return QualityMetrics(
                completeness=0.0,
                consistency=0.0,
                accuracy=0.0,
                uniqueness=0.0,
                timestamp=datetime.now()
            )
            
        # Calculate completeness
        completeness_scores = []
        for item in data:
            present_fields = self.required_fields & set(item.keys())
            completeness_scores.append(len(present_fields) / len(self.required_fields))
            
        completeness = np.mean(completeness_scores)
        
        # Calculate consistency
        method_consistency = self._calculate_consistency(
            [item.get("method", "").upper() for item in data]
        )
        version_consistency = self._calculate_consistency(
            [str(item.get("version", "")) for item in data]
        )
        
        consistency = np.mean([method_consistency, version_consistency])
        
        # Calculate accuracy
        accuracy_scores = []
        for item in data:
            scores = []
            
            # Check path format
            if "path" in item:
                scores.append(bool(re.match(r'^/[\w\-/{}]*$', item["path"])))
                
            # Check method validity
            if "method" in item:
                scores.append(item["method"].upper() in self.valid_methods)
                
            # Check version format
            if "version" in item:
                scores.append(bool(re.match(r'^\d+(\.\d+)*$', str(item["version"]))))
                
            accuracy_scores.append(np.mean(scores) if scores else 0.0)
            
        accuracy = np.mean(accuracy_scores)
        
        # Calculate uniqueness
        unique_paths = len({
            item.get("path")
            for item in data
            if "path" in item
        })
        uniqueness = unique_paths / len(data)
        
        return QualityMetrics(
            completeness=completeness,
            consistency=consistency,
            accuracy=accuracy,
            uniqueness=uniqueness,
            timestamp=datetime.now()
        )
        
    def _validate_item(self, item: Dict, strict: bool) -> List[str]:
        """Validate single API data item.
        
        Args:
            item: API data dictionary.
            strict: Whether to enforce strict validation.
            
        Returns:
            List of error messages.
        """
        errors = []
        
        # Check required fields
        missing_fields = self.required_fields - set(item.keys())
        if missing_fields:
            errors.append(f"Missing required fields: {', '.join(missing_fields)}")
            
        # Validate path
        if "path" in item:
            if not isinstance(item["path"], str):
                errors.append("Path must be a string")
            elif not re.match(r'^/[\w\-/{}]*$', item["path"]):
                errors.append("Invalid path format")
                
        # Validate method
        if "method" in item:
            if not isinstance(item["method"], str):
                errors.append("Method must be a string")
            elif item["method"].upper() not in self.valid_methods:
                errors.append(f"Invalid method: {item['method']}")
                
        # Validate version
        if "version" in item:
            version_str = str(item["version"])
            if not re.match(r'^\d+(\.\d+)*$', version_str):
                errors.append(f"Invalid version format: {version_str}")
                
        # Validate description
        if "description" in item:
            if not isinstance(item["description"], str):
                errors.append("Description must be a string")
            elif strict and len(item["description"]) < 10:
                errors.append("Description too short")
                
        return errors
        
    def _calculate_consistency(self, values: List[str]) -> float:
        """Calculate consistency score for a list of values.
        
        Args:
            values: List of values to check consistency.
            
        Returns:
            Consistency score between 0 and 1.
        """
        if not values:
            return 0.0
            
        # Count occurrences
        value_counts = {}
        for value in values:
            value_counts[value] = value_counts.get(value, 0) + 1
            
        # Find most common value
        most_common_count = max(value_counts.values())
        
        return most_common_count / len(values) 