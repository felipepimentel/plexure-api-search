"""Query template matching module."""

import logging
import re
from typing import Dict, List, Optional, Tuple

import numpy as np
from sentence_transformers import util

from ..config import config_instance
from ..monitoring.metrics import MetricsManager
from ..services.models import ModelService

logger = logging.getLogger(__name__)

class QueryTemplate:
    """Query template with parameters."""

    def __init__(
        self,
        template: str,
        parameters: Optional[Dict[str, str]] = None,
        score: float = 0.0,
    ):
        """Initialize query template.
        
        Args:
            template: Template string
            parameters: Template parameters
            score: Match score
        """
        self.template = template
        self.parameters = parameters or {}
        self.score = score

class TemplateManager:
    """Query template manager."""

    def __init__(self):
        """Initialize template manager."""
        self.model_service = ModelService()
        self.metrics = MetricsManager()
        self.initialized = False
        self.templates = self._get_default_templates()

    def initialize(self) -> None:
        """Initialize template manager."""
        if self.initialized:
            return

        try:
            # Initialize model service
            self.model_service.initialize()
            self.initialized = True
            logger.info("Template manager initialized")

        except Exception as e:
            logger.error(f"Failed to initialize template manager: {e}")
            raise

    def cleanup(self) -> None:
        """Clean up template manager."""
        self.model_service.cleanup()
        self.initialized = False
        logger.info("Template manager cleaned up")

    def match(
        self,
        query: str,
        min_score: float = 0.7,
    ) -> List[QueryTemplate]:
        """Match query against templates.
        
        Args:
            query: Search query
            min_score: Minimum similarity score
            
        Returns:
            List of matching templates
        """
        if not self.initialized:
            self.initialize()

        try:
            # Start timer
            start_time = self.metrics.start_timer()

            # Encode query
            query_embedding = self.model_service.encode(query)

            # Match against templates
            matches = []
            for template in self.templates:
                # Calculate similarity
                template_embedding = self.model_service.encode(template)
                similarity = util.pytorch_cos_sim(
                    query_embedding,
                    template_embedding,
                )[0][0].item()

                # Add if above threshold
                if similarity >= min_score:
                    # Extract parameters
                    parameters = self._extract_parameters(query, template)
                    matches.append(
                        QueryTemplate(
                            template=template,
                            parameters=parameters,
                            score=similarity,
                        )
                    )

            # Sort by score
            matches.sort(key=lambda x: x.score, reverse=True)

            # Stop timer
            self.metrics.stop_timer(
                start_time,
                "template_matching",
                {"query": query},
            )

            return matches

        except Exception as e:
            logger.error(f"Failed to match templates: {e}")
            self.metrics.increment(
                "template_errors",
                {"query": query},
            )
            return []

    def _extract_parameters(
        self,
        query: str,
        template: str,
    ) -> Dict[str, str]:
        """Extract parameters from query based on template.
        
        Args:
            query: Search query
            template: Template string
            
        Returns:
            Extracted parameters
        """
        parameters = {}

        # Convert template to regex pattern
        pattern = template
        for param in ["resource", "action", "filter", "sort"]:
            pattern = pattern.replace(f"{{{param}}}", f"(?P<{param}>[\\w\\s]+)")
        pattern = f"^{pattern}$"

        # Try to match
        match = re.match(pattern, query, re.IGNORECASE)
        if match:
            parameters = match.groupdict()

        return parameters

    def _get_default_templates(self) -> List[str]:
        """Get default query templates.
        
        Returns:
            List of template strings
        """
        return [
            "find {resource}",
            "search for {resource}",
            "show me {resource}",
            "list {resource}",
            "get {resource}",
            "{action} {resource}",
            "filter {resource} by {filter}",
            "sort {resource} by {sort}",
            "{resource} with {filter}",
            "{resource} ordered by {sort}",
        ]

# Global instance
template_manager = TemplateManager() 