"""Business-focused API endpoint recommendations."""

from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional, Set
import logging
import numpy as np

from ..config import config_instance
from ..monitoring import metrics_manager
from ..analytics.usage_analytics import BusinessTrend, UsageMetrics
from ..documentation.business_docs import BusinessUseCase

logger = logging.getLogger(__name__)

@dataclass
class BusinessOpportunity:
    """Business opportunity recommendation."""

    title: str
    description: str
    potential_value: float
    implementation_effort: float
    time_to_value: str
    required_endpoints: List[str]
    success_metrics: List[str]
    risks: List[str]
    mitigation_steps: List[str]
    estimated_roi: float

@dataclass
class EndpointRecommendation:
    """API endpoint recommendation."""

    endpoint_id: str
    relevance_score: float
    business_value: float
    implementation_complexity: float
    opportunities: List[BusinessOpportunity]
    synergies: List[Dict[str, float]]
    prerequisites: List[str]
    next_steps: List[str]

class BusinessRecommender:
    """Generates business-focused API recommendations."""

    def __init__(self):
        """Initialize recommender."""
        self.metrics = metrics_manager
        self._init_patterns()

    def _init_patterns(self):
        """Initialize recommendation patterns."""
        self.business_patterns = {
            "revenue_generation": {
                "indicators": ["payment", "billing", "subscription", "pricing"],
                "value_multiplier": 1.5,
                "required_endpoints": ["payment", "customer", "product"],
                "success_metrics": [
                    "Revenue Growth",
                    "Customer Lifetime Value",
                    "Conversion Rate",
                    "Average Transaction Value",
                ],
            },
            "cost_reduction": {
                "indicators": ["automation", "optimization", "efficiency", "workflow"],
                "value_multiplier": 1.3,
                "required_endpoints": ["process", "workflow", "integration"],
                "success_metrics": [
                    "Cost Reduction",
                    "Process Efficiency",
                    "Resource Utilization",
                    "Error Rate Reduction",
                ],
            },
            "customer_experience": {
                "indicators": ["user", "profile", "preference", "personalization"],
                "value_multiplier": 1.4,
                "required_endpoints": ["user", "profile", "analytics"],
                "success_metrics": [
                    "Customer Satisfaction",
                    "User Engagement",
                    "Feature Adoption",
                    "Support Ticket Reduction",
                ],
            },
            "market_expansion": {
                "indicators": ["location", "language", "currency", "region"],
                "value_multiplier": 1.6,
                "required_endpoints": ["localization", "payment", "compliance"],
                "success_metrics": [
                    "Market Penetration",
                    "Geographic Revenue",
                    "Local User Growth",
                    "Regional Adoption",
                ],
            },
        }

    def get_recommendations(
        self,
        current_endpoint: str,
        usage_metrics: UsageMetrics,
        trends: List[BusinessTrend],
        similar_endpoints: List[str],
    ) -> List[EndpointRecommendation]:
        """Get business-focused endpoint recommendations.

        Args:
            current_endpoint: Current endpoint ID
            usage_metrics: Usage metrics
            trends: Business trends
            similar_endpoints: Similar endpoints

        Returns:
            List of endpoint recommendations
        """
        try:
            recommendations = []
            
            # Analyze current endpoint
            current_patterns = self._identify_patterns(current_endpoint)
            current_value = self._calculate_business_value(
                current_endpoint, usage_metrics, current_patterns
            )
            
            # Analyze similar endpoints
            for endpoint in similar_endpoints:
                try:
                    # Calculate relevance and value
                    patterns = self._identify_patterns(endpoint)
                    relevance = self._calculate_relevance(
                        current_patterns, patterns
                    )
                    value = self._calculate_business_value(
                        endpoint, usage_metrics, patterns
                    )
                    
                    # Generate opportunities
                    opportunities = self._identify_opportunities(
                        endpoint, patterns, trends
                    )
                    
                    # Calculate implementation complexity
                    complexity = self._estimate_complexity(
                        endpoint, patterns, current_endpoint
                    )
                    
                    # Find endpoint synergies
                    synergies = self._find_synergies(
                        endpoint, similar_endpoints, patterns
                    )
                    
                    # Get implementation prerequisites
                    prerequisites = self._get_prerequisites(
                        endpoint, patterns, current_endpoint
                    )
                    
                    # Generate next steps
                    next_steps = self._generate_next_steps(
                        endpoint, opportunities, complexity
                    )
                    
                    # Create recommendation
                    recommendation = EndpointRecommendation(
                        endpoint_id=endpoint,
                        relevance_score=relevance,
                        business_value=value,
                        implementation_complexity=complexity,
                        opportunities=opportunities,
                        synergies=synergies,
                        prerequisites=prerequisites,
                        next_steps=next_steps,
                    )
                    
                    recommendations.append(recommendation)

                except Exception as e:
                    logger.error(f"Failed to analyze endpoint {endpoint}: {e}")
                    continue
            
            # Sort by business value and relevance
            recommendations.sort(
                key=lambda x: (x.business_value, x.relevance_score),
                reverse=True
            )
            
            return recommendations

        except Exception as e:
            logger.error(f"Failed to get recommendations: {e}")
            return []

    def _identify_patterns(self, endpoint_id: str) -> Set[str]:
        """Identify business patterns in endpoint."""
        patterns = set()
        
        try:
            endpoint_lower = endpoint_id.lower()
            
            for pattern, config in self.business_patterns.items():
                if any(ind in endpoint_lower for ind in config["indicators"]):
                    patterns.add(pattern)

        except Exception as e:
            logger.error(f"Failed to identify patterns: {e}")
        
        return patterns

    def _calculate_business_value(
        self,
        endpoint_id: str,
        metrics: UsageMetrics,
        patterns: Set[str],
    ) -> float:
        """Calculate business value score."""
        try:
            base_value = metrics.business_value
            
            # Apply pattern multipliers
            for pattern in patterns:
                if pattern in self.business_patterns:
                    base_value *= self.business_patterns[pattern]["value_multiplier"]
            
            return min(max(base_value, 0.0), 1.0)

        except Exception as e:
            logger.error(f"Failed to calculate business value: {e}")
            return 0.0

    def _calculate_relevance(
        self,
        current_patterns: Set[str],
        target_patterns: Set[str],
    ) -> float:
        """Calculate pattern relevance score."""
        try:
            if not current_patterns or not target_patterns:
                return 0.0
            
            # Calculate Jaccard similarity
            intersection = len(current_patterns & target_patterns)
            union = len(current_patterns | target_patterns)
            
            return intersection / union if union > 0 else 0.0

        except Exception as e:
            logger.error(f"Failed to calculate relevance: {e}")
            return 0.0

    def _identify_opportunities(
        self,
        endpoint_id: str,
        patterns: Set[str],
        trends: List[BusinessTrend],
    ) -> List[BusinessOpportunity]:
        """Identify business opportunities."""
        opportunities = []
        
        try:
            # Analyze each pattern
            for pattern in patterns:
                if pattern not in self.business_patterns:
                    continue
                
                config = self.business_patterns[pattern]
                
                # Check for relevant trends
                relevant_trends = [
                    t for t in trends
                    if endpoint_id in t.affected_endpoints
                    and t.impact_score > 0.5
                ]
                
                if relevant_trends:
                    # Create opportunity based on trends
                    opportunity = BusinessOpportunity(
                        title=f"{pattern.title()} Opportunity",
                        description=self._generate_opportunity_description(
                            pattern, relevant_trends
                        ),
                        potential_value=max(t.impact_score for t in relevant_trends),
                        implementation_effort=self._estimate_implementation_effort(
                            pattern, config
                        ),
                        time_to_value=self._estimate_time_to_value(pattern),
                        required_endpoints=config["required_endpoints"],
                        success_metrics=config["success_metrics"],
                        risks=self._identify_risks(pattern),
                        mitigation_steps=self._generate_mitigation_steps(pattern),
                        estimated_roi=self._estimate_roi(
                            pattern,
                            max(t.impact_score for t in relevant_trends),
                        ),
                    )
                    
                    opportunities.append(opportunity)

        except Exception as e:
            logger.error(f"Failed to identify opportunities: {e}")
        
        return opportunities

    def _generate_opportunity_description(
        self,
        pattern: str,
        trends: List[BusinessTrend],
    ) -> str:
        """Generate opportunity description."""
        try:
            trend_impacts = [
                f"{t.description} (Impact: {t.impact_score:.1%})"
                for t in trends[:3]
            ]
            
            return f"""
Business Opportunity: {pattern.replace('_', ' ').title()}

Market Trends:
{chr(10).join(f'- {impact}' for impact in trend_impacts)}

This opportunity aligns with current market trends and business patterns,
offering significant potential for growth and value creation.
"""
        except Exception as e:
            logger.error(f"Failed to generate description: {e}")
            return ""

    def _estimate_implementation_effort(
        self,
        pattern: str,
        config: Dict,
    ) -> float:
        """Estimate implementation effort."""
        try:
            # Base effort by number of required endpoints
            base_effort = len(config["required_endpoints"]) * 0.2
            
            # Adjust for pattern complexity
            if pattern == "revenue_generation":
                base_effort *= 1.2  # Higher complexity
            elif pattern == "cost_reduction":
                base_effort *= 0.8  # Lower complexity
            
            return min(max(base_effort, 0.1), 1.0)

        except Exception as e:
            logger.error(f"Failed to estimate effort: {e}")
            return 0.5

    def _estimate_time_to_value(self, pattern: str) -> str:
        """Estimate time to realize business value."""
        estimates = {
            "revenue_generation": "2-3 months",
            "cost_reduction": "1-2 months",
            "customer_experience": "2-4 months",
            "market_expansion": "3-6 months",
        }
        return estimates.get(pattern, "2-4 months")

    def _identify_risks(self, pattern: str) -> List[str]:
        """Identify implementation risks."""
        common_risks = [
            "Integration complexity",
            "Resource availability",
            "Technical dependencies",
        ]
        
        pattern_risks = {
            "revenue_generation": [
                "Market acceptance",
                "Pricing strategy",
                "Competition response",
            ],
            "cost_reduction": [
                "Process disruption",
                "Change management",
                "Quality maintenance",
            ],
            "customer_experience": [
                "User adoption",
                "Performance impact",
                "Support requirements",
            ],
            "market_expansion": [
                "Market readiness",
                "Regulatory compliance",
                "Cultural adaptation",
            ],
        }
        
        return common_risks + pattern_risks.get(pattern, [])

    def _generate_mitigation_steps(self, pattern: str) -> List[str]:
        """Generate risk mitigation steps."""
        common_steps = [
            "Detailed planning and analysis",
            "Phased implementation approach",
            "Regular progress monitoring",
        ]
        
        pattern_steps = {
            "revenue_generation": [
                "Market validation",
                "Competitive analysis",
                "Pricing optimization",
            ],
            "cost_reduction": [
                "Process baseline measurement",
                "Impact analysis",
                "Fallback planning",
            ],
            "customer_experience": [
                "User feedback collection",
                "Performance testing",
                "Support team training",
            ],
            "market_expansion": [
                "Market research",
                "Compliance review",
                "Local partnership development",
            ],
        }
        
        return common_steps + pattern_steps.get(pattern, [])

    def _estimate_roi(self, pattern: str, impact_score: float) -> float:
        """Estimate ROI for opportunity."""
        try:
            # Base ROI multipliers
            multipliers = {
                "revenue_generation": 2.0,
                "cost_reduction": 1.5,
                "customer_experience": 1.3,
                "market_expansion": 1.8,
            }
            
            base_roi = impact_score * multipliers.get(pattern, 1.0)
            
            # Adjust for implementation time
            time_factor = {
                "revenue_generation": 0.8,  # Faster returns
                "cost_reduction": 0.9,
                "customer_experience": 0.7,
                "market_expansion": 0.6,  # Slower returns
            }
            
            return base_roi * time_factor.get(pattern, 0.8)

        except Exception as e:
            logger.error(f"Failed to estimate ROI: {e}")
            return 0.0

    def _estimate_complexity(
        self,
        endpoint_id: str,
        patterns: Set[str],
        current_endpoint: str,
    ) -> float:
        """Estimate implementation complexity."""
        try:
            # Base complexity by number of patterns
            base_complexity = len(patterns) * 0.2
            
            # Adjust for pattern types
            if "revenue_generation" in patterns:
                base_complexity += 0.2
            if "market_expansion" in patterns:
                base_complexity += 0.3
            
            # Adjust for similarity with current endpoint
            current_patterns = self._identify_patterns(current_endpoint)
            similarity = self._calculate_relevance(current_patterns, patterns)
            
            # Lower complexity if similar to current endpoint
            base_complexity *= (1 - similarity * 0.3)
            
            return min(max(base_complexity, 0.1), 1.0)

        except Exception as e:
            logger.error(f"Failed to estimate complexity: {e}")
            return 0.5

    def _find_synergies(
        self,
        endpoint_id: str,
        similar_endpoints: List[str],
        patterns: Set[str],
    ) -> List[Dict[str, float]]:
        """Find endpoint synergies."""
        synergies = []
        
        try:
            for other_endpoint in similar_endpoints:
                if other_endpoint == endpoint_id:
                    continue
                
                other_patterns = self._identify_patterns(other_endpoint)
                synergy_score = self._calculate_relevance(patterns, other_patterns)
                
                if synergy_score > 0.3:  # Minimum synergy threshold
                    synergies.append({
                        "endpoint": other_endpoint,
                        "score": synergy_score,
                        "type": self._determine_synergy_type(patterns, other_patterns),
                    })
            
            # Sort by synergy score
            synergies.sort(key=lambda x: x["score"], reverse=True)

        except Exception as e:
            logger.error(f"Failed to find synergies: {e}")
        
        return synergies

    def _determine_synergy_type(
        self,
        patterns1: Set[str],
        patterns2: Set[str],
    ) -> str:
        """Determine type of synergy between pattern sets."""
        if patterns1 == patterns2:
            return "complementary"
        elif patterns1 & patterns2:
            return "supplementary"
        else:
            return "independent"

    def _get_prerequisites(
        self,
        endpoint_id: str,
        patterns: Set[str],
        current_endpoint: str,
    ) -> List[str]:
        """Get implementation prerequisites."""
        prerequisites = set()
        
        try:
            # Add pattern-specific prerequisites
            for pattern in patterns:
                if pattern in self.business_patterns:
                    prerequisites.update(
                        self.business_patterns[pattern]["required_endpoints"]
                    )
            
            # Add technical prerequisites
            prerequisites.update([
                "API authentication",
                "Error handling",
                "Monitoring setup",
            ])
            
            # Add business prerequisites
            if "revenue_generation" in patterns:
                prerequisites.update([
                    "Billing system integration",
                    "Revenue tracking setup",
                ])
            if "market_expansion" in patterns:
                prerequisites.update([
                    "Market analysis",
                    "Compliance review",
                ])

        except Exception as e:
            logger.error(f"Failed to get prerequisites: {e}")
        
        return sorted(list(prerequisites))

    def _generate_next_steps(
        self,
        endpoint_id: str,
        opportunities: List[BusinessOpportunity],
        complexity: float,
    ) -> List[str]:
        """Generate implementation next steps."""
        steps = [
            "Review technical documentation",
            "Assess resource requirements",
            "Create implementation plan",
        ]
        
        try:
            # Add opportunity-specific steps
            for opportunity in opportunities:
                if opportunity.potential_value > 0.7:
                    steps.extend([
                        f"Analyze {opportunity.title} requirements",
                        f"Plan {opportunity.title} implementation",
                    ])
            
            # Add complexity-based steps
            if complexity > 0.7:
                steps.extend([
                    "Conduct technical feasibility study",
                    "Plan phased implementation",
                    "Set up extended testing",
                ])
            elif complexity > 0.4:
                steps.extend([
                    "Review implementation guides",
                    "Plan standard testing",
                ])

        except Exception as e:
            logger.error(f"Failed to generate next steps: {e}")
        
        return steps 