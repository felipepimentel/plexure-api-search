"""Business-focused API workflow templates."""

from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional, Set
import logging

from ..config import config_instance
from ..monitoring.metrics_manager import metrics_manager
from ..analytics.usage_analytics import BusinessTrend, UsageMetrics
from ..recommendations.business_recommendations import BusinessOpportunity

logger = logging.getLogger(__name__)

@dataclass
class WorkflowStep:
    """API workflow step."""

    endpoint_id: str
    operation: str
    description: str
    parameters: Dict[str, str]
    expected_response: Dict[str, str]
    error_handling: List[str]
    business_impact: str
    success_criteria: List[str]

@dataclass
class WorkflowTemplate:
    """API workflow template."""

    title: str
    description: str
    business_value: str
    target_audience: str
    steps: List[WorkflowStep]
    prerequisites: List[str]
    success_metrics: List[str]
    estimated_effort: str
    implementation_guide: List[str]
    best_practices: List[str]

class BusinessWorkflowGenerator:
    """Generates business-focused API workflow templates."""

    def __init__(self):
        """Initialize workflow generator."""
        self.metrics = metrics_manager
        self._init_templates()

    def _init_templates(self):
        """Initialize workflow templates."""
        self.workflow_patterns = {
            "revenue_optimization": {
                "title": "Revenue Optimization Workflow",
                "description": "Optimize revenue through integrated payment and customer workflows",
                "value_prop": "Increase revenue through optimized payment flows",
                "required_endpoints": ["payment", "customer", "product"],
                "success_metrics": [
                    "Conversion Rate",
                    "Average Transaction Value",
                    "Customer Lifetime Value",
                ],
            },
            "customer_onboarding": {
                "title": "Customer Onboarding Workflow",
                "description": "Streamline customer onboarding and activation",
                "value_prop": "Reduce time-to-value for new customers",
                "required_endpoints": ["customer", "profile", "preferences"],
                "success_metrics": [
                    "Onboarding Completion Rate",
                    "Time to First Value",
                    "Initial Engagement Score",
                ],
            },
            "order_processing": {
                "title": "Order Processing Workflow",
                "description": "Automate order processing and fulfillment",
                "value_prop": "Increase operational efficiency",
                "required_endpoints": ["order", "inventory", "shipping"],
                "success_metrics": [
                    "Processing Time",
                    "Error Rate",
                    "Customer Satisfaction",
                ],
            },
            "analytics_integration": {
                "title": "Analytics Integration Workflow",
                "description": "Implement comprehensive analytics tracking",
                "value_prop": "Data-driven decision making",
                "required_endpoints": ["events", "analytics", "reporting"],
                "success_metrics": [
                    "Data Completeness",
                    "Insight Generation",
                    "Report Utilization",
                ],
            },
        }

    def generate_workflow(
        self,
        business_opportunity: BusinessOpportunity,
        available_endpoints: List[str],
        usage_metrics: Optional[UsageMetrics] = None,
    ) -> WorkflowTemplate:
        """Generate workflow template for business opportunity.

        Args:
            business_opportunity: Business opportunity
            available_endpoints: Available API endpoints
            usage_metrics: Optional usage metrics

        Returns:
            Workflow template
        """
        try:
            # Identify workflow pattern
            pattern = self._identify_workflow_pattern(business_opportunity)
            if not pattern:
                raise ValueError("No suitable workflow pattern found")
            
            # Get pattern config
            config = self.workflow_patterns[pattern]
            
            # Validate required endpoints
            available_set = set(available_endpoints)
            required_set = set(config["required_endpoints"])
            if not required_set.issubset(available_set):
                missing = required_set - available_set
                raise ValueError(f"Missing required endpoints: {missing}")
            
            # Generate workflow steps
            steps = self._generate_workflow_steps(
                pattern, config, available_endpoints
            )
            
            # Create workflow template
            template = WorkflowTemplate(
                title=config["title"],
                description=config["description"],
                business_value=config["value_prop"],
                target_audience=self._determine_target_audience(pattern),
                steps=steps,
                prerequisites=self._get_prerequisites(pattern, config),
                success_metrics=config["success_metrics"],
                estimated_effort=self._estimate_effort(pattern, steps),
                implementation_guide=self._generate_implementation_guide(
                    pattern, steps
                ),
                best_practices=self._get_best_practices(pattern),
            )
            
            return template

        except Exception as e:
            logger.error(f"Failed to generate workflow: {e}")
            raise

    def _identify_workflow_pattern(
        self,
        opportunity: BusinessOpportunity,
    ) -> Optional[str]:
        """Identify suitable workflow pattern."""
        try:
            # Match based on opportunity title and metrics
            title_lower = opportunity.title.lower()
            
            if "revenue" in title_lower:
                return "revenue_optimization"
            elif "customer" in title_lower:
                return "customer_onboarding"
            elif "order" in title_lower or "process" in title_lower:
                return "order_processing"
            elif "analytics" in title_lower or "insight" in title_lower:
                return "analytics_integration"
            
            # Match based on required endpoints
            endpoint_set = set(opportunity.required_endpoints)
            for pattern, config in self.workflow_patterns.items():
                required_set = set(config["required_endpoints"])
                if required_set.issubset(endpoint_set):
                    return pattern
            
            return None

        except Exception as e:
            logger.error(f"Failed to identify workflow pattern: {e}")
            return None

    def _generate_workflow_steps(
        self,
        pattern: str,
        config: Dict,
        available_endpoints: List[str],
    ) -> List[WorkflowStep]:
        """Generate workflow steps."""
        steps = []
        
        try:
            if pattern == "revenue_optimization":
                steps.extend([
                    WorkflowStep(
                        endpoint_id="customer",
                        operation="GET",
                        description="Retrieve customer profile and preferences",
                        parameters={
                            "customer_id": "string",
                            "include_preferences": "boolean",
                        },
                        expected_response={
                            "customer_profile": "object",
                            "preferences": "object",
                        },
                        error_handling=[
                            "Handle customer not found",
                            "Validate response data",
                        ],
                        business_impact="Personalized customer experience",
                        success_criteria=[
                            "Valid customer data retrieved",
                            "Preferences available",
                        ],
                    ),
                    WorkflowStep(
                        endpoint_id="product",
                        operation="GET",
                        description="Get personalized product recommendations",
                        parameters={
                            "customer_id": "string",
                            "limit": "integer",
                        },
                        expected_response={
                            "recommendations": "array",
                            "relevance_scores": "object",
                        },
                        error_handling=[
                            "Handle empty recommendations",
                            "Validate product availability",
                        ],
                        business_impact="Increased conversion probability",
                        success_criteria=[
                            "Relevant recommendations generated",
                            "Products in stock",
                        ],
                    ),
                    WorkflowStep(
                        endpoint_id="payment",
                        operation="POST",
                        description="Process optimized payment transaction",
                        parameters={
                            "customer_id": "string",
                            "amount": "number",
                            "currency": "string",
                        },
                        expected_response={
                            "transaction_id": "string",
                            "status": "string",
                        },
                        error_handling=[
                            "Handle payment failures",
                            "Implement retry logic",
                            "Record transaction metrics",
                        ],
                        business_impact="Direct revenue generation",
                        success_criteria=[
                            "Payment processed successfully",
                            "Transaction recorded",
                        ],
                    ),
                ])
            
            elif pattern == "customer_onboarding":
                steps.extend([
                    WorkflowStep(
                        endpoint_id="customer",
                        operation="POST",
                        description="Create new customer profile",
                        parameters={
                            "email": "string",
                            "name": "string",
                            "organization": "string",
                        },
                        expected_response={
                            "customer_id": "string",
                            "status": "string",
                        },
                        error_handling=[
                            "Validate input data",
                            "Handle duplicate customers",
                        ],
                        business_impact="New customer acquisition",
                        success_criteria=[
                            "Customer profile created",
                            "Welcome email sent",
                        ],
                    ),
                    WorkflowStep(
                        endpoint_id="preferences",
                        operation="POST",
                        description="Set initial customer preferences",
                        parameters={
                            "customer_id": "string",
                            "preferences": "object",
                        },
                        expected_response={
                            "status": "string",
                            "updated_preferences": "object",
                        },
                        error_handling=[
                            "Validate preferences format",
                            "Handle update conflicts",
                        ],
                        business_impact="Personalized experience setup",
                        success_criteria=[
                            "Preferences saved",
                            "Personalization active",
                        ],
                    ),
                ])
            
            elif pattern == "order_processing":
                steps.extend([
                    WorkflowStep(
                        endpoint_id="order",
                        operation="POST",
                        description="Create new order",
                        parameters={
                            "customer_id": "string",
                            "items": "array",
                            "shipping_address": "object",
                        },
                        expected_response={
                            "order_id": "string",
                            "status": "string",
                        },
                        error_handling=[
                            "Validate order data",
                            "Check inventory availability",
                        ],
                        business_impact="Order capture and processing",
                        success_criteria=[
                            "Order created successfully",
                            "Inventory reserved",
                        ],
                    ),
                    WorkflowStep(
                        endpoint_id="inventory",
                        operation="PUT",
                        description="Update inventory levels",
                        parameters={
                            "order_id": "string",
                            "items": "array",
                        },
                        expected_response={
                            "status": "string",
                            "updated_inventory": "object",
                        },
                        error_handling=[
                            "Handle insufficient inventory",
                            "Implement compensation logic",
                        ],
                        business_impact="Inventory accuracy",
                        success_criteria=[
                            "Inventory updated",
                            "Stock levels accurate",
                        ],
                    ),
                ])
            
            elif pattern == "analytics_integration":
                steps.extend([
                    WorkflowStep(
                        endpoint_id="events",
                        operation="POST",
                        description="Track business events",
                        parameters={
                            "event_type": "string",
                            "event_data": "object",
                            "timestamp": "string",
                        },
                        expected_response={
                            "event_id": "string",
                            "status": "string",
                        },
                        error_handling=[
                            "Validate event format",
                            "Handle tracking failures",
                        ],
                        business_impact="Business event tracking",
                        success_criteria=[
                            "Event recorded",
                            "Data validated",
                        ],
                    ),
                    WorkflowStep(
                        endpoint_id="analytics",
                        operation="GET",
                        description="Generate business insights",
                        parameters={
                            "metric_type": "string",
                            "time_range": "object",
                        },
                        expected_response={
                            "insights": "array",
                            "metrics": "object",
                        },
                        error_handling=[
                            "Handle data gaps",
                            "Validate insight quality",
                        ],
                        business_impact="Data-driven decisions",
                        success_criteria=[
                            "Insights generated",
                            "Metrics calculated",
                        ],
                    ),
                ])

        except Exception as e:
            logger.error(f"Failed to generate workflow steps: {e}")
        
        return steps

    def _determine_target_audience(self, pattern: str) -> str:
        """Determine workflow target audience."""
        audiences = {
            "revenue_optimization": "Sales and Revenue Teams",
            "customer_onboarding": "Customer Success Teams",
            "order_processing": "Operations Teams",
            "analytics_integration": "Business Intelligence Teams",
        }
        return audiences.get(pattern, "Development Teams")

    def _get_prerequisites(
        self,
        pattern: str,
        config: Dict,
    ) -> List[str]:
        """Get workflow prerequisites."""
        common_prereqs = [
            "API authentication setup",
            "Error handling implementation",
            "Monitoring configuration",
        ]
        
        pattern_prereqs = {
            "revenue_optimization": [
                "Payment gateway integration",
                "Customer data management",
                "Product catalog setup",
            ],
            "customer_onboarding": [
                "Email service integration",
                "Customer database setup",
                "Preference management system",
            ],
            "order_processing": [
                "Inventory management system",
                "Shipping integration",
                "Order tracking system",
            ],
            "analytics_integration": [
                "Event tracking setup",
                "Data warehouse connection",
                "Reporting system integration",
            ],
        }
        
        return common_prereqs + pattern_prereqs.get(pattern, [])

    def _estimate_effort(
        self,
        pattern: str,
        steps: List[WorkflowStep],
    ) -> str:
        """Estimate implementation effort."""
        # Base effort by number of steps
        base_effort = len(steps) * 2  # 2 days per step
        
        # Adjust for pattern complexity
        multipliers = {
            "revenue_optimization": 1.5,
            "customer_onboarding": 1.2,
            "order_processing": 1.3,
            "analytics_integration": 1.4,
        }
        
        total_days = base_effort * multipliers.get(pattern, 1.0)
        
        if total_days <= 5:
            return "1 week"
        elif total_days <= 10:
            return "2 weeks"
        elif total_days <= 15:
            return "3 weeks"
        else:
            return "4+ weeks"

    def _generate_implementation_guide(
        self,
        pattern: str,
        steps: List[WorkflowStep],
    ) -> List[str]:
        """Generate implementation guide."""
        guide = [
            "1. Review Prerequisites",
            "2. Set Up Development Environment",
            "3. Implement Authentication",
        ]
        
        # Add step-specific instructions
        for i, step in enumerate(steps, 4):
            guide.append(
                f"{i}. Implement {step.endpoint_id} Integration:"
                f"\n   - {step.description}"
                f"\n   - Handle errors: {', '.join(step.error_handling)}"
                f"\n   - Verify: {', '.join(step.success_criteria)}"
            )
        
        # Add testing and deployment steps
        guide.extend([
            f"{len(guide) + 1}. Implement End-to-End Testing",
            f"{len(guide) + 2}. Set Up Monitoring",
            f"{len(guide) + 3}. Deploy to Production",
            f"{len(guide) + 4}. Monitor Business Metrics",
        ])
        
        return guide

    def _get_best_practices(self, pattern: str) -> List[str]:
        """Get workflow best practices."""
        common_practices = [
            "Implement comprehensive error handling",
            "Add detailed logging",
            "Set up monitoring alerts",
            "Document integration points",
        ]
        
        pattern_practices = {
            "revenue_optimization": [
                "Implement idempotent transactions",
                "Add fraud detection",
                "Monitor conversion metrics",
                "Optimize payment flows",
            ],
            "customer_onboarding": [
                "Implement progressive profiling",
                "Add engagement tracking",
                "Optimize for conversion",
                "Monitor dropout points",
            ],
            "order_processing": [
                "Implement transaction rollback",
                "Add inventory checks",
                "Monitor processing times",
                "Optimize for scale",
            ],
            "analytics_integration": [
                "Implement data validation",
                "Add data quality checks",
                "Monitor data completeness",
                "Optimize query performance",
            ],
        }
        
        return common_practices + pattern_practices.get(pattern, []) 