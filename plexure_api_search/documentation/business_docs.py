"""Business-focused API documentation generator."""

from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional, Set
import logging

from ..config import config_instance
from ..monitoring import metrics_manager
from ..analytics.usage_analytics import BusinessTrend, UsageMetrics

logger = logging.getLogger(__name__)

@dataclass
class BusinessUseCase:
    """Business use case documentation."""

    title: str
    description: str
    target_audience: str
    business_value: str
    implementation_time: str
    required_endpoints: List[str]
    sample_workflow: List[str]
    success_metrics: List[str]
    considerations: List[str]
    next_steps: List[str]

@dataclass
class BusinessGuide:
    """Business implementation guide."""

    title: str
    overview: str
    use_cases: List[BusinessUseCase]
    technical_requirements: List[str]
    business_requirements: List[str]
    implementation_steps: List[Dict[str, str]]
    success_stories: List[Dict[str, str]]
    roi_calculation: Dict[str, float]
    timeline: List[Dict[str, str]]
    resources: List[Dict[str, str]]

class BusinessDocGenerator:
    """Generates business-focused API documentation."""

    def __init__(self):
        """Initialize documentation generator."""
        self.metrics = metrics_manager
        self._init_templates()

    def _init_templates(self):
        """Initialize documentation templates."""
        self.use_case_templates = {
            "revenue_generation": {
                "title": "Revenue Generation with {endpoint}",
                "description": "Implement {endpoint} to create new revenue streams",
                "target_audience": "Business Development, Product Management",
                "business_value": "Direct revenue generation through API monetization",
                "implementation_time": "2-4 weeks",
                "success_metrics": [
                    "Monthly Recurring Revenue (MRR)",
                    "Customer Lifetime Value (CLV)",
                    "API Usage Growth",
                    "Revenue per API Call",
                ],
            },
            "cost_optimization": {
                "title": "Cost Optimization with {endpoint}",
                "description": "Optimize operations using {endpoint}",
                "target_audience": "Operations, Finance",
                "business_value": "Reduced operational costs and improved efficiency",
                "implementation_time": "4-8 weeks",
                "success_metrics": [
                    "Cost per Transaction",
                    "Processing Time Reduction",
                    "Resource Utilization",
                    "Error Rate Reduction",
                ],
            },
            "customer_experience": {
                "title": "Enhanced Customer Experience with {endpoint}",
                "description": "Improve customer satisfaction using {endpoint}",
                "target_audience": "Customer Success, Product Management",
                "business_value": "Improved customer satisfaction and retention",
                "implementation_time": "3-6 weeks",
                "success_metrics": [
                    "Customer Satisfaction Score",
                    "Net Promoter Score",
                    "Customer Retention Rate",
                    "Feature Adoption Rate",
                ],
            },
            "market_expansion": {
                "title": "Market Expansion with {endpoint}",
                "description": "Enter new markets using {endpoint}",
                "target_audience": "Business Development, Sales",
                "business_value": "Market share growth and new customer acquisition",
                "implementation_time": "6-12 weeks",
                "success_metrics": [
                    "Market Penetration Rate",
                    "New Customer Acquisition",
                    "Geographic Coverage",
                    "Revenue from New Markets",
                ],
            },
        }

    def generate_business_docs(
        self,
        endpoint_id: str,
        usage_metrics: UsageMetrics,
        trends: List[BusinessTrend],
    ) -> BusinessGuide:
        """Generate business-focused documentation.

        Args:
            endpoint_id: Endpoint identifier
            usage_metrics: Usage metrics
            trends: Business trends

        Returns:
            Business implementation guide
        """
        try:
            # Identify primary use cases
            use_cases = self._identify_use_cases(endpoint_id, usage_metrics, trends)
            
            # Generate implementation guide
            guide = BusinessGuide(
                title=f"Business Implementation Guide: {endpoint_id}",
                overview=self._generate_overview(endpoint_id, usage_metrics),
                use_cases=use_cases,
                technical_requirements=self._get_technical_requirements(endpoint_id),
                business_requirements=self._get_business_requirements(use_cases),
                implementation_steps=self._generate_implementation_steps(use_cases),
                success_stories=self._get_success_stories(endpoint_id),
                roi_calculation=self._calculate_roi(usage_metrics),
                timeline=self._generate_timeline(use_cases),
                resources=self._get_resources(endpoint_id),
            )
            
            return guide

        except Exception as e:
            logger.error(f"Failed to generate business docs: {e}")
            raise

    def _identify_use_cases(
        self,
        endpoint_id: str,
        usage_metrics: UsageMetrics,
        trends: List[BusinessTrend],
    ) -> List[BusinessUseCase]:
        """Identify relevant business use cases."""
        use_cases = []
        
        try:
            # Analyze metrics and trends
            high_revenue = usage_metrics.revenue_generated > 1000
            high_adoption = usage_metrics.unique_users > 100
            growing_usage = any(
                t.trend_type == "adoption_growth" and endpoint_id in t.affected_endpoints
                for t in trends
            )
            
            # Generate relevant use cases
            if high_revenue or any(
                t.trend_type == "revenue_growth" and endpoint_id in t.affected_endpoints
                for t in trends
            ):
                use_cases.append(
                    self._create_revenue_use_case(endpoint_id, usage_metrics)
                )
            
            if usage_metrics.error_rate < 0.1 and usage_metrics.avg_latency < 200:
                use_cases.append(
                    self._create_optimization_use_case(endpoint_id, usage_metrics)
                )
            
            if high_adoption or growing_usage:
                use_cases.append(
                    self._create_experience_use_case(endpoint_id, usage_metrics)
                )
            
            if usage_metrics.business_value > 0.7:
                use_cases.append(
                    self._create_expansion_use_case(endpoint_id, usage_metrics)
                )

        except Exception as e:
            logger.error(f"Failed to identify use cases: {e}")
        
        return use_cases

    def _create_revenue_use_case(
        self,
        endpoint_id: str,
        metrics: UsageMetrics,
    ) -> BusinessUseCase:
        """Create revenue generation use case."""
        template = self.use_case_templates["revenue_generation"]
        
        return BusinessUseCase(
            title=template["title"].format(endpoint=endpoint_id),
            description=template["description"].format(endpoint=endpoint_id),
            target_audience=template["target_audience"],
            business_value=template["business_value"],
            implementation_time=template["implementation_time"],
            required_endpoints=[endpoint_id],
            sample_workflow=[
                "1. Integrate API endpoint",
                "2. Set up usage tracking",
                "3. Implement billing system",
                "4. Monitor revenue metrics",
            ],
            success_metrics=template["success_metrics"],
            considerations=[
                "Pricing strategy optimization",
                "Usage limits and quotas",
                "Revenue sharing models",
                "Market competitiveness",
            ],
            next_steps=[
                "Define pricing tiers",
                "Set up billing integration",
                "Create revenue dashboards",
                "Plan marketing strategy",
            ],
        )

    def _create_optimization_use_case(
        self,
        endpoint_id: str,
        metrics: UsageMetrics,
    ) -> BusinessUseCase:
        """Create cost optimization use case."""
        template = self.use_case_templates["cost_optimization"]
        
        return BusinessUseCase(
            title=template["title"].format(endpoint=endpoint_id),
            description=template["description"].format(endpoint=endpoint_id),
            target_audience=template["target_audience"],
            business_value=template["business_value"],
            implementation_time=template["implementation_time"],
            required_endpoints=[endpoint_id],
            sample_workflow=[
                "1. Analyze current costs",
                "2. Implement API integration",
                "3. Set up monitoring",
                "4. Measure cost savings",
            ],
            success_metrics=template["success_metrics"],
            considerations=[
                "Resource optimization",
                "Process automation",
                "Error handling",
                "Scalability planning",
            ],
            next_steps=[
                "Identify optimization targets",
                "Create implementation plan",
                "Set up monitoring",
                "Define success metrics",
            ],
        )

    def _create_experience_use_case(
        self,
        endpoint_id: str,
        metrics: UsageMetrics,
    ) -> BusinessUseCase:
        """Create customer experience use case."""
        template = self.use_case_templates["customer_experience"]
        
        return BusinessUseCase(
            title=template["title"].format(endpoint=endpoint_id),
            description=template["description"].format(endpoint=endpoint_id),
            target_audience=template["target_audience"],
            business_value=template["business_value"],
            implementation_time=template["implementation_time"],
            required_endpoints=[endpoint_id],
            sample_workflow=[
                "1. Map customer journey",
                "2. Implement API features",
                "3. Set up feedback collection",
                "4. Monitor satisfaction",
            ],
            success_metrics=template["success_metrics"],
            considerations=[
                "User experience design",
                "Performance optimization",
                "Error handling",
                "Customer feedback",
            ],
            next_steps=[
                "Define experience metrics",
                "Create feedback system",
                "Plan A/B testing",
                "Set up monitoring",
            ],
        )

    def _create_expansion_use_case(
        self,
        endpoint_id: str,
        metrics: UsageMetrics,
    ) -> BusinessUseCase:
        """Create market expansion use case."""
        template = self.use_case_templates["market_expansion"]
        
        return BusinessUseCase(
            title=template["title"].format(endpoint=endpoint_id),
            description=template["description"].format(endpoint=endpoint_id),
            target_audience=template["target_audience"],
            business_value=template["business_value"],
            implementation_time=template["implementation_time"],
            required_endpoints=[endpoint_id],
            sample_workflow=[
                "1. Market analysis",
                "2. API implementation",
                "3. Launch preparation",
                "4. Market monitoring",
            ],
            success_metrics=template["success_metrics"],
            considerations=[
                "Market requirements",
                "Competitive analysis",
                "Scalability planning",
                "Regulatory compliance",
            ],
            next_steps=[
                "Define target markets",
                "Create expansion plan",
                "Set up monitoring",
                "Plan marketing strategy",
            ],
        )

    def _generate_overview(
        self,
        endpoint_id: str,
        metrics: UsageMetrics,
    ) -> str:
        """Generate business overview."""
        return f"""
Business Overview: {endpoint_id}

Key Metrics:
- Total Usage: {metrics.total_calls:,} calls
- Success Rate: {metrics.success_rate:.1%}
- Unique Users: {metrics.unique_users:,}
- Revenue Generated: ${metrics.revenue_generated:,.2f}
- Business Value Score: {metrics.business_value:.2f}

This API endpoint provides significant business capabilities for your organization.
Current metrics indicate strong {self._get_performance_indicator(metrics)} performance
with opportunities for {self._get_growth_opportunities(metrics)}.
"""

    def _get_performance_indicator(self, metrics: UsageMetrics) -> str:
        """Get performance indicator description."""
        if metrics.business_value > 0.8:
            return "exceptional"
        elif metrics.business_value > 0.6:
            return "strong"
        elif metrics.business_value > 0.4:
            return "moderate"
        else:
            return "developing"

    def _get_growth_opportunities(self, metrics: UsageMetrics) -> str:
        """Get growth opportunity description."""
        opportunities = []
        
        if metrics.revenue_generated < 1000:
            opportunities.append("revenue growth")
        if metrics.unique_users < 100:
            opportunities.append("user acquisition")
        if metrics.error_rate > 0.1:
            opportunities.append("reliability improvement")
        if metrics.avg_latency > 200:
            opportunities.append("performance optimization")
        
        if not opportunities:
            return "continued optimization"
        
        return " and ".join(opportunities)

    def _get_technical_requirements(self, endpoint_id: str) -> List[str]:
        """Get technical requirements."""
        return [
            "API authentication credentials",
            "Secure HTTPS connection",
            "JSON request/response handling",
            "Error handling implementation",
            "Rate limiting compliance",
            "Data validation",
            "Logging and monitoring",
        ]

    def _get_business_requirements(
        self,
        use_cases: List[BusinessUseCase],
    ) -> List[str]:
        """Get business requirements."""
        requirements = set()
        
        for use_case in use_cases:
            if "revenue" in use_case.title.lower():
                requirements.update([
                    "Billing system integration",
                    "Revenue tracking capability",
                    "Usage monitoring",
                ])
            if "optimization" in use_case.title.lower():
                requirements.update([
                    "Cost tracking system",
                    "Performance monitoring",
                    "Process automation capability",
                ])
            if "experience" in use_case.title.lower():
                requirements.update([
                    "Customer feedback system",
                    "User behavior tracking",
                    "Support system integration",
                ])
            if "expansion" in use_case.title.lower():
                requirements.update([
                    "Market analysis capability",
                    "Competitive monitoring",
                    "Regulatory compliance",
                ])
        
        return sorted(list(requirements))

    def _generate_implementation_steps(
        self,
        use_cases: List[BusinessUseCase],
    ) -> List[Dict[str, str]]:
        """Generate implementation steps."""
        steps = []
        
        # Add common initial steps
        steps.extend([
            {
                "phase": "Planning",
                "step": "Requirements Analysis",
                "description": "Define technical and business requirements",
                "duration": "1-2 weeks",
            },
            {
                "phase": "Planning",
                "step": "Architecture Design",
                "description": "Design integration architecture",
                "duration": "1-2 weeks",
            },
        ])
        
        # Add use case specific steps
        for use_case in use_cases:
            if "revenue" in use_case.title.lower():
                steps.extend([
                    {
                        "phase": "Implementation",
                        "step": "Revenue System Integration",
                        "description": "Integrate billing and revenue tracking",
                        "duration": "2-3 weeks",
                    },
                    {
                        "phase": "Implementation",
                        "step": "Usage Tracking",
                        "description": "Implement usage monitoring",
                        "duration": "1-2 weeks",
                    },
                ])
            if "optimization" in use_case.title.lower():
                steps.extend([
                    {
                        "phase": "Implementation",
                        "step": "Process Automation",
                        "description": "Implement automated workflows",
                        "duration": "2-4 weeks",
                    },
                    {
                        "phase": "Implementation",
                        "step": "Performance Monitoring",
                        "description": "Set up monitoring systems",
                        "duration": "1-2 weeks",
                    },
                ])
        
        # Add common final steps
        steps.extend([
            {
                "phase": "Testing",
                "step": "Integration Testing",
                "description": "Test all integrations",
                "duration": "1-2 weeks",
            },
            {
                "phase": "Deployment",
                "step": "Production Deployment",
                "description": "Deploy to production",
                "duration": "1 week",
            },
            {
                "phase": "Post-Launch",
                "step": "Monitoring",
                "description": "Monitor performance and metrics",
                "duration": "Ongoing",
            },
        ])
        
        return steps

    def _get_success_stories(self, endpoint_id: str) -> List[Dict[str, str]]:
        """Get relevant success stories."""
        # TODO: Load from database
        return [
            {
                "company": "Example Corp",
                "industry": "Technology",
                "challenge": "Needed to optimize API operations",
                "solution": f"Implemented {endpoint_id} with custom integration",
                "results": "30% cost reduction, 50% faster processing",
                "timeline": "3 months",
            },
            {
                "company": "Sample Inc",
                "industry": "Finance",
                "challenge": "Required secure API integration",
                "solution": f"Deployed {endpoint_id} with enhanced security",
                "results": "99.9% uptime, zero security incidents",
                "timeline": "2 months",
            },
        ]

    def _calculate_roi(self, metrics: UsageMetrics) -> Dict[str, float]:
        """Calculate ROI metrics."""
        try:
            # Calculate costs
            implementation_cost = 50000  # Estimated implementation cost
            operational_cost = metrics.total_calls * 0.001  # Cost per call
            maintenance_cost = implementation_cost * 0.2  # 20% maintenance
            
            # Calculate benefits
            revenue = metrics.revenue_generated
            cost_savings = metrics.total_calls * 0.005  # Estimated savings
            productivity_gain = metrics.total_calls * 0.002  # Productivity value
            
            # Calculate ROI
            total_cost = implementation_cost + operational_cost + maintenance_cost
            total_benefit = revenue + cost_savings + productivity_gain
            roi = ((total_benefit - total_cost) / total_cost) * 100
            
            return {
                "implementation_cost": implementation_cost,
                "operational_cost": operational_cost,
                "maintenance_cost": maintenance_cost,
                "revenue": revenue,
                "cost_savings": cost_savings,
                "productivity_gain": productivity_gain,
                "total_cost": total_cost,
                "total_benefit": total_benefit,
                "roi_percentage": roi,
                "payback_months": (total_cost / (total_benefit / 12))
                if total_benefit > 0 else float("inf"),
            }

        except Exception as e:
            logger.error(f"Failed to calculate ROI: {e}")
            return {}

    def _generate_timeline(
        self,
        use_cases: List[BusinessUseCase],
    ) -> List[Dict[str, str]]:
        """Generate implementation timeline."""
        timeline = []
        current_week = 0
        
        # Planning phase
        timeline.extend([
            {
                "phase": "Planning",
                "activity": "Requirements Analysis",
                "start_week": str(current_week),
                "duration": "2 weeks",
            },
            {
                "phase": "Planning",
                "activity": "Architecture Design",
                "start_week": str(current_week + 2),
                "duration": "2 weeks",
            },
        ])
        current_week += 4
        
        # Implementation phase
        for use_case in use_cases:
            timeline.append({
                "phase": "Implementation",
                "activity": f"Implement {use_case.title}",
                "start_week": str(current_week),
                "duration": use_case.implementation_time,
            })
            current_week += int(use_case.implementation_time.split("-")[0])
        
        # Testing and deployment
        timeline.extend([
            {
                "phase": "Testing",
                "activity": "Integration Testing",
                "start_week": str(current_week),
                "duration": "2 weeks",
            },
            {
                "phase": "Deployment",
                "activity": "Production Deployment",
                "start_week": str(current_week + 2),
                "duration": "1 week",
            },
            {
                "phase": "Post-Launch",
                "activity": "Monitoring and Optimization",
                "start_week": str(current_week + 3),
                "duration": "Ongoing",
            },
        ])
        
        return timeline

    def _get_resources(self, endpoint_id: str) -> List[Dict[str, str]]:
        """Get implementation resources."""
        return [
            {
                "type": "Documentation",
                "title": "Technical Documentation",
                "url": f"/docs/api/{endpoint_id}",
                "description": "Detailed API documentation",
            },
            {
                "type": "Guide",
                "title": "Implementation Guide",
                "url": f"/guides/{endpoint_id}",
                "description": "Step-by-step implementation guide",
            },
            {
                "type": "Sample",
                "title": "Code Samples",
                "url": f"/samples/{endpoint_id}",
                "description": "Implementation examples",
            },
            {
                "type": "Support",
                "title": "Support Resources",
                "url": "/support",
                "description": "Technical support contacts",
            },
        ] 