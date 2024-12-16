"""Business-focused API usage analytics."""

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Set, Tuple
import logging
import numpy as np

from ..config import config_instance
from ..monitoring import metrics_manager

logger = logging.getLogger(__name__)

@dataclass
class UsageMetrics:
    """API usage metrics with business context."""

    endpoint_id: str
    total_calls: int
    success_rate: float
    error_rate: float
    avg_latency: float
    peak_qps: float
    unique_users: int
    revenue_generated: float
    business_value: float
    timestamp: datetime

@dataclass
class BusinessTrend:
    """Business trend analysis."""

    trend_type: str
    description: str
    impact_score: float
    affected_endpoints: List[str]
    recommendations: List[str]
    metrics: Dict[str, float]
    timestamp: datetime

class UsageAnalytics:
    """Analyzes API usage patterns for business insights."""

    def __init__(self):
        """Initialize analytics."""
        self.metrics = metrics_manager
        self._init_storage()

    def _init_storage(self):
        """Initialize data storage."""
        self.usage_history: Dict[str, List[UsageMetrics]] = {}
        self.business_trends: List[BusinessTrend] = []
        self._load_historical_data()

    def _load_historical_data(self):
        """Load historical usage data."""
        try:
            # TODO: Load from database
            pass
        except Exception as e:
            logger.error(f"Failed to load historical data: {e}")

    def record_usage(
        self,
        endpoint_id: str,
        success: bool,
        latency: float,
        user_id: str,
        revenue: float = 0.0,
    ) -> None:
        """Record API usage event.

        Args:
            endpoint_id: Endpoint identifier
            success: Whether the call succeeded
            latency: Request latency in milliseconds
            user_id: User identifier
            revenue: Revenue generated (if applicable)
        """
        try:
            # Get or create metrics for endpoint
            if endpoint_id not in self.usage_history:
                self.usage_history[endpoint_id] = []

            history = self.usage_history[endpoint_id]
            
            # Calculate metrics
            total_calls = len(history) + 1
            success_rate = (
                sum(1 for m in history if m.success_rate > 0.9) + (1 if success else 0)
            ) / total_calls
            error_rate = 1 - success_rate
            avg_latency = (
                sum(m.avg_latency for m in history[-100:]) + latency
            ) / (min(len(history[-100:]) + 1, 100))
            
            # Calculate QPS for last minute
            recent = [
                m for m in history
                if m.timestamp > datetime.now() - timedelta(minutes=1)
            ]
            peak_qps = max(len(recent) + 1, self._get_peak_qps(endpoint_id))
            
            # Track unique users
            unique_users = len(
                {m.endpoint_id for m in history[-1000:]} | {user_id}
            )
            
            # Calculate business value
            business_value = self._calculate_business_value(
                success_rate=success_rate,
                error_rate=error_rate,
                avg_latency=avg_latency,
                peak_qps=peak_qps,
                unique_users=unique_users,
                revenue=revenue,
            )
            
            # Create metrics
            metrics = UsageMetrics(
                endpoint_id=endpoint_id,
                total_calls=total_calls,
                success_rate=success_rate,
                error_rate=error_rate,
                avg_latency=avg_latency,
                peak_qps=peak_qps,
                unique_users=unique_users,
                revenue_generated=revenue,
                business_value=business_value,
                timestamp=datetime.now(),
            )
            
            # Update history
            history.append(metrics)
            
            # Trim old data
            if len(history) > 10000:  # Keep last 10K events
                history = history[-10000:]
            
            # Update trends
            self._update_trends()

        except Exception as e:
            logger.error(f"Failed to record usage: {e}")

    def _get_peak_qps(self, endpoint_id: str) -> float:
        """Get peak QPS for endpoint."""
        try:
            history = self.usage_history.get(endpoint_id, [])
            if not history:
                return 0.0
            
            # Calculate QPS in 1-second windows for last minute
            now = datetime.now()
            minute_ago = now - timedelta(minutes=1)
            recent = [m for m in history if m.timestamp > minute_ago]
            
            if not recent:
                return 0.0
            
            # Group by second
            qps_by_second = {}
            for metric in recent:
                second = metric.timestamp.replace(microsecond=0)
                qps_by_second[second] = qps_by_second.get(second, 0) + 1
            
            return max(qps_by_second.values())

        except Exception as e:
            logger.error(f"Failed to calculate peak QPS: {e}")
            return 0.0

    def _calculate_business_value(
        self,
        success_rate: float,
        error_rate: float,
        avg_latency: float,
        peak_qps: float,
        unique_users: int,
        revenue: float,
    ) -> float:
        """Calculate business value score.

        Args:
            success_rate: API success rate
            error_rate: API error rate
            avg_latency: Average latency
            peak_qps: Peak queries per second
            unique_users: Number of unique users
            revenue: Revenue generated

        Returns:
            Business value score between 0 and 1
        """
        try:
            # Weight factors
            weights = {
                "success_rate": 0.2,
                "performance": 0.15,
                "adoption": 0.25,
                "revenue": 0.3,
                "scalability": 0.1
            }
            
            # Calculate component scores
            success_score = success_rate
            performance_score = 1.0 - min(avg_latency / 1000.0, 1.0)  # Normalize to 0-1
            adoption_score = min(unique_users / 1000.0, 1.0)  # Normalize to 0-1
            revenue_score = min(revenue / 10000.0, 1.0)  # Normalize to 0-1
            scalability_score = min(peak_qps / 100.0, 1.0)  # Normalize to 0-1
            
            # Calculate weighted score
            business_value = (
                weights["success_rate"] * success_score +
                weights["performance"] * performance_score +
                weights["adoption"] * adoption_score +
                weights["revenue"] * revenue_score +
                weights["scalability"] * scalability_score
            )
            
            return min(max(business_value, 0.0), 1.0)

        except Exception as e:
            logger.error(f"Failed to calculate business value: {e}")
            return 0.0

    def _update_trends(self) -> None:
        """Update business trends analysis."""
        try:
            trends = []
            
            # Analyze adoption trends
            adoption_trends = self._analyze_adoption_trends()
            if adoption_trends:
                trends.extend(adoption_trends)
            
            # Analyze performance trends
            performance_trends = self._analyze_performance_trends()
            if performance_trends:
                trends.extend(performance_trends)
            
            # Analyze revenue trends
            revenue_trends = self._analyze_revenue_trends()
            if revenue_trends:
                trends.extend(revenue_trends)
            
            # Analyze usage patterns
            pattern_trends = self._analyze_usage_patterns()
            if pattern_trends:
                trends.extend(pattern_trends)
            
            # Update stored trends
            self.business_trends = trends

        except Exception as e:
            logger.error(f"Failed to update trends: {e}")

    def _analyze_adoption_trends(self) -> List[BusinessTrend]:
        """Analyze API adoption trends."""
        trends = []
        
        try:
            for endpoint_id, history in self.usage_history.items():
                if not history:
                    continue
                
                # Analyze last 7 days
                week_ago = datetime.now() - timedelta(days=7)
                recent = [m for m in history if m.timestamp > week_ago]
                
                if not recent:
                    continue
                
                # Calculate adoption metrics
                daily_users = {}
                for metric in recent:
                    day = metric.timestamp.date()
                    if day not in daily_users:
                        daily_users[day] = set()
                    daily_users[day].add(metric.endpoint_id)
                
                daily_counts = [len(users) for users in daily_users.values()]
                if len(daily_counts) < 2:
                    continue
                
                # Calculate growth rate
                growth_rate = (daily_counts[-1] - daily_counts[0]) / daily_counts[0]
                
                # Generate trend if significant growth
                if growth_rate > 0.2:  # 20% growth
                    trends.append(
                        BusinessTrend(
                            trend_type="adoption_growth",
                            description=f"Strong adoption growth for endpoint {endpoint_id}",
                            impact_score=min(growth_rate, 1.0),
                            affected_endpoints=[endpoint_id],
                            recommendations=[
                                "Consider scaling infrastructure",
                                "Analyze user feedback for improvements",
                                "Plan for premium features",
                                "Document success patterns",
                            ],
                            metrics={
                                "growth_rate": growth_rate,
                                "daily_active_users": daily_counts[-1],
                                "weekly_active_users": len(
                                    {u for users in daily_users.values() for u in users}
                                ),
                            },
                            timestamp=datetime.now(),
                        )
                    )
                elif growth_rate < -0.2:  # 20% decline
                    trends.append(
                        BusinessTrend(
                            trend_type="adoption_decline",
                            description=f"Significant adoption decline for endpoint {endpoint_id}",
                            impact_score=min(abs(growth_rate), 1.0),
                            affected_endpoints=[endpoint_id],
                            recommendations=[
                                "Investigate user feedback",
                                "Check for technical issues",
                                "Review documentation clarity",
                                "Consider user outreach",
                            ],
                            metrics={
                                "decline_rate": abs(growth_rate),
                                "daily_active_users": daily_counts[-1],
                                "peak_users": max(daily_counts),
                            },
                            timestamp=datetime.now(),
                        )
                    )

        except Exception as e:
            logger.error(f"Failed to analyze adoption trends: {e}")
        
        return trends

    def _analyze_performance_trends(self) -> List[BusinessTrend]:
        """Analyze API performance trends."""
        trends = []
        
        try:
            for endpoint_id, history in self.usage_history.items():
                if not history:
                    continue
                
                # Analyze last 24 hours
                day_ago = datetime.now() - timedelta(days=1)
                recent = [m for m in history if m.timestamp > day_ago]
                
                if not recent:
                    continue
                
                # Calculate performance metrics
                hourly_latency = {}
                hourly_errors = {}
                for metric in recent:
                    hour = metric.timestamp.replace(minute=0, second=0, microsecond=0)
                    if hour not in hourly_latency:
                        hourly_latency[hour] = []
                        hourly_errors[hour] = []
                    hourly_latency[hour].append(metric.avg_latency)
                    hourly_errors[hour].append(metric.error_rate)
                
                # Calculate trends
                latency_trend = np.mean(
                    [np.mean(lats) for lats in hourly_latency.values()]
                )
                error_trend = np.mean(
                    [np.mean(errs) for errs in hourly_errors.values()]
                )
                
                # Generate trends for significant issues
                if latency_trend > 500:  # High latency
                    trends.append(
                        BusinessTrend(
                            trend_type="high_latency",
                            description=f"Performance degradation in endpoint {endpoint_id}",
                            impact_score=min(latency_trend / 1000.0, 1.0),
                            affected_endpoints=[endpoint_id],
                            recommendations=[
                                "Optimize endpoint implementation",
                                "Review caching strategy",
                                "Check database queries",
                                "Monitor system resources",
                            ],
                            metrics={
                                "avg_latency": latency_trend,
                                "peak_latency": max(
                                    max(lats) for lats in hourly_latency.values()
                                ),
                                "error_rate": error_trend,
                            },
                            timestamp=datetime.now(),
                        )
                    )
                
                if error_trend > 0.1:  # High error rate
                    trends.append(
                        BusinessTrend(
                            trend_type="high_errors",
                            description=f"Elevated error rates in endpoint {endpoint_id}",
                            impact_score=min(error_trend * 2, 1.0),
                            affected_endpoints=[endpoint_id],
                            recommendations=[
                                "Investigate error patterns",
                                "Improve error handling",
                                "Update documentation",
                                "Set up error alerting",
                            ],
                            metrics={
                                "error_rate": error_trend,
                                "peak_errors": max(
                                    max(errs) for errs in hourly_errors.values()
                                ),
                                "success_rate": 1 - error_trend,
                            },
                            timestamp=datetime.now(),
                        )
                    )

        except Exception as e:
            logger.error(f"Failed to analyze performance trends: {e}")
        
        return trends

    def _analyze_revenue_trends(self) -> List[BusinessTrend]:
        """Analyze revenue trends."""
        trends = []
        
        try:
            for endpoint_id, history in self.usage_history.items():
                if not history:
                    continue
                
                # Analyze last 30 days
                month_ago = datetime.now() - timedelta(days=30)
                recent = [m for m in history if m.timestamp > month_ago]
                
                if not recent:
                    continue
                
                # Calculate daily revenue
                daily_revenue = {}
                for metric in recent:
                    day = metric.timestamp.date()
                    if day not in daily_revenue:
                        daily_revenue[day] = 0.0
                    daily_revenue[day] += metric.revenue_generated
                
                # Calculate revenue metrics
                revenue_values = list(daily_revenue.values())
                if len(revenue_values) < 2:
                    continue
                
                avg_revenue = np.mean(revenue_values)
                revenue_growth = (
                    revenue_values[-1] - revenue_values[0]
                ) / max(revenue_values[0], 1.0)
                
                # Generate trends for significant patterns
                if revenue_growth > 0.3:  # 30% growth
                    trends.append(
                        BusinessTrend(
                            trend_type="revenue_growth",
                            description=f"Strong revenue growth from endpoint {endpoint_id}",
                            impact_score=min(revenue_growth, 1.0),
                            affected_endpoints=[endpoint_id],
                            recommendations=[
                                "Analyze growth drivers",
                                "Plan for scaling",
                                "Consider premium features",
                                "Document success patterns",
                            ],
                            metrics={
                                "revenue_growth": revenue_growth,
                                "avg_daily_revenue": avg_revenue,
                                "peak_daily_revenue": max(revenue_values),
                            },
                            timestamp=datetime.now(),
                        )
                    )
                elif revenue_growth < -0.3:  # 30% decline
                    trends.append(
                        BusinessTrend(
                            trend_type="revenue_decline",
                            description=f"Significant revenue decline from endpoint {endpoint_id}",
                            impact_score=min(abs(revenue_growth), 1.0),
                            affected_endpoints=[endpoint_id],
                            recommendations=[
                                "Investigate usage patterns",
                                "Review pricing strategy",
                                "Check competitor offerings",
                                "Plan retention campaign",
                            ],
                            metrics={
                                "revenue_decline": abs(revenue_growth),
                                "avg_daily_revenue": avg_revenue,
                                "min_daily_revenue": min(revenue_values),
                            },
                            timestamp=datetime.now(),
                        )
                    )

        except Exception as e:
            logger.error(f"Failed to analyze revenue trends: {e}")
        
        return trends

    def _analyze_usage_patterns(self) -> List[BusinessTrend]:
        """Analyze usage patterns."""
        trends = []
        
        try:
            # Analyze endpoint correlations
            correlations = self._find_endpoint_correlations()
            for endpoints, correlation in correlations:
                if correlation > 0.8:  # Strong correlation
                    trends.append(
                        BusinessTrend(
                            trend_type="usage_pattern",
                            description="Strong usage correlation between endpoints",
                            impact_score=correlation,
                            affected_endpoints=list(endpoints),
                            recommendations=[
                                "Consider endpoint bundling",
                                "Update documentation",
                                "Create workflow templates",
                                "Optimize for common patterns",
                            ],
                            metrics={
                                "correlation": correlation,
                                "pattern_strength": correlation,
                            },
                            timestamp=datetime.now(),
                        )
                    )
            
            # Analyze usage spikes
            spikes = self._find_usage_spikes()
            for endpoint_id, spike_score in spikes:
                trends.append(
                    BusinessTrend(
                        trend_type="usage_spike",
                        description=f"Unusual usage pattern detected for endpoint {endpoint_id}",
                        impact_score=spike_score,
                        affected_endpoints=[endpoint_id],
                        recommendations=[
                            "Monitor system resources",
                            "Review rate limiting",
                            "Check for abuse patterns",
                            "Plan for load balancing",
                        ],
                        metrics={
                            "spike_score": spike_score,
                            "normal_usage": self._get_normal_usage(endpoint_id),
                        },
                        timestamp=datetime.now(),
                    )
                )

        except Exception as e:
            logger.error(f"Failed to analyze usage patterns: {e}")
        
        return trends

    def _find_endpoint_correlations(self) -> List[Tuple[Set[str], float]]:
        """Find correlated endpoint usage patterns."""
        correlations = []
        
        try:
            # Get hourly usage counts
            hourly_usage = {}
            for endpoint_id, history in self.usage_history.items():
                usage_by_hour = {}
                for metric in history:
                    hour = metric.timestamp.replace(minute=0, second=0, microsecond=0)
                    usage_by_hour[hour] = usage_by_hour.get(hour, 0) + 1
                hourly_usage[endpoint_id] = usage_by_hour
            
            # Calculate correlations between endpoints
            endpoints = list(hourly_usage.keys())
            for i in range(len(endpoints)):
                for j in range(i + 1, len(endpoints)):
                    endpoint1 = endpoints[i]
                    endpoint2 = endpoints[j]
                    
                    # Get common hours
                    hours = set(hourly_usage[endpoint1].keys()) & set(
                        hourly_usage[endpoint2].keys()
                    )
                    if len(hours) < 24:  # Require at least 24 hours of data
                        continue
                    
                    # Calculate correlation
                    usage1 = [hourly_usage[endpoint1].get(h, 0) for h in hours]
                    usage2 = [hourly_usage[endpoint2].get(h, 0) for h in hours]
                    correlation = np.corrcoef(usage1, usage2)[0, 1]
                    
                    if not np.isnan(correlation):
                        correlations.append(({endpoint1, endpoint2}, abs(correlation)))

        except Exception as e:
            logger.error(f"Failed to find endpoint correlations: {e}")
        
        return correlations

    def _find_usage_spikes(self) -> List[Tuple[str, float]]:
        """Find unusual usage spikes."""
        spikes = []
        
        try:
            for endpoint_id, history in self.usage_history.items():
                if not history:
                    continue
                
                # Analyze last hour
                hour_ago = datetime.now() - timedelta(hours=1)
                recent = [m for m in history if m.timestamp > hour_ago]
                
                if not recent:
                    continue
                
                # Calculate baseline
                baseline = self._get_normal_usage(endpoint_id)
                if baseline == 0:
                    continue
                
                # Calculate current usage
                current = len(recent)
                
                # Calculate spike score
                spike_score = current / baseline if baseline > 0 else 0
                if spike_score > 3:  # 3x normal usage
                    spikes.append((endpoint_id, min(spike_score / 10, 1.0)))

        except Exception as e:
            logger.error(f"Failed to find usage spikes: {e}")
        
        return spikes

    def _get_normal_usage(self, endpoint_id: str) -> float:
        """Get normal usage level for endpoint."""
        try:
            history = self.usage_history.get(endpoint_id, [])
            if not history:
                return 0.0
            
            # Calculate average hourly usage for last 7 days
            week_ago = datetime.now() - timedelta(days=7)
            recent = [m for m in history if m.timestamp > week_ago]
            
            if not recent:
                return 0.0
            
            # Group by hour
            hourly_counts = {}
            for metric in recent:
                hour = metric.timestamp.replace(minute=0, second=0, microsecond=0)
                hourly_counts[hour] = hourly_counts.get(hour, 0) + 1
            
            return np.mean(list(hourly_counts.values()))

        except Exception as e:
            logger.error(f"Failed to get normal usage: {e}")
            return 0.0

    def get_business_insights(
        self,
        window: timedelta = timedelta(days=30),
    ) -> Dict[str, Any]:
        """Get business insights from usage analytics.

        Args:
            window: Time window for analysis

        Returns:
            Dictionary of business insights
        """
        try:
            cutoff = datetime.now() - window
            recent_trends = [
                t for t in self.business_trends if t.timestamp > cutoff
            ]
            
            # Group trends by type
            trends_by_type = {}
            for trend in recent_trends:
                if trend.trend_type not in trends_by_type:
                    trends_by_type[trend.trend_type] = []
                trends_by_type[trend.trend_type].append({
                    "description": trend.description,
                    "impact_score": trend.impact_score,
                    "affected_endpoints": trend.affected_endpoints,
                    "recommendations": trend.recommendations,
                    "metrics": trend.metrics,
                    "timestamp": trend.timestamp.isoformat(),
                })
            
            # Calculate aggregate metrics
            total_calls = sum(
                m.total_calls for metrics in self.usage_history.values()
                for m in metrics if m.timestamp > cutoff
            )
            
            avg_success_rate = np.mean([
                m.success_rate
                for metrics in self.usage_history.values()
                for m in metrics if m.timestamp > cutoff
            ])
            
            total_revenue = sum(
                m.revenue_generated
                for metrics in self.usage_history.values()
                for m in metrics if m.timestamp > cutoff
            )
            
            return {
                "trends": trends_by_type,
                "metrics": {
                    "total_calls": total_calls,
                    "avg_success_rate": avg_success_rate,
                    "total_revenue": total_revenue,
                    "active_endpoints": len(self.usage_history),
                },
                "window": str(window),
                "generated_at": datetime.now().isoformat(),
            }

        except Exception as e:
            logger.error(f"Failed to get business insights: {e}")
            return {
                "error": str(e),
                "generated_at": datetime.now().isoformat(),
            } 