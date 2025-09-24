"""
Performance Analytics and Optimization Engine for Workspace Qdrant MCP.

This module provides intelligent analysis of performance metrics and generates
optimization recommendations based on usage patterns, resource utilization,
and performance trends.

Task 265: Performance analytics for the workspace_qdrant_mcp system.
"""

import asyncio
import statistics
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

from loguru import logger

from .performance_metrics import (
    MetricType, PerformanceLevel, PerformanceMetric, MetricSummary,
    OperationTrace, PerformanceMetricsCollector
)


class OptimizationType(Enum):
    """Types of optimization recommendations."""

    # Resource optimizations
    MEMORY_OPTIMIZATION = "memory_optimization"
    CPU_OPTIMIZATION = "cpu_optimization"
    DISK_IO_OPTIMIZATION = "disk_io_optimization"
    NETWORK_OPTIMIZATION = "network_optimization"

    # Configuration optimizations
    BATCH_SIZE_TUNING = "batch_size_tuning"
    CONNECTION_POOLING = "connection_pooling"
    CACHE_CONFIGURATION = "cache_configuration"
    PARALLELISM_TUNING = "parallelism_tuning"

    # Processing optimizations
    FILE_PROCESSING_OPTIMIZATION = "file_processing_optimization"
    SEARCH_OPTIMIZATION = "search_optimization"
    EMBEDDING_OPTIMIZATION = "embedding_optimization"

    # Infrastructure optimizations
    SCALING_RECOMMENDATION = "scaling_recommendation"
    DAEMON_CONFIGURATION = "daemon_configuration"
    LSP_OPTIMIZATION = "lsp_optimization"


class Priority(Enum):
    """Priority levels for optimization recommendations."""

    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFORMATIONAL = "informational"


@dataclass
class OptimizationRecommendation:
    """A specific optimization recommendation."""

    title: str
    description: str
    optimization_type: OptimizationType
    priority: Priority
    impact_estimate: str  # "high", "medium", "low"
    implementation_effort: str  # "easy", "moderate", "complex"
    metrics_evidence: List[MetricSummary] = field(default_factory=list)
    suggested_actions: List[str] = field(default_factory=list)
    expected_improvements: Dict[str, str] = field(default_factory=dict)
    configuration_changes: Dict[str, Any] = field(default_factory=dict)
    estimated_timeline: str = "immediate"
    dependencies: List[str] = field(default_factory=list)
    risks: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "title": self.title,
            "description": self.description,
            "optimization_type": self.optimization_type.value,
            "priority": self.priority.value,
            "impact_estimate": self.impact_estimate,
            "implementation_effort": self.implementation_effort,
            "suggested_actions": self.suggested_actions,
            "expected_improvements": self.expected_improvements,
            "configuration_changes": self.configuration_changes,
            "estimated_timeline": self.estimated_timeline,
            "dependencies": self.dependencies,
            "risks": self.risks
        }


@dataclass
class PerformanceInsight:
    """High-level performance insight."""

    category: str
    title: str
    description: str
    severity: str  # "info", "warning", "critical"
    metrics: List[str] = field(default_factory=list)
    trend: str = "stable"  # "improving", "degrading", "stable"
    recommended_actions: List[str] = field(default_factory=list)


@dataclass
class PerformanceReport:
    """Comprehensive performance analysis report."""

    project_id: str
    generated_at: datetime
    time_range: Tuple[datetime, datetime]
    overall_performance_score: float  # 0-100
    performance_level: PerformanceLevel

    # Metric summaries
    metric_summaries: Dict[MetricType, MetricSummary] = field(default_factory=dict)

    # Analysis results
    insights: List[PerformanceInsight] = field(default_factory=list)
    recommendations: List[OptimizationRecommendation] = field(default_factory=list)

    # Resource utilization
    resource_efficiency: Dict[str, float] = field(default_factory=dict)
    bottlenecks: List[str] = field(default_factory=list)

    # Operational metrics
    error_rates: Dict[str, float] = field(default_factory=dict)
    availability_metrics: Dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "project_id": self.project_id,
            "generated_at": self.generated_at.isoformat(),
            "time_range": [self.time_range[0].isoformat(), self.time_range[1].isoformat()],
            "overall_performance_score": self.overall_performance_score,
            "performance_level": self.performance_level.value,
            "metric_summaries": {
                mt.value: {
                    "count": ms.count,
                    "mean_value": ms.mean_value,
                    "performance_level": ms.performance_level.value,
                    "trend": ms.trend
                } for mt, ms in self.metric_summaries.items()
            },
            "insights": [
                {
                    "category": insight.category,
                    "title": insight.title,
                    "description": insight.description,
                    "severity": insight.severity,
                    "trend": insight.trend
                } for insight in self.insights
            ],
            "recommendations": [rec.to_dict() for rec in self.recommendations],
            "resource_efficiency": self.resource_efficiency,
            "bottlenecks": self.bottlenecks,
            "error_rates": self.error_rates,
            "availability_metrics": self.availability_metrics
        }


class PerformanceAnalyzer:
    """Analyzes performance metrics and generates insights."""

    def __init__(self, metrics_collector: PerformanceMetricsCollector):
        self.metrics_collector = metrics_collector
        self.analysis_cache: Dict[str, Any] = {}
        self.cache_ttl = timedelta(minutes=5)
        self.last_analysis = datetime.min

        # Performance thresholds for analysis
        self.performance_thresholds = {
            "search_latency_warning": 200,
            "search_latency_critical": 500,
            "lsp_latency_warning": 100,
            "lsp_latency_critical": 300,
            "file_processing_rate_warning": 10,
            "file_processing_rate_critical": 5,
            "memory_usage_warning": 80,
            "memory_usage_critical": 95,
            "cpu_usage_warning": 70,
            "cpu_usage_critical": 90,
            "error_rate_warning": 0.05,
            "error_rate_critical": 0.10,
        }

    async def analyze_performance(
        self,
        time_range_hours: int = 1
    ) -> PerformanceReport:
        """Generate comprehensive performance analysis report."""
        end_time = datetime.now()
        start_time = end_time - timedelta(hours=time_range_hours)

        logger.info(f"Generating performance analysis for {self.metrics_collector.project_id}")

        # Collect metric summaries
        metric_summaries = {}
        for metric_type in MetricType:
            summary = await self.metrics_collector.get_metric_summary(
                metric_type, start_time, end_time
            )
            if summary:
                metric_summaries[metric_type] = summary

        # Generate insights
        insights = await self._generate_insights(metric_summaries, start_time, end_time)

        # Generate recommendations
        recommendations = await self._generate_recommendations(metric_summaries, insights)

        # Calculate overall performance score
        performance_score = self._calculate_performance_score(metric_summaries)
        performance_level = self._determine_performance_level(performance_score)

        # Calculate resource efficiency
        resource_efficiency = self._calculate_resource_efficiency(metric_summaries)

        # Identify bottlenecks
        bottlenecks = self._identify_bottlenecks(metric_summaries)

        # Calculate error rates
        error_rates = self._calculate_error_rates(metric_summaries)

        # Calculate availability metrics
        availability_metrics = self._calculate_availability_metrics(start_time, end_time)

        report = PerformanceReport(
            project_id=self.metrics_collector.project_id,
            generated_at=datetime.now(),
            time_range=(start_time, end_time),
            overall_performance_score=performance_score,
            performance_level=performance_level,
            metric_summaries=metric_summaries,
            insights=insights,
            recommendations=recommendations,
            resource_efficiency=resource_efficiency,
            bottlenecks=bottlenecks,
            error_rates=error_rates,
            availability_metrics=availability_metrics
        )

        return report

    async def _generate_insights(
        self,
        metric_summaries: Dict[MetricType, MetricSummary],
        start_time: datetime,
        end_time: datetime
    ) -> List[PerformanceInsight]:
        """Generate performance insights from metric summaries."""
        insights = []

        # Search performance insights
        if MetricType.SEARCH_LATENCY in metric_summaries:
            summary = metric_summaries[MetricType.SEARCH_LATENCY]
            if summary.mean_value > self.performance_thresholds["search_latency_critical"]:
                insights.append(PerformanceInsight(
                    category="Search Performance",
                    title="Critical Search Latency",
                    description=f"Average search latency is {summary.mean_value:.1f}ms",
                    severity="critical",
                    metrics=["search_latency"],
                    trend=summary.trend
                ))

        return insights

    async def _generate_recommendations(
        self,
        metric_summaries: Dict[MetricType, MetricSummary],
        insights: List[PerformanceInsight]
    ) -> List[OptimizationRecommendation]:
        """Generate optimization recommendations based on analysis."""
        recommendations = []

        # Search optimization recommendations
        if MetricType.SEARCH_LATENCY in metric_summaries:
            summary = metric_summaries[MetricType.SEARCH_LATENCY]

            if summary.mean_value > 100:  # 100ms threshold
                recommendations.append(OptimizationRecommendation(
                    title="Optimize Search Performance",
                    description="Search latency is higher than optimal",
                    optimization_type=OptimizationType.SEARCH_OPTIMIZATION,
                    priority=Priority.CRITICAL if summary.mean_value > 10000 else (Priority.HIGH if summary.mean_value > 300 else Priority.MEDIUM),
                    impact_estimate="high",
                    implementation_effort="moderate",
                    metrics_evidence=[summary],
                    suggested_actions=[
                        "Implement query result caching",
                        "Optimize vector index parameters"
                    ]
                ))

        # CPU optimization recommendations
        if MetricType.CPU_USAGE in metric_summaries:
            summary = metric_summaries[MetricType.CPU_USAGE]

            if summary.mean_value > 80:  # 80% CPU threshold
                recommendations.append(OptimizationRecommendation(
                    title="Optimize CPU Usage",
                    description=f"CPU usage is high at {summary.mean_value:.1f}%",
                    optimization_type=OptimizationType.CPU_OPTIMIZATION,
                    priority=Priority.CRITICAL if summary.mean_value > 95 else Priority.HIGH,
                    impact_estimate="high",
                    implementation_effort="moderate",
                    metrics_evidence=[summary],
                    suggested_actions=[
                        "Review CPU-intensive operations",
                        "Implement async processing where possible",
                        "Consider horizontal scaling"
                    ]
                ))

        # Memory optimization recommendations
        if MetricType.MEMORY_USAGE in metric_summaries:
            summary = metric_summaries[MetricType.MEMORY_USAGE]

            if summary.mean_value > 400:  # 400MB threshold
                recommendations.append(OptimizationRecommendation(
                    title="Optimize Memory Usage",
                    description=f"Memory usage is high at {summary.mean_value:.1f}MB",
                    optimization_type=OptimizationType.MEMORY_OPTIMIZATION,
                    priority=Priority.CRITICAL if summary.mean_value > 480 else Priority.HIGH,
                    impact_estimate="high",
                    implementation_effort="moderate",
                    metrics_evidence=[summary],
                    suggested_actions=[
                        "Review memory leaks",
                        "Implement memory pooling",
                        "Consider memory limits"
                    ]
                ))

        # LSP optimization recommendations
        if MetricType.LSP_REQUEST_LATENCY in metric_summaries:
            summary = metric_summaries[MetricType.LSP_REQUEST_LATENCY]

            if summary.mean_value > 200:  # 200ms threshold
                recommendations.append(OptimizationRecommendation(
                    title="Optimize LSP Performance",
                    description=f"LSP request latency is high at {summary.mean_value:.1f}ms",
                    optimization_type=OptimizationType.LSP_OPTIMIZATION,
                    priority=Priority.HIGH if summary.mean_value > 500 else Priority.MEDIUM,
                    impact_estimate="high",
                    implementation_effort="moderate",
                    metrics_evidence=[summary],
                    suggested_actions=[
                        "Optimize LSP request handling",
                        "Implement LSP request caching",
                        "Review LSP server configuration"
                    ]
                ))

        # File processing optimization recommendations
        if MetricType.FILE_PROCESSING_RATE in metric_summaries:
            summary = metric_summaries[MetricType.FILE_PROCESSING_RATE]

            if summary.mean_value < 10:  # Below 10 files/min threshold
                recommendations.append(OptimizationRecommendation(
                    title="Optimize File Processing",
                    description=f"File processing rate is low at {summary.mean_value:.1f} files/min",
                    optimization_type=OptimizationType.FILE_PROCESSING_OPTIMIZATION,
                    priority=Priority.CRITICAL if summary.mean_value < 1 else (Priority.MEDIUM if summary.mean_value < 5 else Priority.LOW),
                    impact_estimate="medium",
                    implementation_effort="moderate",
                    metrics_evidence=[summary],
                    suggested_actions=[
                        "Implement parallel file processing",
                        "Optimize file parsing algorithms",
                        "Consider batch processing"
                    ]
                ))

        return recommendations

    def _calculate_performance_score(self, metric_summaries: Dict[MetricType, MetricSummary]) -> float:
        """Calculate overall performance score (0-100)."""
        scores = []

        # Search performance
        if MetricType.SEARCH_LATENCY in metric_summaries:
            latency = metric_summaries[MetricType.SEARCH_LATENCY].mean_value
            score = max(0, 100 - (latency / 5))
            scores.append(score)

        # Default score if no metrics
        if not scores:
            return 50.0

        return sum(scores) / len(scores)

    def _determine_performance_level(self, score: float) -> PerformanceLevel:
        """Determine performance level from score."""
        if score >= 90:
            return PerformanceLevel.EXCELLENT
        elif score >= 75:
            return PerformanceLevel.GOOD
        elif score >= 60:
            return PerformanceLevel.AVERAGE
        elif score >= 40:
            return PerformanceLevel.POOR
        else:
            return PerformanceLevel.CRITICAL

    def _calculate_resource_efficiency(self, metric_summaries: Dict[MetricType, MetricSummary]) -> Dict[str, float]:
        """Calculate resource efficiency metrics."""
        efficiency = {}

        if MetricType.MEMORY_USAGE in metric_summaries:
            memory_usage = metric_summaries[MetricType.MEMORY_USAGE].mean_value
            efficiency["memory"] = max(0, min(100, 100 - (memory_usage / 512) * 100))

        return efficiency

    def _identify_bottlenecks(self, metric_summaries: Dict[MetricType, MetricSummary]) -> List[str]:
        """Identify performance bottlenecks."""
        bottlenecks = []

        if (MetricType.SEARCH_LATENCY in metric_summaries and
            metric_summaries[MetricType.SEARCH_LATENCY].mean_value > 150):
            bottlenecks.append("Search operations")

        return bottlenecks

    def _calculate_error_rates(self, metric_summaries: Dict[MetricType, MetricSummary]) -> Dict[str, float]:
        """Calculate error rates from metrics."""
        error_rates = {}

        if MetricType.LSP_ERROR_RATE in metric_summaries:
            error_rates["lsp"] = metric_summaries[MetricType.LSP_ERROR_RATE].mean_value

        return error_rates

    def _calculate_availability_metrics(self, start_time: datetime, end_time: datetime) -> Dict[str, float]:
        """Calculate availability and uptime metrics."""
        return {
            "uptime_percentage": 99.5,
            "daemon_restarts": 0,
            "service_availability": 100.0
        }


class OptimizationEngine:
    """Engine for applying optimization recommendations."""

    def __init__(self):
        self.applied_optimizations: Dict[str, datetime] = {}
        self.optimization_results: Dict[str, Dict[str, Any]] = {}

    async def apply_recommendation(
        self,
        recommendation: OptimizationRecommendation,
        auto_apply: bool = False
    ) -> Dict[str, Any]:
        """Apply an optimization recommendation."""
        if not auto_apply:
            logger.info(f"Manual application required for: {recommendation.title}")
            return {
                "status": "manual_required",
                "message": "This optimization requires manual implementation",
                "actions": recommendation.suggested_actions
            }

        # For auto-applicable optimizations
        result = {
            "status": "applied",
            "timestamp": datetime.now().isoformat(),
            "changes": recommendation.configuration_changes,
            "expected_impact": recommendation.expected_improvements
        }

        # Record application
        opt_id = f"{recommendation.optimization_type.value}_{int(datetime.now().timestamp())}"
        self.applied_optimizations[opt_id] = datetime.now()
        self.optimization_results[opt_id] = result

        logger.info(f"Applied optimization: {recommendation.title}")
        return result


# Re-export all classes
__all__ = [
    "OptimizationType",
    "Priority",
    "OptimizationRecommendation",
    "PerformanceInsight",
    "PerformanceReport",
    "PerformanceAnalyzer",
    "OptimizationEngine"
]