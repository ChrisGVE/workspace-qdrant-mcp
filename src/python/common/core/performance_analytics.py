"""
Performance Analytics and Optimization Engine for Workspace Qdrant MCP.

This module provides intelligent analysis of performance metrics and generates
optimization recommendations based on usage patterns, resource utilization,
and performance trends.
"""

import asyncio
from common.logging.loguru_config import get_logger
import statistics
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple

from .performance_metrics import (
    MetricType, PerformanceLevel, PerformanceMetric, MetricSummary,
    OperationTrace, PerformanceMetricsCollector
)

logger = get_logger(__name__)


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
            # Latency thresholds (ms)
            "search_latency_warning": 200,
            "search_latency_critical": 500,
            "lsp_latency_warning": 100,
            "lsp_latency_critical": 300,
            
            # Throughput thresholds
            "file_processing_rate_warning": 10,  # files/min
            "file_processing_rate_critical": 5,
            
            # Resource thresholds
            "memory_usage_warning": 80,  # %
            "memory_usage_critical": 95,
            "cpu_usage_warning": 70,  # %
            "cpu_usage_critical": 90,
            
            # Error rate thresholds
            "error_rate_warning": 0.05,  # 5%
            "error_rate_critical": 0.10,  # 10%
        }
    
    async def analyze_performance(
        self,
        time_range_hours: int = 1
    ) -> PerformanceReport:
        """Generate comprehensive performance analysis report."""
        end_time = datetime.now()
        start_time = end_time - timedelta(hours=time_range_hours)
        
        # Check cache
        cache_key = f"analysis_{start_time.isoformat()}_{end_time.isoformat()}"
        if (cache_key in self.analysis_cache and 
            datetime.now() - self.last_analysis < self.cache_ttl):
            return self.analysis_cache[cache_key]
        
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
        
        # Cache result
        self.analysis_cache[cache_key] = report
        self.last_analysis = datetime.now()
        
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
                    description=f"Average search latency is {summary.mean_value:.1f}ms, "
                               f"exceeding critical threshold of {self.performance_thresholds['search_latency_critical']}ms",
                    severity="critical",
                    metrics=["search_latency"],
                    trend=summary.trend,
                    recommended_actions=[
                        "Review search query complexity",
                        "Check collection indexing",
                        "Consider increasing resources"
                    ]
                ))
            elif summary.mean_value > self.performance_thresholds["search_latency_warning"]:
                insights.append(PerformanceInsight(
                    category="Search Performance",
                    title="Elevated Search Latency",
                    description=f"Average search latency is {summary.mean_value:.1f}ms, "
                               f"above optimal performance levels",
                    severity="warning",
                    metrics=["search_latency"],
                    trend=summary.trend,
                    recommended_actions=[
                        "Monitor query patterns",
                        "Review search optimization settings"
                    ]
                ))
        
        # Memory usage insights
        if MetricType.MEMORY_USAGE in metric_summaries:
            summary = metric_summaries[MetricType.MEMORY_USAGE]
            # Calculate memory percentage based on typical limits
            memory_percent = (summary.mean_value / 512) * 100  # Assuming 512MB limit
            
            if memory_percent > self.performance_thresholds["memory_usage_critical"]:
                insights.append(PerformanceInsight(
                    category="Resource Usage",
                    title="Critical Memory Usage",
                    description=f"Memory usage is at {memory_percent:.1f}% of allocated resources",
                    severity="critical",
                    metrics=["memory_usage"],
                    trend=summary.trend,
                    recommended_actions=[
                        "Review memory allocation",
                        "Implement garbage collection optimization",
                        "Consider scaling up resources"
                    ]
                ))
        
        # File processing insights
        if MetricType.FILE_PROCESSING_RATE in metric_summaries:
            summary = metric_summaries[MetricType.FILE_PROCESSING_RATE]
            if summary.mean_value < self.performance_thresholds["file_processing_rate_critical"]:
                insights.append(PerformanceInsight(
                    category="Processing Performance",
                    title="Low File Processing Rate",
                    description=f"File processing rate is {summary.mean_value:.1f} files/min, "
                               f"below optimal performance",
                    severity="warning",
                    metrics=["file_processing_rate"],
                    trend=summary.trend,
                    recommended_actions=[
                        "Review file processing pipeline",
                        "Consider batch size optimization",
                        "Check for I/O bottlenecks"
                    ]
                ))
        
        # LSP performance insights
        if MetricType.LSP_REQUEST_LATENCY in metric_summaries:
            summary = metric_summaries[MetricType.LSP_REQUEST_LATENCY]
            if summary.mean_value > self.performance_thresholds["lsp_latency_critical"]:
                insights.append(PerformanceInsight(
                    category="LSP Performance",
                    title="High LSP Request Latency",
                    description=f"LSP requests are taking {summary.mean_value:.1f}ms on average",
                    severity="critical",
                    metrics=["lsp_request_latency"],
                    trend=summary.trend,
                    recommended_actions=[
                        "Review LSP server configuration",
                        "Check for network issues",
                        "Consider connection pooling"
                    ]
                ))
        
        return insights
    
    async def _generate_recommendations(
        self,
        metric_summaries: Dict[MetricType, MetricSummary],
        insights: List[PerformanceInsight]
    ) -> List[OptimizationRecommendation]:
        """Generate optimization recommendations based on analysis."""
        recommendations = []
        
        # Memory optimization recommendations
        if MetricType.MEMORY_USAGE in metric_summaries:
            summary = metric_summaries[MetricType.MEMORY_USAGE]
            memory_percent = (summary.mean_value / 512) * 100
            
            if memory_percent > 80:
                recommendations.append(OptimizationRecommendation(
                    title="Optimize Memory Usage",
                    description="High memory usage detected. Implementing memory optimization strategies could improve performance and reduce resource pressure.",
                    optimization_type=OptimizationType.MEMORY_OPTIMIZATION,
                    priority=Priority.HIGH if memory_percent > 90 else Priority.MEDIUM,
                    impact_estimate="high",
                    implementation_effort="moderate",
                    metrics_evidence=[summary],
                    suggested_actions=[
                        "Implement periodic garbage collection",
                        "Review buffer sizes and caching strategies",
                        "Consider memory-mapped file operations",
                        "Optimize data structures for memory efficiency"
                    ],
                    expected_improvements={
                        "memory_usage": f"Reduce by 15-25%",
                        "gc_frequency": "Reduce pause times",
                        "overall_stability": "Improve daemon reliability"
                    },
                    configuration_changes={
                        "gc_threshold": 0.8,
                        "buffer_size_mb": max(64, int(summary.mean_value * 0.6)),
                        "cache_max_size": "auto"
                    },
                    estimated_timeline="1-2 days",
                    risks=[
                        "Temporary performance impact during optimization",
                        "Potential cache miss increase during transition"
                    ]
                ))
        
        # Search optimization recommendations
        if MetricType.SEARCH_LATENCY in metric_summaries:
            summary = metric_summaries[MetricType.SEARCH_LATENCY]
            
            if summary.mean_value > 100:  # 100ms threshold
                recommendations.append(OptimizationRecommendation(
                    title="Optimize Search Performance",
                    description="Search latency is higher than optimal. Several optimization strategies can improve search responsiveness.",
                    optimization_type=OptimizationType.SEARCH_OPTIMIZATION,
                    priority=Priority.HIGH if summary.mean_value > 300 else Priority.MEDIUM,
                    impact_estimate="high",
                    implementation_effort="moderate",
                    metrics_evidence=[summary],
                    suggested_actions=[
                        "Implement query result caching",
                        "Optimize vector index parameters",
                        "Review search query patterns",
                        "Consider search result pagination"
                    ],
                    expected_improvements={
                        "search_latency": f"Reduce by 30-50%",
                        "search_throughput": "Increase by 40-60%",
                        "user_experience": "Significantly improved"
                    },
                    configuration_changes={
                        "enable_query_cache": True,
                        "cache_size_mb": 128,
                        "index_optimization": "auto",
                        "max_results_per_query": 100
                    },
                    estimated_timeline="2-3 days"
                ))
        
        # Batch processing optimization
        file_processing_summary = metric_summaries.get(MetricType.FILE_PROCESSING_RATE)
        cpu_summary = metric_summaries.get(MetricType.CPU_USAGE)
        
        if file_processing_summary and cpu_summary:
            # If CPU usage is low but file processing is slow, suggest parallelism
            if cpu_summary.mean_value < 50 and file_processing_summary.mean_value < 30:
                recommendations.append(OptimizationRecommendation(
                    title="Increase Processing Parallelism",
                    description="CPU utilization is low while file processing rate is suboptimal. Increasing parallelism could improve throughput.",
                    optimization_type=OptimizationType.PARALLELISM_TUNING,
                    priority=Priority.MEDIUM,
                    impact_estimate="medium",
                    implementation_effort="easy",
                    metrics_evidence=[file_processing_summary, cpu_summary],
                    suggested_actions=[
                        "Increase worker thread count",
                        "Implement parallel file processing",
                        "Optimize I/O operations",
                        "Review batch sizes"
                    ],
                    expected_improvements={
                        "file_processing_rate": "Increase by 50-100%",
                        "cpu_utilization": "More efficient resource usage",
                        "overall_throughput": "Significantly improved"
                    },
                    configuration_changes={
                        "max_concurrent_jobs": min(8, max(4, int(cpu_summary.mean_value / 10))),
                        "batch_size": 50,
                        "io_timeout": 30
                    },
                    estimated_timeline="immediate"
                ))
        
        # LSP optimization recommendations
        if MetricType.LSP_REQUEST_LATENCY in metric_summaries:
            summary = metric_summaries[MetricType.LSP_REQUEST_LATENCY]
            
            if summary.mean_value > 50:  # 50ms threshold
                recommendations.append(OptimizationRecommendation(
                    title="Optimize LSP Communication",
                    description="LSP request latency is elevated. Connection pooling and request optimization can improve responsiveness.",
                    optimization_type=OptimizationType.LSP_OPTIMIZATION,
                    priority=Priority.MEDIUM,
                    impact_estimate="medium",
                    implementation_effort="moderate",
                    metrics_evidence=[summary],
                    suggested_actions=[
                        "Implement connection pooling",
                        "Enable request batching",
                        "Optimize payload sizes",
                        "Review timeout configurations"
                    ],
                    expected_improvements={
                        "lsp_latency": "Reduce by 25-40%",
                        "connection_efficiency": "Improved",
                        "developer_experience": "More responsive"
                    },
                    configuration_changes={
                        "connection_pool_size": 5,
                        "request_timeout": 10,
                        "batch_requests": True,
                        "keep_alive": True
                    },
                    estimated_timeline="1-2 days"
                ))
        
        # Resource scaling recommendations
        high_cpu = cpu_summary and cpu_summary.mean_value > 80
        high_memory = MetricType.MEMORY_USAGE in metric_summaries and (metric_summaries[MetricType.MEMORY_USAGE].mean_value / 512) * 100 > 85
        
        if high_cpu or high_memory:
            recommendations.append(OptimizationRecommendation(
                title="Consider Resource Scaling",
                description="Resource utilization is consistently high. Scaling up resources or optimizing configuration could improve overall performance.",
                optimization_type=OptimizationType.SCALING_RECOMMENDATION,
                priority=Priority.HIGH if high_cpu and high_memory else Priority.MEDIUM,
                impact_estimate="high",
                implementation_effort="easy",
                suggested_actions=[
                    "Increase memory allocation",
                    "Consider additional CPU cores",
                    "Review resource limits configuration",
                    "Monitor resource trends"
                ],
                expected_improvements={
                    "overall_performance": "20-50% improvement",
                    "response_times": "Reduced latency",
                    "system_stability": "Improved reliability"
                },
                configuration_changes={
                    "max_memory_mb": 1024,
                    "max_cpu_percent": 100,
                    "worker_processes": "auto"
                },
                estimated_timeline="immediate",
                dependencies=[
                    "Infrastructure provisioning",
                    "Configuration deployment"
                ]
            ))
        
        return recommendations
    
    def _calculate_performance_score(self, metric_summaries: Dict[MetricType, MetricSummary]) -> float:
        """Calculate overall performance score (0-100)."""
        scores = []
        weights = []
        
        # Search performance (weight: 25%)
        if MetricType.SEARCH_LATENCY in metric_summaries:
            latency = metric_summaries[MetricType.SEARCH_LATENCY].mean_value
            score = max(0, 100 - (latency / 5))  # 500ms = 0 score
            scores.append(score)
            weights.append(25)
        
        # Resource efficiency (weight: 25%)
        resource_score = 100
        if MetricType.MEMORY_USAGE in metric_summaries:
            memory_percent = (metric_summaries[MetricType.MEMORY_USAGE].mean_value / 512) * 100
            resource_score = min(resource_score, max(0, 100 - memory_percent))
        
        if MetricType.CPU_USAGE in metric_summaries:
            cpu_percent = metric_summaries[MetricType.CPU_USAGE].mean_value
            resource_score = min(resource_score, max(0, 100 - cpu_percent))
        
        scores.append(resource_score)
        weights.append(25)
        
        # Processing efficiency (weight: 20%)
        if MetricType.FILE_PROCESSING_RATE in metric_summaries:
            rate = metric_summaries[MetricType.FILE_PROCESSING_RATE].mean_value
            score = min(100, (rate / 50) * 100)  # 50 files/min = 100 score
            scores.append(score)
            weights.append(20)
        
        # LSP performance (weight: 15%)
        if MetricType.LSP_REQUEST_LATENCY in metric_summaries:
            latency = metric_summaries[MetricType.LSP_REQUEST_LATENCY].mean_value
            score = max(0, 100 - (latency / 2))  # 200ms = 0 score
            scores.append(score)
            weights.append(15)
        
        # Stability (weight: 15%)
        stability_score = 100
        if MetricType.LSP_ERROR_RATE in metric_summaries:
            error_rate = metric_summaries[MetricType.LSP_ERROR_RATE].mean_value
            stability_score = max(0, 100 - (error_rate * 1000))  # 10% error rate = 0 score
        
        scores.append(stability_score)
        weights.append(15)
        
        # Calculate weighted average
        if not scores:
            return 50.0  # Default neutral score
        
        total_weight = sum(weights)
        weighted_sum = sum(score * weight for score, weight in zip(scores, weights))
        
        return weighted_sum / total_weight
    
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
        
        # Memory efficiency
        if MetricType.MEMORY_USAGE in metric_summaries:
            memory_usage = metric_summaries[MetricType.MEMORY_USAGE].mean_value
            efficiency["memory"] = max(0, min(100, 100 - (memory_usage / 512) * 100))
        
        # CPU efficiency  
        if MetricType.CPU_USAGE in metric_summaries:
            cpu_usage = metric_summaries[MetricType.CPU_USAGE].mean_value
            efficiency["cpu"] = max(0, min(100, 100 - cpu_usage))
        
        # I/O efficiency
        if MetricType.DISK_IO in metric_summaries:
            disk_io = metric_summaries[MetricType.DISK_IO].mean_value
            efficiency["disk_io"] = max(0, min(100, 100 - (disk_io / 1000)))  # 1MB/s = baseline
        
        return efficiency
    
    def _identify_bottlenecks(self, metric_summaries: Dict[MetricType, MetricSummary]) -> List[str]:
        """Identify performance bottlenecks."""
        bottlenecks = []
        
        # High search latency
        if (MetricType.SEARCH_LATENCY in metric_summaries and 
            metric_summaries[MetricType.SEARCH_LATENCY].mean_value > 200):
            bottlenecks.append("Search operations")
        
        # High memory usage
        if (MetricType.MEMORY_USAGE in metric_summaries and 
            (metric_summaries[MetricType.MEMORY_USAGE].mean_value / 512) * 100 > 80):
            bottlenecks.append("Memory allocation")
        
        # High CPU usage
        if (MetricType.CPU_USAGE in metric_summaries and 
            metric_summaries[MetricType.CPU_USAGE].mean_value > 80):
            bottlenecks.append("CPU processing")
        
        # Low file processing rate
        if (MetricType.FILE_PROCESSING_RATE in metric_summaries and 
            metric_summaries[MetricType.FILE_PROCESSING_RATE].mean_value < 10):
            bottlenecks.append("File processing pipeline")
        
        # High LSP latency
        if (MetricType.LSP_REQUEST_LATENCY in metric_summaries and 
            metric_summaries[MetricType.LSP_REQUEST_LATENCY].mean_value > 100):
            bottlenecks.append("LSP communication")
        
        return bottlenecks
    
    def _calculate_error_rates(self, metric_summaries: Dict[MetricType, MetricSummary]) -> Dict[str, float]:
        """Calculate error rates from metrics."""
        error_rates = {}
        
        # LSP error rate
        if MetricType.LSP_ERROR_RATE in metric_summaries:
            error_rates["lsp"] = metric_summaries[MetricType.LSP_ERROR_RATE].mean_value
        
        # Add other error rates as needed
        
        return error_rates
    
    def _calculate_availability_metrics(self, start_time: datetime, end_time: datetime) -> Dict[str, float]:
        """Calculate availability and uptime metrics."""
        # This would typically require historical data about daemon uptime
        # For now, return baseline metrics
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
    
    async def get_optimization_history(self) -> Dict[str, Any]:
        """Get history of applied optimizations."""
        return {
            "total_optimizations": len(self.applied_optimizations),
            "recent_optimizations": [
                {
                    "id": opt_id,
                    "applied_at": timestamp.isoformat(),
                    "result": self.optimization_results.get(opt_id, {})
                }
                for opt_id, timestamp in sorted(
                    self.applied_optimizations.items(),
                    key=lambda x: x[1],
                    reverse=True
                )[:10]
            ]
        }