"""
Integrated Performance Monitoring System for Workspace Qdrant MCP.

This module provides the main performance monitoring coordinator that integrates
metrics collection, analytics, storage, and optimization recommendations into
a unified system for daemon performance management.
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Set
from dataclasses import dataclass

from .performance_metrics import (
    MetricType, PerformanceMetricsCollector, PerformanceProfiler, 
    PerformanceMetric, OperationTrace
)
from .performance_analytics import (
    PerformanceAnalyzer, PerformanceReport, OptimizationRecommendation,
    OptimizationEngine, Priority
)
from .performance_storage import get_performance_storage, PerformanceStorage
from .resource_manager import ResourceAlert, ResourceUsage

logger = logging.getLogger(__name__)


@dataclass
class PerformanceAlert:
    """Enhanced performance alert with optimization context."""
    
    timestamp: datetime
    project_id: str
    alert_type: str  # "threshold", "trend", "anomaly"
    severity: str  # "info", "warning", "critical"
    metric_type: str
    current_value: float
    threshold_value: Optional[float] = None
    trend: Optional[str] = None
    message: str = ""
    recommendations: List[str] = None
    auto_actionable: bool = False
    
    def __post_init__(self):
        if self.recommendations is None:
            self.recommendations = []


class PerformanceMonitor:
    """
    Comprehensive performance monitoring system for daemon instances.
    
    Coordinates metrics collection, analysis, storage, and optimization
    recommendations for optimal daemon performance.
    """
    
    def __init__(self, project_id: str):
        self.project_id = project_id
        self.running = False
        
        # Core components
        self.metrics_collector = PerformanceMetricsCollector(project_id)
        self.analyzer = PerformanceAnalyzer(self.metrics_collector)
        self.optimization_engine = OptimizationEngine()
        self.storage: Optional[PerformanceStorage] = None
        
        # Monitoring configuration
        self.collection_interval = 10.0  # seconds
        self.analysis_interval = 300.0  # 5 minutes
        self.storage_batch_size = 100
        self.alert_cooldown = timedelta(minutes=15)
        
        # Active monitoring tasks
        self._collection_task: Optional[asyncio.Task] = None
        self._analysis_task: Optional[asyncio.Task] = None
        self._storage_task: Optional[asyncio.Task] = None
        
        # Alert management
        self.active_alerts: Dict[str, PerformanceAlert] = {}
        self.alert_callbacks: List[callable] = []
        self.last_alert_times: Dict[str, datetime] = {}
        
        # Performance thresholds
        self.alert_thresholds = {
            MetricType.SEARCH_LATENCY: {
                "warning": 200,  # ms
                "critical": 500
            },
            MetricType.MEMORY_USAGE: {
                "warning": 400,  # MB
                "critical": 480
            },
            MetricType.CPU_USAGE: {
                "warning": 70,  # %
                "critical": 90
            },
            MetricType.LSP_REQUEST_LATENCY: {
                "warning": 100,  # ms
                "critical": 300
            },
            MetricType.FILE_PROCESSING_RATE: {
                "warning": 10,  # files/min (inverted - low is bad)
                "critical": 5
            }
        }
        
        # Metrics buffer for batch processing
        self._metrics_buffer: List[PerformanceMetric] = []
        self._traces_buffer: List[OperationTrace] = []
        
        # Setup callbacks
        self.metrics_collector.add_metric_callback(self._on_metric_collected)
        self.metrics_collector.add_operation_callback(self._on_operation_completed)
    
    async def start(self):
        """Start the performance monitoring system."""
        if self.running:
            logger.warning(f"Performance monitor already running for {self.project_id}")
            return
        
        logger.info(f"Starting performance monitor for {self.project_id}")
        
        # Initialize storage
        self.storage = await get_performance_storage(self.project_id)
        
        # Start monitoring tasks
        self.running = True
        self._collection_task = asyncio.create_task(self._collection_loop())
        self._analysis_task = asyncio.create_task(self._analysis_loop())
        self._storage_task = asyncio.create_task(self._storage_loop())
        
        logger.info(f"Performance monitor started for {self.project_id}")
    
    async def stop(self):
        """Stop the performance monitoring system."""
        if not self.running:
            return
        
        logger.info(f"Stopping performance monitor for {self.project_id}")
        
        self.running = False
        
        # Cancel monitoring tasks
        for task in [self._collection_task, self._analysis_task, self._storage_task]:
            if task and not task.done():
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
        
        # Flush remaining data to storage
        await self._flush_to_storage()
        
        logger.info(f"Performance monitor stopped for {self.project_id}")
    
    def add_alert_callback(self, callback: callable):
        """Add callback for performance alerts."""
        self.alert_callbacks.append(callback)
    
    async def record_search_performance(
        self,
        query: str,
        result_count: int,
        latency_ms: float,
        relevance_score: Optional[float] = None,
        operation_id: Optional[str] = None
    ):
        """Record search performance metrics."""
        await self.metrics_collector.record_search_performance(
            query, result_count, latency_ms, relevance_score, operation_id
        )
    
    async def record_lsp_performance(
        self,
        method: str,
        latency_ms: float,
        response_size_bytes: int,
        success: bool,
        operation_id: Optional[str] = None
    ):
        """Record LSP operation performance."""
        await self.metrics_collector.record_lsp_performance(
            method, latency_ms, response_size_bytes, success, operation_id
        )
    
    async def record_file_processing(
        self,
        file_path: str,
        processing_time_seconds: float,
        file_size_bytes: int,
        success: bool,
        operation_id: Optional[str] = None
    ):
        """Record file processing performance."""
        await self.metrics_collector.record_file_processing(
            file_path, processing_time_seconds, file_size_bytes, success, operation_id
        )
    
    def profile_operation(self, operation_type: str, context: Optional[Dict[str, Any]] = None) -> PerformanceProfiler:
        """Create a performance profiler for an operation."""
        return self.metrics_collector.profile_operation(operation_type, context)
    
    async def get_current_performance_report(self, time_range_hours: int = 1) -> PerformanceReport:
        """Get current performance analysis report."""
        return await self.analyzer.analyze_performance(time_range_hours)
    
    async def get_optimization_recommendations(self) -> List[OptimizationRecommendation]:
        """Get current optimization recommendations."""
        report = await self.get_current_performance_report()
        return report.recommendations
    
    async def apply_optimization(
        self,
        recommendation: OptimizationRecommendation,
        auto_apply: bool = False
    ) -> Dict[str, Any]:
        """Apply an optimization recommendation."""
        return await self.optimization_engine.apply_recommendation(recommendation, auto_apply)
    
    async def get_performance_summary(self) -> Dict[str, Any]:
        """Get a summary of current performance status."""
        report = await self.get_current_performance_report()
        
        # Get recent alerts
        recent_alerts = [
            alert for alert in self.active_alerts.values()
            if alert.timestamp > datetime.now() - timedelta(hours=1)
        ]
        
        # Get storage stats
        storage_stats = await self.storage.get_storage_stats() if self.storage else {}
        
        return {
            "project_id": self.project_id,
            "performance_score": report.overall_performance_score,
            "performance_level": report.performance_level.value,
            "active_alerts": len(self.active_alerts),
            "recent_alerts": len(recent_alerts),
            "recommendations_count": len(report.recommendations),
            "high_priority_recommendations": len([
                r for r in report.recommendations 
                if r.priority in [Priority.CRITICAL, Priority.HIGH]
            ]),
            "resource_efficiency": report.resource_efficiency,
            "bottlenecks": report.bottlenecks,
            "storage_stats": storage_stats,
            "monitoring_status": "running" if self.running else "stopped"
        }
    
    async def _collection_loop(self):
        """Main metrics collection loop."""
        logger.debug(f"Started collection loop for {self.project_id}")
        
        try:
            while self.running:
                try:
                    # Collect system resource metrics
                    await self.metrics_collector.record_system_resources()
                    
                    await asyncio.sleep(self.collection_interval)
                    
                except asyncio.CancelledError:
                    break
                except Exception as e:
                    logger.error(f"Error in collection loop: {e}")
                    await asyncio.sleep(5.0)  # Brief pause before retry
        
        except asyncio.CancelledError:
            logger.debug(f"Collection loop cancelled for {self.project_id}")
    
    async def _analysis_loop(self):
        """Main performance analysis loop."""
        logger.debug(f"Started analysis loop for {self.project_id}")
        
        try:
            while self.running:
                try:
                    # Generate performance report
                    report = await self.analyzer.analyze_performance()
                    
                    # Store the report
                    if self.storage:
                        await self.storage.store_performance_report(report)
                    
                    # Check for new alerts based on analysis
                    await self._check_performance_alerts(report)
                    
                    await asyncio.sleep(self.analysis_interval)
                    
                except asyncio.CancelledError:
                    break
                except Exception as e:
                    logger.error(f"Error in analysis loop: {e}")
                    await asyncio.sleep(30.0)  # Longer pause for analysis errors
        
        except asyncio.CancelledError:
            logger.debug(f"Analysis loop cancelled for {self.project_id}")
    
    async def _storage_loop(self):
        """Main storage loop for batching metrics to persistent storage."""
        logger.debug(f"Started storage loop for {self.project_id}")
        
        try:
            while self.running:
                try:
                    await asyncio.sleep(30.0)  # Store every 30 seconds
                    await self._flush_to_storage()
                    
                except asyncio.CancelledError:
                    break
                except Exception as e:
                    logger.error(f"Error in storage loop: {e}")
                    await asyncio.sleep(10.0)
        
        except asyncio.CancelledError:
            logger.debug(f"Storage loop cancelled for {self.project_id}")
    
    def _on_metric_collected(self, metric: PerformanceMetric):
        """Handle new metric collection."""
        self._metrics_buffer.append(metric)
        
        # Check for immediate alerts
        asyncio.create_task(self._check_metric_alerts(metric))
    
    def _on_operation_completed(self, trace: OperationTrace):
        """Handle operation completion."""
        self._traces_buffer.append(trace)
        
        # Log operation performance
        if trace.duration:
            logger.debug(
                f"Operation {trace.operation_type} completed in {trace.duration:.2f}s "
                f"with {len(trace.metrics)} metrics"
            )
    
    async def _flush_to_storage(self):
        """Flush buffered data to storage."""
        if not self.storage:
            return
        
        try:
            # Store metrics batch
            if self._metrics_buffer:
                metrics_to_store = self._metrics_buffer[:self.storage_batch_size]
                self._metrics_buffer = self._metrics_buffer[self.storage_batch_size:]
                await self.storage.store_metrics_batch(metrics_to_store)
            
            # Store operation traces
            for trace in self._traces_buffer:
                await self.storage.store_operation_trace(trace)
            self._traces_buffer.clear()
            
        except Exception as e:
            logger.error(f"Failed to flush data to storage: {e}")
    
    async def _check_metric_alerts(self, metric: PerformanceMetric):
        """Check if a metric triggers any alerts."""
        if metric.metric_type not in self.alert_thresholds:
            return
        
        thresholds = self.alert_thresholds[metric.metric_type]
        alert_key = f"{metric.metric_type.value}_{self.project_id}"
        
        # Check cooldown
        if (alert_key in self.last_alert_times and 
            datetime.now() - self.last_alert_times[alert_key] < self.alert_cooldown):
            return
        
        alert = None
        
        # Check thresholds
        if "critical" in thresholds:
            if (metric.metric_type == MetricType.FILE_PROCESSING_RATE and 
                metric.value < thresholds["critical"]):
                # Inverted threshold for processing rate
                alert = PerformanceAlert(
                    timestamp=datetime.now(),
                    project_id=self.project_id,
                    alert_type="threshold",
                    severity="critical",
                    metric_type=metric.metric_type.value,
                    current_value=metric.value,
                    threshold_value=thresholds["critical"],
                    message=f"File processing rate critically low: {metric.value:.1f} files/min",
                    recommendations=[
                        "Review file processing pipeline",
                        "Check for I/O bottlenecks",
                        "Consider increasing parallelism"
                    ]
                )
            elif (metric.metric_type != MetricType.FILE_PROCESSING_RATE and 
                  metric.value > thresholds["critical"]):
                alert = PerformanceAlert(
                    timestamp=datetime.now(),
                    project_id=self.project_id,
                    alert_type="threshold",
                    severity="critical",
                    metric_type=metric.metric_type.value,
                    current_value=metric.value,
                    threshold_value=thresholds["critical"],
                    message=f"{metric.metric_type.value} critically high: {metric.value:.1f} {metric.unit}",
                    recommendations=self._get_metric_recommendations(metric.metric_type)
                )
        
        elif "warning" in thresholds:
            if (metric.metric_type == MetricType.FILE_PROCESSING_RATE and 
                metric.value < thresholds["warning"]):
                alert = PerformanceAlert(
                    timestamp=datetime.now(),
                    project_id=self.project_id,
                    alert_type="threshold",
                    severity="warning",
                    metric_type=metric.metric_type.value,
                    current_value=metric.value,
                    threshold_value=thresholds["warning"],
                    message=f"File processing rate low: {metric.value:.1f} files/min",
                    recommendations=self._get_metric_recommendations(metric.metric_type)
                )
            elif (metric.metric_type != MetricType.FILE_PROCESSING_RATE and 
                  metric.value > thresholds["warning"]):
                alert = PerformanceAlert(
                    timestamp=datetime.now(),
                    project_id=self.project_id,
                    alert_type="threshold",
                    severity="warning",
                    metric_type=metric.metric_type.value,
                    current_value=metric.value,
                    threshold_value=thresholds["warning"],
                    message=f"{metric.metric_type.value} elevated: {metric.value:.1f} {metric.unit}",
                    recommendations=self._get_metric_recommendations(metric.metric_type)
                )
        
        if alert:
            self.active_alerts[alert_key] = alert
            self.last_alert_times[alert_key] = datetime.now()
            await self._notify_alert(alert)
    
    async def _check_performance_alerts(self, report: PerformanceReport):
        """Check for performance alerts based on analysis report."""
        # Check for degrading trends
        for metric_type, summary in report.metric_summaries.items():
            if summary.trend == "degrading":
                alert_key = f"trend_{metric_type.value}_{self.project_id}"
                
                if (alert_key not in self.last_alert_times or 
                    datetime.now() - self.last_alert_times[alert_key] > self.alert_cooldown):
                    
                    alert = PerformanceAlert(
                        timestamp=datetime.now(),
                        project_id=self.project_id,
                        alert_type="trend",
                        severity="warning",
                        metric_type=metric_type.value,
                        current_value=summary.mean_value,
                        trend=summary.trend,
                        message=f"{metric_type.value} showing degrading trend",
                        recommendations=self._get_metric_recommendations(metric_type)
                    )
                    
                    self.active_alerts[alert_key] = alert
                    self.last_alert_times[alert_key] = datetime.now()
                    await self._notify_alert(alert)
    
    def _get_metric_recommendations(self, metric_type: MetricType) -> List[str]:
        """Get recommendations for specific metric types."""
        recommendations = {
            MetricType.SEARCH_LATENCY: [
                "Review search query complexity",
                "Check collection indexing",
                "Consider query result caching"
            ],
            MetricType.MEMORY_USAGE: [
                "Review memory allocation",
                "Implement garbage collection optimization",
                "Check for memory leaks"
            ],
            MetricType.CPU_USAGE: [
                "Review CPU-intensive operations",
                "Consider load balancing",
                "Optimize algorithms"
            ],
            MetricType.LSP_REQUEST_LATENCY: [
                "Check LSP server configuration",
                "Review network connectivity",
                "Consider connection pooling"
            ],
            MetricType.FILE_PROCESSING_RATE: [
                "Review file processing pipeline",
                "Check for I/O bottlenecks",
                "Consider increasing parallelism"
            ]
        }
        
        return recommendations.get(metric_type, ["Review system configuration"])
    
    async def _notify_alert(self, alert: PerformanceAlert):
        """Notify alert callbacks about new alert."""
        logger.warning(f"Performance alert: {alert.message}")
        
        for callback in self.alert_callbacks:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(alert)
                else:
                    callback(alert)
            except Exception as e:
                logger.error(f"Alert callback failed: {e}")
    
    async def resolve_alert(self, alert_key: str):
        """Resolve an active alert."""
        if alert_key in self.active_alerts:
            alert = self.active_alerts[alert_key]
            del self.active_alerts[alert_key]
            logger.info(f"Resolved alert: {alert.message}")
    
    async def get_active_alerts(self) -> List[PerformanceAlert]:
        """Get list of active alerts."""
        return list(self.active_alerts.values())


# Global performance monitors per project
_performance_monitors: Dict[str, PerformanceMonitor] = {}
_monitor_lock = asyncio.Lock()


async def get_performance_monitor(project_id: str) -> PerformanceMonitor:
    """Get or create a performance monitor for a project."""
    global _performance_monitors
    
    async with _monitor_lock:
        if project_id not in _performance_monitors:
            monitor = PerformanceMonitor(project_id)
            _performance_monitors[project_id] = monitor
            # Auto-start the monitor
            await monitor.start()
        
        return _performance_monitors[project_id]


async def stop_performance_monitor(project_id: str):
    """Stop and remove a performance monitor."""
    global _performance_monitors
    
    async with _monitor_lock:
        if project_id in _performance_monitors:
            monitor = _performance_monitors[project_id]
            await monitor.stop()
            del _performance_monitors[project_id]


async def get_all_performance_summaries() -> Dict[str, Dict[str, Any]]:
    """Get performance summaries for all active monitors."""
    summaries = {}
    
    async with _monitor_lock:
        for project_id, monitor in _performance_monitors.items():
            try:
                summaries[project_id] = await monitor.get_performance_summary()
            except Exception as e:
                logger.error(f"Failed to get summary for {project_id}: {e}")
                summaries[project_id] = {"error": str(e)}
    
    return summaries


async def cleanup_all_performance_monitors():
    """Stop and cleanup all performance monitors."""
    global _performance_monitors
    
    async with _monitor_lock:
        for monitor in list(_performance_monitors.values()):
            try:
                await monitor.stop()
            except Exception as e:
                logger.error(f"Failed to stop monitor: {e}")
        
        _performance_monitors.clear()