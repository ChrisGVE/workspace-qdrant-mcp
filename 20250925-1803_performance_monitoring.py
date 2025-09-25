#!/usr/bin/env python3
"""
Performance Monitoring and Optimization System for Workspace Qdrant MCP

Implements comprehensive performance monitoring with automatic optimization,
health reporting, real-time metrics collection, and resource management.

Created: 2025-09-25T18:03:00+02:00
"""

import asyncio
import gc
import logging
import psutil
import threading
import time
import tracemalloc
from collections import defaultdict, deque
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Callable, Union
import json
import statistics
import weakref


class MetricType(Enum):
    """Performance metric types"""
    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    TIMER = "timer"


class HealthStatus(Enum):
    """System health status levels"""
    HEALTHY = "healthy"
    WARNING = "warning"
    CRITICAL = "critical"
    UNKNOWN = "unknown"


class OptimizationType(Enum):
    """Types of optimization recommendations"""
    MEMORY = "memory"
    CPU = "cpu"
    IO = "io"
    NETWORK = "network"
    CONFIGURATION = "configuration"
    ALGORITHM = "algorithm"


@dataclass
class PerformanceMetric:
    """Individual performance metric"""
    name: str
    value: float
    timestamp: datetime
    metric_type: MetricType
    tags: Dict[str, str]
    metadata: Dict[str, Any]


@dataclass
class HealthReport:
    """System health report"""
    timestamp: datetime
    overall_status: HealthStatus
    component_health: Dict[str, HealthStatus]
    metrics: Dict[str, float]
    alerts: List[str]
    recommendations: List[str]
    uptime: float


@dataclass
class OptimizationRecommendation:
    """Performance optimization recommendation"""
    recommendation_id: str
    optimization_type: OptimizationType
    description: str
    impact_estimate: float
    implementation_difficulty: int  # 1-10 scale
    estimated_benefit: str
    code_changes_required: List[str]
    configuration_changes: Dict[str, Any]
    priority: int  # 1-10 scale


@dataclass
class BenchmarkResult:
    """Performance benchmark result"""
    test_name: str
    timestamp: datetime
    duration: float
    throughput: float
    memory_usage: Dict[str, float]
    cpu_usage: float
    success_rate: float
    error_count: int
    metadata: Dict[str, Any]


class MetricsCollector:
    """Real-time metrics collection system"""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.logger = logging.getLogger(__name__)
        self.metrics_buffer: deque = deque(maxlen=self.config.get('buffer_size', 10000))
        self.metric_aggregators: Dict[str, Callable] = {}
        self.collection_interval = self.config.get('collection_interval', 1.0)
        self.is_collecting = False
        self._collection_thread: Optional[threading.Thread] = None
        self._lock = threading.RLock()

        # System metrics
        self.process = psutil.Process()
        self.system_start_time = time.time()

        # Memory tracking
        if self.config.get('enable_memory_profiling', False):
            tracemalloc.start()

    def start_collection(self):
        """Start metrics collection in background thread"""
        if self.is_collecting:
            return

        self.is_collecting = True
        self._collection_thread = threading.Thread(target=self._collection_loop, daemon=True)
        self._collection_thread.start()
        self.logger.info("Metrics collection started")

    def stop_collection(self):
        """Stop metrics collection"""
        self.is_collecting = False
        if self._collection_thread and self._collection_thread.is_alive():
            self._collection_thread.join(timeout=5.0)
        self.logger.info("Metrics collection stopped")

    def _collection_loop(self):
        """Main collection loop running in background thread"""
        try:
            while self.is_collecting:
                try:
                    self._collect_system_metrics()
                    time.sleep(self.collection_interval)
                except Exception as e:
                    self.logger.error(f"Error in metrics collection loop: {e}")
                    time.sleep(self.collection_interval * 2)  # Back off on error

        except Exception as e:
            self.logger.error(f"Fatal error in collection loop: {e}")
            self.is_collecting = False

    def _collect_system_metrics(self):
        """Collect system-level performance metrics"""
        timestamp = datetime.now()

        try:
            # Memory metrics
            memory_info = self.process.memory_info()
            self.record_metric("memory.rss", memory_info.rss / 1024 / 1024, MetricType.GAUGE, timestamp)
            self.record_metric("memory.vms", memory_info.vms / 1024 / 1024, MetricType.GAUGE, timestamp)

            # CPU metrics
            cpu_percent = self.process.cpu_percent()
            self.record_metric("cpu.usage", cpu_percent, MetricType.GAUGE, timestamp)

            # System memory
            system_memory = psutil.virtual_memory()
            self.record_metric("system.memory.total", system_memory.total / 1024 / 1024 / 1024, MetricType.GAUGE, timestamp)
            self.record_metric("system.memory.available", system_memory.available / 1024 / 1024 / 1024, MetricType.GAUGE, timestamp)
            self.record_metric("system.memory.percent", system_memory.percent, MetricType.GAUGE, timestamp)

            # System CPU
            system_cpu = psutil.cpu_percent()
            self.record_metric("system.cpu.usage", system_cpu, MetricType.GAUGE, timestamp)

            # Disk I/O
            io_counters = self.process.io_counters()
            self.record_metric("io.read_bytes", io_counters.read_bytes, MetricType.COUNTER, timestamp)
            self.record_metric("io.write_bytes", io_counters.write_bytes, MetricType.COUNTER, timestamp)

            # Thread count
            self.record_metric("threads.count", self.process.num_threads(), MetricType.GAUGE, timestamp)

            # File descriptors (Unix only)
            try:
                self.record_metric("files.open", self.process.num_fds(), MetricType.GAUGE, timestamp)
            except (AttributeError, psutil.AccessDenied):
                pass  # Not available on Windows or access denied

            # Uptime
            uptime = time.time() - self.system_start_time
            self.record_metric("uptime", uptime, MetricType.GAUGE, timestamp)

            # Memory profiling (if enabled)
            if self.config.get('enable_memory_profiling', False) and tracemalloc.is_tracing():
                current, peak = tracemalloc.get_traced_memory()
                self.record_metric("memory.traced.current", current / 1024 / 1024, MetricType.GAUGE, timestamp)
                self.record_metric("memory.traced.peak", peak / 1024 / 1024, MetricType.GAUGE, timestamp)

        except Exception as e:
            self.logger.error(f"Error collecting system metrics: {e}")

    def record_metric(self, name: str, value: float, metric_type: MetricType,
                     timestamp: Optional[datetime] = None, tags: Optional[Dict[str, str]] = None,
                     metadata: Optional[Dict[str, Any]] = None):
        """Record a performance metric with thread safety"""
        if timestamp is None:
            timestamp = datetime.now()

        metric = PerformanceMetric(
            name=name,
            value=value,
            timestamp=timestamp,
            metric_type=metric_type,
            tags=tags or {},
            metadata=metadata or {}
        )

        with self._lock:
            self.metrics_buffer.append(metric)

            # Apply aggregator if configured
            if name in self.metric_aggregators:
                try:
                    self.metric_aggregators[name](metric)
                except Exception as e:
                    self.logger.error(f"Error in metric aggregator for {name}: {e}")

    def get_metrics(self, metric_name: Optional[str] = None,
                   time_range: Optional[Tuple[datetime, datetime]] = None) -> List[PerformanceMetric]:
        """Retrieve metrics with optional filtering"""
        with self._lock:
            metrics = list(self.metrics_buffer)

        # Filter by metric name
        if metric_name:
            metrics = [m for m in metrics if m.name == metric_name]

        # Filter by time range
        if time_range:
            start_time, end_time = time_range
            metrics = [m for m in metrics if start_time <= m.timestamp <= end_time]

        return metrics

    def get_metric_summary(self, metric_name: str,
                          time_range: Optional[Tuple[datetime, datetime]] = None) -> Dict[str, Any]:
        """Get statistical summary of a metric"""
        metrics = self.get_metrics(metric_name, time_range)

        if not metrics:
            return {"error": "No metrics found", "count": 0}

        values = [m.value for m in metrics]

        try:
            summary = {
                "count": len(values),
                "min": min(values),
                "max": max(values),
                "mean": statistics.mean(values),
                "median": statistics.median(values),
                "std_dev": statistics.stdev(values) if len(values) > 1 else 0.0,
                "latest": values[-1] if values else None,
                "first_timestamp": metrics[0].timestamp.isoformat(),
                "last_timestamp": metrics[-1].timestamp.isoformat()
            }

            # Add percentiles if enough data
            if len(values) >= 10:
                sorted_values = sorted(values)
                summary.update({
                    "p50": statistics.median(sorted_values),
                    "p95": sorted_values[int(len(sorted_values) * 0.95)],
                    "p99": sorted_values[int(len(sorted_values) * 0.99)]
                })

            return summary

        except Exception as e:
            self.logger.error(f"Error calculating metric summary for {metric_name}: {e}")
            return {"error": str(e), "count": len(values)}

    def register_aggregator(self, metric_name: str, aggregator_func: Callable[[PerformanceMetric], None]):
        """Register custom metric aggregator"""
        self.metric_aggregators[metric_name] = aggregator_func


class PerformanceOptimizer:
    """Automatic performance optimization system"""

    def __init__(self, metrics_collector: MetricsCollector, config: Optional[Dict[str, Any]] = None):
        self.metrics_collector = metrics_collector
        self.config = config or {}
        self.logger = logging.getLogger(__name__)
        self.optimization_history: List[OptimizationRecommendation] = []
        self.applied_optimizations: Set[str] = set()

        # Optimization thresholds
        self.memory_threshold = self.config.get('memory_threshold_mb', 500)
        self.cpu_threshold = self.config.get('cpu_threshold_percent', 80)
        self.response_time_threshold = self.config.get('response_time_threshold_ms', 100)

    def analyze_performance(self, time_range: Optional[Tuple[datetime, datetime]] = None) -> List[OptimizationRecommendation]:
        """Analyze performance and generate optimization recommendations"""
        recommendations = []

        try:
            # Memory analysis
            memory_recommendations = self._analyze_memory_usage(time_range)
            recommendations.extend(memory_recommendations)

            # CPU analysis
            cpu_recommendations = self._analyze_cpu_usage(time_range)
            recommendations.extend(cpu_recommendations)

            # I/O analysis
            io_recommendations = self._analyze_io_performance(time_range)
            recommendations.extend(io_recommendations)

            # Response time analysis
            response_recommendations = self._analyze_response_times(time_range)
            recommendations.extend(response_recommendations)

            # Sort by priority (highest first)
            recommendations.sort(key=lambda x: x.priority, reverse=True)

            # Store in history
            self.optimization_history.extend(recommendations)

            return recommendations

        except Exception as e:
            self.logger.error(f"Error in performance analysis: {e}")
            return []

    def _analyze_memory_usage(self, time_range: Optional[Tuple[datetime, datetime]] = None) -> List[OptimizationRecommendation]:
        """Analyze memory usage and generate recommendations"""
        recommendations = []

        try:
            memory_summary = self.metrics_collector.get_metric_summary("memory.rss", time_range)

            if "error" in memory_summary:
                return recommendations

            max_memory = memory_summary.get("max", 0)
            mean_memory = memory_summary.get("mean", 0)
            current_memory = memory_summary.get("latest", 0)

            # High memory usage
            if current_memory > self.memory_threshold:
                recommendations.append(OptimizationRecommendation(
                    recommendation_id=f"memory_high_{int(time.time())}",
                    optimization_type=OptimizationType.MEMORY,
                    description=f"Memory usage is {current_memory:.1f}MB, exceeding threshold of {self.memory_threshold}MB",
                    impact_estimate=max(0, current_memory - self.memory_threshold),
                    implementation_difficulty=3,
                    estimated_benefit="Reduce memory usage by 20-40%",
                    code_changes_required=[
                        "Implement object pooling for frequently created objects",
                        "Add explicit garbage collection triggers",
                        "Review and optimize data structures",
                        "Implement memory-mapped files for large datasets"
                    ],
                    configuration_changes={
                        "gc_threshold": "lower",
                        "buffer_size": "reduce",
                        "cache_size": "optimize"
                    },
                    priority=8 if current_memory > self.memory_threshold * 1.5 else 6
                ))

            # Memory growth trend
            if mean_memory > 0 and current_memory > mean_memory * 1.3:
                recommendations.append(OptimizationRecommendation(
                    recommendation_id=f"memory_growth_{int(time.time())}",
                    optimization_type=OptimizationType.MEMORY,
                    description="Memory usage shows upward trend, possible memory leak",
                    impact_estimate=current_memory - mean_memory,
                    implementation_difficulty=5,
                    estimated_benefit="Prevent memory leaks and stabilize usage",
                    code_changes_required=[
                        "Audit for circular references",
                        "Review event listener cleanup",
                        "Check weak reference usage",
                        "Implement memory profiling"
                    ],
                    configuration_changes={
                        "enable_memory_profiling": True,
                        "gc_debug": True
                    },
                    priority=7
                ))

            # Memory profiling recommendations
            if self.config.get('enable_memory_profiling', False):
                traced_summary = self.metrics_collector.get_metric_summary("memory.traced.current", time_range)
                if "error" not in traced_summary:
                    traced_current = traced_summary.get("latest", 0)
                    if traced_current > current_memory * 0.8:  # Traced memory close to RSS
                        recommendations.append(OptimizationRecommendation(
                            recommendation_id=f"memory_profiling_{int(time.time())}",
                            optimization_type=OptimizationType.MEMORY,
                            description="High traced memory usage indicates potential optimization opportunities",
                            impact_estimate=traced_current * 0.2,
                            implementation_difficulty=4,
                            estimated_benefit="Optimize memory allocation patterns",
                            code_changes_required=[
                                "Analyze memory allocation hotspots",
                                "Optimize data structure choices",
                                "Implement lazy loading patterns",
                                "Review string concatenation usage"
                            ],
                            configuration_changes={},
                            priority=5
                        ))

            return recommendations

        except Exception as e:
            self.logger.error(f"Error analyzing memory usage: {e}")
            return []

    def _analyze_cpu_usage(self, time_range: Optional[Tuple[datetime, datetime]] = None) -> List[OptimizationRecommendation]:
        """Analyze CPU usage and generate recommendations"""
        recommendations = []

        try:
            cpu_summary = self.metrics_collector.get_metric_summary("cpu.usage", time_range)

            if "error" in cpu_summary:
                return recommendations

            max_cpu = cpu_summary.get("max", 0)
            mean_cpu = cpu_summary.get("mean", 0)
            current_cpu = cpu_summary.get("latest", 0)

            # High CPU usage
            if current_cpu > self.cpu_threshold:
                recommendations.append(OptimizationRecommendation(
                    recommendation_id=f"cpu_high_{int(time.time())}",
                    optimization_type=OptimizationType.CPU,
                    description=f"CPU usage is {current_cpu:.1f}%, exceeding threshold of {self.cpu_threshold}%",
                    impact_estimate=current_cpu - self.cpu_threshold,
                    implementation_difficulty=4,
                    estimated_benefit="Reduce CPU usage by 15-30%",
                    code_changes_required=[
                        "Profile CPU-intensive operations",
                        "Implement async/await for I/O operations",
                        "Optimize algorithmic complexity",
                        "Add caching for expensive computations"
                    ],
                    configuration_changes={
                        "async_processing": True,
                        "thread_pool_size": "optimize",
                        "batch_size": "increase"
                    },
                    priority=7 if current_cpu > self.cpu_threshold * 1.3 else 5
                ))

            # Sustained high CPU
            if mean_cpu > self.cpu_threshold * 0.7:
                recommendations.append(OptimizationRecommendation(
                    recommendation_id=f"cpu_sustained_{int(time.time())}",
                    optimization_type=OptimizationType.CPU,
                    description="Sustained high CPU usage indicates processing bottleneck",
                    impact_estimate=mean_cpu - (self.cpu_threshold * 0.7),
                    implementation_difficulty=6,
                    estimated_benefit="Improve overall system responsiveness",
                    code_changes_required=[
                        "Implement parallel processing",
                        "Add CPU usage monitoring hooks",
                        "Optimize hot code paths",
                        "Consider algorithm improvements"
                    ],
                    configuration_changes={
                        "enable_multiprocessing": True,
                        "cpu_affinity": "optimize"
                    },
                    priority=6
                ))

            return recommendations

        except Exception as e:
            self.logger.error(f"Error analyzing CPU usage: {e}")
            return []

    def _analyze_io_performance(self, time_range: Optional[Tuple[datetime, datetime]] = None) -> List[OptimizationRecommendation]:
        """Analyze I/O performance and generate recommendations"""
        recommendations = []

        try:
            read_summary = self.metrics_collector.get_metric_summary("io.read_bytes", time_range)
            write_summary = self.metrics_collector.get_metric_summary("io.write_bytes", time_range)

            if "error" not in read_summary and "error" not in write_summary:
                read_rate = read_summary.get("max", 0) - read_summary.get("min", 0)
                write_rate = write_summary.get("max", 0) - write_summary.get("min", 0)

                # High I/O activity
                high_io_threshold = 100 * 1024 * 1024  # 100MB
                if read_rate > high_io_threshold or write_rate > high_io_threshold:
                    recommendations.append(OptimizationRecommendation(
                        recommendation_id=f"io_high_{int(time.time())}",
                        optimization_type=OptimizationType.IO,
                        description=f"High I/O activity detected (Read: {read_rate/1024/1024:.1f}MB, Write: {write_rate/1024/1024:.1f}MB)",
                        impact_estimate=max(read_rate, write_rate) / 1024 / 1024,
                        implementation_difficulty=5,
                        estimated_benefit="Reduce I/O bottleneck and improve response times",
                        code_changes_required=[
                            "Implement I/O buffering and batching",
                            "Add async I/O operations",
                            "Optimize file access patterns",
                            "Consider in-memory caching"
                        ],
                        configuration_changes={
                            "io_buffer_size": "increase",
                            "async_io": True,
                            "cache_enabled": True
                        },
                        priority=6
                    ))

            return recommendations

        except Exception as e:
            self.logger.error(f"Error analyzing I/O performance: {e}")
            return []

    def _analyze_response_times(self, time_range: Optional[Tuple[datetime, datetime]] = None) -> List[OptimizationRecommendation]:
        """Analyze response times and generate recommendations"""
        recommendations = []

        try:
            # Look for response time metrics
            response_metrics = [m for m in self.metrics_collector.get_metrics(time_range=time_range)
                              if "response_time" in m.name or "latency" in m.name]

            if not response_metrics:
                return recommendations

            # Group by metric name
            metric_groups = defaultdict(list)
            for metric in response_metrics:
                metric_groups[metric.name].append(metric.value)

            for metric_name, values in metric_groups.items():
                if not values:
                    continue

                mean_time = statistics.mean(values)
                max_time = max(values)
                p95_time = sorted(values)[int(len(values) * 0.95)] if len(values) > 10 else max_time

                # High response times
                if p95_time > self.response_time_threshold:
                    recommendations.append(OptimizationRecommendation(
                        recommendation_id=f"response_time_{metric_name}_{int(time.time())}",
                        optimization_type=OptimizationType.ALGORITHM,
                        description=f"{metric_name} P95 response time is {p95_time:.1f}ms, exceeding {self.response_time_threshold}ms threshold",
                        impact_estimate=p95_time - self.response_time_threshold,
                        implementation_difficulty=4,
                        estimated_benefit="Improve user experience and system throughput",
                        code_changes_required=[
                            "Profile slow operations",
                            "Implement result caching",
                            "Optimize database queries",
                            "Add connection pooling"
                        ],
                        configuration_changes={
                            "connection_pool_size": "increase",
                            "query_cache_enabled": True,
                            "timeout_ms": "optimize"
                        },
                        priority=7 if p95_time > self.response_time_threshold * 2 else 5
                    ))

            return recommendations

        except Exception as e:
            self.logger.error(f"Error analyzing response times: {e}")
            return []

    def apply_optimization(self, recommendation: OptimizationRecommendation) -> bool:
        """Apply an optimization recommendation (placeholder for implementation)"""
        try:
            # This is a placeholder - in practice, you'd implement the actual optimizations
            self.logger.info(f"Applying optimization: {recommendation.description}")

            # For demonstration, we'll just apply configuration changes
            for key, value in recommendation.configuration_changes.items():
                self.config[key] = value
                self.logger.info(f"Applied config change: {key} = {value}")

            self.applied_optimizations.add(recommendation.recommendation_id)
            return True

        except Exception as e:
            self.logger.error(f"Error applying optimization {recommendation.recommendation_id}: {e}")
            return False

    def get_optimization_history(self) -> List[OptimizationRecommendation]:
        """Get history of optimization recommendations"""
        return self.optimization_history.copy()


class HealthMonitor:
    """System health monitoring and reporting"""

    def __init__(self, metrics_collector: MetricsCollector, config: Optional[Dict[str, Any]] = None):
        self.metrics_collector = metrics_collector
        self.config = config or {}
        self.logger = logging.getLogger(__name__)

        # Health thresholds
        self.thresholds = {
            'memory_warning_mb': self.config.get('memory_warning_mb', 400),
            'memory_critical_mb': self.config.get('memory_critical_mb', 500),
            'cpu_warning_percent': self.config.get('cpu_warning_percent', 70),
            'cpu_critical_percent': self.config.get('cpu_critical_percent', 90),
            'response_warning_ms': self.config.get('response_warning_ms', 100),
            'response_critical_ms': self.config.get('response_critical_ms', 500)
        }

        self.health_history: deque = deque(maxlen=self.config.get('health_history_size', 1000))

    def generate_health_report(self) -> HealthReport:
        """Generate comprehensive health report"""
        try:
            timestamp = datetime.now()
            component_health = {}
            alerts = []
            recommendations = []
            metrics = {}

            # System uptime
            uptime = self.metrics_collector.get_metric_summary("uptime").get("latest", 0)

            # Memory health
            memory_health, memory_alerts, memory_recommendations, memory_metrics = self._assess_memory_health()
            component_health["memory"] = memory_health
            alerts.extend(memory_alerts)
            recommendations.extend(memory_recommendations)
            metrics.update(memory_metrics)

            # CPU health
            cpu_health, cpu_alerts, cpu_recommendations, cpu_metrics = self._assess_cpu_health()
            component_health["cpu"] = cpu_health
            alerts.extend(cpu_alerts)
            recommendations.extend(cpu_recommendations)
            metrics.update(cpu_metrics)

            # I/O health
            io_health, io_alerts, io_recommendations, io_metrics = self._assess_io_health()
            component_health["io"] = io_health
            alerts.extend(io_alerts)
            recommendations.extend(io_recommendations)
            metrics.update(io_metrics)

            # Response time health
            response_health, response_alerts, response_recommendations, response_metrics = self._assess_response_health()
            component_health["response_times"] = response_health
            alerts.extend(response_alerts)
            recommendations.extend(response_recommendations)
            metrics.update(response_metrics)

            # Overall health assessment
            health_values = list(component_health.values())
            if HealthStatus.CRITICAL in health_values:
                overall_status = HealthStatus.CRITICAL
            elif HealthStatus.WARNING in health_values:
                overall_status = HealthStatus.WARNING
            elif HealthStatus.HEALTHY in health_values:
                overall_status = HealthStatus.HEALTHY
            else:
                overall_status = HealthStatus.UNKNOWN

            # Create health report
            health_report = HealthReport(
                timestamp=timestamp,
                overall_status=overall_status,
                component_health=component_health,
                metrics=metrics,
                alerts=alerts,
                recommendations=list(set(recommendations)),  # Remove duplicates
                uptime=uptime
            )

            # Store in history
            self.health_history.append(health_report)

            return health_report

        except Exception as e:
            self.logger.error(f"Error generating health report: {e}")
            return HealthReport(
                timestamp=datetime.now(),
                overall_status=HealthStatus.UNKNOWN,
                component_health={},
                metrics={},
                alerts=[f"Health monitoring error: {str(e)}"],
                recommendations=["Fix health monitoring system"],
                uptime=0.0
            )

    def _assess_memory_health(self) -> Tuple[HealthStatus, List[str], List[str], Dict[str, float]]:
        """Assess memory health status"""
        try:
            memory_summary = self.metrics_collector.get_metric_summary("memory.rss")

            if "error" in memory_summary:
                return HealthStatus.UNKNOWN, ["Memory metrics unavailable"], [], {}

            current_memory = memory_summary.get("latest", 0)
            max_memory = memory_summary.get("max", 0)
            mean_memory = memory_summary.get("mean", 0)

            alerts = []
            recommendations = []
            metrics = {
                "memory_current_mb": current_memory,
                "memory_max_mb": max_memory,
                "memory_mean_mb": mean_memory
            }

            # Determine health status
            if current_memory >= self.thresholds['memory_critical_mb']:
                health_status = HealthStatus.CRITICAL
                alerts.append(f"Critical: Memory usage at {current_memory:.1f}MB")
                recommendations.append("Immediately reduce memory usage")
            elif current_memory >= self.thresholds['memory_warning_mb']:
                health_status = HealthStatus.WARNING
                alerts.append(f"Warning: Memory usage at {current_memory:.1f}MB")
                recommendations.append("Monitor memory usage closely")
            else:
                health_status = HealthStatus.HEALTHY

            # Memory growth trend
            if mean_memory > 0 and current_memory > mean_memory * 1.2:
                if health_status == HealthStatus.HEALTHY:
                    health_status = HealthStatus.WARNING
                alerts.append("Memory usage trending upward")
                recommendations.append("Investigate potential memory leaks")

            return health_status, alerts, recommendations, metrics

        except Exception as e:
            self.logger.error(f"Error assessing memory health: {e}")
            return HealthStatus.UNKNOWN, [f"Memory assessment error: {e}"], [], {}

    def _assess_cpu_health(self) -> Tuple[HealthStatus, List[str], List[str], Dict[str, float]]:
        """Assess CPU health status"""
        try:
            cpu_summary = self.metrics_collector.get_metric_summary("cpu.usage")

            if "error" in cpu_summary:
                return HealthStatus.UNKNOWN, ["CPU metrics unavailable"], [], {}

            current_cpu = cpu_summary.get("latest", 0)
            max_cpu = cpu_summary.get("max", 0)
            mean_cpu = cpu_summary.get("mean", 0)

            alerts = []
            recommendations = []
            metrics = {
                "cpu_current_percent": current_cpu,
                "cpu_max_percent": max_cpu,
                "cpu_mean_percent": mean_cpu
            }

            # Determine health status
            if current_cpu >= self.thresholds['cpu_critical_percent']:
                health_status = HealthStatus.CRITICAL
                alerts.append(f"Critical: CPU usage at {current_cpu:.1f}%")
                recommendations.append("Immediately reduce CPU load")
            elif current_cpu >= self.thresholds['cpu_warning_percent']:
                health_status = HealthStatus.WARNING
                alerts.append(f"Warning: CPU usage at {current_cpu:.1f}%")
                recommendations.append("Optimize CPU-intensive operations")
            else:
                health_status = HealthStatus.HEALTHY

            # Sustained high CPU
            if mean_cpu >= self.thresholds['cpu_warning_percent']:
                if health_status == HealthStatus.HEALTHY:
                    health_status = HealthStatus.WARNING
                alerts.append("Sustained high CPU usage detected")
                recommendations.append("Profile and optimize performance bottlenecks")

            return health_status, alerts, recommendations, metrics

        except Exception as e:
            self.logger.error(f"Error assessing CPU health: {e}")
            return HealthStatus.UNKNOWN, [f"CPU assessment error: {e}"], [], {}

    def _assess_io_health(self) -> Tuple[HealthStatus, List[str], List[str], Dict[str, float]]:
        """Assess I/O health status"""
        try:
            # This is a simplified I/O health assessment
            # In practice, you'd analyze I/O rates, queue depths, etc.

            alerts = []
            recommendations = []
            metrics = {}

            # For now, assume I/O is healthy unless other indicators suggest otherwise
            health_status = HealthStatus.HEALTHY

            return health_status, alerts, recommendations, metrics

        except Exception as e:
            self.logger.error(f"Error assessing I/O health: {e}")
            return HealthStatus.UNKNOWN, [f"I/O assessment error: {e}"], [], {}

    def _assess_response_health(self) -> Tuple[HealthStatus, List[str], List[str], Dict[str, float]]:
        """Assess response time health status"""
        try:
            # Look for response time metrics
            response_metrics = [m for m in self.metrics_collector.get_metrics()
                              if "response_time" in m.name or "latency" in m.name]

            if not response_metrics:
                return HealthStatus.UNKNOWN, ["No response time metrics available"], [], {}

            # Calculate aggregate response time stats
            recent_metrics = [m for m in response_metrics
                            if m.timestamp > datetime.now() - timedelta(minutes=5)]

            if not recent_metrics:
                return HealthStatus.UNKNOWN, ["No recent response time data"], [], {}

            values = [m.value for m in recent_metrics]
            mean_response = statistics.mean(values)
            max_response = max(values)
            p95_response = sorted(values)[int(len(values) * 0.95)] if len(values) > 10 else max_response

            alerts = []
            recommendations = []
            metrics = {
                "response_time_mean_ms": mean_response,
                "response_time_max_ms": max_response,
                "response_time_p95_ms": p95_response
            }

            # Determine health status
            if p95_response >= self.thresholds['response_critical_ms']:
                health_status = HealthStatus.CRITICAL
                alerts.append(f"Critical: P95 response time at {p95_response:.1f}ms")
                recommendations.append("Immediately investigate performance bottlenecks")
            elif p95_response >= self.thresholds['response_warning_ms']:
                health_status = HealthStatus.WARNING
                alerts.append(f"Warning: P95 response time at {p95_response:.1f}ms")
                recommendations.append("Optimize slow operations")
            else:
                health_status = HealthStatus.HEALTHY

            return health_status, alerts, recommendations, metrics

        except Exception as e:
            self.logger.error(f"Error assessing response health: {e}")
            return HealthStatus.UNKNOWN, [f"Response time assessment error: {e}"], [], {}

    def get_health_trends(self, hours: int = 24) -> Dict[str, Any]:
        """Get health trends over specified time period"""
        try:
            cutoff_time = datetime.now() - timedelta(hours=hours)
            recent_reports = [r for r in self.health_history if r.timestamp >= cutoff_time]

            if not recent_reports:
                return {"error": "No health data available for specified time period"}

            trends = {
                "time_period_hours": hours,
                "total_reports": len(recent_reports),
                "status_distribution": defaultdict(int),
                "component_trends": defaultdict(lambda: defaultdict(int)),
                "metric_trends": defaultdict(list),
                "alert_frequency": defaultdict(int)
            }

            for report in recent_reports:
                # Overall status distribution
                trends["status_distribution"][report.overall_status.value] += 1

                # Component health trends
                for component, status in report.component_health.items():
                    trends["component_trends"][component][status.value] += 1

                # Metric trends
                for metric, value in report.metrics.items():
                    trends["metric_trends"][metric].append(value)

                # Alert frequency
                for alert in report.alerts:
                    alert_type = alert.split(":")[0] if ":" in alert else "general"
                    trends["alert_frequency"][alert_type] += 1

            # Calculate metric statistics
            for metric, values in trends["metric_trends"].items():
                if values:
                    trends["metric_trends"][metric] = {
                        "min": min(values),
                        "max": max(values),
                        "mean": statistics.mean(values),
                        "latest": values[-1],
                        "trend": "increasing" if len(values) > 1 and values[-1] > values[0] else "stable"
                    }

            return dict(trends)

        except Exception as e:
            self.logger.error(f"Error getting health trends: {e}")
            return {"error": str(e)}


class PerformanceBenchmark:
    """Performance benchmarking system"""

    def __init__(self, metrics_collector: MetricsCollector, config: Optional[Dict[str, Any]] = None):
        self.metrics_collector = metrics_collector
        self.config = config or {}
        self.logger = logging.getLogger(__name__)
        self.benchmark_results: List[BenchmarkResult] = []

    async def run_benchmark(self, test_name: str, test_func: Callable,
                           iterations: int = 100, **kwargs) -> BenchmarkResult:
        """Run performance benchmark test"""
        try:
            self.logger.info(f"Starting benchmark: {test_name} ({iterations} iterations)")

            start_time = time.time()
            start_memory = self.metrics_collector.process.memory_info().rss

            successful_runs = 0
            error_count = 0
            execution_times = []

            for i in range(iterations):
                try:
                    iteration_start = time.time()

                    if asyncio.iscoroutinefunction(test_func):
                        await test_func(**kwargs)
                    else:
                        test_func(**kwargs)

                    execution_times.append(time.time() - iteration_start)
                    successful_runs += 1

                except Exception as e:
                    error_count += 1
                    self.logger.debug(f"Benchmark iteration {i} failed: {e}")

            end_time = time.time()
            end_memory = self.metrics_collector.process.memory_info().rss

            # Calculate metrics
            total_duration = end_time - start_time
            throughput = successful_runs / total_duration if total_duration > 0 else 0
            success_rate = successful_runs / iterations if iterations > 0 else 0

            memory_usage = {
                "start_mb": start_memory / 1024 / 1024,
                "end_mb": end_memory / 1024 / 1024,
                "delta_mb": (end_memory - start_memory) / 1024 / 1024
            }

            # CPU usage during test (approximate)
            cpu_usage = self.metrics_collector.process.cpu_percent()

            # Additional metrics
            metadata = {
                "iterations": iterations,
                "successful_runs": successful_runs,
                "avg_execution_time": statistics.mean(execution_times) if execution_times else 0,
                "min_execution_time": min(execution_times) if execution_times else 0,
                "max_execution_time": max(execution_times) if execution_times else 0,
                "p95_execution_time": sorted(execution_times)[int(len(execution_times) * 0.95)] if len(execution_times) > 10 else max(execution_times) if execution_times else 0
            }

            result = BenchmarkResult(
                test_name=test_name,
                timestamp=datetime.now(),
                duration=total_duration,
                throughput=throughput,
                memory_usage=memory_usage,
                cpu_usage=cpu_usage,
                success_rate=success_rate,
                error_count=error_count,
                metadata=metadata
            )

            self.benchmark_results.append(result)

            self.logger.info(f"Benchmark completed: {test_name} - "
                           f"Throughput: {throughput:.2f}/s, Success Rate: {success_rate:.1%}")

            return result

        except Exception as e:
            self.logger.error(f"Error running benchmark {test_name}: {e}")
            return BenchmarkResult(
                test_name=test_name,
                timestamp=datetime.now(),
                duration=0,
                throughput=0,
                memory_usage={},
                cpu_usage=0,
                success_rate=0,
                error_count=iterations,
                metadata={"error": str(e)}
            )

    def get_benchmark_history(self, test_name: Optional[str] = None) -> List[BenchmarkResult]:
        """Get benchmark history, optionally filtered by test name"""
        if test_name:
            return [r for r in self.benchmark_results if r.test_name == test_name]
        return self.benchmark_results.copy()

    def compare_benchmarks(self, test_name: str, baseline_timestamp: Optional[datetime] = None) -> Dict[str, Any]:
        """Compare benchmark results against baseline"""
        try:
            test_results = self.get_benchmark_history(test_name)

            if not test_results:
                return {"error": "No benchmark results found for test"}

            if baseline_timestamp:
                baseline_results = [r for r in test_results if r.timestamp <= baseline_timestamp]
                current_results = [r for r in test_results if r.timestamp > baseline_timestamp]
            else:
                # Use first result as baseline
                baseline_results = test_results[:1]
                current_results = test_results[1:]

            if not baseline_results or not current_results:
                return {"error": "Insufficient data for comparison"}

            baseline = baseline_results[-1]  # Most recent baseline
            current = current_results[-1]    # Most recent current

            comparison = {
                "test_name": test_name,
                "baseline": {
                    "timestamp": baseline.timestamp.isoformat(),
                    "throughput": baseline.throughput,
                    "duration": baseline.duration,
                    "success_rate": baseline.success_rate,
                    "memory_delta_mb": baseline.memory_usage.get("delta_mb", 0)
                },
                "current": {
                    "timestamp": current.timestamp.isoformat(),
                    "throughput": current.throughput,
                    "duration": current.duration,
                    "success_rate": current.success_rate,
                    "memory_delta_mb": current.memory_usage.get("delta_mb", 0)
                },
                "changes": {}
            }

            # Calculate changes
            if baseline.throughput > 0:
                comparison["changes"]["throughput_percent"] = ((current.throughput - baseline.throughput) / baseline.throughput) * 100

            if baseline.duration > 0:
                comparison["changes"]["duration_percent"] = ((current.duration - baseline.duration) / baseline.duration) * 100

            comparison["changes"]["success_rate_percent"] = (current.success_rate - baseline.success_rate) * 100
            comparison["changes"]["memory_delta_mb"] = current.memory_usage.get("delta_mb", 0) - baseline.memory_usage.get("delta_mb", 0)

            # Performance assessment
            performance_score = 0
            if comparison["changes"].get("throughput_percent", 0) > 0:
                performance_score += 1
            if comparison["changes"].get("duration_percent", 0) < 0:
                performance_score += 1
            if comparison["changes"].get("success_rate_percent", 0) >= 0:
                performance_score += 1
            if comparison["changes"].get("memory_delta_mb", 0) <= 0:
                performance_score += 1

            comparison["performance_assessment"] = {
                "score": performance_score,
                "max_score": 4,
                "rating": "excellent" if performance_score >= 3 else "good" if performance_score >= 2 else "needs_improvement"
            }

            return comparison

        except Exception as e:
            self.logger.error(f"Error comparing benchmarks: {e}")
            return {"error": str(e)}


class PerformanceMonitoringSystem:
    """Main performance monitoring system orchestrating all components"""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.logger = logging.getLogger(__name__)

        # Initialize components
        self.metrics_collector = MetricsCollector(self.config.get('metrics', {}))
        self.optimizer = PerformanceOptimizer(self.metrics_collector, self.config.get('optimizer', {}))
        self.health_monitor = HealthMonitor(self.metrics_collector, self.config.get('health', {}))
        self.benchmark = PerformanceBenchmark(self.metrics_collector, self.config.get('benchmark', {}))

        # Auto-start metrics collection
        if self.config.get('auto_start', True):
            self.metrics_collector.start_collection()

    def get_performance_dashboard(self) -> Dict[str, Any]:
        """Get comprehensive performance dashboard data"""
        try:
            dashboard = {
                "timestamp": datetime.now().isoformat(),
                "health_report": asdict(self.health_monitor.generate_health_report()),
                "optimization_recommendations": [asdict(rec) for rec in self.optimizer.analyze_performance()],
                "key_metrics": {},
                "recent_benchmarks": [],
                "system_info": {}
            }

            # Key metrics
            dashboard["key_metrics"] = {
                "memory": self.metrics_collector.get_metric_summary("memory.rss"),
                "cpu": self.metrics_collector.get_metric_summary("cpu.usage"),
                "uptime": self.metrics_collector.get_metric_summary("uptime")
            }

            # Recent benchmarks
            recent_benchmarks = [r for r in self.benchmark.benchmark_results
                               if r.timestamp > datetime.now() - timedelta(hours=24)]
            dashboard["recent_benchmarks"] = [asdict(r) for r in recent_benchmarks[-10:]]

            # System info
            dashboard["system_info"] = {
                "cpu_count": psutil.cpu_count(),
                "memory_total_gb": psutil.virtual_memory().total / 1024 / 1024 / 1024,
                "platform": psutil.WINDOWS if psutil.WINDOWS else psutil.LINUX if psutil.LINUX else "unknown"
            }

            return dashboard

        except Exception as e:
            self.logger.error(f"Error generating performance dashboard: {e}")
            return {
                "timestamp": datetime.now().isoformat(),
                "error": str(e)
            }

    def shutdown(self):
        """Shutdown the performance monitoring system"""
        try:
            self.metrics_collector.stop_collection()
            self.logger.info("Performance monitoring system shutdown complete")
        except Exception as e:
            self.logger.error(f"Error during shutdown: {e}")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.shutdown()


# Main execution function for testing
async def main():
    """Main function for testing the performance monitoring system"""
    # Initialize system
    config = {
        'auto_start': True,
        'metrics': {
            'collection_interval': 0.5,
            'enable_memory_profiling': True
        },
        'optimizer': {
            'memory_threshold_mb': 100,  # Lower for testing
            'cpu_threshold_percent': 50
        }
    }

    with PerformanceMonitoringSystem(config) as monitor:
        # Let it collect some metrics
        await asyncio.sleep(2)

        # Run a simple benchmark
        def test_computation():
            # Simple computation for benchmarking
            result = sum(i**2 for i in range(1000))
            return result

        benchmark_result = await monitor.benchmark.run_benchmark(
            "test_computation", test_computation, iterations=50
        )

        print(f"Benchmark Result: {benchmark_result.throughput:.2f} ops/sec")

        # Generate dashboard
        dashboard = monitor.get_performance_dashboard()
        print(json.dumps(dashboard, indent=2, default=str))


if __name__ == "__main__":
    asyncio.run(main())