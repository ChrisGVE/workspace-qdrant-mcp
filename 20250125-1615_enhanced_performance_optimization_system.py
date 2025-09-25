#!/usr/bin/env python3
"""
Enhanced Performance Monitoring and Optimization System

This module implements comprehensive performance monitoring with automatic
optimization triggers, health reporting, bottleneck analysis, and resource
usage monitoring for the workspace-qdrant-mcp project.

Task 265: Comprehensive performance monitoring and optimization implementation
"""

import asyncio
import json
import statistics
import time
import psutil
import threading
from collections import defaultdict, deque
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Callable, Any, Set, Tuple, Union
from enum import Enum, auto

from loguru import logger


class PerformanceLevel(Enum):
    """Performance level enumeration for metrics classification."""
    EXCELLENT = auto()
    GOOD = auto()
    ACCEPTABLE = auto()
    DEGRADED = auto()
    CRITICAL = auto()


class OptimizationTrigger(Enum):
    """Optimization trigger types."""
    THRESHOLD_EXCEEDED = auto()
    TREND_DEGRADATION = auto()
    RESOURCE_PRESSURE = auto()
    USER_TRIGGERED = auto()
    SCHEDULED = auto()


@dataclass
class ResourceMetric:
    """Resource usage metric for comprehensive monitoring."""
    timestamp: datetime
    cpu_percent: float
    memory_bytes: int
    memory_percent: float
    disk_io_read: int
    disk_io_write: int
    network_sent: int
    network_recv: int
    open_files: int
    thread_count: int
    load_average: float

    def is_under_pressure(self, thresholds: Dict[str, float]) -> bool:
        """Check if system is under resource pressure."""
        return (
            self.cpu_percent > thresholds.get('cpu_percent', 80) or
            self.memory_percent > thresholds.get('memory_percent', 85) or
            self.load_average > thresholds.get('load_average', 2.0) or
            self.open_files > thresholds.get('open_files', 1000)
        )


@dataclass
class BottleneckAnalysis:
    """Analysis result for performance bottlenecks."""
    timestamp: datetime
    bottleneck_type: str  # 'cpu', 'memory', 'io', 'network', 'concurrency'
    severity: str  # 'low', 'medium', 'high', 'critical'
    affected_operations: List[str]
    root_cause: str
    impact_score: float  # 0.0 - 1.0
    recommended_actions: List[str]
    estimated_improvement: float  # % improvement expected

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'timestamp': self.timestamp.isoformat(),
            'bottleneck_type': self.bottleneck_type,
            'severity': self.severity,
            'affected_operations': self.affected_operations,
            'root_cause': self.root_cause,
            'impact_score': self.impact_score,
            'recommended_actions': self.recommended_actions,
            'estimated_improvement': self.estimated_improvement
        }


@dataclass
class OptimizationAction:
    """Represents an optimization action to be taken."""
    action_id: str
    trigger: OptimizationTrigger
    action_type: str
    description: str
    parameters: Dict[str, Any]
    expected_benefit: float
    risk_level: str  # 'low', 'medium', 'high'
    reversible: bool
    auto_executable: bool

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'action_id': self.action_id,
            'trigger': self.trigger.name,
            'action_type': self.action_type,
            'description': self.description,
            'parameters': self.parameters,
            'expected_benefit': self.expected_benefit,
            'risk_level': self.risk_level,
            'reversible': self.reversible,
            'auto_executable': self.auto_executable
        }


@dataclass
class HealthReport:
    """Comprehensive system health report."""
    timestamp: datetime
    overall_health: str  # 'excellent', 'good', 'degraded', 'critical'
    health_score: float  # 0.0 - 100.0
    component_health: Dict[str, str]
    active_alerts: List[Dict]
    performance_trends: Dict[str, str]  # 'improving', 'stable', 'degrading'
    resource_utilization: ResourceMetric
    bottlenecks: List[BottleneckAnalysis]
    optimization_recommendations: List[OptimizationAction]
    uptime_seconds: float

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'timestamp': self.timestamp.isoformat(),
            'overall_health': self.overall_health,
            'health_score': self.health_score,
            'component_health': self.component_health,
            'active_alerts': self.active_alerts,
            'performance_trends': self.performance_trends,
            'resource_utilization': {
                'cpu_percent': self.resource_utilization.cpu_percent,
                'memory_percent': self.resource_utilization.memory_percent,
                'memory_bytes': self.resource_utilization.memory_bytes,
                'disk_io_read': self.resource_utilization.disk_io_read,
                'disk_io_write': self.resource_utilization.disk_io_write,
                'network_sent': self.resource_utilization.network_sent,
                'network_recv': self.resource_utilization.network_recv,
                'open_files': self.resource_utilization.open_files,
                'thread_count': self.resource_utilization.thread_count,
                'load_average': self.resource_utilization.load_average
            },
            'bottlenecks': [b.to_dict() for b in self.bottlenecks],
            'optimization_recommendations': [r.to_dict() for r in self.optimization_recommendations],
            'uptime_seconds': self.uptime_seconds
        }


class ResourceMonitor:
    """Advanced resource usage monitoring with trend analysis."""

    def __init__(self, monitoring_interval: float = 5.0):
        self.monitoring_interval = monitoring_interval
        self.metrics_history: deque = deque(maxlen=1000)  # Last 1000 measurements
        self.process = psutil.Process()
        self.start_time = time.time()
        self.running = False
        self._monitor_task: Optional[asyncio.Task] = None

        # Threshold defaults
        self.pressure_thresholds = {
            'cpu_percent': 80.0,
            'memory_percent': 85.0,
            'load_average': 2.0,
            'open_files': 1000
        }

    async def start_monitoring(self):
        """Start resource monitoring."""
        if self.running:
            return

        self.running = True
        self._monitor_task = asyncio.create_task(self._monitor_loop())
        logger.info("Resource monitoring started", interval=self.monitoring_interval)

    async def stop_monitoring(self):
        """Stop resource monitoring."""
        if not self.running:
            return

        self.running = False
        if self._monitor_task:
            self._monitor_task.cancel()
            try:
                await self._monitor_task
            except asyncio.CancelledError:
                pass
        logger.info("Resource monitoring stopped")

    async def _monitor_loop(self):
        """Main monitoring loop."""
        while self.running:
            try:
                metric = await self._collect_resource_metric()
                self.metrics_history.append(metric)
                await asyncio.sleep(self.monitoring_interval)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("Resource monitoring error", error=str(e))
                await asyncio.sleep(self.monitoring_interval)

    async def _collect_resource_metric(self) -> ResourceMetric:
        """Collect comprehensive resource metrics."""
        try:
            # Process-specific metrics
            with self.process.oneshot():
                cpu_percent = self.process.cpu_percent()
                memory_info = self.process.memory_info()
                memory_percent = self.process.memory_percent()
                open_files = len(self.process.open_files())
                threads = self.process.num_threads()
                io_counters = self.process.io_counters()

            # System-wide metrics
            load_avg = psutil.getloadavg()[0] if hasattr(psutil, 'getloadavg') else 0.0
            net_io = psutil.net_io_counters()

            return ResourceMetric(
                timestamp=datetime.now(),
                cpu_percent=cpu_percent,
                memory_bytes=memory_info.rss,
                memory_percent=memory_percent,
                disk_io_read=io_counters.read_bytes,
                disk_io_write=io_counters.write_bytes,
                network_sent=net_io.bytes_sent if net_io else 0,
                network_recv=net_io.bytes_recv if net_io else 0,
                open_files=open_files,
                thread_count=threads,
                load_average=load_avg
            )
        except Exception as e:
            logger.warning("Failed to collect resource metric", error=str(e))
            return ResourceMetric(
                timestamp=datetime.now(),
                cpu_percent=0.0,
                memory_bytes=0,
                memory_percent=0.0,
                disk_io_read=0,
                disk_io_write=0,
                network_sent=0,
                network_recv=0,
                open_files=0,
                thread_count=0,
                load_average=0.0
            )

    def get_current_metrics(self) -> Optional[ResourceMetric]:
        """Get the most recent resource metrics."""
        return self.metrics_history[-1] if self.metrics_history else None

    def get_resource_trends(self, window_minutes: int = 30) -> Dict[str, str]:
        """Analyze resource usage trends."""
        if len(self.metrics_history) < 2:
            return {}

        cutoff = datetime.now() - timedelta(minutes=window_minutes)
        recent_metrics = [m for m in self.metrics_history if m.timestamp > cutoff]

        if len(recent_metrics) < 2:
            return {}

        trends = {}

        # CPU trend
        cpu_values = [m.cpu_percent for m in recent_metrics]
        cpu_trend = self._calculate_trend(cpu_values)
        trends['cpu'] = cpu_trend

        # Memory trend
        memory_values = [m.memory_percent for m in recent_metrics]
        memory_trend = self._calculate_trend(memory_values)
        trends['memory'] = memory_trend

        # Load average trend
        load_values = [m.load_average for m in recent_metrics]
        load_trend = self._calculate_trend(load_values)
        trends['load'] = load_trend

        return trends

    def _calculate_trend(self, values: List[float]) -> str:
        """Calculate trend direction from values."""
        if len(values) < 2:
            return 'stable'

        # Simple linear trend calculation
        x = list(range(len(values)))
        n = len(values)

        sum_x = sum(x)
        sum_y = sum(values)
        sum_xy = sum(x[i] * values[i] for i in range(n))
        sum_x2 = sum(xi * xi for xi in x)

        try:
            slope = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x * sum_x)

            if slope > 0.1:
                return 'increasing'
            elif slope < -0.1:
                return 'decreasing'
            else:
                return 'stable'
        except ZeroDivisionError:
            return 'stable'

    def is_under_pressure(self) -> bool:
        """Check if system is currently under resource pressure."""
        current = self.get_current_metrics()
        return current.is_under_pressure(self.pressure_thresholds) if current else False


class BottleneckDetector:
    """Advanced bottleneck detection and analysis system."""

    def __init__(self, resource_monitor: ResourceMonitor):
        self.resource_monitor = resource_monitor
        self.detected_bottlenecks: List[BottleneckAnalysis] = []
        self.analysis_interval = 60.0  # Analyze every minute
        self.running = False
        self._analysis_task: Optional[asyncio.Task] = None

        # Bottleneck thresholds
        self.bottleneck_thresholds = {
            'cpu_high': 90.0,
            'cpu_sustained': 70.0,
            'memory_high': 90.0,
            'memory_sustained': 80.0,
            'load_high': 3.0,
            'io_high_read': 100 * 1024 * 1024,  # 100MB/s
            'io_high_write': 100 * 1024 * 1024,  # 100MB/s
        }

    async def start_detection(self):
        """Start bottleneck detection."""
        if self.running:
            return

        self.running = True
        self._analysis_task = asyncio.create_task(self._analysis_loop())
        logger.info("Bottleneck detection started")

    async def stop_detection(self):
        """Stop bottleneck detection."""
        if not self.running:
            return

        self.running = False
        if self._analysis_task:
            self._analysis_task.cancel()
            try:
                await self._analysis_task
            except asyncio.CancelledError:
                pass
        logger.info("Bottleneck detection stopped")

    async def _analysis_loop(self):
        """Main bottleneck analysis loop."""
        while self.running:
            try:
                bottlenecks = await self._analyze_bottlenecks()
                if bottlenecks:
                    self.detected_bottlenecks.extend(bottlenecks)
                    # Keep only recent bottlenecks (last 24 hours)
                    cutoff = datetime.now() - timedelta(hours=24)
                    self.detected_bottlenecks = [
                        b for b in self.detected_bottlenecks
                        if b.timestamp > cutoff
                    ]

                await asyncio.sleep(self.analysis_interval)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("Bottleneck analysis error", error=str(e))
                await asyncio.sleep(self.analysis_interval)

    async def _analyze_bottlenecks(self) -> List[BottleneckAnalysis]:
        """Analyze current system state for bottlenecks."""
        bottlenecks = []
        current_time = datetime.now()

        # Get recent metrics for analysis (last 5 minutes)
        recent_metrics = [
            m for m in self.resource_monitor.metrics_history
            if current_time - m.timestamp < timedelta(minutes=5)
        ]

        if len(recent_metrics) < 10:  # Need sufficient data
            return bottlenecks

        # CPU bottleneck detection
        cpu_values = [m.cpu_percent for m in recent_metrics]
        avg_cpu = statistics.mean(cpu_values)
        max_cpu = max(cpu_values)

        if max_cpu > self.bottleneck_thresholds['cpu_high']:
            bottlenecks.append(BottleneckAnalysis(
                timestamp=current_time,
                bottleneck_type='cpu',
                severity='critical' if max_cpu > 95 else 'high',
                affected_operations=['all_operations'],
                root_cause=f'CPU usage peaked at {max_cpu:.1f}%',
                impact_score=min(max_cpu / 100.0, 1.0),
                recommended_actions=[
                    'Reduce concurrent operations',
                    'Optimize CPU-intensive algorithms',
                    'Consider CPU scaling'
                ],
                estimated_improvement=max_cpu - 70.0
            ))
        elif avg_cpu > self.bottleneck_thresholds['cpu_sustained']:
            bottlenecks.append(BottleneckAnalysis(
                timestamp=current_time,
                bottleneck_type='cpu',
                severity='medium',
                affected_operations=['compute_operations'],
                root_cause=f'Sustained CPU usage at {avg_cpu:.1f}%',
                impact_score=avg_cpu / 100.0,
                recommended_actions=[
                    'Optimize algorithms',
                    'Implement caching',
                    'Profile CPU usage'
                ],
                estimated_improvement=avg_cpu - 50.0
            ))

        # Memory bottleneck detection
        memory_values = [m.memory_percent for m in recent_metrics]
        avg_memory = statistics.mean(memory_values)
        max_memory = max(memory_values)

        if max_memory > self.bottleneck_thresholds['memory_high']:
            bottlenecks.append(BottleneckAnalysis(
                timestamp=current_time,
                bottleneck_type='memory',
                severity='critical' if max_memory > 95 else 'high',
                affected_operations=['memory_operations', 'caching'],
                root_cause=f'Memory usage peaked at {max_memory:.1f}%',
                impact_score=min(max_memory / 100.0, 1.0),
                recommended_actions=[
                    'Implement garbage collection',
                    'Reduce cache sizes',
                    'Memory leak investigation',
                    'Consider memory scaling'
                ],
                estimated_improvement=max_memory - 60.0
            ))

        # I/O bottleneck detection
        io_read_values = [m.disk_io_read for m in recent_metrics]
        io_write_values = [m.disk_io_write for m in recent_metrics]

        if len(io_read_values) > 1:
            read_rate = (io_read_values[-1] - io_read_values[0]) / len(io_read_values)
            write_rate = (io_write_values[-1] - io_write_values[0]) / len(io_write_values)

            if read_rate > self.bottleneck_thresholds['io_high_read']:
                bottlenecks.append(BottleneckAnalysis(
                    timestamp=current_time,
                    bottleneck_type='io',
                    severity='medium',
                    affected_operations=['file_operations', 'database_operations'],
                    root_cause=f'High disk read rate: {read_rate / (1024*1024):.1f} MB/s',
                    impact_score=min(read_rate / self.bottleneck_thresholds['io_high_read'], 1.0),
                    recommended_actions=[
                        'Implement read caching',
                        'Optimize file access patterns',
                        'Consider SSD upgrade'
                    ],
                    estimated_improvement=30.0
                ))

            if write_rate > self.bottleneck_thresholds['io_high_write']:
                bottlenecks.append(BottleneckAnalysis(
                    timestamp=current_time,
                    bottleneck_type='io',
                    severity='medium',
                    affected_operations=['file_operations', 'logging'],
                    root_cause=f'High disk write rate: {write_rate / (1024*1024):.1f} MB/s',
                    impact_score=min(write_rate / self.bottleneck_thresholds['io_high_write'], 1.0),
                    recommended_actions=[
                        'Batch write operations',
                        'Reduce logging frequency',
                        'Implement write caching'
                    ],
                    estimated_improvement=25.0
                ))

        return bottlenecks

    def get_recent_bottlenecks(self, hours: int = 24) -> List[BottleneckAnalysis]:
        """Get recent bottlenecks within specified time window."""
        cutoff = datetime.now() - timedelta(hours=hours)
        return [b for b in self.detected_bottlenecks if b.timestamp > cutoff]


class AutoOptimizer:
    """Automatic optimization system with intelligent triggers."""

    def __init__(self, resource_monitor: ResourceMonitor, bottleneck_detector: BottleneckDetector):
        self.resource_monitor = resource_monitor
        self.bottleneck_detector = bottleneck_detector
        self.optimization_history: List[Dict] = []
        self.running = False
        self._optimizer_task: Optional[asyncio.Task] = None
        self.optimization_interval = 120.0  # Check every 2 minutes

        # Auto-optimization settings
        self.auto_optimization_enabled = True
        self.max_auto_actions_per_hour = 5
        self.auto_action_cooldown = timedelta(minutes=30)
        self.last_auto_actions: deque = deque(maxlen=100)

    async def start_optimization(self):
        """Start automatic optimization."""
        if self.running:
            return

        self.running = True
        self._optimizer_task = asyncio.create_task(self._optimization_loop())
        logger.info("Auto-optimization started")

    async def stop_optimization(self):
        """Stop automatic optimization."""
        if not self.running:
            return

        self.running = False
        if self._optimizer_task:
            self._optimizer_task.cancel()
            try:
                await self._optimizer_task
            except asyncio.CancelledError:
                pass
        logger.info("Auto-optimization stopped")

    async def _optimization_loop(self):
        """Main optimization loop."""
        while self.running:
            try:
                if self.auto_optimization_enabled:
                    await self._evaluate_and_optimize()
                await asyncio.sleep(self.optimization_interval)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("Auto-optimization error", error=str(e))
                await asyncio.sleep(self.optimization_interval)

    async def _evaluate_and_optimize(self):
        """Evaluate system state and trigger optimizations if needed."""
        current_time = datetime.now()

        # Check rate limiting
        recent_actions = [
            action for action in self.last_auto_actions
            if current_time - action['timestamp'] < timedelta(hours=1)
        ]

        if len(recent_actions) >= self.max_auto_actions_per_hour:
            logger.debug("Auto-optimization rate limited")
            return

        # Check for resource pressure
        if self.resource_monitor.is_under_pressure():
            await self._handle_resource_pressure()

        # Check for detected bottlenecks
        recent_bottlenecks = self.bottleneck_detector.get_recent_bottlenecks(hours=1)
        critical_bottlenecks = [b for b in recent_bottlenecks if b.severity == 'critical']

        if critical_bottlenecks:
            await self._handle_critical_bottlenecks(critical_bottlenecks)

    async def _handle_resource_pressure(self):
        """Handle resource pressure situations."""
        current_metrics = self.resource_monitor.get_current_metrics()
        if not current_metrics:
            return

        actions_taken = []

        # Handle high memory usage
        if current_metrics.memory_percent > 85:
            action = await self._trigger_memory_optimization()
            if action:
                actions_taken.append(action)

        # Handle high CPU usage
        if current_metrics.cpu_percent > 85:
            action = await self._trigger_cpu_optimization()
            if action:
                actions_taken.append(action)

        # Record actions
        if actions_taken:
            self.last_auto_actions.append({
                'timestamp': datetime.now(),
                'trigger': 'resource_pressure',
                'actions': actions_taken
            })

            logger.info("Auto-optimization triggered by resource pressure",
                       actions_count=len(actions_taken))

    async def _trigger_memory_optimization(self) -> Optional[str]:
        """Trigger memory optimization actions."""
        # This would integrate with actual system optimizations
        logger.info("Triggering memory optimization")

        # Example optimization actions:
        # - Clear caches
        # - Reduce buffer sizes
        # - Trigger garbage collection
        # - Reduce concurrent operations

        return "memory_optimization"

    async def _trigger_cpu_optimization(self) -> Optional[str]:
        """Trigger CPU optimization actions."""
        # This would integrate with actual system optimizations
        logger.info("Triggering CPU optimization")

        # Example optimization actions:
        # - Reduce thread pool sizes
        # - Lower operation concurrency
        # - Enable CPU throttling
        # - Defer non-critical operations

        return "cpu_optimization"

    async def _handle_critical_bottlenecks(self, bottlenecks: List[BottleneckAnalysis]):
        """Handle critical bottlenecks with automatic actions."""
        for bottleneck in bottlenecks:
            action = await self._create_optimization_action(bottleneck)
            if action and action.auto_executable:
                await self._execute_optimization_action(action)

                self.last_auto_actions.append({
                    'timestamp': datetime.now(),
                    'trigger': 'critical_bottleneck',
                    'bottleneck_type': bottleneck.bottleneck_type,
                    'action': action.action_id
                })

    async def _create_optimization_action(self, bottleneck: BottleneckAnalysis) -> Optional[OptimizationAction]:
        """Create optimization action for a bottleneck."""
        action_id = f"auto_opt_{int(time.time())}"

        if bottleneck.bottleneck_type == 'memory' and bottleneck.severity == 'critical':
            return OptimizationAction(
                action_id=action_id,
                trigger=OptimizationTrigger.THRESHOLD_EXCEEDED,
                action_type='memory_pressure_relief',
                description='Automatic memory pressure relief',
                parameters={'force_gc': True, 'clear_caches': True},
                expected_benefit=bottleneck.estimated_improvement,
                risk_level='low',
                reversible=True,
                auto_executable=True
            )
        elif bottleneck.bottleneck_type == 'cpu' and bottleneck.severity == 'critical':
            return OptimizationAction(
                action_id=action_id,
                trigger=OptimizationTrigger.THRESHOLD_EXCEEDED,
                action_type='cpu_load_reduction',
                description='Automatic CPU load reduction',
                parameters={'reduce_concurrency': True, 'throttle_operations': True},
                expected_benefit=bottleneck.estimated_improvement,
                risk_level='medium',
                reversible=True,
                auto_executable=True
            )

        return None

    async def _execute_optimization_action(self, action: OptimizationAction):
        """Execute an optimization action."""
        logger.info("Executing optimization action",
                   action_id=action.action_id,
                   action_type=action.action_type)

        # Record execution
        self.optimization_history.append({
            'timestamp': datetime.now().isoformat(),
            'action': action.to_dict(),
            'status': 'executed'
        })


class EnhancedPerformanceMonitor:
    """
    Comprehensive performance monitoring system with automatic optimization,
    health reporting, and bottleneck analysis.
    """

    def __init__(self, project_id: str):
        self.project_id = project_id
        self.start_time = time.time()

        # Core components
        self.resource_monitor = ResourceMonitor()
        self.bottleneck_detector = BottleneckDetector(self.resource_monitor)
        self.auto_optimizer = AutoOptimizer(self.resource_monitor, self.bottleneck_detector)

        # Health monitoring
        self.health_alerts: List[Dict] = []
        self.component_health: Dict[str, str] = {}

        # Performance thresholds
        self.health_thresholds = {
            'excellent': 90.0,
            'good': 75.0,
            'acceptable': 60.0,
            'degraded': 40.0
        }

        logger.info("Enhanced performance monitor initialized", project_id=project_id)

    async def start_monitoring(self):
        """Start all monitoring components."""
        await self.resource_monitor.start_monitoring()
        await self.bottleneck_detector.start_detection()
        await self.auto_optimizer.start_optimization()

        logger.info("Enhanced performance monitoring started")

    async def stop_monitoring(self):
        """Stop all monitoring components."""
        await self.auto_optimizer.stop_optimization()
        await self.bottleneck_detector.stop_detection()
        await self.resource_monitor.stop_monitoring()

        logger.info("Enhanced performance monitoring stopped")

    async def generate_health_report(self) -> HealthReport:
        """Generate comprehensive system health report."""
        current_time = datetime.now()

        # Get current resource metrics
        current_resources = self.resource_monitor.get_current_metrics()
        if not current_resources:
            current_resources = ResourceMetric(
                timestamp=current_time,
                cpu_percent=0.0, memory_bytes=0, memory_percent=0.0,
                disk_io_read=0, disk_io_write=0, network_sent=0, network_recv=0,
                open_files=0, thread_count=0, load_average=0.0
            )

        # Calculate overall health score
        health_score = self._calculate_health_score(current_resources)

        # Determine overall health level
        overall_health = self._determine_health_level(health_score)

        # Get component health
        component_health = self._assess_component_health(current_resources)

        # Get performance trends
        performance_trends = self.resource_monitor.get_resource_trends()

        # Get active alerts
        active_alerts = self._get_active_alerts()

        # Get recent bottlenecks
        recent_bottlenecks = self.bottleneck_detector.get_recent_bottlenecks(hours=24)

        # Generate optimization recommendations
        optimization_recommendations = await self._generate_optimization_recommendations(current_resources, recent_bottlenecks)

        # Calculate uptime
        uptime = time.time() - self.start_time

        return HealthReport(
            timestamp=current_time,
            overall_health=overall_health,
            health_score=health_score,
            component_health=component_health,
            active_alerts=active_alerts,
            performance_trends=performance_trends,
            resource_utilization=current_resources,
            bottlenecks=recent_bottlenecks,
            optimization_recommendations=optimization_recommendations,
            uptime_seconds=uptime
        )

    def _calculate_health_score(self, resources: ResourceMetric) -> float:
        """Calculate overall system health score (0-100)."""
        scores = []

        # CPU health score (inverted - lower usage is better)
        cpu_score = max(0, 100 - resources.cpu_percent * 1.2)
        scores.append(cpu_score)

        # Memory health score
        memory_score = max(0, 100 - resources.memory_percent * 1.1)
        scores.append(memory_score)

        # Load average health score
        load_score = max(0, 100 - (resources.load_average * 25))
        scores.append(load_score)

        # File descriptor health score
        fd_score = max(0, 100 - (resources.open_files / 10))
        scores.append(fd_score)

        return statistics.mean(scores)

    def _determine_health_level(self, score: float) -> str:
        """Determine health level from score."""
        if score >= self.health_thresholds['excellent']:
            return 'excellent'
        elif score >= self.health_thresholds['good']:
            return 'good'
        elif score >= self.health_thresholds['acceptable']:
            return 'acceptable'
        elif score >= self.health_thresholds['degraded']:
            return 'degraded'
        else:
            return 'critical'

    def _assess_component_health(self, resources: ResourceMetric) -> Dict[str, str]:
        """Assess health of individual components."""
        return {
            'cpu': self._determine_health_level(max(0, 100 - resources.cpu_percent * 1.2)),
            'memory': self._determine_health_level(max(0, 100 - resources.memory_percent * 1.1)),
            'disk_io': 'good',  # Would need more sophisticated analysis
            'network': 'good',  # Would need more sophisticated analysis
            'processes': self._determine_health_level(max(0, 100 - (resources.thread_count / 2)))
        }

    def _get_active_alerts(self) -> List[Dict]:
        """Get currently active health alerts."""
        # Return recent alerts (last hour)
        cutoff = datetime.now() - timedelta(hours=1)
        return [
            alert for alert in self.health_alerts
            if datetime.fromisoformat(alert.get('timestamp', '')) > cutoff
        ]

    async def _generate_optimization_recommendations(
        self,
        resources: ResourceMetric,
        bottlenecks: List[BottleneckAnalysis]
    ) -> List[OptimizationAction]:
        """Generate optimization recommendations based on current state."""
        recommendations = []

        # High memory usage recommendations
        if resources.memory_percent > 80:
            recommendations.append(OptimizationAction(
                action_id=f"mem_opt_{int(time.time())}",
                trigger=OptimizationTrigger.THRESHOLD_EXCEEDED,
                action_type='memory_optimization',
                description='Reduce memory usage through cache management',
                parameters={
                    'clear_caches': True,
                    'reduce_buffer_sizes': True,
                    'force_gc': True
                },
                expected_benefit=20.0,
                risk_level='low',
                reversible=True,
                auto_executable=True
            ))

        # High CPU usage recommendations
        if resources.cpu_percent > 80:
            recommendations.append(OptimizationAction(
                action_id=f"cpu_opt_{int(time.time())}",
                trigger=OptimizationTrigger.THRESHOLD_EXCEEDED,
                action_type='cpu_optimization',
                description='Reduce CPU load through operation throttling',
                parameters={
                    'reduce_thread_pool_size': True,
                    'throttle_background_tasks': True,
                    'defer_non_critical': True
                },
                expected_benefit=25.0,
                risk_level='medium',
                reversible=True,
                auto_executable=False  # Manual approval needed
            ))

        # Bottleneck-based recommendations
        for bottleneck in bottlenecks:
            if bottleneck.severity in ['high', 'critical']:
                recommendations.append(OptimizationAction(
                    action_id=f"bottleneck_opt_{int(time.time())}",
                    trigger=OptimizationTrigger.THRESHOLD_EXCEEDED,
                    action_type=f'{bottleneck.bottleneck_type}_optimization',
                    description=f'Address {bottleneck.bottleneck_type} bottleneck: {bottleneck.root_cause}',
                    parameters={'target_bottleneck': bottleneck.to_dict()},
                    expected_benefit=bottleneck.estimated_improvement,
                    risk_level='medium',
                    reversible=True,
                    auto_executable=bottleneck.severity == 'critical'
                ))

        return recommendations

    @asynccontextmanager
    async def performance_context(self, operation_name: str):
        """Context manager for performance monitoring of operations."""
        start_time = time.time()
        start_resources = self.resource_monitor.get_current_metrics()

        try:
            yield
        finally:
            end_time = time.time()
            end_resources = self.resource_monitor.get_current_metrics()

            duration = end_time - start_time

            # Log performance info
            logger.debug("Operation completed",
                        operation=operation_name,
                        duration=duration,
                        cpu_change=end_resources.cpu_percent - start_resources.cpu_percent if start_resources and end_resources else 0,
                        memory_change=end_resources.memory_percent - start_resources.memory_percent if start_resources and end_resources else 0)


# Export main classes
__all__ = [
    'PerformanceLevel',
    'OptimizationTrigger',
    'ResourceMetric',
    'BottleneckAnalysis',
    'OptimizationAction',
    'HealthReport',
    'ResourceMonitor',
    'BottleneckDetector',
    'AutoOptimizer',
    'EnhancedPerformanceMonitor'
]