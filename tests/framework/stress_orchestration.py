"""
Stress Testing Orchestration for Multi-Component Coordination

This module provides specialized orchestration for stress testing scenarios,
coordinating multiple system components under load, simulating failures, and
tracking recovery times and performance degradation.

Features:
- Multi-component stress test coordination
- Load pattern simulation (CONSTANT, RAMP_UP, SPIKE, SUSTAINED)
- Failure injection (CRASH, HANG, SLOW, NETWORK_PARTITION)
- Recovery time measurement
- Performance degradation tracking
- Resource constraint enforcement
- Resource exhaustion simulation
- Stability checkpoint validation
- 24+ hour stability testing with checkpoint/resume
- Memory leak detection and resource monitoring
"""

import asyncio
import json
import logging
import signal
import sqlite3
import statistics
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Optional

import psutil

from .integration import (
    ComponentConfig,
    ComponentController,
    ComponentInstance,
    ComponentState,
    DockerController,
    IntegrationTestCoordinator,
    ProcessController,
)
from .orchestration import (
    OrchestrationConfig,
    OrchestrationResult,
    PipelineStage,
    TestOrchestrator,
)
from .resource_exhaustion import (
    ResourceExhaustionScenario,
    ResourceExhaustionSimulator,
)


class LoadPattern(Enum):
    """Load patterns for stress testing."""
    CONSTANT = "constant"      # Steady load throughout test
    RAMP_UP = "ramp_up"        # Gradually increase load
    SPIKE = "spike"            # Sudden bursts of load
    SUSTAINED = "sustained"     # Long-term sustained load


class FailureMode(Enum):
    """Failure modes for component failure injection."""
    CRASH = "crash"                        # Process termination (SIGKILL)
    HANG = "hang"                          # Process freeze (SIGSTOP)
    SLOW = "slow"                          # Performance degradation
    NETWORK_PARTITION = "network_partition"  # Network isolation


# Extend PipelineStage with stress-specific stages
class StressPipelineStage(Enum):
    """Additional pipeline stages for stress testing."""
    RESOURCE_BASELINE = "resource_baseline"        # Capture baseline metrics
    LOAD_RAMP = "load_ramp"                       # Gradually increase load
    STRESS_EXECUTION = "stress_execution"          # Run at full stress
    RESOURCE_EXHAUSTION = "resource_exhaustion"    # Simulate resource exhaustion
    FAILURE_INJECTION = "failure_injection"        # Inject failures
    RECOVERY_VALIDATION = "recovery_validation"    # Verify recovery
    DEGRADATION_ANALYSIS = "degradation_analysis"  # Analyze performance


@dataclass
class StressTestConfig(OrchestrationConfig):
    """Configuration for stress testing orchestration."""
    load_pattern: LoadPattern = LoadPattern.CONSTANT
    duration_hours: float = 24.0
    resource_constraints: dict[str, Any] = field(default_factory=lambda: {
        "memory_mb": 1024,
        "cpu_percent": 80,
        "disk_io_mb_s": 100
    })
    failure_injection_enabled: bool = False
    resource_exhaustion_scenario: ResourceExhaustionScenario | None = None
    performance_thresholds: dict[str, float] = field(default_factory=lambda: {
        "p50_ms": 100.0,
        "p95_ms": 500.0,
        "p99_ms": 1000.0,
        "error_rate_percent": 1.0
    })
    stability_checkpoints_minutes: int = 30


@dataclass
class StabilityTestConfig:
    """Configuration for 24+ hour stability testing."""
    duration_hours: float = 24.0
    checkpoint_interval_minutes: int = 30
    resource_monitoring_interval_seconds: int = 60
    memory_leak_detection_enabled: bool = True
    performance_thresholds: dict[str, float] = field(default_factory=lambda: {
        "p50_ms": 100.0,
        "p95_ms": 500.0,
        "p99_ms": 1000.0,
        "error_rate_percent": 1.0
    })
    auto_stop_on_degradation: bool = False
    memory_leak_threshold_mb_per_hour: float = 5.0


@dataclass
class ComponentStressConfig:
    """Stress testing configuration for a system component."""
    component_name: str
    resource_limits: dict[str, Any] = field(default_factory=lambda: {
        "memory_mb": 512,
        "cpu_percent": 50
    })
    failure_modes: list[str] = field(default_factory=lambda: ["crash"])
    health_check_endpoint: str | None = None
    recovery_timeout_seconds: float = 30.0


@dataclass
class ResourceMetrics:
    """Resource usage metrics at a point in time."""
    timestamp: float
    cpu_percent: float
    memory_rss_mb: float
    memory_vms_mb: float
    memory_available_mb: float
    disk_read_mb: float
    disk_write_mb: float
    net_sent_mb: float
    net_recv_mb: float
    open_fds: int
    component_name: str | None = None


@dataclass
class MemoryLeakReport:
    """Memory leak detection report."""
    detected: bool
    growth_rate_mb_per_hour: float
    confidence: float  # 0-1
    affected_component: str | None
    samples_analyzed: int
    start_memory_mb: float
    end_memory_mb: float
    duration_hours: float


@dataclass
class StabilityTestReport:
    """Comprehensive stability test report."""
    run_id: str
    duration_hours: float
    stability_score: float  # 0-100
    uptime_percentage: float
    total_errors: int
    error_rate_per_hour: float
    memory_leak_detected: bool
    memory_growth_rate_mb_per_hour: float
    performance_degradation_percent: float
    recovery_incidents: list[dict[str, Any]]
    resource_usage_summary: dict[str, dict[str, float]]  # metric -> {avg, min, max, p95}
    checkpoints_completed: int
    test_stopped_reason: str | None = None


@dataclass
class StressTestResult(OrchestrationResult):
    """Results from stress testing orchestration."""
    baseline_metrics: dict[str, Any] = field(default_factory=dict)
    recovery_times: dict[str, float] = field(default_factory=dict)  # component -> seconds
    performance_samples: dict[str, list[float]] = field(default_factory=lambda: defaultdict(list))
    failure_injections: list[dict[str, Any]] = field(default_factory=list)
    stability_violations: list[str] = field(default_factory=list)
    resource_exhaustion_results: dict[str, Any] = field(default_factory=dict)

    @property
    def avg_recovery_time(self) -> float:
        """Calculate average recovery time across components."""
        if not self.recovery_times:
            return 0.0
        return sum(self.recovery_times.values()) / len(self.recovery_times)

    @property
    def max_recovery_time(self) -> float:
        """Get maximum recovery time across components."""
        if not self.recovery_times:
            return 0.0
        return max(self.recovery_times.values())


class ResourceMonitor:
    """
    Continuous resource monitoring for stability testing.

    Tracks CPU, memory, disk I/O, network I/O, and file descriptors
    over time using psutil for system metrics.
    """

    def __init__(self, component_name: str | None = None):
        """Initialize resource monitor.

        Args:
            component_name: Optional component name for tagging metrics
        """
        self.component_name = component_name
        self.metrics_history: list[ResourceMetrics] = []
        self._monitoring_task: asyncio.Task | None = None
        self._stop_event = asyncio.Event()
        self.logger = logging.getLogger(__name__)

        # Track initial resource values for delta calculations
        self._initial_disk_io: tuple[int, int] | None = None
        self._initial_net_io: tuple[int, int] | None = None

    async def start_monitoring(self, interval_seconds: int = 60) -> None:
        """Start continuous resource monitoring.

        Args:
            interval_seconds: Interval between measurements
        """
        self._stop_event.clear()
        self._monitoring_task = asyncio.create_task(
            self._monitor_loop(interval_seconds)
        )
        self.logger.info(f"Started resource monitoring (interval={interval_seconds}s)")

    async def stop_monitoring(self) -> None:
        """Stop resource monitoring."""
        self._stop_event.set()
        if self._monitoring_task:
            await self._monitoring_task
            self._monitoring_task = None
        self.logger.info("Stopped resource monitoring")

    async def _monitor_loop(self, interval_seconds: int) -> None:
        """Main monitoring loop.

        Args:
            interval_seconds: Interval between measurements
        """
        while not self._stop_event.is_set():
            try:
                metrics = self._capture_metrics()
                self.metrics_history.append(metrics)
            except Exception as e:
                self.logger.error(f"Error capturing metrics: {e}")

            try:
                await asyncio.wait_for(
                    self._stop_event.wait(),
                    timeout=interval_seconds
                )
                break  # Stop event was set
            except asyncio.TimeoutError:
                pass  # Continue monitoring

    def _capture_metrics(self) -> ResourceMetrics:
        """Capture current resource metrics.

        Returns:
            ResourceMetrics snapshot
        """
        # Get process or system metrics
        process = psutil.Process()

        # CPU and memory
        cpu_percent = process.cpu_percent(interval=0.1)
        mem_info = process.memory_info()
        virtual_mem = psutil.virtual_memory()

        # Disk I/O
        disk_io = psutil.disk_io_counters()
        if self._initial_disk_io is None:
            self._initial_disk_io = (disk_io.read_bytes, disk_io.write_bytes)
        disk_read_mb = (disk_io.read_bytes - self._initial_disk_io[0]) / (1024 * 1024)
        disk_write_mb = (disk_io.write_bytes - self._initial_disk_io[1]) / (1024 * 1024)

        # Network I/O
        net_io = psutil.net_io_counters()
        if self._initial_net_io is None:
            self._initial_net_io = (net_io.bytes_sent, net_io.bytes_recv)
        net_sent_mb = (net_io.bytes_sent - self._initial_net_io[0]) / (1024 * 1024)
        net_recv_mb = (net_io.bytes_recv - self._initial_net_io[1]) / (1024 * 1024)

        # Open file descriptors
        try:
            open_fds = process.num_fds() if hasattr(process, 'num_fds') else len(process.open_files())
        except (psutil.AccessDenied, AttributeError):
            open_fds = 0

        return ResourceMetrics(
            timestamp=time.time(),
            cpu_percent=cpu_percent,
            memory_rss_mb=mem_info.rss / (1024 * 1024),
            memory_vms_mb=mem_info.vms / (1024 * 1024),
            memory_available_mb=virtual_mem.available / (1024 * 1024),
            disk_read_mb=disk_read_mb,
            disk_write_mb=disk_write_mb,
            net_sent_mb=net_sent_mb,
            net_recv_mb=net_recv_mb,
            open_fds=open_fds,
            component_name=self.component_name
        )

    def get_metrics(self) -> list[ResourceMetrics]:
        """Get all collected metrics.

        Returns:
            List of resource metrics
        """
        return self.metrics_history.copy()

    def detect_memory_leak(
        self,
        threshold_mb_per_hour: float = 5.0
    ) -> MemoryLeakReport | None:
        """Detect memory leaks from collected metrics.

        Args:
            threshold_mb_per_hour: Growth rate threshold for leak detection

        Returns:
            MemoryLeakReport if leak detected, None otherwise
        """

        detector = MemoryLeakDetector()
        return detector.analyze_memory_trend(
            self.metrics_history,
            threshold_mb_per_hour,
            self.component_name
        )


class MemoryLeakDetector:
    """
    Analyzes resource trends to detect memory leaks.

    Uses linear regression to identify sustained memory growth
    over time and determine if it exceeds acceptable thresholds.
    """

    def __init__(self):
        """Initialize memory leak detector."""
        self.logger = logging.getLogger(__name__)

    def analyze_memory_trend(
        self,
        metrics: list[ResourceMetrics],
        threshold_mb_per_hour: float,
        component_name: str | None = None
    ) -> MemoryLeakReport:
        """Analyze memory usage trend to detect leaks.

        Args:
            metrics: List of resource metrics over time
            threshold_mb_per_hour: Growth rate threshold
            component_name: Component being analyzed

        Returns:
            MemoryLeakReport with detection results
        """
        if len(metrics) < 2:
            return MemoryLeakReport(
                detected=False,
                growth_rate_mb_per_hour=0.0,
                confidence=0.0,
                affected_component=component_name,
                samples_analyzed=len(metrics),
                start_memory_mb=0.0,
                end_memory_mb=0.0,
                duration_hours=0.0
            )

        # Extract memory values and timestamps
        memory_values = [m.memory_rss_mb for m in metrics]
        timestamps = [m.timestamp for m in metrics]

        # Calculate growth rate
        growth_rate = self.calculate_growth_rate(memory_values, timestamps)

        # Determine if leak detected
        detected = self.is_leak_detected(growth_rate, threshold_mb_per_hour)

        # Calculate confidence based on data quality
        confidence = self._calculate_confidence(memory_values, growth_rate)

        # Duration
        duration_hours = (timestamps[-1] - timestamps[0]) / 3600.0

        return MemoryLeakReport(
            detected=detected,
            growth_rate_mb_per_hour=growth_rate,
            confidence=confidence,
            affected_component=component_name,
            samples_analyzed=len(metrics),
            start_memory_mb=memory_values[0],
            end_memory_mb=memory_values[-1],
            duration_hours=duration_hours
        )

    def calculate_growth_rate(
        self,
        values: list[float],
        timestamps: list[float]
    ) -> float:
        """Calculate memory growth rate using linear regression.

        Args:
            values: Memory values in MB
            timestamps: Unix timestamps

        Returns:
            Growth rate in MB/hour
        """
        if len(values) < 2:
            return 0.0

        # Convert timestamps to hours from start
        hours = [(t - timestamps[0]) / 3600.0 for t in timestamps]

        # Simple linear regression
        n = len(values)
        sum_x = sum(hours)
        sum_y = sum(values)
        sum_xy = sum(x * y for x, y in zip(hours, values, strict=False))
        sum_x2 = sum(x * x for x in hours)

        # Calculate slope (growth rate per hour)
        denominator = n * sum_x2 - sum_x * sum_x
        if abs(denominator) < 1e-10:
            return 0.0

        slope = (n * sum_xy - sum_x * sum_y) / denominator

        return slope

    def is_leak_detected(
        self,
        growth_rate: float,
        threshold: float
    ) -> bool:
        """Determine if growth rate indicates a leak.

        Args:
            growth_rate: Memory growth rate in MB/hour
            threshold: Threshold for leak detection

        Returns:
            True if leak detected
        """
        return growth_rate > threshold

    def _calculate_confidence(
        self,
        values: list[float],
        growth_rate: float
    ) -> float:
        """Calculate confidence in leak detection.

        Args:
            values: Memory values
            growth_rate: Calculated growth rate

        Returns:
            Confidence score 0-1
        """
        if len(values) < 10:
            # Low confidence with few samples
            return min(len(values) / 10.0, 0.5)

        # Calculate coefficient of variation
        if len(values) >= 2:
            mean = statistics.mean(values)
            if mean > 0:
                stdev = statistics.stdev(values)
                cv = stdev / mean
                # Lower CV = higher confidence (less noise)
                confidence = max(0.0, min(1.0, 1.0 - cv))
            else:
                confidence = 0.5
        else:
            confidence = 0.5

        return confidence


class StabilityMetricsCollector:
    """
    Collects and aggregates stability-specific metrics.

    Tracks uptime, error rates, response times, and resource trends
    to calculate an overall stability score for long-running tests.
    """

    def __init__(self, database_path: Path):
        """Initialize stability metrics collector.

        Args:
            database_path: Path to SQLite database
        """
        self.database_path = database_path
        self.logger = logging.getLogger(__name__)

    def record_checkpoint(
        self,
        run_id: str,
        elapsed_hours: float,
        metrics: dict[str, Any]
    ) -> None:
        """Record stability checkpoint.

        Args:
            run_id: Stability run ID
            elapsed_hours: Hours elapsed since start
            metrics: Checkpoint metrics dictionary
        """
        with sqlite3.connect(self.database_path) as conn:
            conn.execute("""
                INSERT INTO stability_checkpoints
                (run_id, elapsed_hours, checkpoint_time, metrics_json)
                VALUES (?, ?, ?, ?)
            """, (
                run_id,
                elapsed_hours,
                time.time(),
                json.dumps(metrics)
            ))

    def get_stability_score(self, run_id: str) -> float:
        """Calculate stability score for a run.

        Args:
            run_id: Stability run ID

        Returns:
            Stability score 0-100
        """
        with sqlite3.connect(self.database_path) as conn:
            # Get run info
            run_row = conn.execute("""
                SELECT duration_hours, total_errors, uptime_percentage,
                       performance_degradation_percent, memory_leak_detected
                FROM stability_runs
                WHERE run_id = ?
            """, (run_id,)).fetchone()

            if not run_row:
                return 0.0

            duration, errors, uptime, degradation, leak = run_row

            # Calculate weighted score
            # Uptime: 40%
            uptime_score = uptime * 0.4

            # Error rate: 30% (fewer errors = higher score)
            error_rate = errors / max(duration, 1.0)
            error_score = max(0, (1.0 - min(error_rate / 10.0, 1.0)) * 30.0)

            # Performance: 20% (less degradation = higher score)
            perf_score = max(0, (1.0 - min(degradation / 100.0, 1.0)) * 20.0)

            # Memory: 10% (no leak = full score)
            memory_score = 0.0 if leak else 10.0

            return uptime_score + error_score + perf_score + memory_score

    def generate_stability_report(
        self,
        run_id: str,
        resource_metrics: list[ResourceMetrics],
        recovery_incidents: list[dict[str, Any]]
    ) -> StabilityTestReport:
        """Generate comprehensive stability report.

        Args:
            run_id: Stability run ID
            resource_metrics: Collected resource metrics
            recovery_incidents: Recovery incident records

        Returns:
            StabilityTestReport
        """
        with sqlite3.connect(self.database_path) as conn:
            # Get run data
            run_row = conn.execute("""
                SELECT duration_hours, total_errors, uptime_percentage,
                       performance_degradation_percent, memory_leak_detected,
                       memory_growth_rate, checkpoints_completed, stopped_reason
                FROM stability_runs
                WHERE run_id = ?
            """, (run_id,)).fetchone()

            if not run_row:
                raise ValueError(f"Run {run_id} not found")

            (duration, total_errors, uptime, degradation, leak,
             growth_rate, checkpoints, stop_reason) = run_row

            # Calculate resource usage summary
            resource_summary = self._summarize_resources(resource_metrics)

            # Calculate error rate
            error_rate = total_errors / max(duration, 1.0)

            # Get stability score
            stability_score = self.get_stability_score(run_id)

            return StabilityTestReport(
                run_id=run_id,
                duration_hours=duration,
                stability_score=stability_score,
                uptime_percentage=uptime,
                total_errors=total_errors,
                error_rate_per_hour=error_rate,
                memory_leak_detected=bool(leak),
                memory_growth_rate_mb_per_hour=growth_rate,
                performance_degradation_percent=degradation,
                recovery_incidents=recovery_incidents,
                resource_usage_summary=resource_summary,
                checkpoints_completed=checkpoints,
                test_stopped_reason=stop_reason
            )

    def _summarize_resources(
        self,
        metrics: list[ResourceMetrics]
    ) -> dict[str, dict[str, float]]:
        """Summarize resource usage statistics.

        Args:
            metrics: List of resource metrics

        Returns:
            Summary dictionary with avg, min, max, p95 for each metric
        """
        if not metrics:
            return {}

        summary = {}

        # Metrics to summarize
        metric_fields = [
            ('cpu_percent', 'cpu_percent'),
            ('memory_rss_mb', 'memory_rss_mb'),
            ('disk_read_mb', 'disk_read_mb'),
            ('disk_write_mb', 'disk_write_mb'),
            ('net_sent_mb', 'net_sent_mb'),
            ('net_recv_mb', 'net_recv_mb'),
            ('open_fds', 'open_fds')
        ]

        for name, field in metric_fields:
            values = [getattr(m, field) for m in metrics]
            if values:
                summary[name] = {
                    'avg': statistics.mean(values),
                    'min': min(values),
                    'max': max(values),
                    'p95': self._percentile(values, 95)
                }

        return summary

    @staticmethod
    def _percentile(values: list[float], p: int) -> float:
        """Calculate percentile value.

        Args:
            values: List of values
            p: Percentile (0-100)

        Returns:
            Percentile value
        """
        if not values:
            return 0.0
        sorted_values = sorted(values)
        idx = int(len(sorted_values) * p / 100.0)
        return sorted_values[min(idx, len(sorted_values) - 1)]


class StabilityTestManager:
    """
    Manages 24+ hour stability test runs with checkpointing.

    Handles test lifecycle, checkpoint/resume functionality,
    state persistence, and graceful shutdown/recovery.
    """

    def __init__(self, database_path: Path):
        """Initialize stability test manager.

        Args:
            database_path: Path to SQLite database
        """
        self.database_path = database_path
        self.active_runs: dict[str, dict[str, Any]] = {}
        self.resource_monitors: dict[str, ResourceMonitor] = {}
        self.logger = logging.getLogger(__name__)
        self.metrics_collector = StabilityMetricsCollector(database_path)

        # Initialize database tables
        self._init_stability_database()

    def _init_stability_database(self):
        """Initialize stability testing database tables."""
        with sqlite3.connect(self.database_path) as conn:
            # Stability runs
            conn.execute("""
                CREATE TABLE IF NOT EXISTS stability_runs (
                    run_id TEXT PRIMARY KEY,
                    config_json TEXT NOT NULL,
                    start_time REAL NOT NULL,
                    end_time REAL,
                    status TEXT NOT NULL,
                    duration_hours REAL,
                    total_errors INTEGER DEFAULT 0,
                    uptime_percentage REAL DEFAULT 100.0,
                    performance_degradation_percent REAL DEFAULT 0.0,
                    memory_leak_detected INTEGER DEFAULT 0,
                    memory_growth_rate REAL DEFAULT 0.0,
                    checkpoints_completed INTEGER DEFAULT 0,
                    stopped_reason TEXT
                )
            """)

            # Stability checkpoints
            conn.execute("""
                CREATE TABLE IF NOT EXISTS stability_checkpoints (
                    checkpoint_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    run_id TEXT NOT NULL,
                    elapsed_hours REAL NOT NULL,
                    checkpoint_time REAL NOT NULL,
                    metrics_json TEXT NOT NULL,
                    FOREIGN KEY (run_id) REFERENCES stability_runs (run_id)
                )
            """)

            # Resource samples
            conn.execute("""
                CREATE TABLE IF NOT EXISTS resource_samples (
                    sample_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    run_id TEXT NOT NULL,
                    timestamp REAL NOT NULL,
                    component_name TEXT,
                    cpu_percent REAL,
                    memory_rss_mb REAL,
                    memory_vms_mb REAL,
                    memory_available_mb REAL,
                    disk_read_mb REAL,
                    disk_write_mb REAL,
                    net_sent_mb REAL,
                    net_recv_mb REAL,
                    open_fds INTEGER,
                    FOREIGN KEY (run_id) REFERENCES stability_runs (run_id)
                )
            """)

            # Memory leak reports
            conn.execute("""
                CREATE TABLE IF NOT EXISTS memory_leak_reports (
                    report_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    run_id TEXT NOT NULL,
                    detected INTEGER NOT NULL,
                    growth_rate_mb_per_hour REAL NOT NULL,
                    confidence REAL NOT NULL,
                    affected_component TEXT,
                    samples_analyzed INTEGER NOT NULL,
                    start_memory_mb REAL NOT NULL,
                    end_memory_mb REAL NOT NULL,
                    duration_hours REAL NOT NULL,
                    report_time REAL NOT NULL,
                    FOREIGN KEY (run_id) REFERENCES stability_runs (run_id)
                )
            """)

    async def start_stability_run(
        self,
        config: StabilityTestConfig
    ) -> str:
        """Start a new stability test run.

        Args:
            config: Stability test configuration

        Returns:
            Run ID
        """
        run_id = f"stability_{int(time.time() * 1000)}"

        # Create run record
        with sqlite3.connect(self.database_path) as conn:
            conn.execute("""
                INSERT INTO stability_runs
                (run_id, config_json, start_time, status)
                VALUES (?, ?, ?, ?)
            """, (
                run_id,
                json.dumps({
                    'duration_hours': config.duration_hours,
                    'checkpoint_interval_minutes': config.checkpoint_interval_minutes,
                    'resource_monitoring_interval_seconds': config.resource_monitoring_interval_seconds,
                    'memory_leak_detection_enabled': config.memory_leak_detection_enabled,
                    'performance_thresholds': config.performance_thresholds,
                    'auto_stop_on_degradation': config.auto_stop_on_degradation,
                    'memory_leak_threshold_mb_per_hour': config.memory_leak_threshold_mb_per_hour
                }),
                time.time(),
                'running'
            ))

        # Initialize run state
        self.active_runs[run_id] = {
            'config': config,
            'start_time': time.time(),
            'last_checkpoint': time.time(),
            'errors': 0,
            'downtime_seconds': 0.0
        }

        # Start resource monitoring
        monitor = ResourceMonitor()
        self.resource_monitors[run_id] = monitor
        await monitor.start_monitoring(config.resource_monitoring_interval_seconds)

        self.logger.info(f"Started stability run {run_id}")

        return run_id

    async def checkpoint_run(self, run_id: str) -> None:
        """Save checkpoint for stability run.

        Args:
            run_id: Stability run ID
        """
        if run_id not in self.active_runs:
            raise ValueError(f"Run {run_id} not found")

        run_state = self.active_runs[run_id]
        elapsed_hours = (time.time() - run_state['start_time']) / 3600.0

        # Collect checkpoint metrics
        monitor = self.resource_monitors.get(run_id)
        metrics = {
            'elapsed_hours': elapsed_hours,
            'errors': run_state['errors'],
            'downtime_seconds': run_state['downtime_seconds'],
            'resource_samples': len(monitor.get_metrics()) if monitor else 0
        }

        # Save checkpoint
        self.metrics_collector.record_checkpoint(run_id, elapsed_hours, metrics)

        # Update last checkpoint time
        run_state['last_checkpoint'] = time.time()

        # Update checkpoints counter
        with sqlite3.connect(self.database_path) as conn:
            conn.execute("""
                UPDATE stability_runs
                SET checkpoints_completed = checkpoints_completed + 1
                WHERE run_id = ?
            """, (run_id,))

        # Save resource samples to database
        if monitor:
            await self._save_resource_samples(run_id, monitor.get_metrics())

        self.logger.info(f"Checkpointed run {run_id} at {elapsed_hours:.2f} hours")

    async def resume_run(self, run_id: str) -> None:
        """Resume stability run from last checkpoint.

        Args:
            run_id: Stability run ID
        """
        with sqlite3.connect(self.database_path) as conn:
            # Get run info
            run_row = conn.execute("""
                SELECT config_json, start_time, status
                FROM stability_runs
                WHERE run_id = ?
            """, (run_id,)).fetchone()

            if not run_row:
                raise ValueError(f"Run {run_id} not found")

            config_json, start_time, status = run_row

            if status == 'completed':
                raise ValueError(f"Run {run_id} already completed")

            # Load configuration
            config_dict = json.loads(config_json)
            config = StabilityTestConfig(**config_dict)

            # Get last checkpoint
            checkpoint_row = conn.execute("""
                SELECT elapsed_hours, metrics_json
                FROM stability_checkpoints
                WHERE run_id = ?
                ORDER BY elapsed_hours DESC
                LIMIT 1
            """, (run_id,)).fetchone()

            if checkpoint_row:
                elapsed_hours, metrics_json = checkpoint_row
                metrics = json.loads(metrics_json)
            else:
                elapsed_hours = 0.0
                metrics = {'errors': 0, 'downtime_seconds': 0.0}

        # Restore run state
        self.active_runs[run_id] = {
            'config': config,
            'start_time': start_time,
            'last_checkpoint': time.time(),
            'errors': metrics.get('errors', 0),
            'downtime_seconds': metrics.get('downtime_seconds', 0.0)
        }

        # Restart resource monitoring
        monitor = ResourceMonitor()
        self.resource_monitors[run_id] = monitor
        await monitor.start_monitoring(config.resource_monitoring_interval_seconds)

        # Update status
        with sqlite3.connect(self.database_path) as conn:
            conn.execute("""
                UPDATE stability_runs
                SET status = 'running'
                WHERE run_id = ?
            """, (run_id,))

        self.logger.info(f"Resumed run {run_id} from {elapsed_hours:.2f} hours")

    async def stop_run(
        self,
        run_id: str,
        reason: str = "manual_stop"
    ) -> StabilityTestReport:
        """Stop stability run and generate report.

        Args:
            run_id: Stability run ID
            reason: Reason for stopping

        Returns:
            StabilityTestReport
        """
        if run_id not in self.active_runs:
            raise ValueError(f"Run {run_id} not found")

        run_state = self.active_runs[run_id]
        config = run_state['config']

        # Stop resource monitoring
        monitor = self.resource_monitors.get(run_id)
        if monitor:
            await monitor.stop_monitoring()

        # Calculate final metrics
        duration_hours = (time.time() - run_state['start_time']) / 3600.0
        total_errors = run_state['errors']
        downtime_seconds = run_state['downtime_seconds']
        uptime_percentage = 100.0 * (1.0 - downtime_seconds / (duration_hours * 3600.0))

        # Memory leak detection
        memory_leak_detected = False
        growth_rate = 0.0
        if monitor and config.memory_leak_detection_enabled:
            leak_report = monitor.detect_memory_leak(
                config.memory_leak_threshold_mb_per_hour
            )
            if leak_report:
                memory_leak_detected = leak_report.detected
                growth_rate = leak_report.growth_rate_mb_per_hour

                # Save leak report
                await self._save_memory_leak_report(run_id, leak_report)

        # Performance degradation (placeholder - would need actual measurements)
        performance_degradation = 0.0

        # Update run record
        with sqlite3.connect(self.database_path) as conn:
            conn.execute("""
                UPDATE stability_runs
                SET end_time = ?,
                    status = 'completed',
                    duration_hours = ?,
                    total_errors = ?,
                    uptime_percentage = ?,
                    performance_degradation_percent = ?,
                    memory_leak_detected = ?,
                    memory_growth_rate = ?,
                    stopped_reason = ?
                WHERE run_id = ?
            """, (
                time.time(),
                duration_hours,
                total_errors,
                uptime_percentage,
                performance_degradation,
                1 if memory_leak_detected else 0,
                growth_rate,
                reason,
                run_id
            ))

        # Save final resource samples
        if monitor:
            await self._save_resource_samples(run_id, monitor.get_metrics())

        # Generate report
        resource_metrics = monitor.get_metrics() if monitor else []
        recovery_incidents = []  # Would be populated from actual incidents

        report = self.metrics_collector.generate_stability_report(
            run_id,
            resource_metrics,
            recovery_incidents
        )

        # Cleanup
        del self.active_runs[run_id]
        if run_id in self.resource_monitors:
            del self.resource_monitors[run_id]

        self.logger.info(f"Stopped run {run_id}: {reason}")

        return report

    async def _save_resource_samples(
        self,
        run_id: str,
        metrics: list[ResourceMetrics]
    ) -> None:
        """Save resource samples to database.

        Args:
            run_id: Stability run ID
            metrics: List of resource metrics
        """
        with sqlite3.connect(self.database_path) as conn:
            for m in metrics:
                conn.execute("""
                    INSERT INTO resource_samples
                    (run_id, timestamp, component_name, cpu_percent,
                     memory_rss_mb, memory_vms_mb, memory_available_mb,
                     disk_read_mb, disk_write_mb, net_sent_mb, net_recv_mb,
                     open_fds)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    run_id, m.timestamp, m.component_name, m.cpu_percent,
                    m.memory_rss_mb, m.memory_vms_mb, m.memory_available_mb,
                    m.disk_read_mb, m.disk_write_mb, m.net_sent_mb,
                    m.net_recv_mb, m.open_fds
                ))

    async def _save_memory_leak_report(
        self,
        run_id: str,
        report: MemoryLeakReport
    ) -> None:
        """Save memory leak report to database.

        Args:
            run_id: Stability run ID
            report: Memory leak report
        """
        with sqlite3.connect(self.database_path) as conn:
            conn.execute("""
                INSERT INTO memory_leak_reports
                (run_id, detected, growth_rate_mb_per_hour, confidence,
                 affected_component, samples_analyzed, start_memory_mb,
                 end_memory_mb, duration_hours, report_time)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                run_id,
                1 if report.detected else 0,
                report.growth_rate_mb_per_hour,
                report.confidence,
                report.affected_component,
                report.samples_analyzed,
                report.start_memory_mb,
                report.end_memory_mb,
                report.duration_hours,
                time.time()
            ))


class MultiComponentCoordinator:
    """
    Coordinator for managing multiple components during stress testing.

    Handles component lifecycle, failure injection, health monitoring,
    and recovery time measurement for stress test scenarios.
    """

    def __init__(self, integration_coordinator: IntegrationTestCoordinator):
        """Initialize multi-component coordinator.

        Args:
            integration_coordinator: Integration test coordinator for component management
        """
        self.integration = integration_coordinator
        self.components: dict[str, ComponentInstance] = {}
        self.failure_timestamps: dict[str, float] = {}
        self.recovery_timestamps: dict[str, float] = {}
        self.logger = logging.getLogger(__name__)

    async def start_all_components(
        self,
        configs: list[ComponentStressConfig]
    ) -> dict[str, bool]:
        """Start all components for stress testing.

        Args:
            configs: List of component stress configurations

        Returns:
            Dictionary mapping component name to success status
        """
        results = {}

        for stress_config in configs:
            component_name = stress_config.component_name

            # Check if component is registered with integration coordinator
            if component_name not in self.integration._components:
                self.logger.error(f"Component {component_name} not registered")
                results[component_name] = False
                continue

            try:
                # Start component
                success = await self.integration.start_component(component_name)
                results[component_name] = success

                if success:
                    # Store component instance reference
                    self.components[component_name] = self.integration._components[component_name]
                    self.logger.info(f"Started component {component_name} for stress testing")
                else:
                    self.logger.error(f"Failed to start component {component_name}")

            except Exception as e:
                self.logger.error(f"Error starting component {component_name}: {e}")
                results[component_name] = False

        return results

    async def stop_component(
        self,
        component_name: str,
        failure_mode: str
    ) -> float:
        """Stop component with specific failure mode.

        Args:
            component_name: Name of component to stop
            failure_mode: Failure mode to simulate

        Returns:
            Timestamp when component was stopped
        """
        if component_name not in self.components:
            raise ValueError(f"Component {component_name} not found")

        instance = self.components[component_name]
        self.integration._get_controller(instance.config.component_type)

        failure_timestamp = time.time()
        self.failure_timestamps[component_name] = failure_timestamp

        try:
            if failure_mode == FailureMode.CRASH.value:
                # Hard kill (SIGKILL)
                if instance.process:
                    instance.process.kill()
                    self.logger.info(f"Injected CRASH failure for {component_name}")

            elif failure_mode == FailureMode.HANG.value:
                # Freeze process (SIGSTOP)
                if instance.process and instance.pid:
                    import os
                    os.kill(instance.pid, signal.SIGSTOP)
                    self.logger.info(f"Injected HANG failure for {component_name}")

            elif failure_mode == FailureMode.SLOW.value:
                # Simulate slowdown (in real implementation, could throttle CPU/IO)
                self.logger.info(f"Injected SLOW failure for {component_name}")
                # In production, this would use cgroups or resource limits

            elif failure_mode == FailureMode.NETWORK_PARTITION.value:
                # Simulate network partition (in real implementation, use iptables)
                self.logger.info(f"Injected NETWORK_PARTITION failure for {component_name}")
                # In production, this would use network namespace isolation

            instance.state = ComponentState.FAILED

        except Exception as e:
            self.logger.error(f"Error injecting failure for {component_name}: {e}")

        return failure_timestamp

    async def restart_component(self, component_name: str) -> float:
        """Restart component after failure and measure recovery time.

        Args:
            component_name: Name of component to restart

        Returns:
            Recovery time in seconds
        """
        if component_name not in self.components:
            raise ValueError(f"Component {component_name} not found")

        time.time()

        try:
            # Restart component
            success = await self.integration.start_component(component_name)

            if success:
                # Wait for component to become healthy
                recovery_time = await self._wait_for_health(component_name)

                recovery_timestamp = time.time()
                self.recovery_timestamps[component_name] = recovery_timestamp

                self.logger.info(
                    f"Component {component_name} recovered in {recovery_time:.2f}s"
                )

                return recovery_time
            else:
                self.logger.error(f"Failed to restart component {component_name}")
                return -1.0

        except Exception as e:
            self.logger.error(f"Error restarting component {component_name}: {e}")
            return -1.0

    async def _wait_for_health(
        self,
        component_name: str,
        timeout: float = 30.0
    ) -> float:
        """Wait for component to become healthy.

        Args:
            component_name: Name of component to monitor
            timeout: Maximum time to wait for health

        Returns:
            Time taken to become healthy
        """
        start_time = time.time()
        deadline = start_time + timeout

        while time.time() < deadline:
            is_healthy = await self.check_component_health(component_name)

            if is_healthy:
                return time.time() - start_time

            await asyncio.sleep(0.5)

        # Timeout
        return timeout

    async def check_component_health(self, component_name: str) -> bool:
        """Check if component is healthy.

        Args:
            component_name: Name of component to check

        Returns:
            True if component is healthy
        """
        if component_name not in self.components:
            return False

        instance = self.components[component_name]
        controller = self.integration._get_controller(instance.config.component_type)

        try:
            return await controller.health_check(instance)
        except Exception as e:
            self.logger.error(f"Error checking health for {component_name}: {e}")
            return False

    def get_all_component_statuses(self) -> dict[str, str]:
        """Get current status of all components.

        Returns:
            Dictionary mapping component name to state string
        """
        statuses = {}

        for name, instance in self.components.items():
            statuses[name] = instance.state.name

        return statuses


class StressTestOrchestrator(TestOrchestrator):
    """
    Specialized orchestrator for stress testing scenarios.

    Extends TestOrchestrator with stress-specific capabilities including
    load pattern simulation, failure injection, recovery measurement,
    resource exhaustion simulation, and performance degradation tracking.
    """

    def __init__(
        self,
        project_root: Path,
        test_directory: Path,
        config: StressTestConfig | None = None,
        database_path: Path | None = None
    ):
        """Initialize stress test orchestrator.

        Args:
            project_root: Root directory of the project
            test_directory: Directory containing test files
            config: Stress test configuration
            database_path: Path to orchestration database
        """
        # Initialize parent with stress config
        stress_config = config or StressTestConfig()
        super().__init__(project_root, test_directory, stress_config, database_path)

        # Stress-specific state
        self.stress_config = stress_config
        self.multi_coordinator: MultiComponentCoordinator | None = None
        self.baseline_metrics: dict[str, Any] = {}
        self.performance_samples: dict[str, list[float]] = defaultdict(list)

        # Initialize resource exhaustion simulator
        self.resource_simulator = ResourceExhaustionSimulator()

        # Initialize stability test manager
        self.stability_manager = StabilityTestManager(self.database_path)

        # Initialize stress-specific database tables
        self._init_stress_database()

    def _init_stress_database(self):
        """Initialize stress testing database tables."""
        with sqlite3.connect(self.database_path) as conn:
            # Stress test runs
            conn.execute("""
                CREATE TABLE IF NOT EXISTS stress_test_runs (
                    orchestration_id TEXT PRIMARY KEY,
                    load_pattern TEXT NOT NULL,
                    duration_hours REAL NOT NULL,
                    failure_injection_enabled INTEGER NOT NULL,
                    resource_exhaustion_enabled INTEGER NOT NULL,
                    avg_recovery_time REAL,
                    max_recovery_time REAL,
                    stability_violations INTEGER,
                    FOREIGN KEY (orchestration_id) REFERENCES orchestration_runs (id)
                )
            """)

            # Component recovery times
            conn.execute("""
                CREATE TABLE IF NOT EXISTS component_recovery_times (
                    orchestration_id TEXT NOT NULL,
                    component_name TEXT NOT NULL,
                    failure_mode TEXT NOT NULL,
                    failure_timestamp REAL NOT NULL,
                    recovery_timestamp REAL,
                    recovery_time REAL,
                    FOREIGN KEY (orchestration_id) REFERENCES orchestration_runs (id)
                )
            """)

            # Performance metrics
            conn.execute("""
                CREATE TABLE IF NOT EXISTS stress_performance_metrics (
                    orchestration_id TEXT NOT NULL,
                    component_name TEXT NOT NULL,
                    timestamp REAL NOT NULL,
                    metric_name TEXT NOT NULL,
                    metric_value REAL NOT NULL,
                    FOREIGN KEY (orchestration_id) REFERENCES orchestration_runs (id)
                )
            """)

    async def orchestrate_stress_test(
        self,
        components: list[ComponentStressConfig]
    ) -> StressTestResult:
        """Execute stress testing orchestration for multiple components.

        Args:
            components: List of component stress configurations

        Returns:
            Comprehensive stress test results
        """
        orchestration_id = f"stress_{int(time.time() * 1000)}"
        self._current_orchestration = orchestration_id

        result = StressTestResult(
            orchestration_id=orchestration_id,
            start_time=time.time()
        )

        try:
            # Initialize multi-component coordinator
            integration = self._get_component_integration()
            self.multi_coordinator = MultiComponentCoordinator(integration)

            # Register components with integration coordinator
            for stress_config in components:
                # Create ComponentConfig from stress config
                component_config = ComponentConfig(
                    name=stress_config.component_name,
                    component_type=integration._components.get(
                        stress_config.component_name,
                        ComponentConfig(name=stress_config.component_name,
                                      component_type=None)
                    ).config.component_type if stress_config.component_name in integration._components else None,
                    health_check_url=stress_config.health_check_endpoint
                )
                integration.register_component(component_config)

            # Execute stress test pipeline
            await self._execute_stress_pipeline(result, components)

        except Exception as e:
            result.status = PipelineStage.FAILED
            result.errors.append(f"Stress test failed: {str(e)}")
            self.logger.error(f"Stress test {orchestration_id} failed: {e}")

        finally:
            # Cleanup resource simulator
            self.resource_simulator.stop_all_simulations()

            result.end_time = time.time()
            self._save_stress_test_result(result)
            self._current_orchestration = None

        return result

    async def _execute_stress_pipeline(
        self,
        result: StressTestResult,
        components: list[ComponentStressConfig]
    ):
        """Execute stress testing pipeline stages.

        Args:
            result: Stress test result object to populate
            components: List of component configurations
        """
        # Baseline
        await self._stage_resource_baseline(result, components)

        # Load ramp
        if self.stress_config.load_pattern == LoadPattern.RAMP_UP:
            await self._stage_load_ramp(result, components)

        # Stress execution
        await self._stage_stress_execution(result, components)

        # Resource exhaustion
        if self.stress_config.resource_exhaustion_scenario is not None:
            await self._stage_resource_exhaustion(result, components)

        # Failure injection
        if self.stress_config.failure_injection_enabled:
            await self._stage_failure_injection(result, components)

        # Recovery validation
        if self.stress_config.failure_injection_enabled:
            await self._stage_recovery_validation(result, components)

        # Degradation analysis
        await self._stage_degradation_analysis(result, components)

        result.status = PipelineStage.COMPLETED

    async def _stage_resource_baseline(
        self,
        result: StressTestResult,
        components: list[ComponentStressConfig]
    ):
        """Capture baseline resource metrics before stress test.

        Args:
            result: Stress test result object
            components: List of component configurations
        """
        self.logger.info("Capturing resource baseline metrics")

        # Start components if not already started
        start_results = await self.multi_coordinator.start_all_components(components)

        # Capture baseline metrics
        for component_name, started in start_results.items():
            if started:
                is_healthy = await self.multi_coordinator.check_component_health(component_name)
                self.baseline_metrics[component_name] = {
                    "healthy": is_healthy,
                    "timestamp": time.time()
                }

        result.baseline_metrics = self.baseline_metrics
        self.logger.info(f"Baseline metrics captured for {len(self.baseline_metrics)} components")

    async def _stage_load_ramp(
        self,
        result: StressTestResult,
        components: list[ComponentStressConfig]
    ):
        """Gradually increase load during ramp-up period.

        Args:
            result: Stress test result object
            components: List of component configurations
        """
        self.logger.info("Executing load ramp stage")

        # Simulate gradual load increase
        ramp_duration = 60.0  # 1 minute ramp-up
        ramp_steps = 10
        step_duration = ramp_duration / ramp_steps

        for step in range(ramp_steps):
            load_percent = (step + 1) * (100 / ramp_steps)
            self.logger.info(f"Load ramp: {load_percent:.0f}%")

            # In production, this would adjust actual load generation
            await asyncio.sleep(step_duration)

    async def _stage_stress_execution(
        self,
        result: StressTestResult,
        components: list[ComponentStressConfig]
    ):
        """Execute full stress load for configured duration.

        Args:
            result: Stress test result object
            components: List of component configurations
        """
        self.logger.info(f"Executing stress test for {self.stress_config.duration_hours} hours")

        # In production, this would run actual stress workload
        # For testing, we simulate with shorter duration
        test_duration = min(self.stress_config.duration_hours * 3600, 5.0)  # Max 5 seconds for tests

        start_time = time.time()
        checkpoint_interval = self.stress_config.stability_checkpoints_minutes * 60
        last_checkpoint = start_time

        while time.time() - start_time < test_duration:
            # Sample performance metrics
            for component in components:
                # Simulate performance metric sampling
                sample_value = 100.0  # In production, measure actual latency/throughput
                self.performance_samples[component.component_name].append(sample_value)

            # Check stability checkpoint
            if time.time() - last_checkpoint >= checkpoint_interval:
                violations = await self._check_stability_checkpoint(components)
                result.stability_violations.extend(violations)
                last_checkpoint = time.time()

            await asyncio.sleep(0.5)

        result.performance_samples = dict(self.performance_samples)
        self.logger.info(f"Stress execution completed after {time.time() - start_time:.2f}s")

    async def _stage_resource_exhaustion(
        self,
        result: StressTestResult,
        components: list[ComponentStressConfig]
    ):
        """Simulate resource exhaustion during stress test.

        Args:
            result: Stress test result object
            components: List of component configurations
        """
        self.logger.info("Executing resource exhaustion stage")

        scenario = self.stress_config.resource_exhaustion_scenario

        try:
            # Capture pre-exhaustion metrics
            pre_metrics = {}
            for component in components:
                is_healthy = await self.multi_coordinator.check_component_health(
                    component.component_name
                )
                pre_metrics[component.component_name] = {
                    "healthy": is_healthy,
                    "timestamp": time.time()
                }

            # Execute resource exhaustion scenario
            await self.resource_simulator.execute_scenario(scenario)

            # Capture post-exhaustion metrics
            post_metrics = {}
            for component in components:
                is_healthy = await self.multi_coordinator.check_component_health(
                    component.component_name
                )
                post_metrics[component.component_name] = {
                    "healthy": is_healthy,
                    "timestamp": time.time()
                }

            # Store results
            result.resource_exhaustion_results = {
                "scenario": {
                    "memory_target_mb": scenario.memory_target_mb,
                    "cpu_target_percent": scenario.cpu_target_percent,
                    "disk_io_target_mb_per_sec": scenario.disk_io_target_mb_per_sec,
                    "network_target_mb_per_sec": scenario.network_target_mb_per_sec,
                    "duration_seconds": scenario.duration_seconds
                },
                "pre_metrics": pre_metrics,
                "post_metrics": post_metrics
            }

            # Check for stability violations
            for component_name, post in post_metrics.items():
                pre = pre_metrics.get(component_name, {})
                if pre.get("healthy", False) and not post.get("healthy", False):
                    result.stability_violations.append(
                        f"{component_name} became unhealthy during resource exhaustion"
                    )

        except Exception as e:
            self.logger.error(f"Error during resource exhaustion stage: {e}")
            result.errors.append(f"Resource exhaustion failed: {str(e)}")

        finally:
            # Ensure cleanup
            self.resource_simulator.stop_all_simulations()

        self.logger.info("Resource exhaustion stage completed")

    async def _check_stability_checkpoint(
        self,
        components: list[ComponentStressConfig]
    ) -> list[str]:
        """Check stability thresholds at checkpoint.

        Args:
            components: List of component configurations

        Returns:
            List of stability violations
        """
        violations = []

        for component in components:
            is_healthy = await self.multi_coordinator.check_component_health(
                component.component_name
            )

            if not is_healthy:
                violations.append(
                    f"{component.component_name} failed health check at checkpoint"
                )

        return violations

    async def _stage_failure_injection(
        self,
        result: StressTestResult,
        components: list[ComponentStressConfig]
    ):
        """Inject component failures during stress test.

        Args:
            result: Stress test result object
            components: List of component configurations
        """
        self.logger.info("Injecting component failures")

        for component in components:
            for failure_mode in component.failure_modes:
                try:
                    await self.inject_component_failure(
                        component.component_name,
                        failure_mode
                    )

                    result.failure_injections.append({
                        "component": component.component_name,
                        "failure_mode": failure_mode,
                        "timestamp": time.time()
                    })

                except Exception as e:
                    self.logger.error(
                        f"Error injecting failure for {component.component_name}: {e}"
                    )

    async def _stage_recovery_validation(
        self,
        result: StressTestResult,
        components: list[ComponentStressConfig]
    ):
        """Validate component recovery after failures.

        Args:
            result: Stress test result object
            components: List of component configurations
        """
        self.logger.info("Validating component recovery")

        for component in components:
            try:
                recovery_time = await self.measure_recovery_time(component.component_name)

                if recovery_time >= 0:
                    result.recovery_times[component.component_name] = recovery_time

                    if recovery_time > component.recovery_timeout_seconds:
                        result.warnings.append(
                            f"{component.component_name} recovery time "
                            f"({recovery_time:.2f}s) exceeded timeout "
                            f"({component.recovery_timeout_seconds}s)"
                        )

            except Exception as e:
                self.logger.error(
                    f"Error validating recovery for {component.component_name}: {e}"
                )

    async def _stage_degradation_analysis(
        self,
        result: StressTestResult,
        components: list[ComponentStressConfig]
    ):
        """Analyze performance degradation during stress test.

        Args:
            result: Stress test result object
            components: List of component configurations
        """
        self.logger.info("Analyzing performance degradation")

        degradation = self.track_performance_degradation()

        for component_name, samples in degradation.items():
            if len(samples) >= 2:
                initial = samples[0]
                final = samples[-1]
                degradation_percent = ((final - initial) / initial) * 100

                self.logger.info(
                    f"{component_name} performance degradation: {degradation_percent:.1f}%"
                )

                # Check against thresholds
                if degradation_percent > 50.0:  # 50% degradation threshold
                    result.warnings.append(
                        f"{component_name} experienced {degradation_percent:.1f}% "
                        f"performance degradation"
                    )

    async def inject_component_failure(
        self,
        component_name: str,
        failure_mode: str
    ) -> None:
        """Inject failure into component.

        Args:
            component_name: Name of component to affect
            failure_mode: Type of failure to inject
        """
        if not self.multi_coordinator:
            raise RuntimeError("Multi-component coordinator not initialized")

        self.logger.info(f"Injecting {failure_mode} failure into {component_name}")

        await self.multi_coordinator.stop_component(component_name, failure_mode)

    async def measure_recovery_time(self, component_name: str) -> float:
        """Measure time for component to recover from failure.

        Args:
            component_name: Name of component to measure

        Returns:
            Recovery time in seconds, or -1 if failed to recover
        """
        if not self.multi_coordinator:
            raise RuntimeError("Multi-component coordinator not initialized")

        self.logger.info(f"Measuring recovery time for {component_name}")

        recovery_time = await self.multi_coordinator.restart_component(component_name)

        return recovery_time

    def track_performance_degradation(self) -> dict[str, list[float]]:
        """Track performance degradation across components.

        Returns:
            Dictionary mapping component name to performance samples
        """
        return dict(self.performance_samples)

    def _save_stress_test_result(self, result: StressTestResult):
        """Save stress test result to database.

        Args:
            result: Stress test result to save
        """
        # Save base orchestration result
        self._save_orchestration_result(result)

        # Save stress-specific data
        with sqlite3.connect(self.database_path) as conn:
            conn.execute("""
                INSERT INTO stress_test_runs
                (orchestration_id, load_pattern, duration_hours,
                 failure_injection_enabled, resource_exhaustion_enabled,
                 avg_recovery_time, max_recovery_time, stability_violations)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                result.orchestration_id,
                self.stress_config.load_pattern.value,
                self.stress_config.duration_hours,
                1 if self.stress_config.failure_injection_enabled else 0,
                1 if self.stress_config.resource_exhaustion_scenario is not None else 0,
                result.avg_recovery_time,
                result.max_recovery_time,
                len(result.stability_violations)
            ))

            # Save recovery times
            for component_name, recovery_time in result.recovery_times.items():
                failure_timestamp = self.multi_coordinator.failure_timestamps.get(
                    component_name, 0.0
                )
                recovery_timestamp = self.multi_coordinator.recovery_timestamps.get(
                    component_name, 0.0
                )

                conn.execute("""
                    INSERT INTO component_recovery_times
                    (orchestration_id, component_name, failure_mode,
                     failure_timestamp, recovery_timestamp, recovery_time)
                    VALUES (?, ?, ?, ?, ?, ?)
                """, (
                    result.orchestration_id,
                    component_name,
                    "crash",  # TODO: Track actual failure mode
                    failure_timestamp,
                    recovery_timestamp,
                    recovery_time
                ))

            # Save performance metrics
            for component_name, samples in result.performance_samples.items():
                for i, sample in enumerate(samples):
                    conn.execute("""
                        INSERT INTO stress_performance_metrics
                        (orchestration_id, component_name, timestamp,
                         metric_name, metric_value)
                        VALUES (?, ?, ?, ?, ?)
                    """, (
                        result.orchestration_id,
                        component_name,
                        result.start_time + i * 0.5,  # Approximate timestamp
                        "latency_ms",
                        sample
                    ))
