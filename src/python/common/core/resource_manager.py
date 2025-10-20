"""
Resource Coordination and Isolation System for Multi-Instance Daemons.

This module provides resource management and coordination between multiple daemon
instances including memory limits, CPU throttling, shared resource access, and
monitoring with alerting when limits are approached. Enhanced with performance
monitoring integration for comprehensive resource optimization.
"""

import asyncio
import gc
import json
import logging
from collections import deque
from loguru import logger
import os
import psutil
import resource
import signal
import tempfile
import threading
import time
from contextlib import asynccontextmanager
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional, Set, Any, Callable, Deque
from datetime import datetime, timedelta

# logger imported from loguru


@dataclass
class ResourceLimits:
    """Resource limits for a daemon instance."""
    
    # Memory limits (in MB)
    max_memory_mb: int = 512
    memory_warning_threshold: float = 0.8  # 80% of max
    memory_critical_threshold: float = 0.95  # 95% of max
    
    # CPU limits
    max_cpu_percent: float = 50.0  # Max CPU usage percentage
    cpu_warning_threshold: float = 0.8  # 80% of max
    cpu_critical_threshold: float = 0.95  # 95% of max
    
    # File descriptor limits
    max_open_files: int = 1024
    fd_warning_threshold: float = 0.8
    
    # Connection limits
    max_grpc_connections: int = 100
    max_qdrant_connections: int = 10
    
    # Process limits
    max_child_processes: int = 4
    
    # Timeout limits (in seconds)
    processing_timeout: float = 300.0  # 5 minutes
    connection_timeout: float = 30.0


@dataclass
class ResourceUsage:
    """Current resource usage for a daemon instance."""
    
    timestamp: datetime
    project_id: str
    pid: int
    
    # Memory usage
    memory_mb: float = 0.0
    memory_percent: float = 0.0
    
    # CPU usage
    cpu_percent: float = 0.0
    
    # File descriptors
    open_files: int = 0
    
    # Network connections
    active_connections: int = 0
    grpc_connections: int = 0
    qdrant_connections: int = 0
    
    # Process information
    child_processes: int = 0
    thread_count: int = 0
    
    # Status flags
    is_healthy: bool = True
    warnings: List[str] = None
    errors: List[str] = None
    
    def __post_init__(self):
        if self.warnings is None:
            self.warnings = []
        if self.errors is None:
            self.errors = []


@dataclass
class ResourceAlert:
    """Resource usage alert."""
    
    timestamp: datetime
    project_id: str
    alert_type: str  # warning, critical, error
    resource_type: str  # memory, cpu, connections, etc.
    current_value: float
    threshold_value: float
    message: str


class SharedResourcePool:
    """
    Manages shared resources across multiple daemon instances.
    
    Coordinates access to shared resources like Qdrant connections,
    embedding models, and other expensive-to-create resources.
    """
    
    def __init__(self):
        self._qdrant_connections: Dict[str, Any] = {}
        self._embedding_models: Dict[str, Any] = {}
        self._connection_locks: Dict[str, asyncio.Lock] = {}
        self._usage_counts: Dict[str, int] = {}
        self._cleanup_tasks: Set[asyncio.Task] = set()
        self._lock = asyncio.Lock()
    
    async def get_qdrant_connection(self, url: str, timeout: float = 30.0) -> Any:
        """Get or create a shared Qdrant connection."""
        async with self._lock:
            if url not in self._qdrant_connections:
                logger.info(f"Creating new shared Qdrant connection: {url}")
                # Import here to avoid circular dependencies
                from qdrant_client import AsyncQdrantClient
                from .ssl_config import suppress_qdrant_ssl_warnings
                
                with suppress_qdrant_ssl_warnings():
                    client = AsyncQdrantClient(url=url, timeout=timeout)
                self._qdrant_connections[url] = client
                self._connection_locks[url] = asyncio.Lock()
                self._usage_counts[url] = 0
            
            self._usage_counts[url] += 1
            return self._qdrant_connections[url]
    
    async def release_qdrant_connection(self, url: str) -> None:
        """Release a shared Qdrant connection."""
        async with self._lock:
            if url in self._usage_counts:
                self._usage_counts[url] -= 1
                
                # Cleanup connection if no longer used
                if self._usage_counts[url] <= 0:
                    logger.info(f"Cleaning up unused Qdrant connection: {url}")
                    if url in self._qdrant_connections:
                        try:
                            await self._qdrant_connections[url].close()
                        except Exception as e:
                            logger.warning(f"Error closing Qdrant connection: {e}")
                        finally:
                            del self._qdrant_connections[url]
                            del self._connection_locks[url]
                            del self._usage_counts[url]
    
    async def get_embedding_model(self, model_name: str) -> Any:
        """Get or create a shared embedding model."""
        async with self._lock:
            if model_name not in self._embedding_models:
                logger.info(f"Loading shared embedding model: {model_name}")
                # Import here to avoid circular dependencies
                try:
                    from fastembed import TextEmbedding
                    model = TextEmbedding(model_name=model_name)
                    self._embedding_models[model_name] = model
                    # Initialize usage count for reference tracking
                    if f"model_{model_name}" not in self._usage_counts:
                        self._usage_counts[f"model_{model_name}"] = 0
                except ImportError:
                    logger.warning("FastEmbed not available, using fallback")
                    self._embedding_models[model_name] = None

            # Track usage for reference counting
            if f"model_{model_name}" in self._usage_counts:
                self._usage_counts[f"model_{model_name}"] += 1

            return self._embedding_models[model_name]

    async def release_embedding_model(self, model_name: str) -> None:
        """Release a shared embedding model."""
        async with self._lock:
            usage_key = f"model_{model_name}"
            if usage_key in self._usage_counts:
                self._usage_counts[usage_key] -= 1

                # Cleanup model if no longer used
                if self._usage_counts[usage_key] <= 0:
                    logger.info(f"Cleaning up unused embedding model: {model_name}")
                    if model_name in self._embedding_models:
                        model = self._embedding_models[model_name]
                        if model is not None:
                            try:
                                # Attempt cleanup if model has close method
                                if hasattr(model, 'close'):
                                    await model.close()
                            except Exception as e:
                                logger.warning(f"Error closing embedding model: {e}")
                            finally:
                                del self._embedding_models[model_name]
                                del self._usage_counts[usage_key]
    
    async def cleanup_all(self) -> None:
        """Clean up all shared resources."""
        async with self._lock:
            # Close Qdrant connections
            for url, client in self._qdrant_connections.items():
                try:
                    await client.close()
                except Exception as e:
                    logger.warning(f"Error closing Qdrant connection {url}: {e}")

            # Clean up embedding models (fix memory leak)
            for model_name, model in self._embedding_models.items():
                if model is not None:
                    try:
                        # FastEmbed models may have cleanup methods
                        if hasattr(model, 'close'):
                            await model.close()
                        elif hasattr(model, '__del__'):
                            # Trigger cleanup through garbage collection
                            del model
                    except Exception as e:
                        logger.warning(f"Error cleaning up embedding model {model_name}: {e}")

            # Clear all resources
            self._qdrant_connections.clear()
            self._embedding_models.clear()
            self._connection_locks.clear()
            self._usage_counts.clear()

            # Cancel cleanup tasks
            for task in self._cleanup_tasks:
                if not task.done():
                    task.cancel()
            self._cleanup_tasks.clear()


class ResourceMonitor:
    """
    Monitors resource usage for daemon instances.
    
    Tracks memory, CPU, connections, and other resources with
    configurable thresholds and alerting.
    """
    
    def __init__(self, project_id: str, limits: ResourceLimits):
        self.project_id = project_id
        self.limits = limits
        self.pid = os.getpid()
        self.process = psutil.Process(self.pid)

        self._monitoring = False
        self._monitor_task: Optional[asyncio.Task] = None
        self._alert_callbacks: List[Callable[[ResourceAlert], None]] = []
        # Fix memory leak: Use deque with maxlen for O(1) operations instead of list.pop(0)
        self._usage_history: Deque[ResourceUsage] = deque(maxlen=1000)
        self._max_history = 1000  # Keep last 1000 measurements
    
    def add_alert_callback(self, callback: Callable[[ResourceAlert], None]) -> None:
        """Add a callback to be called when alerts are generated."""
        self._alert_callbacks.append(callback)
    
    async def start_monitoring(self, interval: float = 60.0) -> None:
        """
        Start resource monitoring.

        Default interval optimized for low idle CPU usage (60s).
        For more responsive monitoring during active development, use a lower interval (10-30s).
        """
        if self._monitoring:
            return

        self._monitoring = True
        self._monitor_task = asyncio.create_task(self._monitoring_loop(interval))
        logger.info(f"Started resource monitoring for project {self.project_id} with {interval}s interval")
    
    async def stop_monitoring(self) -> None:
        """Stop resource monitoring."""
        self._monitoring = False
        if self._monitor_task:
            self._monitor_task.cancel()
            try:
                await self._monitor_task
            except asyncio.CancelledError:
                pass
        logger.info(f"Stopped resource monitoring for project {self.project_id}")
    
    async def get_current_usage(self) -> ResourceUsage:
        """Get current resource usage."""
        try:
            # Memory usage
            memory_info = self.process.memory_info()
            memory_mb = memory_info.rss / (1024 * 1024)  # Convert to MB
            memory_percent = (memory_mb / self.limits.max_memory_mb) * 100
            
            # CPU usage
            cpu_percent = self.process.cpu_percent()
            
            # File descriptors
            try:
                open_files = self.process.num_fds()
            except (AttributeError, psutil.AccessDenied):
                open_files = 0
            
            # Network connections
            try:
                connections = self.process.connections()
                active_connections = len(connections)
                grpc_connections = len([c for c in connections if c.laddr.port >= 50000])
            except (psutil.AccessDenied, psutil.NoSuchProcess):
                active_connections = 0
                grpc_connections = 0
            
            # Process information
            try:
                child_processes = len(self.process.children())
                thread_count = self.process.num_threads()
            except (psutil.AccessDenied, psutil.NoSuchProcess):
                child_processes = 0
                thread_count = 0
            
            usage = ResourceUsage(
                timestamp=datetime.now(),
                project_id=self.project_id,
                pid=self.pid,
                memory_mb=memory_mb,
                memory_percent=memory_percent,
                cpu_percent=cpu_percent,
                open_files=open_files,
                active_connections=active_connections,
                grpc_connections=grpc_connections,
                child_processes=child_processes,
                thread_count=thread_count
            )
            
            # Check for alerts
            await self._check_thresholds(usage)
            
            return usage
            
        except (psutil.NoSuchProcess, psutil.AccessDenied) as e:
            logger.warning(f"Error getting resource usage: {e}")
            return ResourceUsage(
                timestamp=datetime.now(),
                project_id=self.project_id,
                pid=self.pid,
                is_healthy=False,
                errors=[str(e)]
            )
    
    async def _monitoring_loop(self, interval: float) -> None:
        """Main monitoring loop."""
        try:
            while self._monitoring:
                usage = await self.get_current_usage()

                # Add to history (deque with maxlen automatically handles overflow - O(1))
                self._usage_history.append(usage)

                await asyncio.sleep(interval)
                
        except asyncio.CancelledError:
            logger.debug("Resource monitoring loop cancelled")
        except Exception as e:
            logger.error(f"Error in resource monitoring loop: {e}")
    
    async def _check_thresholds(self, usage: ResourceUsage) -> None:
        """Check resource usage against thresholds and generate alerts."""
        alerts = []
        
        # Memory checks
        if usage.memory_percent >= self.limits.memory_critical_threshold * 100:
            alerts.append(ResourceAlert(
                timestamp=usage.timestamp,
                project_id=self.project_id,
                alert_type="critical",
                resource_type="memory",
                current_value=usage.memory_percent,
                threshold_value=self.limits.memory_critical_threshold * 100,
                message=f"Memory usage {usage.memory_percent:.1f}% exceeds critical threshold"
            ))
        elif usage.memory_percent >= self.limits.memory_warning_threshold * 100:
            alerts.append(ResourceAlert(
                timestamp=usage.timestamp,
                project_id=self.project_id,
                alert_type="warning",
                resource_type="memory",
                current_value=usage.memory_percent,
                threshold_value=self.limits.memory_warning_threshold * 100,
                message=f"Memory usage {usage.memory_percent:.1f}% exceeds warning threshold"
            ))
        
        # CPU checks
        cpu_threshold_percent = self.limits.max_cpu_percent
        if usage.cpu_percent >= cpu_threshold_percent * self.limits.cpu_critical_threshold:
            alerts.append(ResourceAlert(
                timestamp=usage.timestamp,
                project_id=self.project_id,
                alert_type="critical",
                resource_type="cpu",
                current_value=usage.cpu_percent,
                threshold_value=cpu_threshold_percent * self.limits.cpu_critical_threshold,
                message=f"CPU usage {usage.cpu_percent:.1f}% exceeds critical threshold"
            ))
        elif usage.cpu_percent >= cpu_threshold_percent * self.limits.cpu_warning_threshold:
            alerts.append(ResourceAlert(
                timestamp=usage.timestamp,
                project_id=self.project_id,
                alert_type="warning",
                resource_type="cpu",
                current_value=usage.cpu_percent,
                threshold_value=cpu_threshold_percent * self.limits.cpu_warning_threshold,
                message=f"CPU usage {usage.cpu_percent:.1f}% exceeds warning threshold"
            ))
        
        # File descriptor checks
        fd_percent = (usage.open_files / self.limits.max_open_files) * 100
        if fd_percent >= self.limits.fd_warning_threshold * 100:
            alerts.append(ResourceAlert(
                timestamp=usage.timestamp,
                project_id=self.project_id,
                alert_type="warning",
                resource_type="file_descriptors",
                current_value=fd_percent,
                threshold_value=self.limits.fd_warning_threshold * 100,
                message=f"File descriptor usage {fd_percent:.1f}% exceeds threshold"
            ))
        
        # Process alerts
        for alert in alerts:
            for callback in self._alert_callbacks:
                try:
                    callback(alert)
                except Exception as e:
                    logger.error(f"Error in alert callback: {e}")
    
    def get_usage_history(self, limit: Optional[int] = None) -> List[ResourceUsage]:
        """Get resource usage history."""
        if limit:
            return list(self._usage_history)[-limit:]
        return list(self._usage_history)

    def calculate_memory_growth_rate(self, time_window_minutes: int = 30) -> float:
        """
        Calculate memory growth rate in MB/min over time window.

        Args:
            time_window_minutes: Time window to analyze (default: 30 minutes)

        Returns:
            Memory growth rate in MB/min, or 0.0 if insufficient data
        """
        if len(self._usage_history) < 2:
            return 0.0

        # Get samples from time window
        now = datetime.now()
        cutoff = now - timedelta(minutes=time_window_minutes)
        samples = [u for u in self._usage_history if u.timestamp >= cutoff]

        if len(samples) < 2:
            return 0.0

        # Calculate linear regression for memory growth
        oldest = samples[0]
        newest = samples[-1]
        time_diff = (newest.timestamp - oldest.timestamp).total_seconds() / 60.0  # minutes
        memory_diff = newest.memory_mb - oldest.memory_mb

        if time_diff == 0:
            return 0.0

        return memory_diff / time_diff  # MB/min

    def is_memory_leak_detected(
        self,
        threshold_mb_per_min: float = 5.0,
        time_window_minutes: int = 30
    ) -> bool:
        """
        Detect if memory leak is occurring based on growth rate.

        Args:
            threshold_mb_per_min: Maximum acceptable growth rate (default: 5.0 MB/min)
            time_window_minutes: Time window to analyze (default: 30 minutes)

        Returns:
            True if memory leak detected, False otherwise
        """
        growth_rate = self.calculate_memory_growth_rate(time_window_minutes)
        return growth_rate > threshold_mb_per_min


class ResourceManager:
    """
    Main resource coordination system for multi-instance daemons.
    
    Manages resource limits, monitoring, shared resources, and cleanup
    across multiple daemon instances with project isolation.
    """
    
    def __init__(self):
        self.monitors: Dict[str, ResourceMonitor] = {}
        self.shared_pool = SharedResourcePool()
        self.project_limits: Dict[str, ResourceLimits] = {}
        self.alert_history: List[ResourceAlert] = []
        self.cleanup_handlers: List[Callable[[], None]] = []

        # Alert management (fix memory leak from unbounded growth)
        self._max_alert_history = 500  # Maximum alerts to keep
        self._alert_expiry = timedelta(hours=1)  # Expire after 1 hour
        self._enforcement_tasks: Set[asyncio.Task] = set()  # Track enforcement tasks

        # Memory leak detection and automatic cleanup
        self._leak_detection_running = False
        self._leak_detection_task: Optional[asyncio.Task] = None
        self._leak_detection_interval = 300.0  # Check every 5 minutes
        self._leak_threshold_mb_per_min = 5.0  # Alert threshold
        self._gc_interval = 600.0  # Run gc.collect() every 10 minutes
        self._gc_task: Optional[asyncio.Task] = None

        # Registry file for persistence
        temp_dir = Path(tempfile.gettempdir())
        self.registry_file = temp_dir / "wqm_resource_registry.json"

        # Global resource tracking
        self._global_memory_limit_mb = 2048  # 2GB total
        self._global_cpu_limit_percent = 80.0  # 80% total CPU

        self._lock = asyncio.Lock()
    
    async def register_project(
        self, 
        project_id: str, 
        limits: Optional[ResourceLimits] = None
    ) -> ResourceMonitor:
        """Register a project for resource monitoring."""
        async with self._lock:
            if project_id in self.monitors:
                return self.monitors[project_id]
            
            # Use provided limits or defaults
            project_limits = limits or ResourceLimits()
            
            # Adjust limits based on number of active projects
            active_projects = len(self.monitors)
            if active_projects > 0:
                # Distribute resources among projects
                memory_per_project = self._global_memory_limit_mb // (active_projects + 1)
                cpu_per_project = self._global_cpu_limit_percent / (active_projects + 1)
                
                project_limits.max_memory_mb = min(project_limits.max_memory_mb, memory_per_project)
                project_limits.max_cpu_percent = min(project_limits.max_cpu_percent, cpu_per_project)
            
            # Create monitor
            monitor = ResourceMonitor(project_id, project_limits)
            monitor.add_alert_callback(self._handle_alert)
            
            self.monitors[project_id] = monitor
            self.project_limits[project_id] = project_limits
            
            # Start monitoring
            await monitor.start_monitoring()

            # Start memory leak detection if not already running
            if not self._leak_detection_running:
                await self.start_memory_leak_detection()

            logger.info(f"Registered project {project_id} for resource monitoring")
            await self._save_registry()

            return monitor
    
    async def unregister_project(self, project_id: str) -> None:
        """Unregister a project from resource monitoring."""
        async with self._lock:
            if project_id in self.monitors:
                monitor = self.monitors[project_id]
                await monitor.stop_monitoring()

                del self.monitors[project_id]
                del self.project_limits[project_id]

                # Stop leak detection if no more projects
                if not self.monitors and self._leak_detection_running:
                    await self.stop_memory_leak_detection()

                logger.info(f"Unregistered project {project_id} from resource monitoring")
                await self._save_registry()

    async def start_memory_leak_detection(self) -> None:
        """Start automatic memory leak detection and cleanup."""
        if self._leak_detection_running:
            return

        self._leak_detection_running = True
        self._leak_detection_task = asyncio.create_task(
            self._memory_leak_detection_loop()
        )
        self._gc_task = asyncio.create_task(
            self._periodic_gc_loop()
        )
        logger.info(
            f"Started memory leak detection (interval: {self._leak_detection_interval}s, "
            f"threshold: {self._leak_threshold_mb_per_min} MB/min)"
        )

    async def stop_memory_leak_detection(self) -> None:
        """Stop automatic memory leak detection and cleanup."""
        self._leak_detection_running = False

        if self._leak_detection_task:
            self._leak_detection_task.cancel()
            try:
                await self._leak_detection_task
            except asyncio.CancelledError:
                pass
            self._leak_detection_task = None

        if self._gc_task:
            self._gc_task.cancel()
            try:
                await self._gc_task
            except asyncio.CancelledError:
                pass
            self._gc_task = None

        logger.info("Stopped memory leak detection")

    async def _memory_leak_detection_loop(self) -> None:
        """Monitor for memory leaks and trigger cleanup."""
        try:
            while self._leak_detection_running:
                await asyncio.sleep(self._leak_detection_interval)

                for project_id, monitor in list(self.monitors.items()):
                    try:
                        # Check for memory leak
                        if monitor.is_memory_leak_detected(
                            threshold_mb_per_min=self._leak_threshold_mb_per_min
                        ):
                            growth_rate = monitor.calculate_memory_growth_rate()
                            logger.warning(
                                f"Memory leak detected for project {project_id}: "
                                f"{growth_rate:.2f} MB/min (threshold: {self._leak_threshold_mb_per_min} MB/min)"
                            )
                            await self._trigger_cleanup(project_id, growth_rate)
                    except Exception as e:
                        logger.error(f"Error checking memory leak for {project_id}: {e}")

        except asyncio.CancelledError:
            logger.debug("Memory leak detection loop cancelled")
        except Exception as e:
            logger.error(f"Error in memory leak detection loop: {e}")

    async def _periodic_gc_loop(self) -> None:
        """Run periodic garbage collection."""
        try:
            while self._leak_detection_running:
                await asyncio.sleep(self._gc_interval)

                # Run garbage collection
                import gc
                collected = gc.collect()
                logger.debug(f"Periodic garbage collection freed {collected} objects")

        except asyncio.CancelledError:
            logger.debug("Periodic GC loop cancelled")
        except Exception as e:
            logger.error(f"Error in periodic GC loop: {e}")

    async def _trigger_cleanup(self, project_id: str, growth_rate: float) -> None:
        """
        Trigger cleanup for a project with memory leak.

        Args:
            project_id: Project to cleanup
            growth_rate: Current memory growth rate in MB/min
        """
        logger.info(f"Triggering cleanup for project {project_id} (growth: {growth_rate:.2f} MB/min)")

        # Run aggressive garbage collection
        import gc
        gc.collect()
        gc.collect()  # Run twice for cyclic references
        collected = gc.collect()
        logger.info(f"Aggressive garbage collection freed {collected} objects")

        # Cleanup idle shared resources
        await self._cleanup_idle_shared_resources()

        # Enforce resource limits
        await self.enforce_limits(project_id)

        # Log alert
        alert = ResourceAlert(
            timestamp=datetime.now(),
            project_id=project_id,
            alert_type="critical",
            resource_type="memory_leak",
            current_value=growth_rate,
            threshold_value=self._leak_threshold_mb_per_min,
            message=f"Automatic cleanup triggered for memory leak ({growth_rate:.2f} MB/min)"
        )
        self._handle_alert(alert)

    async def _cleanup_idle_shared_resources(self) -> None:
        """Cleanup shared resources with zero usage count."""
        async with self.shared_pool._lock:
            # Cleanup unused Qdrant connections
            idle_urls = [
                url for url, count in self.shared_pool._usage_counts.items()
                if count <= 0 and url in self.shared_pool._qdrant_connections
            ]
            for url in idle_urls:
                try:
                    logger.info(f"Cleaning up idle Qdrant connection: {url}")
                    await self.shared_pool._qdrant_connections[url].close()
                    del self.shared_pool._qdrant_connections[url]
                    del self.shared_pool._connection_locks[url]
                    del self.shared_pool._usage_counts[url]
                except Exception as e:
                    logger.warning(f"Error cleaning up idle connection {url}: {e}")

            # Cleanup unused embedding models
            idle_models = [
                model_name for usage_key, count in self.shared_pool._usage_counts.items()
                if usage_key.startswith("model_") and count <= 0
                for model_name in [usage_key.replace("model_", "")]
                if model_name in self.shared_pool._embedding_models
            ]
            for model_name in idle_models:
                try:
                    logger.info(f"Cleaning up idle embedding model: {model_name}")
                    model = self.shared_pool._embedding_models[model_name]
                    if model is not None and hasattr(model, 'close'):
                        await model.close()
                    del self.shared_pool._embedding_models[model_name]
                    del self.shared_pool._usage_counts[f"model_{model_name}"]
                except Exception as e:
                    logger.warning(f"Error cleaning up idle model {model_name}: {e}")
    
    async def get_project_usage(self, project_id: str) -> Optional[ResourceUsage]:
        """Get current resource usage for a project."""
        if project_id in self.monitors:
            return await self.monitors[project_id].get_current_usage()
        return None
    
    async def get_all_usage(self) -> Dict[str, ResourceUsage]:
        """Get current resource usage for all projects."""
        usage = {}
        for project_id, monitor in self.monitors.items():
            usage[project_id] = await monitor.get_current_usage()
        return usage
    
    async def enforce_limits(self, project_id: str) -> bool:
        """Enforce resource limits for a project."""
        if project_id not in self.monitors:
            return False
        
        usage = await self.get_project_usage(project_id)
        if not usage:
            return False
        
        limits = self.project_limits[project_id]
        actions_taken = False
        
        # Memory limit enforcement
        if usage.memory_mb > limits.max_memory_mb:
            logger.warning(f"Project {project_id} exceeds memory limit, requesting cleanup")
            # Trigger garbage collection
            import gc
            gc.collect()
            actions_taken = True
        
        # CPU limit enforcement (would require process groups in production)
        if usage.cpu_percent > limits.max_cpu_percent:
            logger.warning(f"Project {project_id} exceeds CPU limit")
            # In a real implementation, this would throttle the process
            actions_taken = True
        
        return actions_taken
    
    def _handle_alert(self, alert: ResourceAlert) -> None:
        """Handle resource alerts."""
        self.alert_history.append(alert)

        # Clean up expired alerts (fix memory leak: time-based expiration)
        now = datetime.now()
        self.alert_history = [
            a for a in self.alert_history
            if now - a.timestamp < self._alert_expiry
        ]

        # Limit by count (secondary safeguard)
        if len(self.alert_history) > self._max_alert_history:
            # Keep most recent alerts
            self.alert_history.sort(key=lambda a: a.timestamp, reverse=True)
            self.alert_history = self.alert_history[:self._max_alert_history]
            logger.warning(
                f"Trimmed alert history to {self._max_alert_history} most recent alerts"
            )

        # Log alert
        level = logging.WARNING if alert.alert_type == "warning" else logging.ERROR
        logger.log(level, f"Resource alert for {alert.project_id}: {alert.message}")

        # Trigger enforcement for critical alerts (fix memory leak: track task)
        if alert.alert_type == "critical":
            task = asyncio.create_task(self.enforce_limits(alert.project_id))
            self._enforcement_tasks.add(task)
            # Auto-remove task from set when done
            task.add_done_callback(lambda t: self._enforcement_tasks.discard(t))
    
    @asynccontextmanager
    async def shared_qdrant_connection(self, url: str):
        """Context manager for shared Qdrant connections."""
        connection = await self.shared_pool.get_qdrant_connection(url)
        try:
            yield connection
        finally:
            await self.shared_pool.release_qdrant_connection(url)
    
    async def get_shared_embedding_model(self, model_name: str):
        """Get shared embedding model."""
        return await self.shared_pool.get_embedding_model(model_name)
    
    def add_cleanup_handler(self, handler: Callable[[], None]) -> None:
        """Add a cleanup handler to be called on shutdown."""
        if handler not in self.cleanup_handlers:
            self.cleanup_handlers.append(handler)

    def remove_cleanup_handler(self, handler: Callable[[], None]) -> None:
        """Remove a cleanup handler."""
        try:
            self.cleanup_handlers.remove(handler)
        except ValueError:
            pass  # Handler not in list
    
    async def cleanup_all(self) -> None:
        """Clean up all resources and stop monitoring."""
        logger.info("Cleaning up resource manager")

        # Stop memory leak detection
        if self._leak_detection_running:
            await self.stop_memory_leak_detection()

        # Cancel all enforcement tasks (fix memory leak)
        for task in list(self._enforcement_tasks):
            if not task.done():
                task.cancel()
        if self._enforcement_tasks:
            await asyncio.gather(*self._enforcement_tasks, return_exceptions=True)
        self._enforcement_tasks.clear()

        # Stop all monitors
        for monitor in self.monitors.values():
            await monitor.stop_monitoring()

        # Cleanup shared resources
        await self.shared_pool.cleanup_all()

        # Run cleanup handlers
        for handler in self.cleanup_handlers:
            try:
                handler()
            except Exception as e:
                logger.error(f"Error in cleanup handler: {e}")

        # Clear state
        self.monitors.clear()
        self.project_limits.clear()
        self.cleanup_handlers.clear()
        self.alert_history.clear()
    
    async def _save_registry(self) -> None:
        """Save resource registry to file."""
        try:
            data = {
                "projects": list(self.monitors.keys()),
                "limits": {
                    project_id: asdict(limits)
                    for project_id, limits in self.project_limits.items()
                },
                "global_limits": {
                    "memory_mb": self._global_memory_limit_mb,
                    "cpu_percent": self._global_cpu_limit_percent
                },
                "timestamp": datetime.now().isoformat()
            }
            
            # Write atomically
            temp_file = self.registry_file.with_suffix('.tmp')
            with open(temp_file, 'w') as f:
                json.dump(data, f, indent=2)
            
            temp_file.replace(self.registry_file)
            
        except Exception as e:
            logger.warning(f"Failed to save resource registry: {e}")
    
    async def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status with performance monitoring integration."""
        all_usage = await self.get_all_usage()
        
        total_memory = sum(usage.memory_mb for usage in all_usage.values())
        total_cpu = sum(usage.cpu_percent for usage in all_usage.values())
        
        recent_alerts = [
            alert for alert in self.alert_history 
            if alert.timestamp > datetime.now() - timedelta(hours=1)
        ]
        
        # Get performance monitoring data
        performance_data = {}
        try:
            from .performance_monitor import get_all_performance_summaries
            performance_data = await get_all_performance_summaries()
        except Exception as e:
            logger.debug(f"Performance monitoring not available: {e}")
            performance_data = {}
        
        return {
            "total_projects": len(self.monitors),
            "total_memory_mb": total_memory,
            "total_cpu_percent": total_cpu,
            "global_memory_limit_mb": self._global_memory_limit_mb,
            "global_cpu_limit_percent": self._global_cpu_limit_percent,
            "memory_utilization": (total_memory / self._global_memory_limit_mb) * 100,
            "cpu_utilization": (total_cpu / self._global_cpu_limit_percent) * 100,
            "active_alerts": len(recent_alerts),
            "project_usage": {
                project_id: asdict(usage) for project_id, usage in all_usage.items()
            },
            "recent_alerts": [asdict(alert) for alert in recent_alerts[-10:]],
            "performance_monitoring": performance_data
        }
    
    async def get_performance_recommendations(self) -> Dict[str, Any]:
        """Get performance optimization recommendations for all projects."""
        try:
            from .performance_monitor import _performance_monitors
            
            recommendations = {}
            for project_id, monitor in _performance_monitors.items():
                try:
                    project_recommendations = await monitor.get_optimization_recommendations()
                    recommendations[project_id] = [
                        rec.to_dict() for rec in project_recommendations
                    ]
                except Exception as e:
                    logger.error(f"Failed to get recommendations for {project_id}: {e}")
                    recommendations[project_id] = {"error": str(e)}
            
            return recommendations
        except ImportError:
            return {"error": "Performance monitoring not available"}


# Global resource manager instance
_resource_manager: Optional[ResourceManager] = None


async def get_resource_manager() -> ResourceManager:
    """Get or create the global resource manager."""
    global _resource_manager
    if _resource_manager is None:
        _resource_manager = ResourceManager()
    return _resource_manager


async def register_project_resources(
    project_id: str, 
    limits: Optional[ResourceLimits] = None
) -> ResourceMonitor:
    """Convenience function to register project for resource monitoring."""
    manager = await get_resource_manager()
    return await manager.register_project(project_id, limits)


async def cleanup_project_resources(project_id: str) -> None:
    """Convenience function to cleanup project resources."""
    manager = await get_resource_manager()
    await manager.unregister_project(project_id)