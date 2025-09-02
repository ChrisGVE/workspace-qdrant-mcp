
"""
Health check system for workspace-qdrant-mcp.

Provides comprehensive health monitoring for all system components including
Qdrant connectivity, embedding services, file watchers, and system resources.
Designed for integration with monitoring systems and load balancers.

Health Check Categories:
    - Database connectivity (Qdrant cluster health)
    - Embedding service availability 
    - File system and watch health
    - Memory and resource utilization
    - Configuration validity
    - Background services status

Integration Features:
    - HTTP health endpoints for load balancer probes
    - Detailed diagnostic information for troubleshooting
    - Configurable health thresholds and alerting
    - Circuit breaker patterns for failing services

Example:
    ```python
    from workspace_qdrant_mcp.observability import health_checker_instance
    
    # Check overall system health
    health_status = await health_checker_instance.get_health_status()
    if health_status["status"] == "healthy":
        logger.info("System is operational")
    
    # Get detailed component diagnostics
    diagnostics = await health_checker_instance.get_detailed_diagnostics()
    ```
"""

import asyncio
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Any, Callable, Awaitable
import psutil
from pathlib import Path

from .logger import get_logger
from .metrics import metrics_instance

logger = get_logger(__name__)


class HealthStatus(Enum):
    """Health status levels for components."""
    HEALTHY = "healthy"
    DEGRADED = "degraded" 
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"


@dataclass
class HealthCheck:
    """Individual health check definition."""
    name: str
    check_function: Callable[[], Awaitable[Dict[str, Any]]]
    timeout_seconds: float = 5.0
    critical: bool = True  # If false, failure won't mark system as unhealthy
    enabled: bool = True
    last_check_time: Optional[float] = None
    last_result: Optional[Dict[str, Any]] = None
    consecutive_failures: int = 0
    max_failures: int = 3


@dataclass
class ComponentHealth:
    """Health status for a system component."""
    name: str
    status: HealthStatus
    message: str = ""
    details: Dict[str, Any] = field(default_factory=dict)
    last_check: Optional[float] = None
    response_time: Optional[float] = None
    error: Optional[str] = None


class HealthChecker:
    """Main health checking and monitoring system."""
    
    def __init__(self):
        self.health_checks: Dict[str, HealthCheck] = {}
        self._enabled = True
        self._background_task: Optional[asyncio.Task] = None
        self._check_interval = 30.0  # seconds
        self._system_thresholds = {
            "memory_usage_percent": 90.0,
            "cpu_usage_percent": 95.0,
            "disk_usage_percent": 95.0,
        }
        
        # Initialize standard health checks
        self._initialize_standard_checks()
        
        logger.info("HealthChecker initialized", checks_registered=len(self.health_checks))
    
    def _initialize_standard_checks(self):
        """Initialize standard system health checks."""
        # Register core health checks
        self.register_check("system_resources", self._check_system_resources)
        self.register_check("qdrant_connectivity", self._check_qdrant_connectivity)
        self.register_check("embedding_service", self._check_embedding_service)
        self.register_check("file_watchers", self._check_file_watchers)
        self.register_check("configuration", self._check_configuration)
        
        logger.debug("Standard health checks registered", 
                    checks=list(self.health_checks.keys()))
    
    def register_check(self, name: str, check_function: Callable[[], Awaitable[Dict[str, Any]]], 
                      timeout_seconds: float = 5.0, critical: bool = True):
        """Register a new health check.
        
        Args:
            name: Unique name for the health check
            check_function: Async function that returns health status dict
            timeout_seconds: Maximum time to wait for check completion
            critical: Whether failure of this check marks system as unhealthy
        """
        self.health_checks[name] = HealthCheck(
            name=name,
            check_function=check_function,
            timeout_seconds=timeout_seconds,
            critical=critical
        )
        
        logger.debug("Health check registered", 
                    check_name=name, 
                    critical=critical, 
                    timeout=timeout_seconds)
    
    def unregister_check(self, name: str):
        """Remove a health check."""
        if name in self.health_checks:
            del self.health_checks[name]
            logger.debug("Health check unregistered", check_name=name)
    
    async def run_check(self, name: str) -> ComponentHealth:
        """Run a specific health check.
        
        Args:
            name: Name of the health check to run
            
        Returns:
            ComponentHealth with check results
        """
        if name not in self.health_checks:
            return ComponentHealth(
                name=name,
                status=HealthStatus.UNKNOWN,
                message="Health check not found",
                error="Check not registered"
            )
        
        check = self.health_checks[name]
        if not check.enabled:
            return ComponentHealth(
                name=name,
                status=HealthStatus.UNKNOWN,
                message="Health check disabled"
            )
        
        start_time = time.perf_counter()
        check_time = time.time()
        
        try:
            # Run check with timeout
            result = await asyncio.wait_for(
                check.check_function(), 
                timeout=check.timeout_seconds
            )
            
            response_time = time.perf_counter() - start_time
            
            # Update check tracking
            check.last_check_time = check_time
            check.last_result = result
            check.consecutive_failures = 0
            
            # Parse result
            status = HealthStatus(result.get("status", "unknown"))
            message = result.get("message", "")
            details = result.get("details", {})
            
            # Record metrics
            metrics_instance.increment_counter("health_checks_total", 
                                             check=name, status=status.value)
            metrics_instance.record_histogram("health_check_duration_seconds", 
                                            response_time, check=name)
            
            return ComponentHealth(
                name=name,
                status=status,
                message=message,
                details=details,
                last_check=check_time,
                response_time=response_time
            )
            
        except asyncio.TimeoutError:
            check.consecutive_failures += 1
            error_msg = f"Health check timed out after {check.timeout_seconds}s"
            
            logger.warning("Health check timeout", 
                          check_name=name, 
                          timeout_seconds=check.timeout_seconds,
                          consecutive_failures=check.consecutive_failures)
            
            metrics_instance.increment_counter("health_checks_total", 
                                             check=name, status="timeout")
            
            return ComponentHealth(
                name=name,
                status=HealthStatus.UNHEALTHY,
                message="Check timed out",
                error=error_msg,
                last_check=check_time,
                response_time=check.timeout_seconds
            )
            
        except Exception as e:
            check.consecutive_failures += 1
            error_msg = str(e)
            
            logger.error("Health check failed", 
                        check_name=name, 
                        error=error_msg,
                        consecutive_failures=check.consecutive_failures,
                        exc_info=True)
            
            metrics_instance.increment_counter("health_checks_total", 
                                             check=name, status="error")
            
            return ComponentHealth(
                name=name,
                status=HealthStatus.UNHEALTHY,
                message="Check failed with exception",
                error=error_msg,
                last_check=check_time
            )
    
    async def get_health_status(self) -> Dict[str, Any]:
        """Get overall system health status.
        
        Returns:
            Dict containing overall status and component details
        """
        if not self._enabled:
            return {
                "status": "unknown",
                "message": "Health checking disabled",
                "timestamp": time.time(),
                "components": {}
            }
        
        # Run all health checks concurrently
        check_tasks = {
            name: self.run_check(name) 
            for name in self.health_checks.keys()
            if self.health_checks[name].enabled
        }
        
        component_results = await asyncio.gather(
            *check_tasks.values(),
            return_exceptions=True
        )
        
        components = {}
        overall_status = HealthStatus.HEALTHY
        critical_failures = []
        
        # Process results
        for (name, task), result in zip(check_tasks.items(), component_results):
            if isinstance(result, Exception):
                logger.error("Health check task failed", 
                           check_name=name, 
                           error=str(result))
                components[name] = ComponentHealth(
                    name=name,
                    status=HealthStatus.UNHEALTHY,
                    message="Task execution failed",
                    error=str(result)
                )
                result = components[name]
            else:
                components[name] = result
            
            # Determine overall status
            check = self.health_checks[name]
            if check.critical and result.status == HealthStatus.UNHEALTHY:
                overall_status = HealthStatus.UNHEALTHY
                critical_failures.append(name)
            elif result.status == HealthStatus.DEGRADED and overall_status == HealthStatus.HEALTHY:
                overall_status = HealthStatus.DEGRADED
        
        # Generate overall message
        if overall_status == HealthStatus.HEALTHY:
            message = "All systems operational"
        elif overall_status == HealthStatus.DEGRADED:
            degraded_checks = [name for name, comp in components.items() 
                             if comp.status == HealthStatus.DEGRADED]
            message = f"System degraded: {', '.join(degraded_checks)}"
        else:
            message = f"System unhealthy: {', '.join(critical_failures)}"
        
        # Record overall health metrics
        metrics_instance.set_gauge("system_health_status", 
                                 1 if overall_status == HealthStatus.HEALTHY else 0)
        
        return {
            "status": overall_status.value,
            "message": message,
            "timestamp": time.time(),
            "components": {name: {
                "status": comp.status.value,
                "message": comp.message,
                "details": comp.details,
                "last_check": comp.last_check,
                "response_time": comp.response_time,
                "error": comp.error
            } for name, comp in components.items()}
        }
    
    async def get_detailed_diagnostics(self) -> Dict[str, Any]:
        """Get detailed system diagnostics for troubleshooting.
        
        Returns:
            Comprehensive diagnostic information
        """
        health_status = await self.get_health_status()
        
        # Add additional diagnostic information
        diagnostics = {
            "health_status": health_status,
            "system_info": await self._get_system_info(),
            "check_history": self._get_check_history(),
            "configuration": self._get_health_configuration(),
            "metrics_summary": metrics_instance.get_metrics_summary()
        }
        
        return diagnostics
    
    async def _get_system_info(self) -> Dict[str, Any]:
        """Get system information for diagnostics."""
        try:
            process = psutil.Process()
            
            return {
                "process_id": process.pid,
                "memory_info": process.memory_info()._asdict(),
                "cpu_percent": process.cpu_percent(),
                "create_time": process.create_time(),
                "num_threads": process.num_threads(),
                "status": process.status(),
                "system_cpu_count": psutil.cpu_count(),
                "system_memory": psutil.virtual_memory()._asdict(),
                "system_disk": psutil.disk_usage("/")._asdict() if Path("/").exists() else None
            }
        except Exception as e:
            logger.warning("Failed to get system info", error=str(e))
            return {"error": str(e)}
    
    def _get_check_history(self) -> Dict[str, Any]:
        """Get health check history and statistics."""
        history = {}
        
        for name, check in self.health_checks.items():
            history[name] = {
                "enabled": check.enabled,
                "critical": check.critical,
                "timeout_seconds": check.timeout_seconds,
                "last_check_time": check.last_check_time,
                "consecutive_failures": check.consecutive_failures,
                "max_failures": check.max_failures,
                "last_result": check.last_result
            }
        
        return history
    
    def _get_health_configuration(self) -> Dict[str, Any]:
        """Get current health checker configuration."""
        return {
            "enabled": self._enabled,
            "check_interval": self._check_interval,
            "registered_checks": list(self.health_checks.keys()),
            "system_thresholds": self._system_thresholds,
            "background_monitoring": self._background_task is not None
        }
    
    # Standard health check implementations
    
    async def _check_system_resources(self) -> Dict[str, Any]:
        """Check system resource utilization."""
        try:
            # Memory check
            memory = psutil.virtual_memory()
            memory_percent = memory.percent
            
            # CPU check
            cpu_percent = psutil.cpu_percent(interval=0.1)
            
            # Disk check
            disk_usage = psutil.disk_usage("/")
            disk_percent = (disk_usage.used / disk_usage.total) * 100
            
            # Determine status based on thresholds
            status = HealthStatus.HEALTHY
            issues = []
            
            if memory_percent > self._system_thresholds["memory_usage_percent"]:
                status = HealthStatus.UNHEALTHY
                issues.append(f"High memory usage: {memory_percent:.1f}%")
            elif memory_percent > self._system_thresholds["memory_usage_percent"] * 0.8:
                status = HealthStatus.DEGRADED
                issues.append(f"Elevated memory usage: {memory_percent:.1f}%")
            
            if cpu_percent > self._system_thresholds["cpu_usage_percent"]:
                status = HealthStatus.UNHEALTHY
                issues.append(f"High CPU usage: {cpu_percent:.1f}%")
            elif cpu_percent > self._system_thresholds["cpu_usage_percent"] * 0.8:
                if status == HealthStatus.HEALTHY:
                    status = HealthStatus.DEGRADED
                issues.append(f"Elevated CPU usage: {cpu_percent:.1f}%")
            
            if disk_percent > self._system_thresholds["disk_usage_percent"]:
                status = HealthStatus.UNHEALTHY
                issues.append(f"High disk usage: {disk_percent:.1f}%")
            elif disk_percent > self._system_thresholds["disk_usage_percent"] * 0.8:
                if status == HealthStatus.HEALTHY:
                    status = HealthStatus.DEGRADED
                issues.append(f"Elevated disk usage: {disk_percent:.1f}%")
            
            message = "System resources OK" if not issues else "; ".join(issues)
            
            return {
                "status": status.value,
                "message": message,
                "details": {
                    "memory": {
                        "percent_used": memory_percent,
                        "available_gb": memory.available / (1024**3),
                        "total_gb": memory.total / (1024**3)
                    },
                    "cpu": {
                        "percent_used": cpu_percent,
                        "core_count": psutil.cpu_count()
                    },
                    "disk": {
                        "percent_used": disk_percent,
                        "free_gb": disk_usage.free / (1024**3),
                        "total_gb": disk_usage.total / (1024**3)
                    }
                }
            }
            
        except Exception as e:
            return {
                "status": HealthStatus.UNHEALTHY.value,
                "message": f"Failed to check system resources: {e}",
                "details": {"error": str(e)}
            }
    
    async def _check_qdrant_connectivity(self) -> Dict[str, Any]:
        """Check Qdrant database connectivity and health."""
        try:
            # Import here to avoid circular dependency
            from ..core.client import QdrantWorkspaceClient
            from ..server import workspace_client
            
            if not workspace_client:
                return {
                    "status": HealthStatus.UNHEALTHY.value,
                    "message": "Workspace client not initialized",
                    "details": {}
                }
            
            # Test basic connectivity
            status = await workspace_client.get_status()
            
            if not status.get("connected", False):
                return {
                    "status": HealthStatus.UNHEALTHY.value,
                    "message": "Qdrant connection failed",
                    "details": status
                }
            
            # Check collection health
            collections = status.get("workspace_collections", [])
            collection_count = len(collections)
            
            return {
                "status": HealthStatus.HEALTHY.value,
                "message": f"Qdrant healthy with {collection_count} collections",
                "details": {
                    "qdrant_url": status.get("qdrant_url"),
                    "collections_count": collection_count,
                    "project": status.get("current_project"),
                    "embedding_model": status.get("embedding_info", {}).get("model_name")
                }
            }
            
        except Exception as e:
            return {
                "status": HealthStatus.UNHEALTHY.value,
                "message": f"Qdrant health check failed: {e}",
                "details": {"error": str(e)}
            }
    
    async def _check_embedding_service(self) -> Dict[str, Any]:
        """Check embedding service availability."""
        try:
            from ..server import workspace_client
            
            if not workspace_client:
                return {
                    "status": HealthStatus.UNHEALTHY.value,
                    "message": "Workspace client not initialized",
                    "details": {}
                }
            
            # Test embedding generation with a simple query
            embedding_service = workspace_client.get_embedding_service()
            test_embeddings = await embedding_service.generate_embeddings(
                "health check test", include_sparse=False
            )
            
            if not test_embeddings or "dense" not in test_embeddings:
                return {
                    "status": HealthStatus.UNHEALTHY.value,
                    "message": "Embedding generation failed",
                    "details": {}
                }
            
            dense_dim = len(test_embeddings["dense"])
            
            return {
                "status": HealthStatus.HEALTHY.value,
                "message": "Embedding service operational",
                "details": {
                    "model_name": embedding_service.model_name,
                    "dense_dimensions": dense_dim,
                    "sparse_enabled": test_embeddings.get("sparse") is not None
                }
            }
            
        except Exception as e:
            return {
                "status": HealthStatus.UNHEALTHY.value,
                "message": f"Embedding service check failed: {e}",
                "details": {"error": str(e)}
            }
    
    async def _check_file_watchers(self) -> Dict[str, Any]:
        """Check file watcher system health."""
        try:
            from ..server import watch_tools_manager
            
            if not watch_tools_manager:
                return {
                    "status": HealthStatus.DEGRADED.value,
                    "message": "Watch tools manager not initialized",
                    "details": {}
                }
            
            # Get watch status
            watch_list = await watch_tools_manager.list_watched_folders(
                active_only=False, 
                include_stats=True
            )
            
            if not watch_list.get("success", False):
                return {
                    "status": HealthStatus.DEGRADED.value,
                    "message": "Failed to get watch status",
                    "details": watch_list
                }
            
            summary = watch_list.get("summary", {})
            total_watches = summary.get("total_watches", 0)
            active_watches = summary.get("active_watches", 0)
            error_watches = summary.get("error_watches", 0)
            
            if error_watches > 0:
                status = HealthStatus.DEGRADED
                message = f"{error_watches} of {total_watches} watches have errors"
            elif active_watches == 0 and total_watches > 0:
                status = HealthStatus.DEGRADED
                message = f"No active watches ({total_watches} configured)"
            else:
                status = HealthStatus.HEALTHY
                message = f"File watchers OK ({active_watches} active)"
            
            return {
                "status": status.value,
                "message": message,
                "details": {
                    "total_watches": total_watches,
                    "active_watches": active_watches,
                    "error_watches": error_watches,
                    "paused_watches": summary.get("paused_watches", 0)
                }
            }
            
        except Exception as e:
            return {
                "status": HealthStatus.DEGRADED.value,
                "message": f"File watcher check failed: {e}",
                "details": {"error": str(e)}
            }
    
    async def _check_configuration(self) -> Dict[str, Any]:
        """Check system configuration validity."""
        try:
            from ..core.config import Config
            from ..utils.config_validator import ConfigValidator
            
            config = Config()
            validator = ConfigValidator(config)
            is_valid, results = validator.validate_all()
            
            if not is_valid:
                return {
                    "status": HealthStatus.UNHEALTHY.value,
                    "message": f"Configuration invalid: {len(results['issues'])} issues",
                    "details": {
                        "issues": results["issues"],
                        "warnings": results["warnings"]
                    }
                }
            
            status = HealthStatus.DEGRADED if results["warnings"] else HealthStatus.HEALTHY
            message = "Configuration valid"
            if results["warnings"]:
                message += f" with {len(results['warnings'])} warnings"
            
            return {
                "status": status.value,
                "message": message,
                "details": {
                    "qdrant_url": config.qdrant_url,
                    "embedding_model": config.embedding_model,
                    "warnings": results["warnings"]
                }
            }
            
        except Exception as e:
            return {
                "status": HealthStatus.UNHEALTHY.value,
                "message": f"Configuration check failed: {e}",
                "details": {"error": str(e)}
            }
    
    def start_background_monitoring(self, interval: float = 30.0):
        """Start background health monitoring task.
        
        Args:
            interval: Check interval in seconds
        """
        if self._background_task and not self._background_task.done():
            logger.warning("Background monitoring already running")
            return
        
        self._check_interval = interval
        self._background_task = asyncio.create_task(self._background_monitor())
        
        logger.info("Background health monitoring started", interval_seconds=interval)
    
    def stop_background_monitoring(self):
        """Stop background health monitoring task."""
        if self._background_task and not self._background_task.done():
            self._background_task.cancel()
            logger.info("Background health monitoring stopped")
    
    async def _background_monitor(self):
        """Background monitoring task."""
        try:
            while self._enabled:
                start_time = time.perf_counter()
                
                try:
                    health_status = await self.get_health_status()
                    
                    # Log health status changes
                    if health_status["status"] != "healthy":
                        logger.warning("System health degraded", 
                                     status=health_status["status"],
                                     message=health_status["message"])
                    else:
                        logger.debug("Health check completed", 
                                   status=health_status["status"])
                    
                    # Update system metrics
                    metrics_instance.update_system_metrics()
                    
                except Exception as e:
                    logger.error("Background health check failed", error=str(e))
                
                # Calculate next check time
                check_duration = time.perf_counter() - start_time
                sleep_time = max(0, self._check_interval - check_duration)
                
                await asyncio.sleep(sleep_time)
                
        except asyncio.CancelledError:
            logger.info("Background health monitoring cancelled")
        except Exception as e:
            logger.error("Background health monitoring failed", error=str(e), exc_info=True)


# Global health checker instance
health_checker_instance = HealthChecker()