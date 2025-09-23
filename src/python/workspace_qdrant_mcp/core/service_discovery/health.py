"""
Health Checking and Monitoring

This module provides health checking capabilities for discovered services,
including HTTP health checks, process validation, and service status monitoring.
"""

import asyncio
import os
try:
    import aiohttp
    AIOHTTP_AVAILABLE = True
except ImportError:
    AIOHTTP_AVAILABLE = False
    import urllib.request
    import urllib.error
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Dict, List, Optional, Union, Callable
from loguru import logger

from .exceptions import HealthError
from .registry import ServiceInfo, ServiceStatus

# logger imported from loguru


class HealthStatus(Enum):
    """Health status enumeration"""
    HEALTHY = "healthy"
    UNHEALTHY = "unhealthy"
    UNREACHABLE = "unreachable"
    PROCESS_DEAD = "process_dead"
    CHECKING = "checking"
    UNKNOWN = "unknown"


@dataclass
class HealthCheckResult:
    """Health check result"""
    service_name: str
    status: HealthStatus
    response_time_ms: Optional[float] = None
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    metrics: Dict[str, str] = field(default_factory=dict)
    error_message: Optional[str] = None

    def is_healthy(self) -> bool:
        """Check if this health result indicates the service is healthy"""
        return self.status == HealthStatus.HEALTHY

    def is_reachable(self) -> bool:
        """Check if this health result indicates the service is reachable"""
        return self.status not in (HealthStatus.UNREACHABLE, HealthStatus.PROCESS_DEAD)

    def age_seconds(self) -> float:
        """Get age of this health check result in seconds"""
        try:
            timestamp_dt = datetime.fromisoformat(self.timestamp.replace('Z', '+00:00'))
            return (datetime.now(timezone.utc) - timestamp_dt).total_seconds()
        except (ValueError, AttributeError):
            return float('inf')


@dataclass
class HealthConfig:
    """Health checker configuration"""
    request_timeout: float = 10.0
    check_interval: float = 30.0
    max_failures: int = 3
    validate_process: bool = True
    custom_headers: Dict[str, str] = field(default_factory=dict)


class HealthChecker:
    """Health checker for service monitoring"""

    def __init__(self, config: Optional[HealthConfig] = None):
        """Initialize health checker"""
        self.config = config or HealthConfig()
        self.session: Optional[aiohttp.ClientSession] = None
        self.health_cache: Dict[str, HealthCheckResult] = {}
        self.failure_counters: Dict[str, int] = {}
        self.monitoring_tasks: Dict[str, asyncio.Task] = {}
        
        logger.info(f"Health checker initialized with timeout: {self.config.request_timeout}s")

    async def start(self) -> None:
        """Start the health checker"""
        # Create HTTP session with custom headers
        headers = aiohttp.ClientResponse.headers
        if self.config.custom_headers:
            headers.update(self.config.custom_headers)
        
        timeout = aiohttp.ClientTimeout(total=self.config.request_timeout)
        self.session = aiohttp.ClientSession(
            timeout=timeout,
            headers=self.config.custom_headers
        )
        
        logger.info("Health checker started")

    async def stop(self) -> None:
        """Stop the health checker"""
        # Stop all monitoring tasks
        for task in self.monitoring_tasks.values():
            task.cancel()
        
        if self.monitoring_tasks:
            await asyncio.gather(*self.monitoring_tasks.values(), return_exceptions=True)
        
        self.monitoring_tasks.clear()
        
        # Close HTTP session
        if self.session:
            await self.session.close()
            self.session = None
        
        logger.info("Health checker stopped")

    async def check_service_health(self, service_name: str, service_info: ServiceInfo) -> HealthCheckResult:
        """Perform health check on a service"""
        logger.debug(f"Checking health for service: {service_name}")
        
        start_time = time.time()
        metrics = {}
        status = HealthStatus.CHECKING
        error_message = None

        try:
            # Step 1: Validate process if enabled
            if self.config.validate_process:
                if not self._is_process_running(service_info.pid):
                    status = HealthStatus.PROCESS_DEAD
                    error_message = f"Process {service_info.pid} is not running"
                    
                    result = HealthCheckResult(
                        service_name=service_name,
                        status=status,
                        response_time_ms=(time.time() - start_time) * 1000,
                        metrics=metrics,
                        error_message=error_message
                    )
                    
                    await self._cache_health_result(result)
                    return result

            # Step 2: Perform HTTP health check
            health_url = f"http://{service_info.host}:{service_info.port}{service_info.health_endpoint}"
            
            if not self.session:
                await self.start()
            
            try:
                async with self.session.get(health_url) as response:
                    response_time = (time.time() - start_time) * 1000
                    metrics["response_time_ms"] = str(response_time)
                    metrics["status_code"] = str(response.status)
                    
                    if response.status == 200:
                        # Try to parse response body for additional metrics
                        try:
                            if response.content_type == 'application/json':
                                health_data = await response.json()
                                if isinstance(health_data, dict):
                                    for key, value in health_data.items():
                                        metrics[key] = str(value)
                        except Exception as e:
                            logger.debug(f"Could not parse health response: {e}")
                        
                        status = HealthStatus.HEALTHY
                        await self._reset_failure_counter(service_name)
                    else:
                        status = HealthStatus.UNHEALTHY
                        error_message = f"HTTP {response.status} response"
                        await self._increment_failure_counter(service_name)
                        
            except asyncio.TimeoutError:
                status = HealthStatus.UNREACHABLE
                error_message = "Health check timeout"
                await self._increment_failure_counter(service_name)
                
            except aiohttp.ClientError as e:
                status = HealthStatus.UNREACHABLE
                error_message = f"HTTP request failed: {e}"
                await self._increment_failure_counter(service_name)

        except Exception as e:
            status = HealthStatus.UNKNOWN
            error_message = f"Health check error: {e}"
            logger.error(f"Unexpected error in health check for {service_name}: {e}")

        result = HealthCheckResult(
            service_name=service_name,
            status=status,
            response_time_ms=(time.time() - start_time) * 1000,
            metrics=metrics,
            error_message=error_message
        )

        await self._cache_health_result(result)
        
        logger.debug(f"Health check completed for {service_name}: {status}")
        return result

    async def start_monitoring(self, services: Dict[str, ServiceInfo]) -> None:
        """Start continuous health monitoring for services"""
        logger.info(f"Starting health monitoring for {len(services)} services")
        
        for service_name, service_info in services.items():
            if service_name not in self.monitoring_tasks:
                task = asyncio.create_task(
                    self._monitor_service(service_name, service_info)
                )
                self.monitoring_tasks[service_name] = task

    async def stop_monitoring(self, service_name: str) -> None:
        """Stop monitoring a specific service"""
        if service_name in self.monitoring_tasks:
            self.monitoring_tasks[service_name].cancel()
            try:
                await self.monitoring_tasks[service_name]
            except asyncio.CancelledError:
                pass
            del self.monitoring_tasks[service_name]
            
        logger.debug(f"Stopped monitoring service: {service_name}")

    async def get_service_health(self, service_name: str) -> Optional[HealthCheckResult]:
        """Get cached health status for a service"""
        return self.health_cache.get(service_name)

    async def get_all_health_status(self) -> Dict[str, HealthCheckResult]:
        """Get health status for all monitored services"""
        return self.health_cache.copy()

    async def is_service_healthy(self, service_name: str) -> bool:
        """Check if a service is currently healthy"""
        result = await self.get_service_health(service_name)
        return result.is_healthy() if result else False

    async def get_failure_count(self, service_name: str) -> int:
        """Get failure count for a service"""
        return self.failure_counters.get(service_name, 0)

    async def has_exceeded_max_failures(self, service_name: str) -> bool:
        """Check if service has exceeded maximum failures"""
        return await self.get_failure_count(service_name) >= self.config.max_failures

    async def _monitor_service(self, service_name: str, service_info: ServiceInfo) -> None:
        """Monitor a single service continuously"""
        logger.debug(f"Starting continuous monitoring for: {service_name}")
        
        while True:
            try:
                await self.check_service_health(service_name, service_info)
                await asyncio.sleep(self.config.check_interval)
                
            except asyncio.CancelledError:
                logger.debug(f"Monitoring cancelled for: {service_name}")
                break
                
            except Exception as e:
                logger.error(f"Error monitoring service {service_name}: {e}")
                await asyncio.sleep(self.config.check_interval)

    async def _cache_health_result(self, result: HealthCheckResult) -> None:
        """Cache health check result"""
        self.health_cache[result.service_name] = result

    async def _increment_failure_counter(self, service_name: str) -> None:
        """Increment failure counter for a service"""
        self.failure_counters[service_name] = self.failure_counters.get(service_name, 0) + 1
        
        if self.failure_counters[service_name] >= self.config.max_failures:
            logger.warning(
                f"Service {service_name} has exceeded maximum failures "
                f"({self.failure_counters[service_name]}/{self.config.max_failures})"
            )

    async def _reset_failure_counter(self, service_name: str) -> None:
        """Reset failure counter for a service"""
        if service_name in self.failure_counters and self.failure_counters[service_name] > 0:
            logger.debug(f"Resetting failure counter for service: {service_name}")
            self.failure_counters[service_name] = 0

    @staticmethod
    def _is_process_running(pid: int) -> bool:
        """Check if a process is running by PID"""
        try:
            # Send signal 0 to check if process exists
            os.kill(pid, 0)
            return True
        except (OSError, ProcessLookupError):
            return False


def health_config_with_timeout(timeout_seconds: float) -> HealthConfig:
    """Helper function to create health config with custom timeout"""
    return HealthConfig(request_timeout=timeout_seconds)