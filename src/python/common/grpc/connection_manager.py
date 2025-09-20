"""
gRPC connection management with health monitoring and automatic recovery.

This module provides robust connection management for the gRPC client,
including connection pooling, health checks, and automatic reconnection.
"""

import asyncio
import logging
import time
from contextlib import asynccontextmanager
from datetime import datetime, timedelta
from typing import Any, AsyncIterator, Dict, Optional, List
from dataclasses import dataclass
import weakref
import threading

import grpc
import grpc.aio
from google.protobuf.empty_pb2 import Empty
from grpc import ChannelConnectivity

from .ingestion_pb2_grpc import IngestServiceStub

logger = logging.getLogger(__name__)


@dataclass
class ConnectionMetrics:
    """Metrics for connection monitoring."""
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    connection_errors: int = 0
    auth_failures: int = 0
    total_response_time: float = 0.0
    min_response_time: float = float('inf')
    max_response_time: float = 0.0
    connection_attempts: int = 0
    successful_connections: int = 0

    @property
    def average_response_time(self) -> float:
        """Calculate average response time in milliseconds."""
        if self.successful_requests == 0:
            return 0.0
        return self.total_response_time / self.successful_requests

    @property
    def success_rate(self) -> float:
        """Calculate success rate as percentage."""
        if self.total_requests == 0:
            return 0.0
        return (self.successful_requests / self.total_requests) * 100

    @property
    def connection_success_rate(self) -> float:
        """Calculate connection success rate as percentage."""
        if self.connection_attempts == 0:
            return 0.0
        return (self.successful_connections / self.connection_attempts) * 100

class ConnectionConfig:
    """Enhanced configuration for gRPC connection management."""

    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = 50051,
        max_message_length: int = 100 * 1024 * 1024,  # 100MB
        max_retries: int = 3,
        retry_backoff_multiplier: float = 1.5,
        initial_retry_delay: float = 1.0,
        max_retry_delay: float = 30.0,
        connection_timeout: float = 10.0,
        health_check_interval: float = 30.0,
        idle_timeout: float = 300.0,  # 5 minutes
        keepalive_time: int = 30,
        keepalive_timeout: int = 5,
        keepalive_without_calls: bool = True,
        max_concurrent_streams: int = 100,
        # Enhanced security options
        enable_tls: bool = False,
        tls_cert_path: Optional[str] = None,
        tls_key_path: Optional[str] = None,
        tls_ca_path: Optional[str] = None,
        api_key: Optional[str] = None,
        # Connection pooling options
        enable_connection_pooling: bool = True,
        pool_size: int = 5,
        max_pool_size: int = 10,
        pool_timeout: float = 30.0,
        # Circuit breaker options
        enable_circuit_breaker: bool = True,
        circuit_breaker_failure_threshold: int = 5,
        circuit_breaker_timeout: float = 60.0,
    ):
        # Basic connection settings
        self.host = host
        self.port = port
        self.max_message_length = max_message_length
        self.max_retries = max_retries
        self.retry_backoff_multiplier = retry_backoff_multiplier
        self.initial_retry_delay = initial_retry_delay
        self.max_retry_delay = max_retry_delay
        self.connection_timeout = connection_timeout
        self.health_check_interval = health_check_interval
        self.idle_timeout = idle_timeout
        self.keepalive_time = keepalive_time
        self.keepalive_timeout = keepalive_timeout
        self.keepalive_without_calls = keepalive_without_calls
        self.max_concurrent_streams = max_concurrent_streams

        # Enhanced security settings
        self.enable_tls = enable_tls
        self.tls_cert_path = tls_cert_path
        self.tls_key_path = tls_key_path
        self.tls_ca_path = tls_ca_path
        self.api_key = api_key

        # Connection pooling settings
        self.enable_connection_pooling = enable_connection_pooling
        self.pool_size = pool_size
        self.max_pool_size = max_pool_size
        self.pool_timeout = pool_timeout

        # Circuit breaker settings
        self.enable_circuit_breaker = enable_circuit_breaker
        self.circuit_breaker_failure_threshold = circuit_breaker_failure_threshold
        self.circuit_breaker_timeout = circuit_breaker_timeout

    @property
    def address(self) -> str:
        """Get the full gRPC address."""
        return f"{self.host}:{self.port}"

    def get_channel_options(self) -> list:
        """Get enhanced gRPC channel options based on configuration."""
        options = [
            ("grpc.max_send_message_length", self.max_message_length),
            ("grpc.max_receive_message_length", self.max_message_length),
            ("grpc.keepalive_time_ms", self.keepalive_time * 1000),
            ("grpc.keepalive_timeout_ms", self.keepalive_timeout * 1000),
            ("grpc.keepalive_permit_without_calls", self.keepalive_without_calls),
            ("grpc.http2.max_pings_without_data", 0),
            ("grpc.http2.min_time_between_pings_ms", 10000),
            ("grpc.http2.min_ping_interval_without_data_ms", 300000),
            ("grpc.max_concurrent_streams", self.max_concurrent_streams),
            # Performance optimizations
            ("grpc.so_reuseport", 1),
            ("grpc.tcp_user_timeout", self.connection_timeout * 1000),
            ("grpc.enable_http_proxy", 0),
        ]

        return options


class ConnectionState:
    """Enhanced state tracking for a gRPC connection."""

    def __init__(self):
        self.channel: Optional[grpc.aio.Channel] = None
        self.stub: Optional[IngestServiceStub] = None
        self.last_health_check = datetime.now()
        self.consecutive_failures = 0
        self.is_healthy = False
        self.connection_time = None
        self.last_used = datetime.now()
        self.metrics = ConnectionMetrics()
        self.is_in_use = False
        self.connection_id = None
        self._lock = asyncio.Lock()

        # Circuit breaker state
        self.circuit_breaker_state = "closed"  # closed, open, half-open
        self.circuit_breaker_failures = 0
        self.circuit_breaker_last_failure = None

    async def reset(self):
        """Reset connection state with enhanced cleanup."""
        async with self._lock:
            if self.channel:
                try:
                    await asyncio.wait_for(self.channel.close(), timeout=5.0)
                except asyncio.TimeoutError:
                    logger.warning("Channel close timed out")
                except Exception as e:
                    logger.warning(f"Error closing channel: {e}")

            self.channel = None
            self.stub = None
            self.is_healthy = False
            self.consecutive_failures = 0
            self.connection_time = None
            self.is_in_use = False
            self.connection_id = None
            # Don't reset metrics to preserve historical data

    def record_request(self, duration_ms: float, success: bool = True):
        """Record request metrics."""
        self.metrics.total_requests += 1
        if success:
            self.metrics.successful_requests += 1
            self.metrics.total_response_time += duration_ms
            self.metrics.min_response_time = min(self.metrics.min_response_time, duration_ms)
            self.metrics.max_response_time = max(self.metrics.max_response_time, duration_ms)
        else:
            self.metrics.failed_requests += 1

    def update_circuit_breaker(self, success: bool):
        """Update circuit breaker state based on operation result."""
        current_time = time.time()

        if success:
            self.circuit_breaker_failures = 0
            if self.circuit_breaker_state == "half-open":
                self.circuit_breaker_state = "closed"
        else:
            self.circuit_breaker_failures += 1
            self.circuit_breaker_last_failure = current_time

            if self.circuit_breaker_failures >= 5:  # threshold
                self.circuit_breaker_state = "open"

    def can_attempt_connection(self, circuit_breaker_timeout: float) -> bool:
        """Check if connection attempt is allowed by circuit breaker."""
        if self.circuit_breaker_state == "closed":
            return True
        elif self.circuit_breaker_state == "open":
            if (self.circuit_breaker_last_failure and
                time.time() - self.circuit_breaker_last_failure > circuit_breaker_timeout):
                self.circuit_breaker_state = "half-open"
                return True
            return False
        else:  # half-open
            return True


class GrpcConnectionManager:
    """Manages gRPC connections with health monitoring and recovery."""

    def __init__(self, config: Optional[ConnectionConfig] = None):
        self.config = config or ConnectionConfig()
        self.state = ConnectionState()
        self._health_check_task: Optional[asyncio.Task] = None
        self._shutdown = False

        logger.info("gRPC connection manager initialized", address=self.config.address)

    async def start(self):
        """Start the connection manager and health monitoring."""
        if self._health_check_task is None:
            self._health_check_task = asyncio.create_task(self._health_check_loop())
            logger.info(
                "Health check monitoring started",
                interval=self.config.health_check_interval,
            )

    async def stop(self):
        """Stop the connection manager and clean up resources."""
        logger.info("Stopping gRPC connection manager")
        self._shutdown = True

        if self._health_check_task:
            self._health_check_task.cancel()
            try:
                await self._health_check_task
            except asyncio.CancelledError:
                pass

        await self.state.reset()
        logger.info("gRPC connection manager stopped")

    async def ensure_connected(self) -> IngestServiceStub:
        """Ensure we have a healthy connection and return the stub."""
        if not self.state.is_healthy or self.state.stub is None:
            await self._create_connection()

        # Update last used time
        self.state.last_used = datetime.now()
        return self.state.stub

    async def _create_connection(self):
        """Create a new gRPC connection."""
        async with self.state._lock:
            try:
                # Close existing connection if any
                if self.state.channel:
                    await self.state.channel.close()

                # Create new channel with configuration
                options = self.config.get_channel_options()
                self.state.channel = grpc.aio.insecure_channel(
                    self.config.address, options=options
                )

                # Wait for connection to be ready
                await asyncio.wait_for(
                    self.state.channel.channel_ready(),
                    timeout=self.config.connection_timeout,
                )

                # Create stub
                self.state.stub = IngestServiceStub(self.state.channel)

                # Verify connection with health check
                await self._perform_health_check()

                self.state.connection_time = datetime.now()
                self.state.consecutive_failures = 0
                self.state.is_healthy = True

                logger.info(
                    "gRPC connection established successfully",
                    address=self.config.address,
                )

            except Exception as e:
                logger.error(
                    "Failed to create gRPC connection",
                    address=self.config.address,
                    error=str(e),
                )
                await self.state.reset()
                raise ConnectionError(f"Failed to connect to gRPC server: {e}") from e

    async def _perform_health_check(self) -> bool:
        """Perform health check on the current connection."""
        if not self.state.stub:
            return False

        try:
            from google.protobuf.empty_pb2 import Empty

            # Perform health check with timeout
            response = await asyncio.wait_for(
                self.state.stub.HealthCheck(Empty()), timeout=5.0
            )

            # Check if service is healthy
            is_healthy = hasattr(response, "status") and response.status == 1  # HEALTHY

            if is_healthy:
                self.state.consecutive_failures = 0
                self.state.is_healthy = True
                logger.debug("Health check passed")
            else:
                self.state.consecutive_failures += 1
                self.state.is_healthy = False
                logger.warning(
                    "Health check failed - service unhealthy",
                    consecutive_failures=self.state.consecutive_failures,
                )

            self.state.last_health_check = datetime.now()
            return is_healthy

        except Exception as e:
            self.state.consecutive_failures += 1
            self.state.is_healthy = False
            logger.warning(
                "Health check failed with exception",
                error=str(e),
                consecutive_failures=self.state.consecutive_failures,
            )
            return False

    async def _health_check_loop(self):
        """Background task for periodic health checks."""
        while not self._shutdown:
            try:
                await asyncio.sleep(self.config.health_check_interval)

                if self._shutdown:
                    break

                # Skip health check if connection is not established
                if not self.state.stub:
                    continue

                # Check if connection is idle and should be closed
                if (
                    datetime.now() - self.state.last_used
                ).total_seconds() > self.config.idle_timeout:
                    logger.info("Closing idle connection")
                    await self.state.reset()
                    continue

                # Perform health check
                await self._perform_health_check()

                # If too many consecutive failures, reset connection
                if self.state.consecutive_failures >= 3:
                    logger.warning(
                        "Too many health check failures, resetting connection"
                    )
                    await self.state.reset()

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("Error in health check loop", error=str(e))
                await asyncio.sleep(5)  # Wait before retrying

    @asynccontextmanager
    async def get_stub(self) -> AsyncIterator[IngestServiceStub]:
        """Context manager for getting a gRPC stub with automatic connection management."""
        stub = await self.ensure_connected()
        try:
            yield stub
        except grpc.RpcError as e:
            if e.code() in (grpc.StatusCode.UNAVAILABLE, grpc.StatusCode.CANCELLED):
                # Connection lost, reset state
                logger.warning("gRPC connection lost, resetting", error=str(e))
                await self.state.reset()
            raise

    async def with_retry(self, operation, *args, **kwargs):
        """Execute a gRPC operation with retry logic."""
        retry_delay = self.config.initial_retry_delay

        for attempt in range(self.config.max_retries + 1):
            try:
                async with self.get_stub() as stub:
                    return await operation(stub, *args, **kwargs)

            except grpc.RpcError as e:
                is_retryable = e.code() in (
                    grpc.StatusCode.UNAVAILABLE,
                    grpc.StatusCode.DEADLINE_EXCEEDED,
                    grpc.StatusCode.RESOURCE_EXHAUSTED,
                    grpc.StatusCode.ABORTED,
                )

                if attempt == self.config.max_retries or not is_retryable:
                    logger.error(
                        "gRPC operation failed after retries",
                        attempt=attempt + 1,
                        error=str(e),
                        code=e.code(),
                    )
                    raise

                # Wait before retry
                logger.warning(
                    "gRPC operation failed, retrying",
                    attempt=attempt + 1,
                    retry_delay=retry_delay,
                    error=str(e),
                )
                await asyncio.sleep(retry_delay)

                # Exponential backoff
                retry_delay = min(
                    retry_delay * self.config.retry_backoff_multiplier,
                    self.config.max_retry_delay,
                )

            except Exception as e:
                logger.error("Non-gRPC error in operation", error=str(e))
                raise

    async def with_stream(self, stream_func):
        """Execute a streaming gRPC operation with proper connection management.

        Args:
            stream_func: Async generator function that takes a stub and yields stream items

        Yields:
            Stream items from the gRPC streaming operation
        """
        last_exception = None

        for attempt in range(self.config.max_retries + 1):
            try:
                stub = await self.get_or_create_stub()
                async for item in stream_func(stub):
                    self.state.last_used = datetime.now()
                    yield item
                return  # Stream completed successfully

            except grpc.RpcError as e:
                last_exception = e
                logger.warning(
                    "Streaming gRPC operation failed",
                    attempt=attempt + 1,
                    max_retries=self.config.max_retries + 1,
                    status_code=e.code(),
                    details=e.details(),
                    error=str(e),
                )

                # For streaming operations, don't retry certain errors
                if e.code() in (
                    grpc.StatusCode.INVALID_ARGUMENT,
                    grpc.StatusCode.NOT_FOUND,
                    grpc.StatusCode.ALREADY_EXISTS,
                    grpc.StatusCode.PERMISSION_DENIED,
                    grpc.StatusCode.FAILED_PRECONDITION,
                    grpc.StatusCode.OUT_OF_RANGE,
                    grpc.StatusCode.UNIMPLEMENTED,
                ):
                    logger.error(
                        "Non-retryable gRPC error in streaming operation", error=str(e)
                    )
                    raise

                # Reset connection on failure
                await self.state.reset()

                if attempt < self.config.max_retries:
                    retry_delay = self.config.initial_retry_delay * (
                        self.config.retry_backoff_multiplier**attempt
                    )
                    retry_delay = min(retry_delay, self.config.max_retry_delay)

                    logger.info(
                        "Retrying streaming operation after delay",
                        retry_delay_seconds=retry_delay,
                        attempt=attempt + 1,
                        error=str(e),
                    )
                    await asyncio.sleep(retry_delay)
                else:
                    logger.error(
                        "Max retries exceeded for streaming operation",
                        max_retries=self.config.max_retries,
                        error=str(e),
                    )
                    raise

            except Exception as e:
                logger.error("Non-gRPC error in streaming operation", error=str(e))
                raise

        if last_exception:
            raise last_exception

    def get_connection_info(self) -> Dict[str, Any]:
        """Get comprehensive connection information and metrics."""
        if hasattr(self, '_connection_pool') and self.config.enable_connection_pooling:
            pool_info = {
                "pool_size": len(self._connection_pool),
                "max_pool_size": self.config.max_pool_size,
                "connections_in_use": sum(1 for conn in self._connection_pool if conn.is_in_use),
                "healthy_connections": sum(1 for conn in self._connection_pool if conn.is_healthy),
                "pool_metrics": {
                    "total_requests": sum(conn.metrics.total_requests for conn in self._connection_pool),
                    "avg_success_rate": sum(conn.metrics.success_rate for conn in self._connection_pool) / len(self._connection_pool) if self._connection_pool else 0,
                }
            }
        else:
            pool_info = {
                "pooling_enabled": False,
                "connection_healthy": self.state.is_healthy,
                "connection_metrics": {
                    "total_requests": self.state.metrics.total_requests,
                    "success_rate": self.state.metrics.success_rate,
                    "avg_response_time": self.state.metrics.average_response_time,
                }
            }

        return {
            "address": self.config.address,
            "tls_enabled": getattr(self.config, 'enable_tls', False),
            "api_key_configured": bool(getattr(self.config, 'api_key', None)),
            "circuit_breaker_enabled": getattr(self.config, 'enable_circuit_breaker', False),
            "total_metrics": {
                "total_requests": self._total_metrics.total_requests,
                "success_rate": self._total_metrics.success_rate,
                "avg_response_time": self._total_metrics.average_response_time,
                "connection_success_rate": self._total_metrics.connection_success_rate,
            },
            **pool_info
        }

    async def get_or_create_stub(self) -> IngestServiceStub:
        """Get or create a stub - alias for ensure_connected for backward compatibility."""
        return await self.ensure_connected()
