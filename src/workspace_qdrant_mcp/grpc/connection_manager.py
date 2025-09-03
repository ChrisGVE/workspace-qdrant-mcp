"""
gRPC connection management with health monitoring and automatic recovery.

This module provides robust connection management for the gRPC client,
including connection pooling, health checks, and automatic reconnection.
"""

import asyncio
import logging
from contextlib import asynccontextmanager
from typing import AsyncIterator, Optional, Dict, Any
from datetime import datetime, timedelta

import grpc
import grpc.aio
from grpc import ChannelConnectivity

from .ingestion_pb2_grpc import IngestServiceStub
from google.protobuf.empty_pb2 import Empty

logger = logging.getLogger(__name__)


class ConnectionConfig:
    """Configuration for gRPC connection management."""
    
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
    ):
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
    
    @property
    def address(self) -> str:
        """Get the full gRPC address."""
        return f"{self.host}:{self.port}"
    
    def get_channel_options(self) -> list:
        """Get gRPC channel options based on configuration."""
        return [
            ('grpc.max_send_message_length', self.max_message_length),
            ('grpc.max_receive_message_length', self.max_message_length),
            ('grpc.keepalive_time_ms', self.keepalive_time * 1000),
            ('grpc.keepalive_timeout_ms', self.keepalive_timeout * 1000),
            ('grpc.keepalive_permit_without_calls', self.keepalive_without_calls),
            ('grpc.http2.max_pings_without_data', 0),
            ('grpc.http2.min_time_between_pings_ms', 10000),
            ('grpc.http2.min_ping_interval_without_data_ms', 300000),
            ('grpc.max_concurrent_streams', self.max_concurrent_streams),
        ]


class ConnectionState:
    """Tracks the state of a gRPC connection."""
    
    def __init__(self):
        self.channel: Optional[grpc.aio.Channel] = None
        self.stub: Optional[IngestServiceStub] = None
        self.last_health_check = datetime.now()
        self.consecutive_failures = 0
        self.is_healthy = False
        self.connection_time = None
        self.last_used = datetime.now()
        self._lock = asyncio.Lock()
    
    async def reset(self):
        """Reset connection state."""
        async with self._lock:
            if self.channel:
                await self.channel.close()
            self.channel = None
            self.stub = None
            self.is_healthy = False
            self.consecutive_failures = 0
            self.connection_time = None


class GrpcConnectionManager:
    """Manages gRPC connections with health monitoring and recovery."""
    
    def __init__(self, config: Optional[ConnectionConfig] = None):
        self.config = config or ConnectionConfig()
        self.state = ConnectionState()
        self._health_check_task: Optional[asyncio.Task] = None
        self._shutdown = False
        
        logger.info("gRPC connection manager initialized", 
                   address=self.config.address)
    
    async def start(self):
        """Start the connection manager and health monitoring."""
        if self._health_check_task is None:
            self._health_check_task = asyncio.create_task(self._health_check_loop())
            logger.info("Health check monitoring started",
                       interval=self.config.health_check_interval)
    
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
                    self.config.address,
                    options=options
                )
                
                # Wait for connection to be ready
                await asyncio.wait_for(
                    self.state.channel.channel_ready(),
                    timeout=self.config.connection_timeout
                )
                
                # Create stub
                self.state.stub = IngestServiceStub(self.state.channel)
                
                # Verify connection with health check
                await self._perform_health_check()
                
                self.state.connection_time = datetime.now()
                self.state.consecutive_failures = 0
                self.state.is_healthy = True
                
                logger.info("gRPC connection established successfully",
                           address=self.config.address)
                
            except Exception as e:
                logger.error("Failed to create gRPC connection",
                            address=self.config.address,
                            error=str(e))
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
                self.state.stub.HealthCheck(Empty()),
                timeout=5.0
            )
            
            # Check if service is healthy
            is_healthy = hasattr(response, 'status') and response.status == 1  # HEALTHY
            
            if is_healthy:
                self.state.consecutive_failures = 0
                self.state.is_healthy = True
                logger.debug("Health check passed")
            else:
                self.state.consecutive_failures += 1
                self.state.is_healthy = False
                logger.warning("Health check failed - service unhealthy",
                              consecutive_failures=self.state.consecutive_failures)
            
            self.state.last_health_check = datetime.now()
            return is_healthy
            
        except Exception as e:
            self.state.consecutive_failures += 1
            self.state.is_healthy = False
            logger.warning("Health check failed with exception",
                          error=str(e),
                          consecutive_failures=self.state.consecutive_failures)
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
                if (datetime.now() - self.state.last_used).total_seconds() > self.config.idle_timeout:
                    logger.info("Closing idle connection")
                    await self.state.reset()
                    continue
                
                # Perform health check
                await self._perform_health_check()
                
                # If too many consecutive failures, reset connection
                if self.state.consecutive_failures >= 3:
                    logger.warning("Too many health check failures, resetting connection")
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
                    logger.error("gRPC operation failed after retries",
                               attempt=attempt + 1,
                               error=str(e),
                               code=e.code())
                    raise
                
                # Wait before retry
                logger.warning("gRPC operation failed, retrying",
                              attempt=attempt + 1,
                              retry_delay=retry_delay,
                              error=str(e))
                await asyncio.sleep(retry_delay)
                
                # Exponential backoff
                retry_delay = min(
                    retry_delay * self.config.retry_backoff_multiplier,
                    self.config.max_retry_delay
                )
            
            except Exception as e:
                logger.error("Non-gRPC error in operation", error=str(e))
                raise
    
    def get_connection_info(self) -> Dict[str, Any]:
        """Get information about the current connection."""
        return {
            "address": self.config.address,
            "is_healthy": self.state.is_healthy,
            "consecutive_failures": self.state.consecutive_failures,
            "last_health_check": self.state.last_health_check.isoformat(),
            "connection_time": self.state.connection_time.isoformat() if self.state.connection_time else None,
            "last_used": self.state.last_used.isoformat(),
            "is_connected": self.state.channel is not None,
        }