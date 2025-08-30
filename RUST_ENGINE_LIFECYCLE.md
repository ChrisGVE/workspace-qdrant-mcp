# Rust Engine Lifecycle Management Design

**Version:** 2.0  
**Date:** 2025-08-30  
**Status:** Complete Design Specification

## Overview

This document defines the comprehensive lifecycle management system for the Rust ingestion engine within workspace-qdrant-mcp v2.0. The design ensures graceful startup, reliable operation, and clean shutdown while maintaining data integrity and user experience.

## Lifecycle States

### State Machine

```
┌─────────┐     start()     ┌──────────┐     complete     ┌─────────┐
│ STOPPED │ ─────────────► │ STARTING │ ─────────────► │ RUNNING │
└─────────┘                 └──────────┘                 └─────────┘
     ▲                           │                           │
     │                           │ error                     │
     │                           ▼                           │
     │                      ┌─────────┐                      │
     │              ┌─────► │  ERROR  │                      │
     │              │       └─────────┘                      │
     │              │            │                           │
     │              │            │ reset                     │
     │              │            ▼                           │
     │         ┌──────────┐  ┌─────────┐  stop()       ┌──────────┐
     └─────────│ STOPPING │◄─│ STOPPED │◄──────────────│ RUNNING  │
               └──────────┘  └─────────┘                └──────────┘
                    │                                        │
                    │ error                                  │
                    ▼                                        │
                ┌─────────┐                                  │
                │  ERROR  │                                  │
                └─────────┘                                  │
                                                            │
               ┌──────────────────────────────────────────────┘
               │
               ▼
          ┌──────────┐     force_shutdown()     ┌─────────┐
          │ STOPPING │ ───────────────────────► │ STOPPED │
          └──────────┘                          └─────────┘
```

### State Definitions

- **STOPPED**: Engine is not running, no resources allocated
- **STARTING**: Engine is initializing components and establishing connections
- **RUNNING**: Engine is fully operational and accepting requests
- **STOPPING**: Engine is shutting down gracefully, finishing active tasks
- **ERROR**: Engine encountered a critical error and requires intervention

## Startup Process

### Phase 1: Initialization

```rust
async fn initialize_engine(config: EngineConfig) -> Result<EngineComponents> {
    tracing::info!("Starting engine initialization");
    
    // 1. Validate configuration
    config.validate()?;
    
    // 2. Initialize metrics collector
    let metrics = MetricsCollector::new();
    metrics.start().await?;
    
    // 3. Initialize storage client
    let storage = StorageClient::connect(&config.qdrant_config).await
        .with_context("Failed to connect to Qdrant")?;
    
    // 4. Validate storage connectivity
    storage.health_check().await
        .with_context("Qdrant health check failed")?;
    
    // 5. Initialize embedding service
    let embedder = EmbeddingService::new(&config.embedding_config).await
        .with_context("Failed to initialize embedding service")?;
    
    // 6. Test embedding generation
    embedder.test_embedding("test").await
        .with_context("Embedding generation test failed")?;
    
    // 7. Initialize LSP manager
    let lsp_manager = LspManager::new().await?;
    lsp_manager.detect_available_servers().await?;
    
    tracing::info!("Engine initialization completed successfully");
    Ok(EngineComponents {
        metrics,
        storage,
        embedder,
        lsp_manager,
    })
}
```

### Phase 2: Service Startup

```rust
async fn start_services(components: EngineComponents) -> Result<RunningServices> {
    tracing::info!("Starting engine services");
    
    // 1. Start processing pipeline
    let pipeline = ProcessingPipeline::new(
        components.embedder,
        components.storage,
        components.lsp_manager,
        components.metrics,
    ).await?;
    
    pipeline.start().await?;
    
    // 2. Start file watching system
    let file_watcher = FileWatchingSystem::new(
        Arc::clone(&pipeline),
        Arc::clone(&components.metrics),
    ).await?;
    
    // 3. Start gRPC server
    let grpc_server = GrpcServer::new(
        Arc::clone(&pipeline),
        Arc::clone(&file_watcher),
        Arc::clone(&components.lsp_manager),
        Arc::clone(&components.metrics),
    );
    
    let server_handle = grpc_server.start().await?;
    
    tracing::info!("All services started successfully");
    Ok(RunningServices {
        pipeline,
        file_watcher,
        grpc_handle: server_handle,
    })
}
```

### Phase 3: Health Verification

```rust
async fn verify_engine_health(services: &RunningServices) -> Result<()> {
    tracing::info!("Performing engine health verification");
    
    // 1. Verify gRPC server is responding
    let health_check = services.grpc_handle.health_check().await?;
    if !health_check.is_healthy() {
        return Err(EngineError::HealthCheckFailed("gRPC server unhealthy".into()));
    }
    
    // 2. Verify processing pipeline is ready
    services.pipeline.ready_check().await?;
    
    // 3. Verify file watcher is active
    services.file_watcher.status_check().await?;
    
    // 4. Test end-to-end functionality
    let test_result = services.pipeline.test_processing().await?;
    if !test_result.success {
        return Err(EngineError::HealthCheckFailed("Processing test failed".into()));
    }
    
    tracing::info!("Engine health verification passed");
    Ok(())
}
```

## Graceful Shutdown Process

### Shutdown Orchestration

```rust
pub struct ShutdownOrchestrator {
    state: Arc<RwLock<EngineState>>,
    shutdown_timeout: Duration,
    force_shutdown_timeout: Duration,
}

impl ShutdownOrchestrator {
    pub async fn shutdown(
        &self,
        services: RunningServices,
        reason: ShutdownReason,
    ) -> Result<()> {
        tracing::info!("Starting graceful shutdown: {:?}", reason);
        
        // Update state
        {
            let mut state = self.state.write().await;
            *state = EngineState::Stopping;
        }
        
        // Phase 1: Stop accepting new work
        self.stop_accepting_requests(&services).await?;
        
        // Phase 2: Complete active work
        let completion_result = self.wait_for_completion(&services).await;
        
        // Phase 3: Shutdown services
        self.shutdown_services(services).await?;
        
        // Phase 4: Final cleanup
        self.final_cleanup().await?;
        
        // Update final state
        {
            let mut state = self.state.write().await;
            *state = match completion_result {
                Ok(()) => EngineState::Stopped,
                Err(e) => {
                    tracing::warn!("Shutdown completed with warnings: {}", e);
                    EngineState::Stopped
                }
            };
        }
        
        tracing::info!("Graceful shutdown completed");
        Ok(())
    }
    
    async fn stop_accepting_requests(&self, services: &RunningServices) -> Result<()> {
        tracing::info!("Stopping acceptance of new requests");
        
        // Stop gRPC server from accepting new connections
        services.grpc_handle.stop_accepting().await?;
        
        // Pause file watcher to prevent new file events
        services.file_watcher.pause().await?;
        
        // Stop processing pipeline from accepting new tasks
        services.pipeline.stop_accepting().await?;
        
        tracing::info!("New request acceptance stopped");
        Ok(())
    }
    
    async fn wait_for_completion(&self, services: &RunningServices) -> Result<()> {
        tracing::info!("Waiting for active tasks to complete");
        
        let start_time = Instant::now();
        let mut last_log_time = start_time;
        
        loop {
            let active_tasks = services.pipeline.active_task_count().await;
            let queued_tasks = services.pipeline.queued_task_count().await;
            
            if active_tasks == 0 && queued_tasks == 0 {
                tracing::info!("All tasks completed");
                break;
            }
            
            let elapsed = start_time.elapsed();
            if elapsed >= self.shutdown_timeout {
                tracing::warn!(
                    "Shutdown timeout reached with {} active and {} queued tasks",
                    active_tasks, queued_tasks
                );
                return Err(EngineError::ShutdownTimeout);
            }
            
            // Log progress every 5 seconds
            if last_log_time.elapsed() >= Duration::from_secs(5) {
                tracing::info!(
                    "Waiting for completion: {} active, {} queued tasks ({}s elapsed)",
                    active_tasks, queued_tasks, elapsed.as_secs()
                );
                last_log_time = Instant::now();
            }
            
            tokio::time::sleep(Duration::from_millis(500)).await;
        }
        
        Ok(())
    }
    
    async fn shutdown_services(&self, services: RunningServices) -> Result<()> {
        tracing::info!("Shutting down services");
        
        // Shutdown in reverse dependency order
        
        // 1. Stop gRPC server
        if let Err(e) = services.grpc_handle.shutdown().await {
            tracing::warn!("gRPC server shutdown error: {}", e);
        }
        
        // 2. Stop file watcher
        if let Err(e) = services.file_watcher.shutdown().await {
            tracing::warn!("File watcher shutdown error: {}", e);
        }
        
        // 3. Stop processing pipeline
        if let Err(e) = services.pipeline.shutdown().await {
            tracing::warn!("Processing pipeline shutdown error: {}", e);
        }
        
        tracing::info!("Service shutdown completed");
        Ok(())
    }
    
    async fn final_cleanup(&self) -> Result<()> {
        tracing::info!("Performing final cleanup");
        
        // Cleanup temporary files
        self.cleanup_temp_files().await?;
        
        // Flush logs
        tracing_appender::Reload::reload()?;
        
        tracing::info!("Final cleanup completed");
        Ok(())
    }
}
```

### Force Shutdown

```rust
impl ShutdownOrchestrator {
    pub async fn force_shutdown(&self, services: RunningServices) -> Result<()> {
        tracing::warn!("Force shutdown initiated");
        
        // Update state immediately
        {
            let mut state = self.state.write().await;
            *state = EngineState::Stopping;
        }
        
        // Force stop all services without waiting
        let shutdown_tasks = vec![
            tokio::spawn(async move { services.grpc_handle.force_shutdown().await }),
            tokio::spawn(async move { services.file_watcher.force_shutdown().await }),
            tokio::spawn(async move { services.pipeline.force_shutdown().await }),
        ];
        
        // Wait for force shutdown with timeout
        let timeout = tokio::time::timeout(
            self.force_shutdown_timeout,
            futures::future::join_all(shutdown_tasks)
        ).await;
        
        match timeout {
            Ok(results) => {
                for (i, result) in results.into_iter().enumerate() {
                    if let Err(e) = result {
                        tracing::error!("Force shutdown task {} failed: {}", i, e);
                    }
                }
            }
            Err(_) => {
                tracing::error!("Force shutdown timeout exceeded");
                // At this point, we can't do much more - the process should exit
            }
        }
        
        // Final state update
        {
            let mut state = self.state.write().await;
            *state = EngineState::Stopped;
        }
        
        tracing::warn!("Force shutdown completed");
        Ok(())
    }
}
```

## Python Integration Lifecycle

### MCP Server Integration

```python
# src/workspace_qdrant_mcp/rust_engine/lifecycle.py
"""
Rust engine lifecycle management for Python MCP integration.

This module handles the full lifecycle of the Rust engine from Python,
including startup, health monitoring, and graceful shutdown.
"""

import asyncio
import atexit
import logging
import signal
import time
from typing import Optional, Dict, Any
from enum import Enum

from .engine import RustIngestionEngine
from .grpc_client import EngineGrpcClient
from .config import EngineConfig

logger = logging.getLogger(__name__)

class EngineState(Enum):
    STOPPED = "stopped"
    STARTING = "starting"
    RUNNING = "running"
    STOPPING = "stopping"
    ERROR = "error"

class RustEngineLifecycleManager:
    """Manages the complete lifecycle of the Rust ingestion engine."""
    
    def __init__(self, config: EngineConfig):
        self.config = config
        self.state = EngineState.STOPPED
        self.engine: Optional[RustIngestionEngine] = None
        self.grpc_client: Optional[EngineGrpcClient] = None
        self.startup_time: Optional[float] = None
        self.shutdown_registered = False
        self._lock = asyncio.Lock()
        
    async def start_engine(self) -> bool:
        """Start the Rust engine with full lifecycle management."""
        async with self._lock:
            if self.state in (EngineState.STARTING, EngineState.RUNNING):
                logger.info(f"Engine already {self.state.value}, skipping start")
                return self.state == EngineState.RUNNING
            
            logger.info("Starting Rust ingestion engine")
            self.state = EngineState.STARTING
            start_time = time.time()
            
            try:
                # Phase 1: Initialize engine
                self.engine = RustIngestionEngine(self.config.to_dict())
                
                # Phase 2: Start engine with timeout
                startup_timeout = self.config.startup_timeout_seconds
                await asyncio.wait_for(
                    self.engine.start(),
                    timeout=startup_timeout
                )
                
                # Phase 3: Establish gRPC connection
                grpc_port = self.engine.grpc_port()
                self.grpc_client = EngineGrpcClient(f"127.0.0.1:{grpc_port}")
                await self.grpc_client.connect()
                
                # Phase 4: Health verification
                await self._verify_engine_health()
                
                # Phase 5: Register shutdown hooks
                if not self.shutdown_registered:
                    self._register_shutdown_hooks()
                    self.shutdown_registered = True
                
                self.state = EngineState.RUNNING
                self.startup_time = time.time() - start_time
                
                logger.info(f"Rust engine started successfully in {self.startup_time:.2f}s on port {grpc_port}")
                return True
                
            except asyncio.TimeoutError:
                logger.error(f"Engine startup timeout after {startup_timeout}s")
                self.state = EngineState.ERROR
                await self._cleanup_failed_start()
                return False
                
            except Exception as e:
                logger.error(f"Failed to start Rust engine: {e}")
                self.state = EngineState.ERROR
                await self._cleanup_failed_start()
                return False
    
    async def stop_engine(self, timeout_seconds: int = 30) -> bool:
        """Stop the Rust engine gracefully."""
        async with self._lock:
            if self.state == EngineState.STOPPED:
                logger.info("Engine already stopped")
                return True
            
            if self.state == EngineState.STOPPING:
                logger.info("Engine already stopping, waiting for completion")
                return await self._wait_for_stop(timeout_seconds)
            
            logger.info(f"Stopping Rust engine (timeout: {timeout_seconds}s)")
            self.state = EngineState.STOPPING
            
            try:
                if self.engine:
                    await asyncio.wait_for(
                        self.engine.stop(timeout_seconds),
                        timeout=timeout_seconds + 5  # Extra buffer for cleanup
                    )
                
                await self._cleanup_engine()
                self.state = EngineState.STOPPED
                
                logger.info("Rust engine stopped successfully")
                return True
                
            except asyncio.TimeoutError:
                logger.warning("Engine stop timeout, forcing shutdown")
                return await self.force_stop()
                
            except Exception as e:
                logger.error(f"Error stopping engine: {e}")
                return await self.force_stop()
    
    async def force_stop(self) -> bool:
        """Force stop the engine immediately."""
        logger.warning("Force stopping Rust engine")
        self.state = EngineState.STOPPING
        
        try:
            if self.engine:
                await self.engine.force_stop()
            
            await self._cleanup_engine()
            self.state = EngineState.STOPPED
            
            logger.warning("Engine force stop completed")
            return True
            
        except Exception as e:
            logger.error(f"Error during force stop: {e}")
            self.state = EngineState.ERROR
            return False
    
    async def restart_engine(self) -> bool:
        """Restart the engine with current configuration."""
        logger.info("Restarting Rust engine")
        
        # Stop current engine
        if not await self.stop_engine():
            logger.error("Failed to stop engine for restart")
            return False
        
        # Small delay to ensure cleanup
        await asyncio.sleep(1.0)
        
        # Start with new instance
        return await self.start_engine()
    
    async def get_engine_status(self) -> Dict[str, Any]:
        """Get comprehensive engine status."""
        status = {
            "state": self.state.value,
            "startup_time": self.startup_time,
            "grpc_connected": self.grpc_client is not None and self.grpc_client.connected,
            "uptime": time.time() - self.startup_time if self.startup_time else 0,
        }
        
        if self.grpc_client and self.state == EngineState.RUNNING:
            try:
                engine_status = await self.grpc_client.get_engine_status()
                status.update({
                    "active_tasks": engine_status.active_tasks,
                    "queued_tasks": engine_status.queued_tasks,
                    "resource_usage": {
                        "cpu_percent": engine_status.resource_usage.cpu_usage_percent,
                        "memory_mb": engine_status.resource_usage.memory_usage_bytes / (1024 * 1024),
                        "open_files": engine_status.resource_usage.open_files,
                    },
                    "version": engine_status.version,
                })
            except Exception as e:
                logger.warning(f"Failed to get engine status: {e}")
                status["status_error"] = str(e)
        
        return status
    
    async def health_check(self) -> bool:
        """Perform engine health check."""
        if self.state != EngineState.RUNNING or not self.grpc_client:
            return False
        
        try:
            health_status = await self.grpc_client.get_health_status()
            return health_status.overall_status == "HEALTHY"
        except Exception as e:
            logger.warning(f"Health check failed: {e}")
            return False
    
    async def _verify_engine_health(self) -> None:
        """Verify engine health after startup."""
        max_attempts = 10
        for attempt in range(max_attempts):
            try:
                if await self.health_check():
                    logger.info("Engine health verification passed")
                    return
            except Exception as e:
                logger.debug(f"Health check attempt {attempt + 1} failed: {e}")
            
            if attempt < max_attempts - 1:
                await asyncio.sleep(0.5)
        
        raise RuntimeError("Engine health verification failed")
    
    def _register_shutdown_hooks(self) -> None:
        """Register shutdown hooks for graceful cleanup."""
        def sync_shutdown():
            """Synchronous shutdown for atexit."""
            if self.state in (EngineState.RUNNING, EngineState.STARTING):
                asyncio.run(self.force_stop())
        
        async def signal_shutdown(sig):
            """Asynchronous shutdown for signals."""
            logger.info(f"Received signal {sig}, shutting down engine")
            await self.stop_engine(timeout_seconds=30)
        
        # Register atexit handler
        atexit.register(sync_shutdown)
        
        # Register signal handlers
        for sig in (signal.SIGINT, signal.SIGTERM):
            try:
                asyncio.get_event_loop().add_signal_handler(
                    sig, lambda s=sig: asyncio.create_task(signal_shutdown(s))
                )
            except (NotImplementedError, RuntimeError):
                # Signal handlers not available on Windows
                pass
    
    async def _cleanup_failed_start(self) -> None:
        """Clean up after failed startup."""
        try:
            if self.grpc_client:
                await self.grpc_client.close()
                self.grpc_client = None
            
            if self.engine:
                await self.engine.force_stop()
                self.engine = None
                
        except Exception as e:
            logger.error(f"Error cleaning up failed start: {e}")
    
    async def _cleanup_engine(self) -> None:
        """Clean up engine resources."""
        if self.grpc_client:
            await self.grpc_client.close()
            self.grpc_client = None
        
        self.engine = None
        self.startup_time = None
    
    async def _wait_for_stop(self, timeout_seconds: int) -> bool:
        """Wait for engine to finish stopping."""
        start_time = time.time()
        while self.state == EngineState.STOPPING:
            if time.time() - start_time > timeout_seconds:
                logger.warning("Timeout waiting for engine stop")
                return False
            
            await asyncio.sleep(0.1)
        
        return self.state == EngineState.STOPPED
```

## Connection Recovery

### Automatic Reconnection

```python
class ConnectionRecoveryManager:
    """Manages automatic reconnection to the Rust engine."""
    
    def __init__(self, lifecycle_manager: RustEngineLifecycleManager):
        self.lifecycle_manager = lifecycle_manager
        self.reconnect_attempts = 0
        self.max_reconnect_attempts = 5
        self.reconnect_delay = 1.0  # Start with 1 second
        self.max_reconnect_delay = 30.0
        self.recovery_task: Optional[asyncio.Task] = None
    
    async def handle_connection_lost(self) -> None:
        """Handle lost connection to Rust engine."""
        logger.warning("Connection to Rust engine lost, attempting recovery")
        
        if self.recovery_task and not self.recovery_task.done():
            logger.info("Recovery already in progress")
            return
        
        self.recovery_task = asyncio.create_task(self._recovery_loop())
    
    async def _recovery_loop(self) -> None:
        """Main recovery loop with exponential backoff."""
        while self.reconnect_attempts < self.max_reconnect_attempts:
            self.reconnect_attempts += 1
            delay = min(
                self.reconnect_delay * (2 ** (self.reconnect_attempts - 1)),
                self.max_reconnect_delay
            )
            
            logger.info(f"Recovery attempt {self.reconnect_attempts}/{self.max_reconnect_attempts} in {delay}s")
            await asyncio.sleep(delay)
            
            try:
                # Check if engine is still running
                if await self.lifecycle_manager.health_check():
                    logger.info("Engine connection recovered")
                    self.reconnect_attempts = 0
                    return
                
                # Try to restart engine
                if await self.lifecycle_manager.restart_engine():
                    logger.info("Engine restarted successfully")
                    self.reconnect_attempts = 0
                    return
                
                logger.warning(f"Recovery attempt {self.reconnect_attempts} failed")
                
            except Exception as e:
                logger.warning(f"Recovery attempt {self.reconnect_attempts} error: {e}")
        
        logger.error("All recovery attempts exhausted, engine unavailable")
        # At this point, the MCP server should switch to degraded mode
```

## Monitoring and Diagnostics

### Health Monitoring

```python
class EngineHealthMonitor:
    """Monitors engine health and performance."""
    
    def __init__(self, lifecycle_manager: RustEngineLifecycleManager):
        self.lifecycle_manager = lifecycle_manager
        self.monitoring_task: Optional[asyncio.Task] = None
        self.health_history: List[Dict[str, Any]] = []
        self.alert_thresholds = {
            "cpu_percent": 90.0,
            "memory_mb": 1000.0,
            "response_time_ms": 5000.0,
        }
    
    async def start_monitoring(self, interval_seconds: float = 30.0) -> None:
        """Start continuous health monitoring."""
        if self.monitoring_task and not self.monitoring_task.done():
            logger.warning("Health monitoring already active")
            return
        
        logger.info(f"Starting health monitoring (interval: {interval_seconds}s)")
        self.monitoring_task = asyncio.create_task(
            self._monitoring_loop(interval_seconds)
        )
    
    async def _monitoring_loop(self, interval: float) -> None:
        """Main monitoring loop."""
        while True:
            try:
                status = await self.lifecycle_manager.get_engine_status()
                health_data = {
                    "timestamp": time.time(),
                    "state": status["state"],
                    "cpu_percent": status.get("resource_usage", {}).get("cpu_percent", 0),
                    "memory_mb": status.get("resource_usage", {}).get("memory_mb", 0),
                    "active_tasks": status.get("active_tasks", 0),
                    "queued_tasks": status.get("queued_tasks", 0),
                }
                
                self.health_history.append(health_data)
                
                # Keep only last 100 entries
                if len(self.health_history) > 100:
                    self.health_history = self.health_history[-100:]
                
                # Check for alerts
                self._check_alerts(health_data)
                
            except Exception as e:
                logger.warning(f"Health monitoring error: {e}")
            
            await asyncio.sleep(interval)
    
    def _check_alerts(self, health_data: Dict[str, Any]) -> None:
        """Check health data against alert thresholds."""
        for metric, threshold in self.alert_thresholds.items():
            value = health_data.get(metric, 0)
            if value > threshold:
                logger.warning(f"Health alert: {metric} = {value} exceeds threshold {threshold}")
```

This comprehensive lifecycle management design ensures the Rust engine operates reliably within the Python MCP environment while maintaining data integrity and providing excellent user experience through graceful startup, operation, and shutdown processes.