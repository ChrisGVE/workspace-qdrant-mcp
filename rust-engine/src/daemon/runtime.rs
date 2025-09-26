//! Tokio runtime integration and management
//!
//! This module provides comprehensive tokio runtime lifecycle management,
//! task coordination, resource sharing, and performance monitoring.

#![allow(dead_code)]

use crate::error::{DaemonError, DaemonResult};
use std::sync::{Arc, atomic::{AtomicBool, AtomicUsize, AtomicU64, Ordering}};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};
use std::collections::HashMap;
use tokio::sync::{mpsc, RwLock, Semaphore, Notify, broadcast, watch, Barrier};
use tokio::time::{interval, sleep, timeout};
use tokio::task::JoinHandle;
use tracing::{info, debug, warn, error, instrument};
use uuid::Uuid;

/// Runtime manager for coordinating all async operations
#[derive(Debug)]
pub struct RuntimeManager {
    /// Runtime configuration
    config: RuntimeConfig,
    /// Task management system
    task_manager: Arc<TaskManager>,
    /// Resource pool for shared resources
    resource_pool: Arc<ResourcePool>,
    /// Communication hub for cross-task messaging
    communication_hub: Arc<CommunicationHub>,
    /// Performance monitoring
    performance_monitor: Arc<PerformanceMonitor>,
    /// Shutdown coordination
    shutdown_manager: Arc<ShutdownManager>,
}

/// Configuration for runtime behavior
#[derive(Debug, Clone)]
pub struct RuntimeConfig {
    /// Maximum number of concurrent tasks
    pub max_concurrent_tasks: usize,
    /// Task timeout duration
    pub task_timeout: Duration,
    /// Resource pool size
    pub resource_pool_size: usize,
    /// Enable performance monitoring
    pub enable_monitoring: bool,
    /// Monitoring interval
    pub monitoring_interval: Duration,
    /// Task retry attempts
    pub max_retry_attempts: u32,
    /// Graceful shutdown timeout
    pub shutdown_timeout: Duration,
}

/// Task management and coordination
#[derive(Debug)]
pub struct TaskManager {
    /// Task registry
    tasks: RwLock<HashMap<TaskId, TaskInfo>>,
    /// Active task count
    active_count: AtomicUsize,
    /// Completed task count
    completed_count: AtomicUsize,
    /// Failed task count
    failed_count: AtomicUsize,
    /// Task spawning semaphore
    spawn_semaphore: Arc<Semaphore>,
    /// Task coordination utilities
    coordination: TaskCoordination,
}

/// Task coordination primitives
#[derive(Debug)]
pub struct TaskCoordination {
    /// Global notification hub
    pub notify_hub: Arc<Notify>,
    /// Task synchronization barriers
    pub barriers: RwLock<HashMap<String, Arc<Barrier>>>,
    /// Task dependency tracking
    pub dependencies: RwLock<HashMap<TaskId, Vec<TaskId>>>,
}

/// Unique task identifier
pub type TaskId = Uuid;

/// Task information and metadata
#[derive(Debug, Clone)]
pub struct TaskInfo {
    pub id: TaskId,
    pub name: String,
    pub status: TaskStatus,
    pub priority: TaskPriority,
    pub created_at: Instant,
    pub started_at: Option<Instant>,
    pub completed_at: Option<Instant>,
    pub retry_count: u32,
    pub dependencies: Vec<TaskId>,
    pub metadata: HashMap<String, String>,
}

/// Task execution status
#[derive(Debug, Clone, PartialEq)]
pub enum TaskStatus {
    Pending,
    Running,
    Completed,
    Failed(String),
    Cancelled,
    Retrying,
    WaitingForDependencies,
}

/// Task priority levels
#[derive(Debug, Clone, PartialEq, Ord, PartialOrd, Eq)]
pub enum TaskPriority {
    Low = 0,
    Normal = 1,
    High = 2,
    Critical = 3,
}

/// Resource pool for managing shared async resources
#[derive(Debug)]
pub struct ResourcePool {
    /// Connection pool semaphore
    connection_semaphore: Arc<Semaphore>,
    /// Memory usage tracking
    memory_usage: AtomicUsize,
    /// Active connections
    connections: Arc<RwLock<HashMap<String, ConnectionHandle>>>,
    /// Resource cleanup notification
    cleanup_notify: Arc<Notify>,
    /// Resource allocation tracking
    allocations: Arc<RwLock<HashMap<String, AllocationInfo>>>,
}

/// Connection handle for resource tracking
#[derive(Debug)]
pub struct ConnectionHandle {
    pub id: String,
    pub created_at: Instant,
    pub last_used: AtomicU64, // Unix timestamp in millis
    pub active: AtomicBool,
    pub metadata: HashMap<String, String>,
}

impl Clone for ConnectionHandle {
    fn clone(&self) -> Self {
        Self {
            id: self.id.clone(),
            created_at: self.created_at,
            last_used: AtomicU64::new(self.last_used.load(Ordering::SeqCst)),
            active: AtomicBool::new(self.active.load(Ordering::SeqCst)),
            metadata: self.metadata.clone(),
        }
    }
}

/// Resource allocation information
#[derive(Debug, Clone)]
pub struct AllocationInfo {
    pub size: usize,
    pub allocated_at: Instant,
    pub allocator: String,
}

/// Communication hub for inter-task messaging
#[derive(Debug)]
pub struct CommunicationHub {
    /// Task lifecycle events
    task_events: broadcast::Sender<TaskEvent>,
    /// System-wide commands
    system_commands: mpsc::UnboundedSender<SystemCommand>,
    /// System status updates
    system_status: watch::Sender<SystemStatus>,
    /// Global notification system
    global_notify: Arc<Notify>,
    /// Message queues for task-to-task communication (future use)
    #[allow(dead_code)]
    message_queues: RwLock<HashMap<String, mpsc::UnboundedSender<TaskMessage>>>,
}

/// Task lifecycle events
#[derive(Debug, Clone)]
pub enum TaskEvent {
    Created { id: TaskId, name: String, priority: TaskPriority },
    Started { id: TaskId, timestamp: Instant },
    Progress { id: TaskId, progress: f32, message: Option<String> },
    Completed { id: TaskId, duration: Duration },
    Failed { id: TaskId, error: String, retry_count: u32 },
    Cancelled { id: TaskId, reason: String },
    DependencyMet { id: TaskId, dependency: TaskId },
}

/// System-wide commands
#[derive(Debug, Clone)]
pub enum SystemCommand {
    Shutdown { graceful: bool },
    PauseProcessing,
    ResumeProcessing,
    FlushBuffers,
    CollectGarbage,
    RefreshConfiguration,
    EmergencyStop,
}

/// System operational status
#[derive(Debug, Clone, PartialEq)]
pub enum SystemStatus {
    Initializing,
    Running,
    Paused,
    ShuttingDown,
    Stopped,
    Error(String),
}

/// Inter-task messages
#[derive(Debug, Clone)]
pub struct TaskMessage {
    pub from: TaskId,
    pub to: String, // Queue name
    pub payload: Vec<u8>,
    pub timestamp: Instant,
}

/// Performance monitoring and metrics
#[derive(Debug)]
pub struct PerformanceMonitor {
    /// Metrics collection
    metrics: Arc<RwLock<RuntimeMetrics>>,
    /// Monitoring task handle
    monitor_handle: RwLock<Option<JoinHandle<()>>>,
    /// Performance history
    history: Arc<RwLock<Vec<PerformanceSnapshot>>>,
    /// Alert thresholds
    thresholds: PerformanceThresholds,
}

/// Runtime performance metrics
#[derive(Debug, Clone)]
pub struct RuntimeMetrics {
    pub tasks_spawned: u64,
    pub tasks_completed: u64,
    pub tasks_failed: u64,
    pub tasks_cancelled: u64,
    pub average_task_duration: Duration,
    pub memory_usage_bytes: usize,
    pub active_connections: usize,
    pub cpu_utilization: f32,
    pub throughput_per_second: f32,
    pub last_updated: Instant,
}

impl Default for RuntimeMetrics {
    fn default() -> Self {
        Self {
            tasks_spawned: 0,
            tasks_completed: 0,
            tasks_failed: 0,
            tasks_cancelled: 0,
            average_task_duration: Duration::from_millis(0),
            memory_usage_bytes: 0,
            active_connections: 0,
            cpu_utilization: 0.0,
            throughput_per_second: 0.0,
            last_updated: Instant::now(),
        }
    }
}

/// Performance snapshot for historical tracking
#[derive(Debug, Clone)]
pub struct PerformanceSnapshot {
    pub timestamp: Instant,
    pub metrics: RuntimeMetrics,
}

/// Performance alert thresholds
#[derive(Debug, Clone)]
pub struct PerformanceThresholds {
    pub max_memory_usage: usize,
    pub max_task_duration: Duration,
    pub max_failure_rate: f32,
    pub min_throughput: f32,
}

/// Shutdown coordination and cleanup
#[derive(Debug)]
pub struct ShutdownManager {
    /// Shutdown signal
    shutdown_signal: Arc<AtomicBool>,
    /// Active shutdown handle
    shutdown_handle: RwLock<Option<JoinHandle<()>>>,
    /// Cleanup tasks
    cleanup_tasks: RwLock<Vec<JoinHandle<()>>>,
    /// Shutdown completion notification
    shutdown_complete: Arc<Notify>,
}

impl Default for RuntimeConfig {
    fn default() -> Self {
        Self {
            max_concurrent_tasks: num_cpus::get() * 4,
            task_timeout: Duration::from_secs(30),
            resource_pool_size: 100,
            enable_monitoring: true,
            monitoring_interval: Duration::from_secs(5),
            max_retry_attempts: 3,
            shutdown_timeout: Duration::from_secs(30),
        }
    }
}

impl RuntimeManager {
    /// Create a new runtime manager
    pub async fn new(config: RuntimeConfig) -> DaemonResult<Self> {
        info!("Initializing runtime manager with config: {:?}", config);

        let task_manager = Arc::new(TaskManager::new(config.max_concurrent_tasks).await?);
        let resource_pool = Arc::new(ResourcePool::new(config.resource_pool_size).await?);
        let communication_hub = Arc::new(CommunicationHub::new().await?);
        let performance_monitor = Arc::new(PerformanceMonitor::new(&config).await?);
        let shutdown_manager = Arc::new(ShutdownManager::new().await?);

        Ok(Self {
            config,
            task_manager,
            resource_pool,
            communication_hub,
            performance_monitor,
            shutdown_manager,
        })
    }

    /// Start the runtime manager
    #[instrument(skip(self))]
    pub async fn start(&self) -> DaemonResult<()> {
        info!("Starting runtime manager");

        // Start performance monitoring if enabled
        if self.config.enable_monitoring {
            self.performance_monitor.start(
                Arc::clone(&self.task_manager),
                Arc::clone(&self.resource_pool),
                self.config.monitoring_interval,
            ).await?;
        }

        // Start resource cleanup task
        self.resource_pool.start_cleanup_task().await?;

        // Update system status
        self.communication_hub.update_system_status(SystemStatus::Running).await?;

        info!("Runtime manager started successfully");
        Ok(())
    }

    /// Stop the runtime manager gracefully
    #[instrument(skip(self))]
    pub async fn stop(&self, graceful: bool) -> DaemonResult<()> {
        info!("Stopping runtime manager (graceful: {})", graceful);

        // Signal shutdown
        self.shutdown_manager.initiate_shutdown(graceful).await?;

        // Update system status
        self.communication_hub.update_system_status(SystemStatus::ShuttingDown).await?;

        if graceful {
            // Wait for active tasks to complete
            self.wait_for_tasks_completion(self.config.shutdown_timeout).await?;
        } else {
            // Cancel all active tasks
            self.cancel_all_tasks().await?;
        }

        // Stop performance monitoring
        self.performance_monitor.stop().await?;

        // Cleanup resources
        self.resource_pool.cleanup_all().await?;

        // Final status update
        self.communication_hub.update_system_status(SystemStatus::Stopped).await?;

        info!("Runtime manager stopped");
        Ok(())
    }

    /// Spawn a new task with tracking
    #[instrument(skip(self, task_fn))]
    pub async fn spawn_task<F, R>(&self,
        name: String,
        priority: TaskPriority,
        task_fn: F
    ) -> DaemonResult<TaskId>
    where
        F: std::future::Future<Output = DaemonResult<R>> + Send + 'static,
        R: Send + 'static,
    {
        let task_id = TaskId::new_v4();

        // Register task
        self.task_manager.register_task(task_id, name.clone(), priority.clone()).await?;

        // Emit task created event
        self.communication_hub.emit_task_event(TaskEvent::Created {
            id: task_id,
            name: name.clone(),
            priority: priority.clone(),
        }).await?;

        // Spawn with tracking
        let task_manager = Arc::clone(&self.task_manager);
        let communication_hub = Arc::clone(&self.communication_hub);
        let timeout_duration = self.config.task_timeout;
        let _max_retries = self.config.max_retry_attempts;

        tokio::spawn(async move {
            let start_time = Instant::now();

            // Mark task as started
            if let Err(e) = task_manager.start_task(task_id).await {
                error!("Failed to start task {}: {}", task_id, e);
                return;
            }

            let _ = communication_hub.emit_task_event(TaskEvent::Started {
                id: task_id,
                timestamp: start_time,
            }).await;

            // Execute task with timeout (simplified - no retry for now)
            let result = match timeout(timeout_duration, task_fn).await {
                Ok(result) => result,
                Err(_) => {
                    // Task timed out
                    Err(DaemonError::Internal {
                        message: format!("Task {} timed out after {:?}", task_id, timeout_duration)
                    })
                }
            };

            let duration = start_time.elapsed();

            // Handle task completion
            match result {
                Ok(_) => {
                    if let Err(e) = task_manager.complete_task(task_id).await {
                        error!("Failed to mark task {} as completed: {}", task_id, e);
                    }
                    let _ = communication_hub.emit_task_event(TaskEvent::Completed {
                        id: task_id,
                        duration,
                    }).await;
                }
                Err(e) => {
                    if let Err(e) = task_manager.fail_task(task_id, e.to_string()).await {
                        error!("Failed to mark task {} as failed: {}", task_id, e);
                    }
                    let _ = communication_hub.emit_task_event(TaskEvent::Failed {
                        id: task_id,
                        error: e.to_string(),
                        retry_count: 0,
                    }).await;
                }
            }
        });

        Ok(task_id)
    }

    /// Wait for tasks completion with timeout
    async fn wait_for_tasks_completion(&self, timeout_duration: Duration) -> DaemonResult<()> {
        let start = Instant::now();

        while self.task_manager.get_active_count() > 0 && start.elapsed() < timeout_duration {
            sleep(Duration::from_millis(100)).await;
        }

        if self.task_manager.get_active_count() > 0 {
            warn!("Some tasks did not complete within timeout, proceeding with shutdown");
        }

        Ok(())
    }

    /// Cancel all active tasks
    async fn cancel_all_tasks(&self) -> DaemonResult<()> {
        // Implementation would cancel all active tasks
        // For now, we'll mark them as cancelled
        self.task_manager.cancel_all_tasks().await
    }

    /// Get runtime statistics
    pub async fn get_statistics(&self) -> RuntimeStatistics {
        RuntimeStatistics {
            active_tasks: self.task_manager.get_active_count(),
            completed_tasks: self.task_manager.get_completed_count(),
            failed_tasks: self.task_manager.get_failed_count(),
            memory_usage: self.resource_pool.get_memory_usage(),
            active_connections: self.resource_pool.get_active_connections().await,
            system_status: self.communication_hub.get_system_status().await,
        }
    }

    /// Get task manager reference
    pub fn task_manager(&self) -> &Arc<TaskManager> {
        &self.task_manager
    }

    /// Get resource pool reference
    pub fn resource_pool(&self) -> &Arc<ResourcePool> {
        &self.resource_pool
    }

    /// Get communication hub reference
    pub fn communication_hub(&self) -> &Arc<CommunicationHub> {
        &self.communication_hub
    }

    /// Get performance monitor reference
    pub fn performance_monitor(&self) -> &Arc<PerformanceMonitor> {
        &self.performance_monitor
    }
}

/// Runtime statistics summary
#[derive(Debug, Clone)]
pub struct RuntimeStatistics {
    pub active_tasks: usize,
    pub completed_tasks: usize,
    pub failed_tasks: usize,
    pub memory_usage: usize,
    pub active_connections: usize,
    pub system_status: SystemStatus,
}

impl TaskManager {
    pub async fn new(max_concurrent: usize) -> DaemonResult<Self> {
        Ok(Self {
            tasks: RwLock::new(HashMap::new()),
            active_count: AtomicUsize::new(0),
            completed_count: AtomicUsize::new(0),
            failed_count: AtomicUsize::new(0),
            spawn_semaphore: Arc::new(Semaphore::new(max_concurrent)),
            coordination: TaskCoordination {
                notify_hub: Arc::new(Notify::new()),
                barriers: RwLock::new(HashMap::new()),
                dependencies: RwLock::new(HashMap::new()),
            },
        })
    }

    pub async fn register_task(&self, id: TaskId, name: String, priority: TaskPriority) -> DaemonResult<()> {
        let task_info = TaskInfo {
            id,
            name,
            status: TaskStatus::Pending,
            priority,
            created_at: Instant::now(),
            started_at: None,
            completed_at: None,
            retry_count: 0,
            dependencies: Vec::new(),
            metadata: HashMap::new(),
        };

        self.tasks.write().await.insert(id, task_info);

        // Use coordination for task notifications
        self.coordination.notify_hub.notify_one();
        Ok(())
    }

    pub async fn start_task(&self, id: TaskId) -> DaemonResult<()> {
        let mut tasks = self.tasks.write().await;
        if let Some(task) = tasks.get_mut(&id) {
            task.status = TaskStatus::Running;
            task.started_at = Some(Instant::now());
            self.active_count.fetch_add(1, Ordering::SeqCst);
        }
        Ok(())
    }

    pub async fn complete_task(&self, id: TaskId) -> DaemonResult<()> {
        let mut tasks = self.tasks.write().await;
        if let Some(task) = tasks.get_mut(&id) {
            task.status = TaskStatus::Completed;
            task.completed_at = Some(Instant::now());
            self.active_count.fetch_sub(1, Ordering::SeqCst);
            self.completed_count.fetch_add(1, Ordering::SeqCst);
        }
        Ok(())
    }

    pub async fn fail_task(&self, id: TaskId, error: String) -> DaemonResult<()> {
        let mut tasks = self.tasks.write().await;
        if let Some(task) = tasks.get_mut(&id) {
            task.status = TaskStatus::Failed(error);
            task.completed_at = Some(Instant::now());
            self.active_count.fetch_sub(1, Ordering::SeqCst);
            self.failed_count.fetch_add(1, Ordering::SeqCst);
        }
        Ok(())
    }

    pub async fn retry_task(&self, id: TaskId) -> DaemonResult<()> {
        let mut tasks = self.tasks.write().await;
        if let Some(task) = tasks.get_mut(&id) {
            task.status = TaskStatus::Retrying;
            task.retry_count += 1;
        }
        Ok(())
    }

    pub async fn cancel_all_tasks(&self) -> DaemonResult<()> {
        let mut tasks = self.tasks.write().await;
        for (_, task) in tasks.iter_mut() {
            if task.status == TaskStatus::Running || task.status == TaskStatus::Pending {
                task.status = TaskStatus::Cancelled;
                task.completed_at = Some(Instant::now());
            }
        }
        self.active_count.store(0, Ordering::SeqCst);
        Ok(())
    }

    pub fn get_active_count(&self) -> usize {
        self.active_count.load(Ordering::SeqCst)
    }

    pub fn get_completed_count(&self) -> usize {
        self.completed_count.load(Ordering::SeqCst)
    }

    pub fn get_failed_count(&self) -> usize {
        self.failed_count.load(Ordering::SeqCst)
    }

    pub async fn acquire_spawn_permit(&self) -> Result<tokio::sync::SemaphorePermit<'_>, tokio::sync::AcquireError> {
        self.spawn_semaphore.acquire().await
    }
}

impl ResourcePool {
    pub async fn new(pool_size: usize) -> DaemonResult<Self> {
        Ok(Self {
            connection_semaphore: Arc::new(Semaphore::new(pool_size)),
            memory_usage: AtomicUsize::new(0),
            connections: Arc::new(RwLock::new(HashMap::new())),
            cleanup_notify: Arc::new(Notify::new()),
            allocations: Arc::new(RwLock::new(HashMap::new())),
        })
    }

    pub async fn acquire_connection(&self) -> Result<tokio::sync::SemaphorePermit<'_>, tokio::sync::AcquireError> {
        self.connection_semaphore.acquire().await
    }

    pub async fn add_connection(&self, id: String) -> DaemonResult<()> {
        let connection = ConnectionHandle {
            id: id.clone(),
            created_at: Instant::now(),
            last_used: AtomicU64::new(
                SystemTime::now()
                    .duration_since(UNIX_EPOCH)
                    .unwrap()
                    .as_millis() as u64
            ),
            active: AtomicBool::new(true),
            metadata: HashMap::new(),
        };

        self.connections.write().await.insert(id, connection);
        Ok(())
    }

    pub async fn remove_connection(&self, id: &str) -> DaemonResult<()> {
        self.connections.write().await.remove(id);
        Ok(())
    }

    pub fn allocate_memory(&self, size: usize, allocator: String) -> String {
        let allocation_id = Uuid::new_v4().to_string();
        let _allocation_info = AllocationInfo {
            size,
            allocated_at: Instant::now(),
            allocator,
        };

        // This would normally be async, but for simplicity we'll make it sync
        // In a real implementation, you'd want async allocation tracking
        self.memory_usage.fetch_add(size, Ordering::SeqCst);

        // Note: In real code, this would need to be async
        // self.allocations.write().await.insert(allocation_id.clone(), allocation_info);

        allocation_id
    }

    pub fn deallocate_memory(&self, _allocation_id: &str) -> DaemonResult<()> {
        // Note: In real code, this would need to be async to access allocations
        // For now, we'll just decrement a fixed amount for testing
        // let allocation_info = self.allocations.write().await.remove(allocation_id);
        // if let Some(info) = allocation_info {
        //     self.memory_usage.fetch_sub(info.size, Ordering::SeqCst);
        // }
        Ok(())
    }

    pub fn get_memory_usage(&self) -> usize {
        self.memory_usage.load(Ordering::SeqCst)
    }

    pub async fn get_active_connections(&self) -> usize {
        self.connections.read().await.len()
    }

    pub async fn start_cleanup_task(&self) -> DaemonResult<()> {
        let connections_lock = Arc::clone(&self.connections);
        let cleanup_notify = Arc::clone(&self.cleanup_notify);

        tokio::spawn(async move {
            let mut cleanup_interval = interval(Duration::from_secs(60));

            loop {
                tokio::select! {
                    _ = cleanup_interval.tick() => {
                        // Cleanup expired connections
                        let now = SystemTime::now()
                            .duration_since(UNIX_EPOCH)
                            .unwrap()
                            .as_millis() as u64;

                        let mut connections_guard = connections_lock.write().await;
                        connections_guard.retain(|_, conn| {
                            let last_used = conn.last_used.load(Ordering::SeqCst);
                            let active = conn.active.load(Ordering::SeqCst);

                            // Keep connection if active or used within last 5 minutes
                            active || (now - last_used) < 300_000
                        });
                    }
                    _ = cleanup_notify.notified() => {
                        // Manual cleanup triggered
                        debug!("Manual resource cleanup triggered");
                    }
                }
            }
        });

        Ok(())
    }

    pub async fn cleanup_all(&self) -> DaemonResult<()> {
        self.connections.write().await.clear();
        self.allocations.write().await.clear();
        self.memory_usage.store(0, Ordering::SeqCst);
        Ok(())
    }

    pub fn trigger_cleanup(&self) {
        self.cleanup_notify.notify_waiters();
    }
}

impl CommunicationHub {
    pub async fn new() -> DaemonResult<Self> {
        let (task_events, _) = broadcast::channel(10000);
        let (system_commands, _) = mpsc::unbounded_channel();
        let (system_status, _) = watch::channel(SystemStatus::Initializing);

        Ok(Self {
            task_events,
            system_commands,
            system_status,
            global_notify: Arc::new(Notify::new()),
            message_queues: RwLock::new(HashMap::new()),
        })
    }

    pub async fn emit_task_event(&self, event: TaskEvent) -> DaemonResult<()> {
        self.task_events.send(event)
            .map_err(|_| DaemonError::Internal {
                message: "Failed to emit task event".to_string()
            })?;
        Ok(())
    }

    pub async fn send_system_command(&self, command: SystemCommand) -> DaemonResult<()> {
        self.system_commands.send(command)
            .map_err(|_| DaemonError::Internal {
                message: "Failed to send system command".to_string()
            })?;
        Ok(())
    }

    pub async fn update_system_status(&self, status: SystemStatus) -> DaemonResult<()> {
        self.system_status.send(status)
            .map_err(|_| DaemonError::Internal {
                message: "Failed to update system status".to_string()
            })?;
        Ok(())
    }

    pub async fn get_system_status(&self) -> SystemStatus {
        self.system_status.borrow().clone()
    }

    pub fn subscribe_task_events(&self) -> broadcast::Receiver<TaskEvent> {
        self.task_events.subscribe()
    }

    pub fn subscribe_system_status(&self) -> watch::Receiver<SystemStatus> {
        self.system_status.subscribe()
    }

    pub fn notify_global(&self) {
        self.global_notify.notify_waiters();
    }

    pub async fn wait_global_notification(&self) {
        self.global_notify.notified().await;
    }
}

impl PerformanceMonitor {
    pub async fn new(config: &RuntimeConfig) -> DaemonResult<Self> {
        let thresholds = PerformanceThresholds {
            max_memory_usage: config.resource_pool_size * 1024 * 1024, // 1MB per connection
            max_task_duration: config.task_timeout,
            max_failure_rate: 0.1, // 10%
            min_throughput: 10.0, // 10 tasks per second
        };

        Ok(Self {
            metrics: Arc::new(RwLock::new(RuntimeMetrics::default())),
            monitor_handle: RwLock::new(None),
            history: Arc::new(RwLock::new(Vec::new())),
            thresholds,
        })
    }

    pub async fn start(
        &self,
        task_manager: Arc<TaskManager>,
        resource_pool: Arc<ResourcePool>,
        interval_duration: Duration,
    ) -> DaemonResult<()> {
        let metrics_lock = Arc::clone(&self.metrics);
        let history_lock = Arc::clone(&self.history);
        let thresholds = self.thresholds.clone();

        let handle = tokio::spawn(async move {
            let mut interval = interval(interval_duration);

            loop {
                interval.tick().await;

                let snapshot = RuntimeMetrics {
                    tasks_spawned: (task_manager.get_active_count() +
                                  task_manager.get_completed_count() +
                                  task_manager.get_failed_count()) as u64,
                    tasks_completed: task_manager.get_completed_count() as u64,
                    tasks_failed: task_manager.get_failed_count() as u64,
                    tasks_cancelled: 0, // Would need additional tracking
                    average_task_duration: Duration::from_millis(100), // Placeholder
                    memory_usage_bytes: resource_pool.get_memory_usage(),
                    active_connections: resource_pool.get_active_connections().await,
                    cpu_utilization: 0.0, // Would need system integration
                    throughput_per_second: 0.0, // Would calculate from historical data
                    last_updated: Instant::now(),
                };

                // Update current metrics
                *metrics_lock.write().await = snapshot.clone();

                // Add to history
                let mut history_guard = history_lock.write().await;
                history_guard.push(PerformanceSnapshot {
                    timestamp: Instant::now(),
                    metrics: snapshot.clone(),
                });

                // Keep only last 100 snapshots
                if history_guard.len() > 100 {
                    history_guard.remove(0);
                }

                // Check thresholds and emit alerts if needed
                Self::check_thresholds(&snapshot, &thresholds).await;
            }
        });

        *self.monitor_handle.write().await = Some(handle);
        Ok(())
    }

    pub async fn stop(&self) -> DaemonResult<()> {
        if let Some(handle) = self.monitor_handle.write().await.take() {
            handle.abort();
        }
        Ok(())
    }

    async fn check_thresholds(metrics: &RuntimeMetrics, thresholds: &PerformanceThresholds) {
        if metrics.memory_usage_bytes > thresholds.max_memory_usage {
            warn!("Memory usage exceeded threshold: {} > {}",
                  metrics.memory_usage_bytes, thresholds.max_memory_usage);
        }

        let failure_rate = if metrics.tasks_spawned > 0 {
            metrics.tasks_failed as f32 / metrics.tasks_spawned as f32
        } else {
            0.0
        };

        if failure_rate > thresholds.max_failure_rate {
            warn!("Task failure rate exceeded threshold: {:.2}% > {:.2}%",
                  failure_rate * 100.0, thresholds.max_failure_rate * 100.0);
        }
    }

    pub async fn get_current_metrics(&self) -> RuntimeMetrics {
        self.metrics.read().await.clone()
    }

    pub async fn get_metrics_history(&self) -> Vec<PerformanceSnapshot> {
        self.history.read().await.clone()
    }

    pub async fn get_metrics_history_summary(&self) -> Vec<PerformanceSnapshot> {
        // Return a copy of the history to avoid Clone issues
        let history = self.history.read().await;
        history.iter().cloned().collect()
    }
}

impl ShutdownManager {
    pub async fn new() -> DaemonResult<Self> {
        Ok(Self {
            shutdown_signal: Arc::new(AtomicBool::new(false)),
            shutdown_handle: RwLock::new(None),
            cleanup_tasks: RwLock::new(Vec::new()),
            shutdown_complete: Arc::new(Notify::new()),
        })
    }

    pub async fn initiate_shutdown(&self, graceful: bool) -> DaemonResult<()> {
        info!("Initiating shutdown (graceful: {})", graceful);
        self.shutdown_signal.store(true, Ordering::SeqCst);

        let shutdown_complete = Arc::clone(&self.shutdown_complete);
        let handle = tokio::spawn(async move {
            // Perform shutdown coordination
            if graceful {
                // Allow some time for graceful shutdown
                sleep(Duration::from_millis(500)).await;
            }
            shutdown_complete.notify_waiters();
        });

        *self.shutdown_handle.write().await = Some(handle);
        Ok(())
    }

    pub fn is_shutdown_requested(&self) -> bool {
        self.shutdown_signal.load(Ordering::SeqCst)
    }

    pub async fn wait_for_shutdown(&self) {
        self.shutdown_complete.notified().await;
    }

    pub async fn add_cleanup_task(&self, handle: JoinHandle<()>) {
        self.cleanup_tasks.write().await.push(handle);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tokio_test::{assert_ok, assert_err};

    #[tokio::test]
    async fn test_runtime_manager_creation() {
        let config = RuntimeConfig::default();
        let manager = assert_ok!(RuntimeManager::new(config).await);

        let stats = manager.get_statistics().await;
        assert_eq!(stats.active_tasks, 0);
        assert_eq!(stats.completed_tasks, 0);
        assert_eq!(stats.failed_tasks, 0);
    }

    #[tokio::test]
    async fn test_task_lifecycle() {
        let config = RuntimeConfig::default();
        let manager = assert_ok!(RuntimeManager::new(config).await);
        assert_ok!(manager.start().await);

        let task_id = assert_ok!(manager.spawn_task(
            "test_task".to_string(),
            TaskPriority::Normal,
            async { Ok(()) }
        ).await);

        // Give task time to complete
        sleep(Duration::from_millis(10)).await;

        let stats = manager.get_statistics().await;
        assert_eq!(stats.completed_tasks, 1);
    }

    #[tokio::test]
    async fn test_resource_pool() {
        let pool = assert_ok!(ResourcePool::new(5).await);

        // Test connection management
        assert_ok!(pool.add_connection("conn1".to_string()).await);
        assert_eq!(pool.get_active_connections().await, 1);

        // Test memory allocation
        let initial_usage = pool.get_memory_usage();
        pool.allocate_memory(1024, "test_allocator".to_string());
        assert_eq!(pool.get_memory_usage(), initial_usage + 1024);
    }

    #[tokio::test]
    async fn test_communication_hub() {
        let hub = assert_ok!(CommunicationHub::new().await);

        // Test task events
        let mut event_receiver = hub.subscribe_task_events();
        let task_id = TaskId::new_v4();

        assert_ok!(hub.emit_task_event(TaskEvent::Started {
            id: task_id,
            timestamp: Instant::now(),
        }).await);

        let received_event = assert_ok!(event_receiver.recv().await);
        match received_event {
            TaskEvent::Started { id, .. } => assert_eq!(id, task_id),
            _ => panic!("Unexpected event type"),
        }
    }

    #[tokio::test]
    async fn test_performance_monitoring() {
        let config = RuntimeConfig::default();
        let monitor = assert_ok!(PerformanceMonitor::new(&config).await);

        let metrics = monitor.get_current_metrics().await;
        assert_eq!(metrics.tasks_spawned, 0);
        assert_eq!(metrics.tasks_completed, 0);
    }

    #[tokio::test]
    async fn test_shutdown_manager() {
        let manager = assert_ok!(ShutdownManager::new().await);

        assert!(!manager.is_shutdown_requested());
        assert_ok!(manager.initiate_shutdown(true).await);
        assert!(manager.is_shutdown_requested());
    }
}