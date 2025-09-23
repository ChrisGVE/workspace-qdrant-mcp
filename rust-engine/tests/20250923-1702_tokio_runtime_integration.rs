//! Comprehensive unit tests for tokio runtime integration
//!
//! This test suite validates tokio runtime lifecycle, task management,
//! resource sharing, and cleanup mechanisms using TDD approach.

use std::sync::{Arc, atomic::{AtomicBool, AtomicUsize, Ordering}};
use std::time::{Duration, Instant};
use std::collections::HashMap;
use tokio::sync::{mpsc, RwLock, Semaphore, Notify, oneshot, broadcast, watch};
use tokio::time::{timeout, sleep, interval};
use tokio_test::{assert_ok, assert_err, assert_pending, assert_ready};
use futures_util::future::{join_all, select_all};
use workspace_qdrant_daemon::daemon::WorkspaceDaemon;
use workspace_qdrant_daemon::config::*;
use workspace_qdrant_daemon::error::DaemonResult;

/// Shared test utilities for tokio runtime testing
pub struct RuntimeTestHarness {
    pub runtime_handle: tokio::runtime::Handle,
    pub task_tracker: Arc<TaskTracker>,
    pub resource_pool: Arc<ResourcePool>,
    pub communication_hub: Arc<CommunicationHub>,
}

/// Task tracking utility for runtime tests
#[derive(Debug)]
pub struct TaskTracker {
    pub active_tasks: AtomicUsize,
    pub completed_tasks: AtomicUsize,
    pub failed_tasks: AtomicUsize,
    pub tasks_registry: RwLock<HashMap<String, TaskInfo>>,
}

/// Individual task information
#[derive(Debug, Clone)]
pub struct TaskInfo {
    pub id: String,
    pub status: TaskStatus,
    pub created_at: Instant,
    pub completed_at: Option<Instant>,
    pub priority: TaskPriority,
}

#[derive(Debug, Clone, PartialEq)]
pub enum TaskStatus {
    Pending,
    Running,
    Completed,
    Failed,
    Cancelled,
}

#[derive(Debug, Clone, PartialEq)]
pub enum TaskPriority {
    Low,
    Normal,
    High,
    Critical,
}

/// Resource pooling for async operations
#[derive(Debug)]
pub struct ResourcePool {
    pub permits: Arc<Semaphore>,
    pub connections: RwLock<Vec<ConnectionHandle>>,
    pub memory_tracker: AtomicUsize,
}

#[derive(Debug, Clone)]
pub struct ConnectionHandle {
    pub id: String,
    pub active: AtomicBool,
    pub last_used: RwLock<Instant>,
}

/// Communication hub for cross-task messaging
#[derive(Debug)]
pub struct CommunicationHub {
    pub task_events: broadcast::Sender<TaskEvent>,
    pub system_commands: mpsc::UnboundedSender<SystemCommand>,
    pub status_updates: watch::Sender<SystemStatus>,
    pub notification_hub: Arc<Notify>,
}

#[derive(Debug, Clone)]
pub enum TaskEvent {
    Started(String),
    Completed(String),
    Failed(String, String),
    Progress(String, f32),
}

#[derive(Debug, Clone)]
pub enum SystemCommand {
    Shutdown,
    PauseProcessing,
    ResumeProcessing,
    FlushBuffers,
    CollectGarbage,
}

#[derive(Debug, Clone, PartialEq)]
pub enum SystemStatus {
    Initializing,
    Running,
    Paused,
    ShuttingDown,
    Stopped,
}

// Shared test utility macros (as mentioned in requirements)
macro_rules! tokio_test_async {
    ($test_name:ident, $test_body:expr) => {
        #[tokio::test]
        async fn $test_name() {
            let harness = create_test_harness().await;
            $test_body(harness).await;
        }
    };
}

macro_rules! concurrent_async_test {
    ($test_name:ident, $concurrency:expr, $test_body:expr) => {
        #[tokio::test]
        async fn $test_name() {
            let harness = create_test_harness().await;
            let mut handles = Vec::new();

            for i in 0..$concurrency {
                let harness_clone = harness.clone();
                let handle = tokio::spawn(async move {
                    $test_body(harness_clone, i).await
                });
                handles.push(handle);
            }

            let results = join_all(handles).await;
            for result in results {
                assert_ok!(result);
            }
        }
    };
}

impl RuntimeTestHarness {
    pub fn clone(&self) -> Self {
        Self {
            runtime_handle: self.runtime_handle.clone(),
            task_tracker: Arc::clone(&self.task_tracker),
            resource_pool: Arc::clone(&self.resource_pool),
            communication_hub: Arc::clone(&self.communication_hub),
        }
    }
}

impl TaskTracker {
    pub fn new() -> Self {
        Self {
            active_tasks: AtomicUsize::new(0),
            completed_tasks: AtomicUsize::new(0),
            failed_tasks: AtomicUsize::new(0),
            tasks_registry: RwLock::new(HashMap::new()),
        }
    }

    pub async fn start_task(&self, id: String, priority: TaskPriority) -> DaemonResult<()> {
        self.active_tasks.fetch_add(1, Ordering::SeqCst);

        let task_info = TaskInfo {
            id: id.clone(),
            status: TaskStatus::Running,
            created_at: Instant::now(),
            completed_at: None,
            priority,
        };

        self.tasks_registry.write().await.insert(id, task_info);
        Ok(())
    }

    pub async fn complete_task(&self, id: &str) -> DaemonResult<()> {
        self.active_tasks.fetch_sub(1, Ordering::SeqCst);
        self.completed_tasks.fetch_add(1, Ordering::SeqCst);

        if let Some(task_info) = self.tasks_registry.write().await.get_mut(id) {
            task_info.status = TaskStatus::Completed;
            task_info.completed_at = Some(Instant::now());
        }
        Ok(())
    }

    pub async fn fail_task(&self, id: &str) -> DaemonResult<()> {
        self.active_tasks.fetch_sub(1, Ordering::SeqCst);
        self.failed_tasks.fetch_add(1, Ordering::SeqCst);

        if let Some(task_info) = self.tasks_registry.write().await.get_mut(id) {
            task_info.status = TaskStatus::Failed;
            task_info.completed_at = Some(Instant::now());
        }
        Ok(())
    }

    pub fn get_active_count(&self) -> usize {
        self.active_tasks.load(Ordering::SeqCst)
    }

    pub fn get_completed_count(&self) -> usize {
        self.completed_tasks.load(Ordering::SeqCst)
    }

    pub fn get_failed_count(&self) -> usize {
        self.failed_tasks.load(Ordering::SeqCst)
    }
}

impl ResourcePool {
    pub fn new(max_permits: usize) -> Self {
        Self {
            permits: Arc::new(Semaphore::new(max_permits)),
            connections: RwLock::new(Vec::new()),
            memory_tracker: AtomicUsize::new(0),
        }
    }

    pub async fn acquire_permit(&self) -> Result<tokio::sync::SemaphorePermit, tokio::sync::AcquireError> {
        self.permits.acquire().await
    }

    pub async fn add_connection(&self, id: String) -> DaemonResult<()> {
        let connection = ConnectionHandle {
            id,
            active: AtomicBool::new(true),
            last_used: RwLock::new(Instant::now()),
        };

        self.connections.write().await.push(connection);
        Ok(())
    }

    pub fn allocate_memory(&self, bytes: usize) {
        self.memory_tracker.fetch_add(bytes, Ordering::SeqCst);
    }

    pub fn deallocate_memory(&self, bytes: usize) {
        self.memory_tracker.fetch_sub(bytes, Ordering::SeqCst);
    }

    pub fn get_memory_usage(&self) -> usize {
        self.memory_tracker.load(Ordering::SeqCst)
    }
}

impl CommunicationHub {
    pub fn new() -> Self {
        let (task_events_tx, _) = broadcast::channel(1000);
        let (system_commands_tx, _) = mpsc::unbounded_channel();
        let (status_updates_tx, _) = watch::channel(SystemStatus::Initializing);

        Self {
            task_events: task_events_tx,
            system_commands: system_commands_tx,
            status_updates: status_updates_tx,
            notification_hub: Arc::new(Notify::new()),
        }
    }

    pub async fn broadcast_task_event(&self, event: TaskEvent) -> DaemonResult<()> {
        self.task_events.send(event)
            .map_err(|_| workspace_qdrant_daemon::error::DaemonError::Internal {
                message: "Failed to broadcast task event".to_string()
            })?;
        Ok(())
    }

    pub async fn send_system_command(&self, command: SystemCommand) -> DaemonResult<()> {
        self.system_commands.send(command)
            .map_err(|_| workspace_qdrant_daemon::error::DaemonError::Internal {
                message: "Failed to send system command".to_string()
            })?;
        Ok(())
    }

    pub async fn update_status(&self, status: SystemStatus) -> DaemonResult<()> {
        self.status_updates.send(status)
            .map_err(|_| workspace_qdrant_daemon::error::DaemonError::Internal {
                message: "Failed to update status".to_string()
            })?;
        Ok(())
    }

    pub fn notify_all(&self) {
        self.notification_hub.notify_waiters();
    }

    pub async fn wait_for_notification(&self) {
        self.notification_hub.notified().await;
    }
}

/// Create test harness for runtime integration tests
pub async fn create_test_harness() -> RuntimeTestHarness {
    let runtime_handle = tokio::runtime::Handle::current();
    let task_tracker = Arc::new(TaskTracker::new());
    let resource_pool = Arc::new(ResourcePool::new(10));
    let communication_hub = Arc::new(CommunicationHub::new());

    RuntimeTestHarness {
        runtime_handle,
        task_tracker,
        resource_pool,
        communication_hub,
    }
}

/// Create test daemon configuration
fn create_test_daemon_config() -> DaemonConfig {
    DaemonConfig {
        server: ServerConfig {
            host: "127.0.0.1".to_string(),
            port: 50053,
            max_connections: 100,
            connection_timeout_secs: 30,
            request_timeout_secs: 60,
            enable_tls: false,
        },
        database: DatabaseConfig {
            sqlite_path: ":memory:".to_string(),
            max_connections: 5,
            connection_timeout_secs: 30,
            enable_wal: true,
        },
        qdrant: QdrantConfig {
            url: "http://localhost:6333".to_string(),
            api_key: None,
            timeout_secs: 30,
            max_retries: 3,
            default_collection: CollectionConfig {
                vector_size: 384,
                distance_metric: "Cosine".to_string(),
                enable_indexing: true,
                replication_factor: 1,
                shard_number: 1,
            },
        },
        processing: ProcessingConfig {
            max_concurrent_tasks: 4,
            default_chunk_size: 1000,
            default_chunk_overlap: 100,
            max_file_size_bytes: 1000000,
            supported_extensions: vec!["txt".to_string(), "md".to_string()],
            enable_lsp: false,
            lsp_timeout_secs: 10,
        },
        file_watcher: FileWatcherConfig {
            enabled: false,
            debounce_ms: 500,
            max_watched_dirs: 100,
            ignore_patterns: vec![],
            recursive: true,
        },
        metrics: MetricsConfig {
            enabled: false,
            collection_interval_secs: 60,
            retention_days: 30,
            enable_prometheus: false,
            prometheus_port: 9090,
        },
        logging: LoggingConfig {
            level: "info".to_string(),
            file_path: None,
            json_format: false,
            max_file_size_mb: 100,
            max_files: 5,
        },
    }
}

// =============================================================================
// RUNTIME LIFECYCLE TESTS
// =============================================================================

#[tokio::test]
async fn test_runtime_handle_access() {
    let harness = create_test_harness().await;

    // Test runtime handle is valid and accessible
    let handle = &harness.runtime_handle;
    assert!(handle.runtime_flavor().is_multi_thread() || handle.runtime_flavor().is_current_thread());

    // Test spawning task on runtime
    let result = handle.spawn(async { "test_task_result" }).await;
    assert_ok!(result);
    assert_eq!(result.unwrap(), "test_task_result");
}

#[tokio::test]
async fn test_runtime_task_spawning() {
    let harness = create_test_harness().await;

    // Test basic task spawning
    let task_id = "spawn_test_1".to_string();
    assert_ok!(harness.task_tracker.start_task(task_id.clone(), TaskPriority::Normal).await);

    let handle = tokio::spawn(async move {
        sleep(Duration::from_millis(10)).await;
        "task_completed"
    });

    let result = assert_ok!(handle.await);
    assert_eq!(result, "task_completed");

    assert_ok!(harness.task_tracker.complete_task(&task_id).await);
    assert_eq!(harness.task_tracker.get_completed_count(), 1);
}

tokio_test_async!(test_runtime_configuration_and_optimization, |harness| async move {
    // Test runtime behavior with different configurations
    let start_time = Instant::now();

    // Spawn multiple tasks to test runtime scheduling
    let mut handles = Vec::new();
    for i in 0..10 {
        let task_id = format!("config_test_{}", i);
        assert_ok!(harness.task_tracker.start_task(task_id.clone(), TaskPriority::Normal).await);

        let tracker = Arc::clone(&harness.task_tracker);
        let handle = tokio::spawn(async move {
            sleep(Duration::from_millis(5)).await;
            assert_ok!(tracker.complete_task(&task_id).await);
        });
        handles.push(handle);
    }

    // Wait for all tasks to complete
    for handle in handles {
        assert_ok!(handle.await);
    }

    let elapsed = start_time.elapsed();
    assert!(elapsed < Duration::from_millis(200)); // Should complete efficiently
    assert_eq!(harness.task_tracker.get_completed_count(), 10);
});

// =============================================================================
// TASK MANAGEMENT TESTS
// =============================================================================

concurrent_async_test!(test_concurrent_task_spawning, 5, |harness, worker_id| async move {
    let task_count = 10;
    let mut handles = Vec::new();

    for i in 0..task_count {
        let task_id = format!("worker_{}_task_{}", worker_id, i);
        assert_ok!(harness.task_tracker.start_task(task_id.clone(), TaskPriority::Normal).await);

        let tracker = Arc::clone(&harness.task_tracker);
        let handle = tokio::spawn(async move {
            sleep(Duration::from_millis(1)).await;
            assert_ok!(tracker.complete_task(&task_id).await);
        });
        handles.push(handle);
    }

    for handle in handles {
        assert_ok!(handle.await);
    }
});

#[tokio::test]
async fn test_task_priority_management() {
    let harness = create_test_harness().await;

    // Test different priority levels
    let priorities = [
        TaskPriority::Critical,
        TaskPriority::High,
        TaskPriority::Normal,
        TaskPriority::Low,
    ];

    for (i, priority) in priorities.iter().enumerate() {
        let task_id = format!("priority_test_{}", i);
        assert_ok!(harness.task_tracker.start_task(task_id.clone(), priority.clone()).await);

        // Verify task is registered with correct priority
        let registry = harness.task_tracker.tasks_registry.read().await;
        let task_info = registry.get(&task_id).unwrap();
        assert_eq!(task_info.priority, *priority);
        assert_eq!(task_info.status, TaskStatus::Running);
    }
}

#[tokio::test]
async fn test_task_cancellation_mechanisms() {
    let harness = create_test_harness().await;

    // Test task cancellation
    let (cancel_tx, cancel_rx) = oneshot::channel::<()>();
    let task_id = "cancellation_test".to_string();

    assert_ok!(harness.task_tracker.start_task(task_id.clone(), TaskPriority::Normal).await);

    let tracker = Arc::clone(&harness.task_tracker);
    let handle = tokio::spawn(async move {
        tokio::select! {
            _ = sleep(Duration::from_secs(10)) => {
                // This should never complete due to cancellation
                assert_ok!(tracker.complete_task(&task_id).await);
            }
            _ = cancel_rx => {
                // Task was cancelled
                assert_ok!(tracker.fail_task(&task_id).await);
            }
        }
    });

    // Cancel the task
    sleep(Duration::from_millis(5)).await;
    let _ = cancel_tx.send(());

    assert_ok!(handle.await);
    assert_eq!(harness.task_tracker.get_failed_count(), 1);
}

#[tokio::test]
async fn test_task_coordination_and_synchronization() {
    let harness = create_test_harness().await;

    let barrier = Arc::new(tokio::sync::Barrier::new(3));
    let completion_order = Arc::new(RwLock::new(Vec::new()));

    let mut handles = Vec::new();

    for i in 0..3 {
        let task_id = format!("sync_test_{}", i);
        assert_ok!(harness.task_tracker.start_task(task_id.clone(), TaskPriority::Normal).await);

        let barrier_clone = Arc::clone(&barrier);
        let order_clone = Arc::clone(&completion_order);
        let tracker = Arc::clone(&harness.task_tracker);

        let handle = tokio::spawn(async move {
            // All tasks wait at barrier
            barrier_clone.wait().await;

            // Record completion order
            order_clone.write().await.push(i);

            assert_ok!(tracker.complete_task(&task_id).await);
        });
        handles.push(handle);
    }

    for handle in handles {
        assert_ok!(handle.await);
    }

    // All tasks should have completed
    assert_eq!(harness.task_tracker.get_completed_count(), 3);

    // Check completion order was recorded
    let order = completion_order.read().await;
    assert_eq!(order.len(), 3);
}

// =============================================================================
// RESOURCE SHARING TESTS
// =============================================================================

#[tokio::test]
async fn test_semaphore_resource_limiting() {
    let harness = create_test_harness().await;

    // Test semaphore limits concurrent access
    let start_time = Instant::now();
    let mut handles = Vec::new();

    // Spawn more tasks than available permits
    for i in 0..20 {
        let pool = Arc::clone(&harness.resource_pool);
        let tracker = Arc::clone(&harness.task_tracker);
        let task_id = format!("semaphore_test_{}", i);

        assert_ok!(tracker.start_task(task_id.clone(), TaskPriority::Normal).await);

        let handle = tokio::spawn(async move {
            let _permit = assert_ok!(pool.acquire_permit().await);
            sleep(Duration::from_millis(10)).await; // Hold permit for 10ms
            assert_ok!(tracker.complete_task(&task_id).await);
        });
        handles.push(handle);
    }

    for handle in handles {
        assert_ok!(handle.await);
    }

    let elapsed = start_time.elapsed();

    // Should take longer due to permit limiting (10 permits, 20 tasks, 10ms each)
    assert!(elapsed >= Duration::from_millis(15));
    assert_eq!(harness.task_tracker.get_completed_count(), 20);
}

#[tokio::test]
async fn test_shared_state_synchronization() {
    let harness = create_test_harness().await;

    let shared_counter = Arc::new(AtomicUsize::new(0));
    let num_tasks = 50;
    let mut handles = Vec::new();

    for i in 0..num_tasks {
        let counter = Arc::clone(&shared_counter);
        let tracker = Arc::clone(&harness.task_tracker);
        let task_id = format!("shared_state_{}", i);

        assert_ok!(tracker.start_task(task_id.clone(), TaskPriority::Normal).await);

        let handle = tokio::spawn(async move {
            // Simulate work that modifies shared state
            for _ in 0..10 {
                counter.fetch_add(1, Ordering::SeqCst);
                sleep(Duration::from_micros(100)).await;
            }
            assert_ok!(tracker.complete_task(&task_id).await);
        });
        handles.push(handle);
    }

    for handle in handles {
        assert_ok!(handle.await);
    }

    // Verify all increments were applied correctly
    assert_eq!(shared_counter.load(Ordering::SeqCst), num_tasks * 10);
    assert_eq!(harness.task_tracker.get_completed_count(), num_tasks);
}

#[tokio::test]
async fn test_cross_task_communication_channels() {
    let harness = create_test_harness().await;

    // Test mpsc communication
    let (tx, mut rx) = mpsc::channel::<String>(100);

    // Producer task
    let producer_tracker = Arc::clone(&harness.task_tracker);
    let producer_handle = tokio::spawn(async move {
        let task_id = "producer_task".to_string();
        assert_ok!(producer_tracker.start_task(task_id.clone(), TaskPriority::Normal).await);

        for i in 0..10 {
            let message = format!("message_{}", i);
            assert_ok!(tx.send(message).await);
        }
        drop(tx); // Close channel

        assert_ok!(producer_tracker.complete_task(&task_id).await);
    });

    // Consumer task
    let consumer_tracker = Arc::clone(&harness.task_tracker);
    let consumer_handle = tokio::spawn(async move {
        let task_id = "consumer_task".to_string();
        assert_ok!(consumer_tracker.start_task(task_id.clone(), TaskPriority::Normal).await);

        let mut message_count = 0;
        while let Some(message) = rx.recv().await {
            assert!(message.starts_with("message_"));
            message_count += 1;
        }

        assert_eq!(message_count, 10);
        assert_ok!(consumer_tracker.complete_task(&task_id).await);
    });

    assert_ok!(producer_handle.await);
    assert_ok!(consumer_handle.await);
    assert_eq!(harness.task_tracker.get_completed_count(), 2);
}

// =============================================================================
// MEMORY MANAGEMENT TESTS
// =============================================================================

#[tokio::test]
async fn test_async_memory_allocation_tracking() {
    let harness = create_test_harness().await;

    // Test memory allocation tracking
    assert_eq!(harness.resource_pool.get_memory_usage(), 0);

    // Simulate memory allocations
    let allocation_sizes = [1024, 2048, 4096, 8192];
    let mut total_allocated = 0;

    for size in allocation_sizes {
        harness.resource_pool.allocate_memory(size);
        total_allocated += size;
        assert_eq!(harness.resource_pool.get_memory_usage(), total_allocated);
    }

    // Simulate memory deallocations
    for size in allocation_sizes {
        harness.resource_pool.deallocate_memory(size);
        total_allocated -= size;
        assert_eq!(harness.resource_pool.get_memory_usage(), total_allocated);
    }

    assert_eq!(harness.resource_pool.get_memory_usage(), 0);
}

#[tokio::test]
async fn test_async_resource_cleanup() {
    let harness = create_test_harness().await;

    // Create connections to test cleanup
    for i in 0..5 {
        let connection_id = format!("connection_{}", i);
        assert_ok!(harness.resource_pool.add_connection(connection_id).await);
    }

    // Verify connections were added
    {
        let connections = harness.resource_pool.connections.read().await;
        assert_eq!(connections.len(), 5);
    }

    // Simulate async cleanup task
    let pool = Arc::clone(&harness.resource_pool);
    let cleanup_handle = tokio::spawn(async move {
        sleep(Duration::from_millis(10)).await;

        let mut connections = pool.connections.write().await;
        connections.clear(); // Simulate resource cleanup
    });

    assert_ok!(cleanup_handle.await);

    // Verify cleanup completed
    {
        let connections = harness.resource_pool.connections.read().await;
        assert_eq!(connections.len(), 0);
    }
}

#[tokio::test]
async fn test_memory_leak_prevention() {
    let harness = create_test_harness().await;

    // Test that tasks don't leak memory
    let initial_memory = harness.resource_pool.get_memory_usage();

    let mut handles = Vec::new();
    for i in 0..100 {
        let task_id = format!("leak_test_{}", i);
        let pool = Arc::clone(&harness.resource_pool);
        let tracker = Arc::clone(&harness.task_tracker);

        assert_ok!(tracker.start_task(task_id.clone(), TaskPriority::Normal).await);

        let handle = tokio::spawn(async move {
            // Simulate temporary memory allocation
            pool.allocate_memory(1024);
            sleep(Duration::from_millis(1)).await;
            pool.deallocate_memory(1024); // Ensure cleanup

            assert_ok!(tracker.complete_task(&task_id).await);
        });
        handles.push(handle);
    }

    for handle in handles {
        assert_ok!(handle.await);
    }

    // Memory usage should return to initial state
    assert_eq!(harness.resource_pool.get_memory_usage(), initial_memory);
    assert_eq!(harness.task_tracker.get_completed_count(), 100);
}

// =============================================================================
// PERFORMANCE MONITORING TESTS
// =============================================================================

#[tokio::test]
async fn test_runtime_performance_monitoring() {
    let harness = create_test_harness().await;

    let start_time = Instant::now();
    let task_count = 1000;

    // Monitor task throughput
    let mut handles = Vec::new();
    for i in 0..task_count {
        let task_id = format!("perf_test_{}", i);
        let tracker = Arc::clone(&harness.task_tracker);

        assert_ok!(tracker.start_task(task_id.clone(), TaskPriority::Normal).await);

        let handle = tokio::spawn(async move {
            // Minimal work to test scheduling overhead
            assert_ok!(tracker.complete_task(&task_id).await);
        });
        handles.push(handle);
    }

    for handle in handles {
        assert_ok!(handle.await);
    }

    let elapsed = start_time.elapsed();
    let tasks_per_second = task_count as f64 / elapsed.as_secs_f64();

    // Should handle at least 1000 tasks per second
    assert!(tasks_per_second > 1000.0, "Task throughput too low: {:.2} tasks/sec", tasks_per_second);
    assert_eq!(harness.task_tracker.get_completed_count(), task_count);
}

#[tokio::test]
async fn test_runtime_latency_measurement() {
    let harness = create_test_harness().await;

    let mut latencies = Vec::new();

    for i in 0..100 {
        let task_id = format!("latency_test_{}", i);
        let tracker = Arc::clone(&harness.task_tracker);

        let task_start = Instant::now();
        assert_ok!(tracker.start_task(task_id.clone(), TaskPriority::Normal).await);

        let handle = tokio::spawn(async move {
            let spawn_latency = task_start.elapsed();
            assert_ok!(tracker.complete_task(&task_id).await);
            spawn_latency
        });

        let latency = assert_ok!(handle.await);
        latencies.push(latency);
    }

    // Calculate average latency
    let total_latency: Duration = latencies.iter().sum();
    let avg_latency = total_latency / latencies.len() as u32;

    // Task spawn latency should be reasonable (< 1ms)
    assert!(avg_latency < Duration::from_millis(1), "Average latency too high: {:?}", avg_latency);
}

// =============================================================================
// ERROR HANDLING TESTS
// =============================================================================

#[tokio::test]
async fn test_task_error_recovery() {
    let harness = create_test_harness().await;

    // Test that errors in one task don't affect others
    let task_id_good = "good_task".to_string();
    let task_id_bad = "bad_task".to_string();

    assert_ok!(harness.task_tracker.start_task(task_id_good.clone(), TaskPriority::Normal).await);
    assert_ok!(harness.task_tracker.start_task(task_id_bad.clone(), TaskPriority::Normal).await);

    let tracker_good = Arc::clone(&harness.task_tracker);
    let tracker_bad = Arc::clone(&harness.task_tracker);

    // Good task
    let good_handle = tokio::spawn(async move {
        sleep(Duration::from_millis(5)).await;
        assert_ok!(tracker_good.complete_task(&task_id_good).await);
    });

    // Bad task that panics
    let bad_handle = tokio::spawn(async move {
        sleep(Duration::from_millis(5)).await;
        assert_ok!(tracker_bad.fail_task(&task_id_bad).await);
        panic!("Simulated task failure");
    });

    // Good task should complete successfully
    assert_ok!(good_handle.await);

    // Bad task should fail but not crash runtime
    assert_err!(bad_handle.await);

    assert_eq!(harness.task_tracker.get_completed_count(), 1);
    assert_eq!(harness.task_tracker.get_failed_count(), 1);
}

#[tokio::test]
async fn test_timeout_handling() {
    let harness = create_test_harness().await;

    // Test task timeout
    let task_id = "timeout_test".to_string();
    assert_ok!(harness.task_tracker.start_task(task_id.clone(), TaskPriority::Normal).await);

    let tracker = Arc::clone(&harness.task_tracker);
    let task = async move {
        sleep(Duration::from_secs(10)).await; // Long running task
        assert_ok!(tracker.complete_task(&task_id).await);
    };

    // Apply timeout
    let result = timeout(Duration::from_millis(50), task).await;
    assert_err!(result); // Should timeout

    // Mark task as failed due to timeout
    assert_ok!(harness.task_tracker.fail_task(&task_id).await);
    assert_eq!(harness.task_tracker.get_failed_count(), 1);
}

// =============================================================================
// INTEGRATION WITH EXISTING DAEMON TESTS
// =============================================================================

#[tokio::test]
async fn test_daemon_with_runtime_integration() {
    let config = create_test_daemon_config();
    let daemon = assert_ok!(WorkspaceDaemon::new(config).await);

    // Test daemon integrates with our runtime utilities
    let harness = create_test_harness().await;

    // Test daemon processor with runtime tracking
    let task_id = "daemon_integration_test".to_string();
    assert_ok!(harness.task_tracker.start_task(task_id.clone(), TaskPriority::Normal).await);

    let processor = daemon.processor();
    let result = processor.process_document("test_integration.rs").await;
    assert_ok!(result);

    assert_ok!(harness.task_tracker.complete_task(&task_id).await);
    assert_eq!(harness.task_tracker.get_completed_count(), 1);
}

#[tokio::test]
async fn test_communication_hub_integration() {
    let harness = create_test_harness().await;

    // Test communication hub with task events
    let mut event_receiver = harness.communication_hub.task_events.subscribe();

    // Send task events
    let events = vec![
        TaskEvent::Started("task_1".to_string()),
        TaskEvent::Progress("task_1".to_string(), 0.5),
        TaskEvent::Completed("task_1".to_string()),
    ];

    for event in events.clone() {
        assert_ok!(harness.communication_hub.broadcast_task_event(event).await);
    }

    // Verify events were received
    for expected_event in events {
        let received_event = assert_ok!(event_receiver.recv().await);
        match (&received_event, &expected_event) {
            (TaskEvent::Started(r), TaskEvent::Started(e)) => assert_eq!(r, e),
            (TaskEvent::Progress(r, rp), TaskEvent::Progress(e, ep)) => {
                assert_eq!(r, e);
                assert!((rp - ep).abs() < f32::EPSILON);
            }
            (TaskEvent::Completed(r), TaskEvent::Completed(e)) => assert_eq!(r, e),
            _ => panic!("Event type mismatch"),
        }
    }
}

#[tokio::test]
async fn test_system_status_monitoring() {
    let harness = create_test_harness().await;

    let mut status_receiver = harness.communication_hub.status_updates.subscribe();

    // Test status transitions
    let statuses = vec![
        SystemStatus::Running,
        SystemStatus::Paused,
        SystemStatus::Running,
        SystemStatus::ShuttingDown,
        SystemStatus::Stopped,
    ];

    for status in statuses.clone() {
        assert_ok!(harness.communication_hub.update_status(status).await);
        let received_status = *status_receiver.borrow_and_update();
        // Note: watch channels may skip intermediate values, so we check the final state
    }

    // Final status should be Stopped
    let final_status = *status_receiver.borrow();
    assert_eq!(final_status, SystemStatus::Stopped);
}

// =============================================================================
// STRESS TESTS
// =============================================================================

#[tokio::test]
async fn test_high_concurrency_stress() {
    let harness = create_test_harness().await;

    let task_count = 10000;
    let mut handles = Vec::new();

    let start_time = Instant::now();

    for i in 0..task_count {
        let task_id = format!("stress_test_{}", i);
        let tracker = Arc::clone(&harness.task_tracker);

        assert_ok!(tracker.start_task(task_id.clone(), TaskPriority::Normal).await);

        let handle = tokio::spawn(async move {
            // Minimal work to test pure scheduling overhead
            assert_ok!(tracker.complete_task(&task_id).await);
        });
        handles.push(handle);
    }

    // Wait for all tasks to complete
    for handle in handles {
        assert_ok!(handle.await);
    }

    let elapsed = start_time.elapsed();

    assert_eq!(harness.task_tracker.get_completed_count(), task_count);

    // Should complete within reasonable time (< 5 seconds)
    assert!(elapsed < Duration::from_secs(5), "Stress test took too long: {:?}", elapsed);
}

#[tokio::test]
async fn test_resource_contention_handling() {
    let harness = create_test_harness().await;

    // Test heavy resource contention
    let contention_tasks = 500;
    let mut handles = Vec::new();

    for i in 0..contention_tasks {
        let task_id = format!("contention_test_{}", i);
        let tracker = Arc::clone(&harness.task_tracker);
        let pool = Arc::clone(&harness.resource_pool);

        assert_ok!(tracker.start_task(task_id.clone(), TaskPriority::Normal).await);

        let handle = tokio::spawn(async move {
            let _permit = assert_ok!(pool.acquire_permit().await);

            // Simulate work under contention
            sleep(Duration::from_millis(1)).await;

            assert_ok!(tracker.complete_task(&task_id).await);
        });
        handles.push(handle);
    }

    for handle in handles {
        assert_ok!(handle.await);
    }

    assert_eq!(harness.task_tracker.get_completed_count(), contention_tasks);
}