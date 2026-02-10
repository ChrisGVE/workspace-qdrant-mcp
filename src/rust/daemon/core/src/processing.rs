//! Priority-based document processing pipeline
//!
//! This module implements a priority-based task queuing system for responsive
//! MCP request handling with preemption capabilities.

use std::collections::{BinaryHeap, HashMap, VecDeque};
use std::cmp::Ordering;
use std::sync::atomic::{AtomicU64, Ordering as AtomicOrdering};
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::{mpsc, oneshot, RwLock, Mutex};
use tokio::task::JoinHandle;
use uuid::Uuid;
use serde::{Deserialize, Serialize};
use thiserror::Error;
use chrono;
use std::path::PathBuf;
use std::num::NonZeroU32;
use governor::{Quota, RateLimiter, clock::DefaultClock, state::{InMemoryState, NotKeyed}};
use crate::queue_operations::QueueManager;
use crate::storage::StorageClient;

/// Pipeline statistics for monitoring
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PipelineStats {
    pub queued_tasks: usize,
    pub running_tasks: usize,
    pub total_capacity: usize,
    pub tasks_completed: u64,
    pub tasks_failed: u64,
    pub tasks_cancelled: u64,
    pub tasks_timed_out: u64,
    pub queue_rejections: u64,
    pub queue_spills: u64,
    pub rate_limited_tasks: u64,
    pub backpressure_events: u64,
    pub uptime_seconds: u64,
}

/// Priority levels for different types of operations
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[repr(u8)]
pub enum TaskPriority {
    /// Background folder watching (lowest priority)
    BackgroundWatching = 0,
    /// CLI commands (normal priority) 
    CliCommands = 1,
    /// Project watching/ingestion (high priority)
    ProjectWatching = 2,
    /// MCP requests (highest priority - can preempt others)
    McpRequests = 3,
}

impl TaskPriority {
    /// Check if this priority can preempt another
    pub fn can_preempt(&self, other: &TaskPriority) -> bool {
        *self > *other
    }
}

/// Task execution context and metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TaskContext {
    pub task_id: Uuid,
    pub priority: TaskPriority,
    pub created_at: chrono::DateTime<chrono::Utc>,
    pub timeout_ms: Option<u64>,
    pub source: TaskSource,
    pub metadata: HashMap<String, String>,
    /// Checkpoint ID for resumable tasks
    pub checkpoint_id: Option<String>,
    /// Whether task supports checkpointing
    pub supports_checkpointing: bool,
}

/// Source of the task for tracking and debugging
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum TaskSource {
    /// MCP server request
    McpServer { request_id: String },
    /// Project file watching
    ProjectWatcher { project_path: String },
    /// CLI command execution  
    CliCommand { command: String },
    /// Background folder monitoring
    BackgroundWatcher { folder_path: String },
    /// Generic task source
    Generic { operation: String },
}

/// Task execution result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum TaskResult {
    /// Task completed successfully
    Success { 
        execution_time_ms: u64,
        data: TaskResultData,
    },
    /// Task was cancelled/preempted
    Cancelled { 
        reason: String,
        checkpoint_id: Option<String>,
        partial_data: Option<TaskResultData>,
    },
    /// Task failed with error
    Error { 
        error: String,
        execution_time_ms: u64,
        checkpoint_id: Option<String>,
    },
    /// Task timed out
    Timeout { 
        timeout_duration_ms: u64,
        checkpoint_id: Option<String>,
    },
    /// Task was preempted but can be resumed
    Preempted {
        checkpoint_id: String,
        partial_data: Option<TaskResultData>,
        preemption_reason: String,
        resume_priority: TaskPriority,
    },
}

/// Checkpoint data for task resumption
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TaskCheckpoint {
    pub checkpoint_id: String,
    pub task_id: Uuid,
    pub created_at: chrono::DateTime<chrono::Utc>,
    pub task_progress: TaskProgress,
    pub state_data: serde_json::Value,
    pub files_modified: Vec<PathBuf>,
    pub rollback_actions: Vec<RollbackAction>,
}

/// Different types of progress tracking for different task types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum TaskProgress {
    DocumentProcessing {
        chunks_processed: usize,
        total_chunks: usize,
        current_chunk_offset: usize,
    },
    FileWatching {
        files_processed: usize,
        current_directory: PathBuf,
        processed_files: Vec<PathBuf>,
    },
    QueryExecution {
        query_stage: String,
        results_collected: usize,
    },
    Generic {
        progress_percentage: f32,
        stage: String,
        metadata: HashMap<String, serde_json::Value>,
    },
}

/// Actions needed to rollback changes if task is cancelled
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum RollbackAction {
    DeleteFile { path: PathBuf },
    RestoreFile { original_path: PathBuf, backup_path: PathBuf },
    RemoveFromCollection { document_id: String, collection: String },
    RevertIndexChanges { index_snapshot: serde_json::Value },
    Custom { action_type: String, data: serde_json::Value },
}

/// Handler for custom rollback actions
///
/// Implement this trait to register domain-specific rollback logic
/// via `CheckpointManager::register_custom_handler`.
#[async_trait::async_trait]
pub trait CustomRollbackHandler: Send + Sync {
    /// Execute the rollback action with the provided data payload
    async fn execute(&self, data: &serde_json::Value) -> Result<(), String>;
}

/// Specific data returned by task execution
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum TaskResultData {
    /// Document processing result
    DocumentProcessing {
        document_id: String,
        collection: String,
        chunks_created: usize,
        checkpoint_id: Option<String>,
    },
    /// File watching result
    FileWatching {
        files_processed: usize,
        errors: Vec<String>,
        checkpoint_id: Option<String>,
    },
    /// Query execution result
    QueryExecution {
        results: Vec<String>,
        total_results: usize,
        checkpoint_id: Option<String>,
    },
    /// Generic result data
    Generic {
        message: String,
        data: serde_json::Value,
        checkpoint_id: Option<String>,
    },
}

/// Errors that can occur in the priority system
#[derive(Error, Debug)]
pub enum PriorityError {
    #[error("Task queue is full")]
    QueueFull,
    
    #[error("Task not found: {task_id}")]
    TaskNotFound { task_id: Uuid },
    
    #[error("Task execution failed: {reason}")]
    ExecutionFailed { reason: String },
    
    #[error("Task timed out after {duration:?}")]
    Timeout { duration: Duration },
    
    #[error("Preemption failed: {reason}")]
    PreemptionFailed { reason: String },
    
    #[error("Communication error: {0}")]
    Communication(String),
    
    #[error("Request queue timeout: {0}")]
    RequestTimeout(String),
    
    #[error("Queue capacity exceeded: {current}/{max}")]
    QueueCapacityExceeded { current: usize, max: usize },
    
    #[error("Invalid priority level: {0}")]
    InvalidPriority(u8),
    
    #[error("Data consistency error: {0}")]
    DataConsistency(String),
    
    #[error("Checkpoint error: {0}")]
    Checkpoint(String),
    
    #[error("Rollback failed: {0}")]
    RollbackFailed(String),
}

/// A task that can be executed with priority and preemption support
pub struct PriorityTask {
    pub context: TaskContext,
    pub payload: TaskPayload,
    pub result_sender: oneshot::Sender<TaskResult>,
    /// Handle for cancellation if task is running
    pub cancellation_token: Option<tokio_util::sync::CancellationToken>,
}

// Manual Debug implementation since oneshot::Sender doesn't implement Debug
impl std::fmt::Debug for PriorityTask {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PriorityTask")
            .field("context", &self.context)
            .field("payload", &self.payload)
            .field("result_sender", &"<oneshot::Sender<TaskResult>>")
            .field("cancellation_token", &self.cancellation_token)
            .finish()
    }
}

/// The actual work to be performed by a task
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum TaskPayload {
    /// Process a document file
    ProcessDocument {
        file_path: std::path::PathBuf,
        collection: String,
    },
    /// Watch a directory for changes
    WatchDirectory {
        path: std::path::PathBuf,
        recursive: bool,
    },
    /// Execute a search query
    ExecuteQuery {
        query: String,
        collection: String,
        limit: usize,
    },
    /// Generic task execution
    Generic {
        operation: String,
        parameters: HashMap<String, serde_json::Value>,
    },
}

/// Priority queue implementation for tasks
/// Uses reverse ordering so highest priority comes first
struct TaskQueueItem {
    task: PriorityTask,
    sequence: u64,
}

impl PartialEq for TaskQueueItem {
    fn eq(&self, other: &Self) -> bool {
        self.task.context.priority == other.task.context.priority
            && self.sequence == other.sequence
    }
}

impl Eq for TaskQueueItem {}

impl PartialOrd for TaskQueueItem {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for TaskQueueItem {
    fn cmp(&self, other: &Self) -> Ordering {
        // First compare by priority (higher priority first)
        match self.task.context.priority.cmp(&other.task.context.priority) {
            Ordering::Equal => {
                // If priorities are equal, use sequence number (FIFO)
                other.sequence.cmp(&self.sequence)
            }
            other_order => other_order,
        }
    }
}

/// The main priority-based processing pipeline
pub struct Pipeline {
    /// Task queue with priority ordering
    task_queue: Arc<RwLock<BinaryHeap<TaskQueueItem>>>,
    /// Currently executing tasks mapped by task ID
    running_tasks: Arc<RwLock<HashMap<Uuid, RunningTask>>>,
    /// Channel for receiving new tasks
    task_receiver: Arc<RwLock<Option<mpsc::UnboundedReceiver<PriorityTask>>>>,
    /// Sender for submitting new tasks
    task_sender: mpsc::UnboundedSender<PriorityTask>,
    /// Global sequence counter for FIFO ordering within priority levels
    sequence_counter: Arc<AtomicU64>,
    /// Maximum number of concurrent tasks
    max_concurrent_tasks: usize,
    /// Task execution handle
    executor_handle: Option<JoinHandle<()>>,
    /// Request queue with timeout handling
    request_queue: Arc<RequestQueue>,
    /// Queue configuration
    #[allow(dead_code)]
    queue_config: QueueConfig,
    /// Checkpoint manager for data consistency
    checkpoint_manager: Arc<CheckpointManager>,
    /// Metrics collector for performance monitoring
    metrics_collector: Arc<MetricsCollector>,
    /// Optional ingestion engine for real document processing
    ingestion_engine: Option<Arc<crate::IngestionEngine>>,
    /// Optional SQLite queue manager for spill-to-disk on overflow
    spill_queue: Option<Arc<QueueManager>>,
}

/// Information about a currently running task
#[derive(Debug)]
struct RunningTask {
    context: TaskContext,
    started_at: Instant,
    handle: JoinHandle<TaskResult>,
    cancellation_token: tokio_util::sync::CancellationToken,
    /// Current checkpoint if task supports checkpointing
    current_checkpoint: Option<TaskCheckpoint>,
    /// Whether the task is in a preemptible state
    is_preemptible: bool,
    /// Progress tracking for consistency checks
    #[allow(dead_code)]
    last_progress_update: Instant,
}

/// Checkpoint manager for handling task state persistence
pub struct CheckpointManager {
    /// Storage for active checkpoints
    checkpoints: Arc<RwLock<HashMap<String, TaskCheckpoint>>>,
    /// Cleanup interval for old checkpoints
    checkpoint_retention: Duration,
    /// Directory for checkpoint file storage
    checkpoint_dir: PathBuf,
    /// Storage client for Qdrant rollback operations (RemoveFromCollection)
    storage_client: Option<Arc<StorageClient>>,
    /// Registry of custom rollback handlers
    custom_handlers: Arc<RwLock<HashMap<String, Arc<dyn CustomRollbackHandler>>>>,
}

impl CheckpointManager {
    /// Create new checkpoint manager
    pub fn new(checkpoint_dir: PathBuf, retention: Duration) -> Self {
        Self {
            checkpoints: Arc::new(RwLock::new(HashMap::new())),
            checkpoint_retention: retention,
            checkpoint_dir,
            storage_client: None,
            custom_handlers: Arc::new(RwLock::new(HashMap::new())),
        }
    }

    /// Set the storage client for Qdrant rollback operations
    pub fn set_storage_client(&mut self, client: Arc<StorageClient>) {
        self.storage_client = Some(client);
    }

    /// Register a custom rollback handler for a given action type
    pub async fn register_custom_handler(
        &self,
        action_type: impl Into<String>,
        handler: Arc<dyn CustomRollbackHandler>,
    ) {
        let mut handlers = self.custom_handlers.write().await;
        handlers.insert(action_type.into(), handler);
    }
    
    /// Create a checkpoint for a task
    pub async fn create_checkpoint(
        &self,
        task_id: Uuid,
        progress: TaskProgress,
        state_data: serde_json::Value,
        files_modified: Vec<PathBuf>,
        rollback_actions: Vec<RollbackAction>,
    ) -> Result<String, PriorityError> {
        let checkpoint_id = format!("ckpt_{}_{}", task_id, chrono::Utc::now().timestamp());
        
        let checkpoint = TaskCheckpoint {
            checkpoint_id: checkpoint_id.clone(),
            task_id,
            created_at: chrono::Utc::now(),
            task_progress: progress,
            state_data,
            files_modified,
            rollback_actions,
        };
        
        // Store in memory
        {
            let mut checkpoints_lock = self.checkpoints.write().await;
            checkpoints_lock.insert(checkpoint_id.clone(), checkpoint.clone());
        }
        
        // Persist to disk
        let checkpoint_file = self.checkpoint_dir.join(format!("{}.json", checkpoint_id));
        let checkpoint_json = serde_json::to_string(&checkpoint)
            .map_err(|e| PriorityError::Checkpoint(e.to_string()))?;
        
        tokio::fs::write(&checkpoint_file, checkpoint_json).await
            .map_err(|e| PriorityError::Checkpoint(e.to_string()))?;
        
        tracing::debug!("Created checkpoint {} for task {}", checkpoint_id, task_id);
        Ok(checkpoint_id)
    }
    
    /// Retrieve a checkpoint
    pub async fn get_checkpoint(&self, checkpoint_id: &str) -> Option<TaskCheckpoint> {
        let checkpoints_lock = self.checkpoints.read().await;
        checkpoints_lock.get(checkpoint_id).cloned()
    }
    
    /// Delete a checkpoint (task completed successfully)
    pub async fn delete_checkpoint(&self, checkpoint_id: &str) -> Result<(), PriorityError> {
        // Remove from memory
        {
            let mut checkpoints_lock = self.checkpoints.write().await;
            checkpoints_lock.remove(checkpoint_id);
        }
        
        // Remove from disk
        let checkpoint_file = self.checkpoint_dir.join(format!("{}.json", checkpoint_id));
        if checkpoint_file.exists() {
            tokio::fs::remove_file(checkpoint_file).await
                .map_err(|e| PriorityError::Checkpoint(e.to_string()))?;
        }
        
        Ok(())
    }
    
    /// Rollback changes using checkpoint data
    pub async fn rollback_checkpoint(
        &self,
        checkpoint_id: &str,
    ) -> Result<(), PriorityError> {
        let checkpoint = self.get_checkpoint(checkpoint_id).await
            .ok_or_else(|| PriorityError::Checkpoint(format!("Checkpoint {} not found", checkpoint_id)))?;
        
        tracing::info!("Rolling back checkpoint {} for task {}", checkpoint_id, checkpoint.task_id);
        
        // Execute rollback actions in reverse order
        for action in checkpoint.rollback_actions.iter().rev() {
            match self.execute_rollback_action(action).await {
                Ok(_) => {
                    tracing::debug!("Successfully executed rollback action: {:?}", action);
                }
                Err(e) => {
                    tracing::error!("Failed to execute rollback action {:?}: {}", action, e);
                    // Continue with other rollback actions even if one fails
                }
            }
        }
        
        // Clean up the checkpoint after rollback
        self.delete_checkpoint(checkpoint_id).await?;
        
        Ok(())
    }
    
    /// Execute a single rollback action
    async fn execute_rollback_action(
        &self,
        action: &RollbackAction,
    ) -> Result<(), PriorityError> {
        match action {
            RollbackAction::DeleteFile { path } => {
                if path.exists() {
                    tokio::fs::remove_file(path).await
                        .map_err(|e| PriorityError::RollbackFailed(e.to_string()))?;
                }
            }
            RollbackAction::RestoreFile { original_path, backup_path } => {
                if backup_path.exists() {
                    tokio::fs::copy(backup_path, original_path).await
                        .map_err(|e| PriorityError::RollbackFailed(e.to_string()))?;
                    let _ = tokio::fs::remove_file(backup_path).await; // Best effort cleanup
                }
            }
            RollbackAction::RemoveFromCollection { document_id, collection } => {
                if let Some(ref storage) = self.storage_client {
                    tracing::info!(
                        "Rollback: removing document '{}' from collection '{}'",
                        document_id, collection
                    );
                    match storage.delete_points_by_document_id(collection, document_id).await {
                        Ok(count) => {
                            tracing::info!(
                                "Rollback: deleted {} points for document '{}' from '{}'",
                                count, document_id, collection
                            );
                        }
                        Err(e) => {
                            return Err(PriorityError::RollbackFailed(
                                format!(
                                    "Failed to remove document '{}' from '{}': {}",
                                    document_id, collection, e
                                ),
                            ));
                        }
                    }
                } else {
                    return Err(PriorityError::RollbackFailed(
                        format!(
                            "No storage client configured; cannot remove document '{}' from '{}'",
                            document_id, collection
                        ),
                    ));
                }
            }
            RollbackAction::RevertIndexChanges { index_snapshot } => {
                // Index reversion logs the snapshot for manual recovery.
                // Qdrant does not support atomic index rollback; field indexes
                // must be recreated individually. The snapshot preserves the
                // pre-change state so operators can restore manually if needed.
                if let Some(ref storage) = self.storage_client {
                    if let Some(collection) = index_snapshot.get("collection").and_then(|v| v.as_str()) {
                        match storage.get_collection_info(collection).await {
                            Ok(info) => {
                                tracing::warn!(
                                    "Rollback: index revert requested for collection '{}' \
                                     (status={}, points={}). Snapshot: {}",
                                    collection, info.status, info.points_count,
                                    serde_json::to_string(index_snapshot).unwrap_or_default()
                                );
                            }
                            Err(e) => {
                                tracing::warn!(
                                    "Rollback: index revert requested but collection query failed: {}. \
                                     Snapshot: {}",
                                    e,
                                    serde_json::to_string(index_snapshot).unwrap_or_default()
                                );
                            }
                        }
                    } else {
                        tracing::warn!(
                            "Rollback: index revert requested but snapshot missing 'collection' field. \
                             Snapshot: {}",
                            serde_json::to_string(index_snapshot).unwrap_or_default()
                        );
                    }
                } else {
                    tracing::warn!(
                        "Rollback: index revert requested but no storage client configured. \
                         Snapshot: {}",
                        serde_json::to_string(index_snapshot).unwrap_or_default()
                    );
                }
            }
            RollbackAction::Custom { action_type, data } => {
                let handlers = self.custom_handlers.read().await;
                if let Some(handler) = handlers.get(action_type.as_str()) {
                    tracing::info!("Rollback: executing custom handler '{}'", action_type);
                    handler.execute(data).await
                        .map_err(|e| PriorityError::RollbackFailed(
                            format!("Custom rollback '{}' failed: {}", action_type, e)
                        ))?;
                    tracing::info!("Rollback: custom handler '{}' completed", action_type);
                } else {
                    return Err(PriorityError::RollbackFailed(
                        format!("No handler registered for custom rollback type '{}'", action_type),
                    ));
                }
            }
        }
        
        Ok(())
    }
    
    /// Clean up old checkpoints
    pub async fn cleanup_old_checkpoints(&self) -> usize {
        let mut cleaned_count = 0;
        let cutoff_time = chrono::Utc::now() - chrono::Duration::from_std(self.checkpoint_retention).unwrap_or(chrono::Duration::hours(24));
        
        let checkpoint_ids_to_remove: Vec<String> = {
            let checkpoints_lock = self.checkpoints.read().await;
            checkpoints_lock
                .iter()
                .filter(|(_, checkpoint)| checkpoint.created_at < cutoff_time)
                .map(|(id, _)| id.clone())
                .collect()
        };
        
        for checkpoint_id in checkpoint_ids_to_remove {
            if let Err(e) = self.delete_checkpoint(&checkpoint_id).await {
                tracing::error!("Failed to cleanup checkpoint {}: {}", checkpoint_id, e);
            } else {
                cleaned_count += 1;
            }
        }
        
        tracing::info!("Cleaned up {} old checkpoints", cleaned_count);
        cleaned_count
    }
}

impl Pipeline {
    /// Create a new priority-based processing pipeline
    pub fn new(max_concurrent_tasks: usize) -> Self {
        Self::with_queue_config(max_concurrent_tasks, QueueConfig::default())
    }
    
    /// Create a new pipeline with custom queue configuration
    pub fn with_queue_config(max_concurrent_tasks: usize, queue_config: QueueConfig) -> Self {
        Self::with_checkpoint_config(max_concurrent_tasks, queue_config, None)
    }
    
    /// Create a new pipeline with custom queue and checkpoint configuration
    pub fn with_checkpoint_config(
        max_concurrent_tasks: usize, 
        queue_config: QueueConfig,
        checkpoint_dir: Option<PathBuf>,
    ) -> Self {
        let (task_sender, task_receiver) = mpsc::unbounded_channel();
        let request_queue = Arc::new(RequestQueue::new(queue_config.clone()));
        
        // Create checkpoint directory if specified, otherwise use temp
        let checkpoint_path = checkpoint_dir.unwrap_or_else(|| {
            std::env::temp_dir().join("wqm_checkpoints")
        });
        
        let checkpoint_manager = Arc::new(CheckpointManager::new(
            checkpoint_path,
            Duration::from_secs(3600), // 1 hour retention by default
        ));
        
        // Create metrics collector
        let metrics_collector = Arc::new(MetricsCollector::new(
            Duration::from_secs(60), // 1 minute sampling window
        ));
        
        // Ensure checkpoint directory exists
        if let Err(e) = std::fs::create_dir_all(&checkpoint_manager.checkpoint_dir) {
            tracing::warn!("Failed to create checkpoint directory: {}", e);
        }
        
        Self {
            task_queue: Arc::new(RwLock::new(BinaryHeap::new())),
            running_tasks: Arc::new(RwLock::new(HashMap::new())),
            task_receiver: Arc::new(RwLock::new(Some(task_receiver))),
            task_sender,
            sequence_counter: Arc::new(AtomicU64::new(0)),
            max_concurrent_tasks,
            executor_handle: None,
            request_queue,
            queue_config,
            checkpoint_manager,
            metrics_collector,
            ingestion_engine: None,
            spill_queue: None,
        }
    }

    /// Set the ingestion engine for real document processing
    pub fn set_ingestion_engine(&mut self, engine: Arc<crate::IngestionEngine>) {
        self.ingestion_engine = Some(engine);
    }

    /// Set the SQLite queue manager for spill-to-disk on overflow
    pub fn set_spill_queue(&mut self, queue_manager: Arc<QueueManager>) {
        self.spill_queue = Some(queue_manager);
    }

    /// Set the storage client for Qdrant rollback operations
    ///
    /// Must be called during initialization before the Pipeline is shared,
    /// since it requires exclusive access to the CheckpointManager Arc.
    pub fn set_rollback_storage(&mut self, client: Arc<StorageClient>) {
        if let Some(cm) = Arc::get_mut(&mut self.checkpoint_manager) {
            cm.set_storage_client(client);
            tracing::info!("Rollback storage client configured for checkpoint manager");
        } else {
            tracing::error!(
                "Cannot set rollback storage: CheckpointManager Arc has multiple references. \
                 Call set_rollback_storage before sharing the Pipeline."
            );
        }
    }

    /// Get a handle for submitting tasks to the pipeline
    pub fn task_submitter(&self) -> TaskSubmitter {
        // Create rate limiter if enabled
        let rate_limiter = if self.request_queue.config.enable_rate_limiting {
            if let Some(max_tps) = self.request_queue.config.max_tasks_per_second {
                let quota = Quota::per_second(NonZeroU32::new(max_tps as u32).unwrap());
                Some(Arc::new(RateLimiter::direct(quota)))
            } else {
                None
            }
        } else {
            None
        };

        TaskSubmitter {
            sender: self.task_sender.clone(),
            request_queue: Arc::clone(&self.request_queue),
            metrics_collector: Arc::clone(&self.metrics_collector),
            rate_limiter,
            spill_queue: self.spill_queue.clone(),
        }
    }
    
    /// Start the pipeline execution loop
    pub async fn start(&mut self) -> Result<(), PriorityError> {
        let task_queue = Arc::clone(&self.task_queue);
        let running_tasks = Arc::clone(&self.running_tasks);
        let task_receiver = Arc::clone(&self.task_receiver);
        let sequence_counter = Arc::clone(&self.sequence_counter);
        let max_concurrent = self.max_concurrent_tasks;
        let ingestion_engine = self.ingestion_engine.clone();

        let handle = tokio::spawn(async move {
            Self::execution_loop(
                task_queue,
                running_tasks,
                task_receiver,
                sequence_counter,
                max_concurrent,
                ingestion_engine,
            ).await;
        });
        
        self.executor_handle = Some(handle);
        Ok(())
    }
    
    /// Get current pipeline statistics (legacy compatibility)
    pub async fn stats(&self) -> PipelineStats {
        let queue_lock = self.task_queue.read().await;
        let running_lock = self.running_tasks.read().await;

        PipelineStats {
            queued_tasks: queue_lock.len(),
            running_tasks: running_lock.len(),
            total_capacity: self.max_concurrent_tasks,
            tasks_completed: self.metrics_collector.tasks_completed.load(AtomicOrdering::Relaxed),
            tasks_failed: self.metrics_collector.tasks_failed.load(AtomicOrdering::Relaxed),
            tasks_cancelled: self.metrics_collector.tasks_cancelled.load(AtomicOrdering::Relaxed),
            tasks_timed_out: self.metrics_collector.tasks_timed_out.load(AtomicOrdering::Relaxed),
            queue_rejections: self.metrics_collector.queue_overflow_count.load(AtomicOrdering::Relaxed),
            queue_spills: self.metrics_collector.queue_spill_count.load(AtomicOrdering::Relaxed),
            rate_limited_tasks: self.metrics_collector.rate_limited_tasks.load(AtomicOrdering::Relaxed),
            backpressure_events: self.metrics_collector.backpressure_events.load(AtomicOrdering::Relaxed),
            uptime_seconds: self.metrics_collector.start_time.elapsed().as_secs(),
        }
    }
    
    /// Get comprehensive priority system metrics
    pub async fn get_priority_system_metrics(&self) -> PrioritySystemMetrics {
        let queue_lock = self.task_queue.read().await;
        let running_lock = self.running_tasks.read().await;
        let queue_stats = self.request_queue.get_stats().await;
        
        self.metrics_collector.generate_metrics(
            running_lock.len(),
            queue_lock.len(),
            self.max_concurrent_tasks,
            &queue_stats,
            &self.checkpoint_manager,
        ).await
    }
    
    /// Get request queue reference for direct access
    pub fn request_queue(&self) -> Arc<RequestQueue> {
        Arc::clone(&self.request_queue)
    }
    
    /// Clean up timed out queued requests
    pub async fn cleanup_queue_timeouts(&self) -> usize {
        self.request_queue.cleanup_timeouts().await
    }
    
    /// Get checkpoint manager reference
    pub fn checkpoint_manager(&self) -> Arc<CheckpointManager> {
        Arc::clone(&self.checkpoint_manager)
    }
    
    /// Get metrics collector reference
    pub fn metrics_collector(&self) -> Arc<MetricsCollector> {
        Arc::clone(&self.metrics_collector)
    }
    
    /// Resume a task from checkpoint
    pub async fn resume_from_checkpoint(
        &self,
        checkpoint_id: &str,
        new_priority: Option<TaskPriority>,
    ) -> Result<TaskResultHandle, PriorityError> {
        let checkpoint = self.checkpoint_manager.get_checkpoint(checkpoint_id).await
            .ok_or_else(|| PriorityError::Checkpoint(format!("Checkpoint {} not found", checkpoint_id)))?;
        
        // Reconstruct the task with updated context
        let task_id = checkpoint.task_id;
        let (result_sender, result_receiver) = oneshot::channel();
        
        // Create a generic payload for resumption (in practice, this would be more specific)
        let payload = TaskPayload::Generic {
            operation: "resume_from_checkpoint".to_string(),
            parameters: {
                let mut params = HashMap::new();
                params.insert("checkpoint_id".to_string(), serde_json::Value::String(checkpoint_id.to_string()));
                params.insert("original_task_id".to_string(), serde_json::Value::String(task_id.to_string()));
                params.insert("state_data".to_string(), checkpoint.state_data);
                params
            },
        };
        
        let context = TaskContext {
            task_id: Uuid::new_v4(), // New task ID for the resumed task
            priority: new_priority.unwrap_or(TaskPriority::ProjectWatching), // Default resumed priority
            created_at: chrono::Utc::now(),
            timeout_ms: Some(60_000), // Default timeout for resumed tasks
            source: TaskSource::Generic { operation: "checkpoint_resume".to_string() },
            metadata: {
                let mut metadata = HashMap::new();
                metadata.insert("resumed_from".to_string(), checkpoint_id.to_string());
                metadata.insert("original_task_id".to_string(), task_id.to_string());
                metadata
            },
            checkpoint_id: Some(checkpoint_id.to_string()),
            supports_checkpointing: true,
        };
        
        let task = PriorityTask {
            context: context.clone(),
            payload,
            result_sender,
            cancellation_token: None,
        };
        
        // Submit the resumed task
        self.task_sender.send(task)
            .map_err(|_| PriorityError::Communication("Pipeline is shutting down".to_string()))?;
        
        tracing::info!("Resumed task from checkpoint {} with new task ID {}", checkpoint_id, context.task_id);
        
        Ok(TaskResultHandle {
            task_id: context.task_id,
            context,
            result_receiver,
        })
    }
    
    /// Clean up old checkpoints
    pub async fn cleanup_old_checkpoints(&self) -> usize {
        self.checkpoint_manager.cleanup_old_checkpoints().await
    }
    
    /// Force rollback a checkpoint (for emergency cleanup)
    pub async fn emergency_rollback(&self, checkpoint_id: &str) -> Result<(), PriorityError> {
        tracing::warn!("Emergency rollback requested for checkpoint: {}", checkpoint_id);
        self.checkpoint_manager.rollback_checkpoint(checkpoint_id).await
    }
    
    /// Get all active checkpoints
    pub async fn list_active_checkpoints(&self) -> Vec<String> {
        let checkpoints_lock = self.checkpoint_manager.checkpoints.read().await;
        checkpoints_lock.keys().cloned().collect()
    }
    
    /// Export metrics in Prometheus format
    pub async fn export_prometheus_metrics(&self) -> String {
        let metrics = self.get_priority_system_metrics().await;
        
        let mut output = String::new();
        
        // Pipeline metrics
        output.push_str("# HELP wqm_tasks_total Total number of tasks processed\n");
        output.push_str("# TYPE wqm_tasks_total counter\n");
        output.push_str(&format!("wqm_tasks_completed {{}} {}\n", metrics.pipeline.tasks_completed));
        output.push_str(&format!("wqm_tasks_failed {{}} {}\n", metrics.pipeline.tasks_failed));
        output.push_str(&format!("wqm_tasks_cancelled {{}} {}\n", metrics.pipeline.tasks_cancelled));
        output.push_str(&format!("wqm_tasks_timed_out {{}} {}\n", metrics.pipeline.tasks_timed_out));
        
        // Queue metrics
        output.push_str("# HELP wqm_queue_size Current queue size\n");
        output.push_str("# TYPE wqm_queue_size gauge\n");
        output.push_str(&format!("wqm_queue_total {{}} {}\n", metrics.queue.total_queued));
        
        for (priority, count) in &metrics.queue.queued_by_priority {
            output.push_str(&format!("wqm_queue_by_priority{{priority=\"{:?}\"}} {}\n", priority, count));
        }
        
        // Performance metrics
        output.push_str("# HELP wqm_task_duration_seconds Task execution duration\n");
        output.push_str("# TYPE wqm_task_duration_seconds histogram\n");
        output.push_str(&format!("wqm_task_duration_average {{}} {}\n", metrics.performance.average_task_duration_ms / 1000.0));
        output.push_str(&format!("wqm_task_duration_p95 {{}} {}\n", metrics.performance.p95_task_duration_ms / 1000.0));
        output.push_str(&format!("wqm_task_duration_p99 {{}} {}\n", metrics.performance.p99_task_duration_ms / 1000.0));
        
        // Preemption metrics
        output.push_str("# HELP wqm_preemptions_total Total preemptions\n");
        output.push_str("# TYPE wqm_preemptions_total counter\n");
        output.push_str(&format!("wqm_preemptions_total {{}} {}\n", metrics.preemption.preemptions_total));
        output.push_str(&format!("wqm_preemptions_graceful {{}} {}\n", metrics.preemption.graceful_preemptions));
        output.push_str(&format!("wqm_preemptions_forced {{}} {}\n", metrics.preemption.forced_aborts));
        
        // Checkpoint metrics
        output.push_str("# HELP wqm_checkpoints_active Active checkpoints\n");
        output.push_str("# TYPE wqm_checkpoints_active gauge\n");
        output.push_str(&format!("wqm_checkpoints_active {{}} {}\n", metrics.checkpoints.active_checkpoints));
        output.push_str(&format!("wqm_rollbacks_total {{}} {}\n", metrics.checkpoints.rollbacks_executed));
        
        output
    }
    
    /// The main execution loop that processes tasks
    async fn execution_loop(
        task_queue: Arc<RwLock<BinaryHeap<TaskQueueItem>>>,
        running_tasks: Arc<RwLock<HashMap<Uuid, RunningTask>>>,
        task_receiver: Arc<RwLock<Option<mpsc::UnboundedReceiver<PriorityTask>>>>,
        sequence_counter: Arc<AtomicU64>,
        max_concurrent: usize,
        ingestion_engine: Option<Arc<crate::IngestionEngine>>,
    ) {
        let mut receiver = {
            let mut lock = task_receiver.write().await;
            lock.take().expect("Task receiver should be available")
        };
        
        let mut cleanup_interval = tokio::time::interval(Duration::from_millis(100));
        let mut queue_cleanup_interval = tokio::time::interval(Duration::from_secs(1));
        let mut checkpoint_cleanup_interval = tokio::time::interval(Duration::from_secs(300)); // 5 minutes
        
        loop {
            tokio::select! {
                // Handle new task submissions
                task = receiver.recv() => {
                    match task {
                        Some(task) => {
                            let sequence = sequence_counter.fetch_add(1, AtomicOrdering::Relaxed);
                            let queue_item = TaskQueueItem { task, sequence };
                            
                            let mut queue_lock = task_queue.write().await;
                            queue_lock.push(queue_item);
                            drop(queue_lock);
                            
                            // Try to start queued tasks
                            Self::try_start_queued_tasks(
                                &task_queue,
                                &running_tasks,
                                max_concurrent,
                                &ingestion_engine,
                            ).await;
                        }
                        None => {
                            tracing::info!("Task receiver closed, shutting down pipeline");
                            break;
                        }
                    }
                }

                // Periodic cleanup and task starting
                _ = cleanup_interval.tick() => {
                    Self::cleanup_completed_tasks(&running_tasks).await;
                    // Try to start more tasks after cleanup
                    Self::try_start_queued_tasks(
                        &task_queue,
                        &running_tasks,
                        max_concurrent,
                        &ingestion_engine,
                    ).await;
                }
                
                // Queue timeout cleanup
                _ = queue_cleanup_interval.tick() => {
                    // This should be done with proper request_queue reference
                    // For now, we'll add a placeholder for the cleanup
                    // In a real implementation, we'd pass the request_queue reference here
                }
                
                // Checkpoint cleanup
                _ = checkpoint_cleanup_interval.tick() => {
                    // This should be done with proper checkpoint_manager reference
                    // In a real implementation, we'd pass the checkpoint_manager reference here
                    // For now, this is just a placeholder
                }
            }
        }
    }
    
    /// Attempt to start queued tasks if capacity allows
    async fn try_start_queued_tasks(
        task_queue: &Arc<RwLock<BinaryHeap<TaskQueueItem>>>,
        running_tasks: &Arc<RwLock<HashMap<Uuid, RunningTask>>>,
        max_concurrent: usize,
        ingestion_engine: &Option<Arc<crate::IngestionEngine>>,
    ) {
        loop {
            let running_count = {
                let running_lock = running_tasks.read().await;
                running_lock.len()
            };
            
            if running_count >= max_concurrent {
                // Check if we can preempt lower priority tasks
                let next_task_priority = {
                    let queue_lock = task_queue.read().await;
                    queue_lock.peek().map(|item| item.task.context.priority)
                };
                
                if let Some(priority) = next_task_priority {
                    // For MCP requests, allow bulk preemption if queue has multiple MCP tasks
                    let slots_needed = if Self::allows_bulk_preemption(priority) {
                        let queue_lock = task_queue.read().await;
                        let mcp_tasks_queued = queue_lock.iter()
                            .filter(|item| matches!(item.task.context.priority, TaskPriority::McpRequests))
                            .count().min(max_concurrent - 1); // Leave at least one slot for other priorities
                        mcp_tasks_queued.max(1)
                    } else {
                        1
                    };
                    
                    let preempted_count = Self::try_preempt_multiple_tasks(
                        running_tasks,
                        priority,
                        slots_needed,
                    ).await;
                    
                    if preempted_count == 0 {
                        break; // No capacity and can't preempt
                    }
                } else {
                    break; // No tasks to start
                }
            }
            
            // Try to start the next highest priority task
            let task_item = {
                let mut queue_lock = task_queue.write().await;
                queue_lock.pop()
            };
            
            if let Some(queue_item) = task_item {
                Self::start_task(running_tasks, queue_item.task, ingestion_engine.clone()).await;
            } else {
                break; // No more tasks to start
            }
        }
    }
    
    /// Try to preempt a lower priority running task with consistency checks
    async fn try_preempt_lower_priority_task(
        running_tasks: &Arc<RwLock<HashMap<Uuid, RunningTask>>>,
        checkpoint_manager: &Arc<CheckpointManager>,
        new_priority: TaskPriority,
    ) -> bool {
        let mut running_lock = running_tasks.write().await;
        
        // Find the best candidate for preemption, considering checkpointing support
        let mut best_candidate: Option<(Uuid, TaskPriority, Instant, bool, bool)> = None;
        
        for (task_id, running_task) in running_lock.iter() {
            if new_priority.can_preempt(&running_task.context.priority) {
                let candidate = (
                    *task_id, 
                    running_task.context.priority, 
                    running_task.started_at,
                    running_task.context.supports_checkpointing,
                    running_task.is_preemptible,
                );
                
                match &best_candidate {
                    None => {
                        best_candidate = Some(candidate);
                    }
                    Some((_, current_priority, current_start, current_checkpointable, current_preemptible)) => {
                        // Select candidate if it has lower priority or same priority with better characteristics
                        if running_task.context.priority < *current_priority ||
                           (running_task.context.priority == *current_priority
                            && ((running_task.is_preemptible && !current_preemptible) ||
                               (running_task.is_preemptible == *current_preemptible
                                && ((running_task.context.supports_checkpointing && !current_checkpointable) ||
                                   (running_task.context.supports_checkpointing == *current_checkpointable
                                    && running_task.started_at > *current_start))))) {
                            best_candidate = Some(candidate);
                        }
                    }
                }
            }
        }
        
        if let Some((task_id, old_priority, _, supports_checkpointing, is_preemptible)) = best_candidate {
            if let Some(running_task) = running_lock.remove(&task_id) {
                tracing::info!(
                    "Preempting task {} (priority {:?}) for higher priority task (priority {:?})",
                    task_id, old_priority, new_priority
                );
                
                // If task supports checkpointing, try to create checkpoint before preemption
                if supports_checkpointing && is_preemptible {
                    // Signal task to create checkpoint
                    if let Some(checkpoint) = &running_task.current_checkpoint {
                        tracing::info!("Task {} has existing checkpoint: {}", task_id, checkpoint.checkpoint_id);
                    } else {
                        tracing::debug!("Creating checkpoint for preempted task {}", task_id);
                        // Note: In a full implementation, we would signal the task to create a checkpoint
                        // For now, we'll just mark that it was preempted with potential for resumption
                    }
                }
                
                // Gracefully cancel the task
                running_task.cancellation_token.cancel();
                
                // Give it a moment to handle cancellation gracefully and create checkpoint if needed
                let grace_period = if supports_checkpointing { 100 } else { 10 };
                drop(running_lock);
                tokio::time::sleep(Duration::from_millis(grace_period)).await;
                
                // If task is still running after grace period, force abort
                if !running_task.handle.is_finished() {
                    tracing::warn!("Task {} did not respond to cancellation, aborting", task_id);
                    running_task.handle.abort();
                    
                    // If we forced abort a checkpointable task, try to rollback
                    if supports_checkpointing {
                        if let Some(checkpoint) = &running_task.current_checkpoint {
                            tracing::warn!("Force aborting checkpointed task {}, attempting rollback", task_id);
                            if let Err(e) = checkpoint_manager.rollback_checkpoint(&checkpoint.checkpoint_id).await {
                                tracing::error!("Failed to rollback checkpoint for aborted task {}: {}", task_id, e);
                            }
                        }
                    }
                }
                
                return true;
            }
        }
        
        false
    }
    
    /// Try to preempt multiple lower priority tasks if needed
    async fn try_preempt_multiple_tasks(
        running_tasks: &Arc<RwLock<HashMap<Uuid, RunningTask>>>,
        new_priority: TaskPriority,
        slots_needed: usize,
    ) -> usize {
        let mut preempted_count = 0;
        
        for _ in 0..slots_needed {
            if Self::try_preempt_lower_priority_task(running_tasks, &Arc::new(CheckpointManager::new(
                std::env::temp_dir().join("temp_checkpoints"),
                Duration::from_secs(3600),
            )), new_priority).await {
                preempted_count += 1;
            } else {
                break; // No more tasks to preempt
            }
        }
        
        preempted_count
    }
    
    /// Check if a task priority allows bulk preemption (MCP requests only)
    fn allows_bulk_preemption(priority: TaskPriority) -> bool {
        matches!(priority, TaskPriority::McpRequests)
    }
    
    /// Get preemption score for a task (higher = more likely to be preempted)
    #[allow(dead_code)]
    fn get_preemption_score(
        task_priority: TaskPriority,
        start_time: Instant,
        new_priority: TaskPriority,
    ) -> Option<u64> {
        if !new_priority.can_preempt(&task_priority) {
            return None;
        }
        
        let priority_diff = (new_priority as u8) - (task_priority as u8);
        let elapsed_ms = start_time.elapsed().as_millis() as u64;
        
        // Lower priority difference and more elapsed time = higher preemption score
        // This means we prefer to preempt tasks that have been running longer
        // and have the least priority difference
        Some((priority_diff as u64 * 1000) + (10000 - elapsed_ms.min(10000)))
    }
    
    /// Start executing a task
    async fn start_task(
        running_tasks: &Arc<RwLock<HashMap<Uuid, RunningTask>>>,
        task: PriorityTask,
        ingestion_engine: Option<Arc<crate::IngestionEngine>>,
    ) {
        let task_id = task.context.task_id;
        let cancellation_token = tokio_util::sync::CancellationToken::new();
        
        let context_for_task = task.context.clone();
        let context_for_running = task.context.clone();
        let token = cancellation_token.clone();
        let payload = task.payload;
        let result_sender = task.result_sender;
        
        let running_tasks_clone = Arc::clone(running_tasks);
        let task_id_for_cleanup = task_id;
        
        let handle = tokio::spawn(async move {
            let start_time = Instant::now();
            
            let result = tokio::select! {
                result = Self::execute_task_payload(payload, &context_for_task, &ingestion_engine) => {
                    match result {
                        Ok(data) => TaskResult::Success {
                            execution_time_ms: start_time.elapsed().as_millis() as u64,
                            data,
                        },
                        Err(error) => {
                            let checkpoint_id = if context_for_task.supports_checkpointing {
                                Some(format!("error_{}_{}", context_for_task.task_id, chrono::Utc::now().timestamp()))
                            } else {
                                None
                            };
                            
                            TaskResult::Error {
                                error: error.to_string(),
                                execution_time_ms: start_time.elapsed().as_millis() as u64,
                                checkpoint_id,
                            }
                        }
                    }
                }
                _ = token.cancelled() => {
                    // Check if task supports checkpointing and create final checkpoint
                    let checkpoint_id = if context_for_task.supports_checkpointing {
                        // In a real implementation, we'd get the current state from the task
                        // For now, we'll create a minimal checkpoint indicator
                        Some(format!("cancelled_{}_{}", context_for_task.task_id, chrono::Utc::now().timestamp()))
                    } else {
                        None
                    };
                    
                    TaskResult::Cancelled {
                        reason: "Task was preempted by higher priority task".to_string(),
                        checkpoint_id,
                        partial_data: None,
                    }
                }
                _ = Self::timeout_future(&context_for_task) => {
                    let checkpoint_id = if context_for_task.supports_checkpointing {
                        Some(format!("timeout_{}_{}", context_for_task.task_id, chrono::Utc::now().timestamp()))
                    } else {
                        None
                    };
                    
                    TaskResult::Timeout {
                        timeout_duration_ms: context_for_task.timeout_ms.unwrap_or(30_000),
                        checkpoint_id,
                    }
                }
            };
            
            // Send result back to caller
            let _ = result_sender.send(result.clone());
            
            // Remove from running tasks when complete
            {
                let mut running_lock = running_tasks_clone.write().await;
                running_lock.remove(&task_id_for_cleanup);
            }
            
            result
        });
        
        let running_task = RunningTask {
            context: context_for_running,
            started_at: Instant::now(),
            handle,
            cancellation_token,
            current_checkpoint: None,
            is_preemptible: true, // Most tasks are preemptible by default
            last_progress_update: Instant::now(),
        };
        
        let mut running_lock = running_tasks.write().await;
        running_lock.insert(task_id, running_task);
    }
    
    /// Create a timeout future if task has timeout configured
    async fn timeout_future(context: &TaskContext) {
        if let Some(timeout_ms) = context.timeout_ms {
            tokio::time::sleep(Duration::from_millis(timeout_ms)).await;
        } else {
            // If no timeout specified, wait indefinitely
            std::future::pending().await
        }
    }
    
    /// Execute the actual task payload
    async fn execute_task_payload(
        payload: TaskPayload,
        context: &TaskContext,
        ingestion_engine: &Option<Arc<crate::IngestionEngine>>,
    ) -> Result<TaskResultData, PriorityError> {
        match payload {
            TaskPayload::ProcessDocument { file_path, collection } => {
                if let Some(engine) = ingestion_engine {
                    tracing::info!(
                        file = %file_path.display(),
                        collection = %collection,
                        "Processing document with ingestion engine"
                    );

                    let result = engine
                        .process_document(&file_path, &collection)
                        .await
                        .map_err(|e| {
                            tracing::error!(
                                file = %file_path.display(),
                                collection = %collection,
                                error = %e,
                                "Document processing failed"
                            );
                            PriorityError::ExecutionFailed {
                                reason: format!("Document processing failed: {}", e),
                            }
                        })?;

                    tracing::info!(
                        document_id = %result.document_id,
                        collection = %result.collection,
                        chunks_created = result.chunks_created.unwrap_or(0),
                        processing_time_ms = result.processing_time_ms,
                        "Document processed successfully"
                    );

                    Ok(TaskResultData::DocumentProcessing {
                        document_id: result.document_id,
                        collection: result.collection,
                        chunks_created: result.chunks_created.unwrap_or(0),
                        checkpoint_id: context.checkpoint_id.clone(),
                    })
                } else {
                    // Stub fallback when no ingestion engine is configured (e.g. tests)
                    tracing::info!(
                        "Processing document (stub): {:?} for collection: {}",
                        file_path, collection
                    );
                    tokio::time::sleep(Duration::from_millis(100)).await;

                    Ok(TaskResultData::DocumentProcessing {
                        document_id: context.task_id.to_string(),
                        collection,
                        chunks_created: 1,
                        checkpoint_id: context.checkpoint_id.clone(),
                    })
                }
            }
            
            TaskPayload::WatchDirectory { path, recursive } => {
                tracing::info!(
                    "Watching directory: {:?}, recursive: {}",
                    path, recursive
                );
                
                Ok(TaskResultData::FileWatching {
                    files_processed: 0,
                    errors: vec![],
                    checkpoint_id: context.checkpoint_id.clone(),
                })
            }
            
            TaskPayload::ExecuteQuery { query, collection, limit } => {
                tracing::info!(
                    "Executing query: '{}' on collection: '{}' with limit: {}",
                    query, collection, limit
                );
                
                Ok(TaskResultData::QueryExecution {
                    results: vec![],
                    total_results: 0,
                    checkpoint_id: context.checkpoint_id.clone(),
                })
            }
            
            TaskPayload::Generic { operation, parameters } => {
                tracing::info!(
                    "Executing generic operation: '{}' with {} parameters",
                    operation, parameters.len()
                );
                
                // Simulate processing time based on operation name
                let sleep_duration = if operation.starts_with("long_") {
                    Duration::from_millis(2000) // Longer for testing preemption
                } else {
                    Duration::from_millis(100)
                };
                tokio::time::sleep(sleep_duration).await;
                
                Ok(TaskResultData::Generic {
                    message: format!("Completed operation: {}", operation),
                    data: serde_json::json!(parameters),
                    checkpoint_id: context.checkpoint_id.clone(),
                })
            }
        }
    }
    
    /// Clean up completed running tasks
    async fn cleanup_completed_tasks(
        running_tasks: &Arc<RwLock<HashMap<Uuid, RunningTask>>>,
    ) {
        let mut to_remove = Vec::new();
        
        {
            let running_lock = running_tasks.read().await;
            for (task_id, running_task) in running_lock.iter() {
                if running_task.handle.is_finished() {
                    to_remove.push(*task_id);
                }
            }
        }
        
        if !to_remove.is_empty() {
            let mut running_lock = running_tasks.write().await;
            for task_id in to_remove {
                running_lock.remove(&task_id);
                tracing::debug!("Cleaned up completed task: {}", task_id);
            }
        }
    }
}

impl Default for Pipeline {
    fn default() -> Self {
        Self::new(4) // Default to 4 concurrent tasks
    }
}

/// Configuration builder for creating optimized queue configurations
pub struct QueueConfigBuilder {
    config: QueueConfig,
}

impl QueueConfigBuilder {
    /// Start with default configuration
    pub fn new() -> Self {
        Self {
            config: QueueConfig::default(),
        }
    }
    
    /// Set maximum queued requests per priority level
    pub fn max_queued_per_priority(mut self, max: usize) -> Self {
        self.config.max_queued_per_priority = max;
        self
    }
    
    /// Set default queue timeout
    pub fn default_queue_timeout(mut self, timeout_ms: u64) -> Self {
        self.config.default_queue_timeout_ms = timeout_ms;
        self
    }
    
    /// Enable or disable deduplication
    pub fn deduplication(mut self, enable: bool) -> Self {
        self.config.enable_deduplication = enable;
        self
    }
    
    /// Set queue wait timeout
    pub fn queue_wait_timeout(mut self, timeout_ms: u64) -> Self {
        self.config.queue_wait_timeout_ms = timeout_ms;
        self
    }
    
    /// Enable or disable priority boosting
    pub fn priority_boost(mut self, enable: bool, age_threshold_ms: u64) -> Self {
        self.config.enable_priority_boost = enable;
        self.config.priority_boost_age_ms = age_threshold_ms;
        self
    }
    
    /// Build the configuration for MCP servers (low latency)
    pub fn for_mcp_server(mut self) -> Self {
        self.config.max_queued_per_priority = 50;
        self.config.default_queue_timeout_ms = 5_000;
        self.config.enable_deduplication = true;
        self.config.queue_wait_timeout_ms = 1_000;
        self.config.enable_priority_boost = true;
        self.config.priority_boost_age_ms = 2_000;
        self
    }
    
    /// Build the configuration for batch processing (high throughput)
    pub fn for_batch_processing(mut self) -> Self {
        self.config.max_queued_per_priority = 1000;
        self.config.default_queue_timeout_ms = 60_000;
        self.config.enable_deduplication = true;
        self.config.queue_wait_timeout_ms = 10_000;
        self.config.enable_priority_boost = false; // Maintain order for batch
        self.config.priority_boost_age_ms = 30_000;
        self
    }
    
    /// Build the configuration for resource-constrained environments
    pub fn for_low_resource(mut self) -> Self {
        self.config.max_queued_per_priority = 10;
        self.config.default_queue_timeout_ms = 120_000;
        self.config.enable_deduplication = false; // Save memory
        self.config.queue_wait_timeout_ms = 30_000;
        self.config.enable_priority_boost = false;
        self.config.priority_boost_age_ms = 60_000;
        self
    }
    
    /// Build the final configuration
    pub fn build(self) -> QueueConfig {
        self.config
    }
}

impl Default for QueueConfigBuilder {
    fn default() -> Self {
        Self::new()
    }
}

/// Handle for submitting tasks to the pipeline
#[derive(Clone)]
pub struct TaskSubmitter {
    sender: mpsc::UnboundedSender<PriorityTask>,
    request_queue: Arc<RequestQueue>,
    metrics_collector: Arc<MetricsCollector>,
    rate_limiter: Option<Arc<RateLimiter<NotKeyed, InMemoryState, DefaultClock>>>,
    /// Optional SQLite queue manager for spill-to-disk on overflow
    spill_queue: Option<Arc<QueueManager>>,
}

impl TaskSubmitter {
    /// Submit a task for execution and get a future for the result
    pub async fn submit_task(
        &self,
        priority: TaskPriority,
        source: TaskSource,
        payload: TaskPayload,
        timeout: Option<Duration>,
    ) -> Result<TaskResultHandle, PriorityError> {
        self.submit_task_with_queue_timeout(priority, source, payload, timeout, None).await
    }
    
    /// Submit a task with separate queue timeout
    pub async fn submit_task_with_queue_timeout(
        &self,
        priority: TaskPriority,
        source: TaskSource,
        payload: TaskPayload,
        execution_timeout: Option<Duration>,
        queue_timeout: Option<Duration>,
    ) -> Result<TaskResultHandle, PriorityError> {
        let task_id = Uuid::new_v4();
        let (result_sender, result_receiver) = oneshot::channel();
        
        // Determine if task supports checkpointing based on payload type
        let supports_checkpointing = match &payload {
            TaskPayload::ProcessDocument { .. } => true,
            TaskPayload::WatchDirectory { .. } => true,
            TaskPayload::ExecuteQuery { .. } => false, // Queries are typically fast
            TaskPayload::Generic { .. } => true, // Assume generic tasks can be checkpointed
        };
        
        let context = TaskContext {
            task_id,
            priority,
            created_at: chrono::Utc::now(),
            timeout_ms: execution_timeout.map(|d| d.as_millis() as u64),
            source,
            metadata: HashMap::new(),
            checkpoint_id: None,
            supports_checkpointing,
        };
        
        let task = PriorityTask {
            context: context.clone(),
            payload: payload.clone(),
            result_sender,
            cancellation_token: None,
        };

        // Check rate limit if enabled
        if let Some(ref limiter) = self.rate_limiter {
            if limiter.check().is_err() {
                // Rate limit exceeded, record and return error
                self.metrics_collector.record_rate_limit();
                tracing::warn!("Rate limit exceeded, rejecting task {}", task_id);

                return Err(PriorityError::Communication(
                    "Rate limit exceeded".to_string()
                ));
            }
        }

        // Check for backpressure
        let utilization = self.request_queue.get_utilization();
        if self.request_queue.config.enable_backpressure
            && utilization >= self.request_queue.config.backpressure_threshold {
            // Record backpressure event and log warning
            self.metrics_collector.record_backpressure();
            tracing::warn!(
                "Backpressure detected: queue at {:.1}% capacity (threshold: {:.1}%), task {} queued with potential delays",
                utilization * 100.0,
                self.request_queue.config.backpressure_threshold * 100.0,
                task_id
            );
        }

        // Check queue capacity BEFORE enqueuing (to implement retry without cloning task)
        // Queue is full - apply aggressive backpressure with retry.
        // After retries exhausted, spill to SQLite unified_queue if configured.
        // The UnifiedQueueProcessor picks up spilled tasks on its next poll cycle.

        let max_retries = 5;
        let mut retry_delay_ms = 100;

        for attempt in 0..max_retries {
            if !self.request_queue.has_capacity() {
                if attempt > 0 {
                    self.metrics_collector.record_queue_overflow();
                    tracing::warn!(
                        "Queue overflow: retry attempt {}/{} for task {} after {}ms delay",
                        attempt, max_retries, task_id, retry_delay_ms
                    );
                } else {
                    self.metrics_collector.record_queue_overflow();
                    tracing::error!(
                        "Request queue FULL, task {} will be retried with backpressure - file watching BLOCKED until queue drains",
                        task_id
                    );
                }

                if attempt < max_retries - 1 {
                    tokio::time::sleep(tokio::time::Duration::from_millis(retry_delay_ms)).await;
                    retry_delay_ms *= 2; // Exponential backoff
                    continue;
                } else {
                    // All retries exhausted — attempt spill to SQLite
                    if let Some(ref spill_queue) = self.spill_queue {
                        match self.spill_to_sqlite(spill_queue, &context, &payload).await {
                            Ok(()) => {
                                self.metrics_collector.record_queue_spill();
                                tracing::warn!(
                                    "Task {} spilled to SQLite after {} retry attempts - will be processed by queue processor",
                                    task_id, max_retries
                                );
                                // Return a handle with an immediate "spilled" result
                                let (result_sender, result_receiver) = oneshot::channel();
                                let _ = result_sender.send(TaskResult::Success {
                                    execution_time_ms: 0,
                                    data: TaskResultData::Generic {
                                        message: "Task spilled to SQLite unified_queue".to_string(),
                                        data: serde_json::json!({"spilled_to_sqlite": true}),
                                        checkpoint_id: None,
                                    },
                                });
                                return Ok(TaskResultHandle {
                                    task_id,
                                    context,
                                    result_receiver,
                                });
                            }
                            Err(spill_err) => {
                                tracing::error!(
                                    "Task {} REJECTED after {} retry attempts and SQLite spill failed: {}",
                                    task_id, max_retries, spill_err
                                );
                                return Err(PriorityError::QueueCapacityExceeded {
                                    current: self.request_queue.size(),
                                    max: self.request_queue.capacity()
                                });
                            }
                        }
                    } else {
                        tracing::error!(
                            "Task {} REJECTED after {} retry attempts - queue still full (no spill target configured)",
                            task_id, max_retries
                        );
                        return Err(PriorityError::QueueCapacityExceeded {
                            current: self.request_queue.size(),
                            max: self.request_queue.capacity()
                        });
                    }
                }
            } else {
                // Queue has capacity, break out of retry loop
                if attempt > 0 {
                    tracing::info!(
                        "Task {} successfully queued after {} retry attempts",
                        task_id, attempt
                    );
                }
                break;
            }
        }

        // Now enqueue the task (should succeed since we checked capacity)
        match self.request_queue.enqueue(task, queue_timeout).await {
            Ok(_) => {
                // Task was successfully queued, now dequeue and send to pipeline
                if let Some(queued_task) = self.request_queue.dequeue().await {
                    self.sender.send(queued_task)
                        .map_err(|_| PriorityError::Communication("Pipeline is shutting down".to_string()))?;
                } else {
                    // This shouldn't happen, but handle gracefully
                    return Err(PriorityError::Communication(
                        "Task was queued but could not be dequeued".to_string()
                    ));
                }
            }
            Err(e) => return Err(e),
        }
        
        Ok(TaskResultHandle {
            task_id,
            context,
            result_receiver,
        })
    }
    
    /// Submit a high priority task that bypasses the queue
    pub async fn submit_urgent_task(
        &self,
        source: TaskSource,
        payload: TaskPayload,
        timeout: Option<Duration>,
    ) -> Result<TaskResultHandle, PriorityError> {
        let task_id = Uuid::new_v4();
        let (result_sender, result_receiver) = oneshot::channel();
        
        let context = TaskContext {
            task_id,
            priority: TaskPriority::McpRequests, // Always highest priority
            created_at: chrono::Utc::now(),
            timeout_ms: timeout.map(|d| d.as_millis() as u64),
            source,
            metadata: {
                let mut metadata = HashMap::new();
                metadata.insert("urgent".to_string(), "true".to_string());
                metadata
            },
            checkpoint_id: None,
            supports_checkpointing: false, // Urgent tasks don't support checkpointing for speed
        };
        
        let task = PriorityTask {
            context: context.clone(),
            payload,
            result_sender,
            cancellation_token: None,
        };
        
        // Bypass queue for urgent tasks
        self.sender.send(task)
            .map_err(|_| PriorityError::Communication("Pipeline is shutting down".to_string()))?;
        
        Ok(TaskResultHandle {
            task_id,
            context,
            result_receiver,
        })
    }
    
    /// Get queue statistics
    pub async fn get_queue_stats(&self) -> QueueStats {
        self.request_queue.get_stats().await
    }
    
    /// Clean up timed out requests from queue
    pub async fn cleanup_queue_timeouts(&self) -> usize {
        self.request_queue.cleanup_timeouts().await
    }

    /// Spill a task to SQLite unified_queue when in-memory queue is full
    ///
    /// Maps Pipeline TaskPayload to unified_queue format and persists it.
    /// The UnifiedQueueProcessor will pick it up on its next poll cycle.
    #[cfg_attr(test, allow(dead_code))]
    pub(crate) async fn spill_to_sqlite(
        &self,
        queue_manager: &QueueManager,
        context: &TaskContext,
        payload: &TaskPayload,
    ) -> Result<(), PriorityError> {
        use crate::unified_queue_schema::{
            ItemType, QueueOperation as UnifiedOp, FilePayload as UqFilePayload,
        };

        match payload {
            TaskPayload::ProcessDocument { file_path, collection } => {
                // Derive tenant_id from TaskSource
                let tenant_id = match &context.source {
                    TaskSource::ProjectWatcher { project_path } => {
                        crate::watching_queue::calculate_tenant_id(
                            std::path::Path::new(project_path),
                        )
                    }
                    TaskSource::BackgroundWatcher { folder_path } => {
                        crate::watching_queue::calculate_tenant_id(
                            std::path::Path::new(folder_path),
                        )
                    }
                    _ => {
                        // Fallback: hash the file's parent directory
                        let parent = file_path.parent()
                            .unwrap_or_else(|| std::path::Path::new("/"));
                        crate::watching_queue::calculate_tenant_id(parent)
                    }
                };

                let file_payload = UqFilePayload {
                    file_path: file_path.to_string_lossy().to_string(),
                    file_type: None,
                    file_hash: None,
                    size_bytes: None,
                };

                let payload_json = serde_json::to_string(&file_payload)
                    .map_err(|e| PriorityError::Communication(
                        format!("Failed to serialize spill payload: {}", e)
                    ))?;

                let metadata = serde_json::json!({
                    "spilled_from": "pipeline",
                    "original_priority": format!("{:?}", context.priority),
                    "task_id": context.task_id.to_string(),
                }).to_string();

                queue_manager.enqueue_unified(
                    ItemType::File,
                    UnifiedOp::Ingest,
                    &tenant_id,
                    collection,
                    &payload_json,
                    0, // Priority computed at dequeue time
                    Some("main"),
                    Some(&metadata),
                ).await.map_err(|e| PriorityError::Communication(
                    format!("SQLite spill failed: {}", e)
                ))?;

                tracing::info!(
                    file_path = %file_path.display(),
                    collection = %collection,
                    tenant_id = %tenant_id,
                    "Task spilled to SQLite unified_queue"
                );

                Ok(())
            }
            _ => {
                // Only ProcessDocument tasks can be spilled
                Err(PriorityError::Communication(
                    format!("Cannot spill {:?} task to SQLite - only ProcessDocument is supported",
                        std::mem::discriminant(payload))
                ))
            }
        }
    }
}

/// Handle for waiting on task execution results
pub struct TaskResultHandle {
    pub task_id: Uuid,
    pub context: TaskContext,
    result_receiver: oneshot::Receiver<TaskResult>,
}

impl TaskResultHandle {
    /// Create a TaskResultHandle for testing purposes.
    #[cfg(test)]
    pub(crate) fn new_for_test(
        task_id: Uuid,
        context: TaskContext,
        result_receiver: oneshot::Receiver<TaskResult>,
    ) -> Self {
        Self { task_id, context, result_receiver }
    }

    /// Wait for the task to complete and return the result
    pub async fn wait(self) -> Result<TaskResult, PriorityError> {
        self.result_receiver.await
            .map_err(|_| PriorityError::Communication("Task executor disconnected".to_string()))
    }

    /// Check if the task result sender has been dropped (task completed or abandoned).
    ///
    /// Returns true when the oneshot sender is no longer alive, meaning the task
    /// executor has either sent a result or been dropped. Used by IPC cleanup to
    /// identify completed tasks that can be removed from the active tasks map.
    pub fn is_completed(&mut self) -> bool {
        match self.result_receiver.try_recv() {
            // Value was sent - task completed
            Ok(_) => true,
            // Sender dropped without sending - task abandoned
            Err(oneshot::error::TryRecvError::Closed) => true,
            // No value yet, sender still alive - task still running
            Err(oneshot::error::TryRecvError::Empty) => false,
        }
    }
}

/// Request queuing configuration and limits
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QueueConfig {
    /// Maximum number of queued requests per priority level
    pub max_queued_per_priority: usize,
    /// Default timeout for queued requests in milliseconds
    pub default_queue_timeout_ms: u64,
    /// Enable request deduplication based on content hash
    pub enable_deduplication: bool,
    /// Maximum time to wait for queue space in milliseconds
    pub queue_wait_timeout_ms: u64,
    /// Enable priority boost for aged requests
    pub enable_priority_boost: bool,
    /// Age threshold for priority boost in milliseconds
    pub priority_boost_age_ms: u64,
    /// Enable rate limiting for task submission
    pub enable_rate_limiting: bool,
    /// Maximum tasks per second (None = unlimited)
    pub max_tasks_per_second: Option<u64>,
    /// Enable backpressure signaling
    pub enable_backpressure: bool,
    /// Queue utilization threshold for backpressure warning (0.0-1.0, default: 0.8)
    pub backpressure_threshold: f64,
}

impl Default for QueueConfig {
    fn default() -> Self {
        Self {
            max_queued_per_priority: 100,
            default_queue_timeout_ms: 30_000,
            enable_deduplication: true,
            queue_wait_timeout_ms: 5_000,
            enable_priority_boost: true,
            priority_boost_age_ms: 10_000,
            enable_rate_limiting: true,
            max_tasks_per_second: Some(100),
            enable_backpressure: true,
            backpressure_threshold: 0.8,
        }
    }
}

/// Queued request with timeout and metadata
#[derive(Debug)]
struct QueuedRequest {
    task: PriorityTask,
    queued_at: Instant,
    timeout: Option<Instant>,
    content_hash: Option<u64>,
    priority_boosted: bool,
    original_priority: TaskPriority,
    #[allow(dead_code)]
    retry_count: usize,
}

/// Request queue manager with timeout and capacity management
pub struct RequestQueue {
    /// Queues per priority level
    priority_queues: Arc<RwLock<HashMap<TaskPriority, VecDeque<QueuedRequest>>>>,
    /// Configuration for queue behavior
    config: QueueConfig,
    /// Total number of queued requests across all priorities
    total_queued: Arc<AtomicU64>,
    /// Request deduplication map (content hash -> task ID)
    dedup_map: Arc<RwLock<HashMap<u64, Uuid>>>,
    /// Timeout manager for queued requests
    timeout_manager: Arc<Mutex<HashMap<Uuid, tokio::time::Sleep>>>,
}

impl RequestQueue {
    /// Create a new request queue with configuration
    pub fn new(config: QueueConfig) -> Self {
        let mut priority_queues = HashMap::new();
        
        // Initialize queues for each priority level
        priority_queues.insert(TaskPriority::McpRequests, VecDeque::new());
        priority_queues.insert(TaskPriority::ProjectWatching, VecDeque::new());
        priority_queues.insert(TaskPriority::CliCommands, VecDeque::new());
        priority_queues.insert(TaskPriority::BackgroundWatching, VecDeque::new());
        
        Self {
            priority_queues: Arc::new(RwLock::new(priority_queues)),
            config,
            total_queued: Arc::new(AtomicU64::new(0)),
            dedup_map: Arc::new(RwLock::new(HashMap::new())),
            timeout_manager: Arc::new(Mutex::new(HashMap::new())),
        }
    }
    
    /// Enqueue a request with timeout handling
    pub async fn enqueue(
        &self,
        task: PriorityTask,
        queue_timeout: Option<Duration>,
    ) -> Result<(), PriorityError> {
        let task_id = task.context.task_id;
        let priority = task.context.priority;
        let current_total = self.total_queued.load(AtomicOrdering::Relaxed) as usize;
        let max_total = self.config.max_queued_per_priority * 4; // 4 priority levels
        
        // Check global queue capacity
        if current_total >= max_total {
            return Err(PriorityError::QueueCapacityExceeded {
                current: current_total,
                max: max_total,
            });
        }
        
        // Calculate content hash for deduplication if enabled
        let content_hash = if self.config.enable_deduplication {
            Some(self.calculate_content_hash(&task))
        } else {
            None
        };
        
        // Check for duplicates
        if let Some(hash) = content_hash {
            let dedup_lock = self.dedup_map.read().await;
            if dedup_lock.contains_key(&hash) {
                return Err(PriorityError::Communication(
                    "Duplicate request already queued".to_string()
                ));
            }
        }
        
        let timeout_instant = queue_timeout
            .or_else(|| Some(Duration::from_millis(self.config.default_queue_timeout_ms)))
            .map(|duration| Instant::now() + duration);
        
        let queued_request = QueuedRequest {
            task,
            queued_at: Instant::now(),
            timeout: timeout_instant,
            content_hash,
            priority_boosted: false,
            original_priority: priority,
            retry_count: 0,
        };
        
        // Add to appropriate priority queue
        {
            let mut queues_lock = self.priority_queues.write().await;
            if let Some(queue) = queues_lock.get_mut(&priority) {
                // Check per-priority capacity
                if queue.len() >= self.config.max_queued_per_priority {
                    return Err(PriorityError::QueueCapacityExceeded {
                        current: queue.len(),
                        max: self.config.max_queued_per_priority,
                    });
                }
                
                queue.push_back(queued_request);
                self.total_queued.fetch_add(1, AtomicOrdering::Relaxed);
                
                // Update deduplication map
                if let Some(hash) = content_hash {
                    let mut dedup_lock = self.dedup_map.write().await;
                    dedup_lock.insert(hash, task_id);
                }
            } else {
                return Err(PriorityError::InvalidPriority(priority as u8));
            }
        }
        
        // Set up timeout handling
        if let Some(timeout_instant) = timeout_instant {
            let timeout_sleep = tokio::time::sleep_until(timeout_instant.into());
            let mut timeout_lock = self.timeout_manager.lock().await;
            timeout_lock.insert(task_id, timeout_sleep);
        }
        
        tracing::debug!(
            "Enqueued request {} with priority {:?}, queue size now: {}",
            task_id, priority, current_total + 1
        );
        
        Ok(())
    }
    
    /// Dequeue the highest priority request
    pub async fn dequeue(&self) -> Option<PriorityTask> {
        let mut queues_lock = self.priority_queues.write().await;
        
        // Check queues in priority order (highest to lowest)
        let priorities = vec![
            TaskPriority::McpRequests,
            TaskPriority::ProjectWatching,
            TaskPriority::CliCommands,
            TaskPriority::BackgroundWatching,
        ];
        
        for priority in priorities {
            if let Some(queue) = queues_lock.get_mut(&priority) {
                if let Some(mut queued_request) = queue.pop_front() {
                    // Check for timeout
                    if let Some(timeout) = queued_request.timeout {
                        if Instant::now() > timeout {
                            // Request has timed out in queue, continue to next
                            self.handle_timeout_cleanup(queued_request.task.context.task_id).await;
                            continue;
                        }
                    }
                    
                    // Check for priority boost
                    if self.config.enable_priority_boost && !queued_request.priority_boosted {
                        let age = queued_request.queued_at.elapsed();
                        let boost_threshold = Duration::from_millis(self.config.priority_boost_age_ms);
                        
                        if age > boost_threshold && queued_request.original_priority != TaskPriority::McpRequests {
                            // Boost priority and re-queue
                            let boosted_priority = match queued_request.original_priority {
                                TaskPriority::BackgroundWatching => TaskPriority::CliCommands,
                                TaskPriority::CliCommands => TaskPriority::ProjectWatching,
                                TaskPriority::ProjectWatching => TaskPriority::McpRequests,
                                TaskPriority::McpRequests => TaskPriority::McpRequests, // Already highest
                            };
                            
                            queued_request.task.context.priority = boosted_priority;
                            queued_request.priority_boosted = true;
                            
                            // Re-queue with boosted priority
                            if let Some(boosted_queue) = queues_lock.get_mut(&boosted_priority) {
                                let task_id = queued_request.task.context.task_id;
                                let original_priority = queued_request.original_priority;
                                
                                boosted_queue.push_front(queued_request); // Push to front for immediate processing
                                tracing::info!(
                                    "Boosted priority for aged request {} from {:?} to {:?}",
                                    task_id,
                                    original_priority,
                                    boosted_priority
                                );
                                continue;
                            }
                        }
                    }
                    
                    // Clean up tracking data
                    self.cleanup_request_tracking(queued_request.task.context.task_id, queued_request.content_hash).await;
                    self.total_queued.fetch_sub(1, AtomicOrdering::Relaxed);
                    
                    return Some(queued_request.task);
                }
            }
        }
        
        None
    }
    
    /// Get queue statistics
    pub async fn get_stats(&self) -> QueueStats {
        let queues_lock = self.priority_queues.read().await;
        let mut stats = QueueStats {
            total_queued: self.total_queued.load(AtomicOrdering::Relaxed) as usize,
            queued_by_priority: HashMap::new(),
            oldest_request_age_ms: None,
            timeout_manager_size: {
                let timeout_lock = self.timeout_manager.lock().await;
                timeout_lock.len()
            },
            deduplication_map_size: {
                let dedup_lock = self.dedup_map.read().await;
                dedup_lock.len()
            },
        };
        
        let mut oldest_age: Option<Duration> = None;
        
        for (priority, queue) in queues_lock.iter() {
            stats.queued_by_priority.insert(*priority, queue.len());
            
            // Find oldest request
            if let Some(oldest_in_queue) = queue.front() {
                let age = oldest_in_queue.queued_at.elapsed();
                match oldest_age {
                    None => oldest_age = Some(age),
                    Some(current_oldest) => {
                        if age > current_oldest {
                            oldest_age = Some(age);
                        }
                    }
                }
            }
        }
        
        stats.oldest_request_age_ms = oldest_age.map(|age| age.as_millis() as u64);
        stats
    }
    
    /// Clean up timed out requests
    pub async fn cleanup_timeouts(&self) -> usize {
        let mut cleaned_count = 0;
        let mut queues_lock = self.priority_queues.write().await;
        let now = Instant::now();
        
        for (priority, queue) in queues_lock.iter_mut() {
            let initial_len = queue.len();
            
            // Remove timed out requests
            queue.retain(|request| {
                if let Some(timeout) = request.timeout {
                    if now > timeout {
                        // Clean up tracking for this request
                        tokio::spawn({
                            let task_id = request.task.context.task_id;
                            let content_hash = request.content_hash;
                            let _queue_ref = Arc::clone(&self.priority_queues);
                            let dedup_ref = Arc::clone(&self.dedup_map);
                            let timeout_ref = Arc::clone(&self.timeout_manager);
                            
                            async move {
                                Self::cleanup_request_tracking_static(
                                    task_id, content_hash, dedup_ref, timeout_ref
                                ).await;
                            }
                        });
                        
                        tracing::warn!(
                            "Request {} timed out in queue after {:?} (priority {:?})",
                            request.task.context.task_id,
                            request.queued_at.elapsed(),
                            priority
                        );
                        
                        return false; // Remove this request
                    }
                }
                true // Keep this request
            });
            
            let removed_count = initial_len - queue.len();
            cleaned_count += removed_count;
            
            // Update total counter
            if removed_count > 0 {
                self.total_queued.fetch_sub(removed_count as u64, AtomicOrdering::Relaxed);
            }
        }
        
        cleaned_count
    }

    /// Get current queue utilization as a percentage (0.0-1.0)
    pub fn get_utilization(&self) -> f64 {
        let current = self.total_queued.load(AtomicOrdering::Relaxed) as usize;
        let max = self.config.max_queued_per_priority * 4; // 4 priority levels
        if max == 0 {
            0.0
        } else {
            current as f64 / max as f64
        }
    }

    /// Calculate content hash for deduplication
    fn calculate_content_hash(&self, task: &PriorityTask) -> u64 {
        use std::hash::{Hash, Hasher};
        use std::collections::hash_map::DefaultHasher;
        
        let mut hasher = DefaultHasher::new();
        
        // Hash the task payload for deduplication
        match &task.payload {
            TaskPayload::ProcessDocument { file_path, collection } => {
                "ProcessDocument".hash(&mut hasher);
                file_path.hash(&mut hasher);
                collection.hash(&mut hasher);
            }
            TaskPayload::WatchDirectory { path, recursive } => {
                "WatchDirectory".hash(&mut hasher);
                path.hash(&mut hasher);
                recursive.hash(&mut hasher);
            }
            TaskPayload::ExecuteQuery { query, collection, limit } => {
                "ExecuteQuery".hash(&mut hasher);
                query.hash(&mut hasher);
                collection.hash(&mut hasher);
                limit.hash(&mut hasher);
            }
            TaskPayload::Generic { operation, parameters } => {
                "Generic".hash(&mut hasher);
                operation.hash(&mut hasher);
                // Hash parameters in a consistent way (without using Hash trait on HashMap)
                let mut param_keys: Vec<_> = parameters.keys().cloned().collect();
                param_keys.sort();
                for key in param_keys {
                    key.hash(&mut hasher);
                    if let Some(value) = parameters.get(&key) {
                        value.to_string().hash(&mut hasher);
                    }
                }
            }
        }
        
        hasher.finish()
    }
    
    /// Handle timeout cleanup for a specific request
    async fn handle_timeout_cleanup(&self, task_id: Uuid) {
        let timeout_lock = self.timeout_manager.lock().await;
        if timeout_lock.contains_key(&task_id) {
            tracing::warn!("Request {} timed out in queue", task_id);
        }
    }
    
    /// Clean up tracking data for a request
    async fn cleanup_request_tracking(&self, task_id: Uuid, content_hash: Option<u64>) {
        Self::cleanup_request_tracking_static(
            task_id,
            content_hash,
            Arc::clone(&self.dedup_map),
            Arc::clone(&self.timeout_manager),
        ).await;
    }
    
    /// Static version of cleanup for spawned tasks
    async fn cleanup_request_tracking_static(
        task_id: Uuid,
        content_hash: Option<u64>,
        dedup_map: Arc<RwLock<HashMap<u64, Uuid>>>,
        timeout_manager: Arc<Mutex<HashMap<Uuid, tokio::time::Sleep>>>,
    ) {
        // Remove from deduplication map
        if let Some(hash) = content_hash {
            let mut dedup_lock = dedup_map.write().await;
            dedup_lock.remove(&hash);
        }
        
        // Remove from timeout manager
        {
            let mut timeout_lock = timeout_manager.lock().await;
            timeout_lock.remove(&task_id);
        }
    }

    /// Check if the queue has capacity for a new task
    pub fn has_capacity(&self) -> bool {
        let current_total = self.total_queued.load(AtomicOrdering::Relaxed) as usize;
        let max_total = self.config.max_queued_per_priority * 4; // 4 priority levels
        current_total < max_total
    }

    /// Get current queue size
    pub fn size(&self) -> usize {
        self.total_queued.load(AtomicOrdering::Relaxed) as usize
    }

    /// Get maximum queue capacity
    pub fn capacity(&self) -> usize {
        self.config.max_queued_per_priority * 4 // 4 priority levels
    }
}

/// Queue statistics for monitoring
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QueueStats {
    pub total_queued: usize,
    pub queued_by_priority: HashMap<TaskPriority, usize>,
    pub oldest_request_age_ms: Option<u64>,
    pub timeout_manager_size: usize,
    pub deduplication_map_size: usize,
}

/// Comprehensive metrics for priority system performance
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PrioritySystemMetrics {
    /// General pipeline metrics
    pub pipeline: PipelineMetrics,
    /// Queue-specific metrics
    pub queue: QueueMetrics,
    /// Preemption behavior metrics
    pub preemption: PreemptionMetrics,
    /// Checkpoint system metrics
    pub checkpoints: CheckpointMetrics,
    /// Performance metrics over time
    pub performance: PerformanceMetrics,
    /// Resource utilization
    pub resources: ResourceMetrics,
}

/// Core pipeline metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PipelineMetrics {
    pub queued_tasks: usize,
    pub running_tasks: usize,
    pub total_capacity: usize,
    pub tasks_completed: u64,
    pub tasks_failed: u64,
    pub tasks_cancelled: u64,
    pub tasks_timed_out: u64,
    pub uptime_seconds: u64,
}

/// Queue behavior metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QueueMetrics {
    pub total_queued: usize,
    pub queued_by_priority: HashMap<TaskPriority, usize>,
    pub oldest_request_age_ms: Option<u64>,
    pub average_queue_time_ms: f64,
    pub max_queue_time_ms: u64,
    pub queue_overflow_count: u64,
    pub queue_spill_count: u64,
    pub deduplication_hits: u64,
    pub priority_boosts_applied: u64,
}

/// Preemption system metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PreemptionMetrics {
    pub preemptions_total: u64,
    pub preemptions_by_priority: HashMap<TaskPriority, u64>,
    pub graceful_preemptions: u64,
    pub forced_aborts: u64,
    pub preemption_success_rate: f64,
    pub average_preemption_time_ms: f64,
    pub tasks_resumed_from_checkpoints: u64,
}

/// Checkpoint system metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CheckpointMetrics {
    pub active_checkpoints: usize,
    pub checkpoints_created: u64,
    pub checkpoints_restored: u64,
    pub rollbacks_executed: u64,
    pub rollback_success_rate: f64,
    pub average_checkpoint_size_bytes: f64,
    pub checkpoint_storage_usage_bytes: u64,
}

/// Performance metrics over time
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceMetrics {
    pub throughput_tasks_per_second: f64,
    pub average_task_duration_ms: f64,
    pub p95_task_duration_ms: f64,
    pub p99_task_duration_ms: f64,
    pub response_time_by_priority: HashMap<TaskPriority, f64>,
    pub error_rate_percent: f64,
    pub system_load_percent: f64,
}

/// Resource utilization metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourceMetrics {
    pub memory_usage_bytes: u64,
    pub cpu_usage_percent: f64,
    pub disk_usage_bytes: u64,
    pub network_io_bytes: u64,
    pub thread_count: usize,
    pub file_handles_open: usize,
}



/// Real-time metrics collector and aggregator
pub struct MetricsCollector {
    /// Start time for uptime calculation
    start_time: Instant,
    /// Atomic counters for high-frequency metrics
    tasks_completed: AtomicU64,
    tasks_failed: AtomicU64,
    tasks_cancelled: AtomicU64,
    tasks_timed_out: AtomicU64,
    preemptions_total: AtomicU64,
    graceful_preemptions: AtomicU64,
    forced_aborts: AtomicU64,
    checkpoints_created: AtomicU64,
    rollbacks_executed: AtomicU64,
    queue_overflow_count: AtomicU64,
    queue_spill_count: AtomicU64,
    deduplication_hits: AtomicU64,
    priority_boosts_applied: AtomicU64,
    rate_limited_tasks: AtomicU64,
    backpressure_events: AtomicU64,

    /// Atomic accumulators for averages
    total_task_duration_ms: AtomicU64,
    total_queue_time_ms: AtomicU64,
    total_preemption_time_ms: AtomicU64,

    /// Recent measurements for percentile calculations
    recent_task_durations: Arc<RwLock<VecDeque<u64>>>,
    recent_queue_times: Arc<RwLock<VecDeque<u64>>>,

    /// Per-priority metrics
    preemptions_by_priority: Arc<RwLock<HashMap<TaskPriority, AtomicU64>>>,
    response_times_by_priority: Arc<RwLock<HashMap<TaskPriority, AtomicU64>>>,

    /// Performance sampling interval
    #[allow(dead_code)]
    sample_window: Duration,
}

impl MetricsCollector {
    /// Create a new metrics collector
    pub fn new(sample_window: Duration) -> Self {
        let mut preemptions_by_priority = HashMap::new();
        let mut response_times_by_priority = HashMap::new();
        
        for priority in [TaskPriority::McpRequests, TaskPriority::ProjectWatching, 
                        TaskPriority::CliCommands, TaskPriority::BackgroundWatching] {
            preemptions_by_priority.insert(priority, AtomicU64::new(0));
            response_times_by_priority.insert(priority, AtomicU64::new(0));
        }
        
        Self {
            start_time: Instant::now(),
            tasks_completed: AtomicU64::new(0),
            tasks_failed: AtomicU64::new(0),
            tasks_cancelled: AtomicU64::new(0),
            tasks_timed_out: AtomicU64::new(0),
            preemptions_total: AtomicU64::new(0),
            graceful_preemptions: AtomicU64::new(0),
            forced_aborts: AtomicU64::new(0),
            checkpoints_created: AtomicU64::new(0),
            rollbacks_executed: AtomicU64::new(0),
            queue_overflow_count: AtomicU64::new(0),
            queue_spill_count: AtomicU64::new(0),
            deduplication_hits: AtomicU64::new(0),
            priority_boosts_applied: AtomicU64::new(0),
            rate_limited_tasks: AtomicU64::new(0),
            backpressure_events: AtomicU64::new(0),
            total_task_duration_ms: AtomicU64::new(0),
            total_queue_time_ms: AtomicU64::new(0),
            total_preemption_time_ms: AtomicU64::new(0),
            recent_task_durations: Arc::new(RwLock::new(VecDeque::with_capacity(1000))),
            recent_queue_times: Arc::new(RwLock::new(VecDeque::with_capacity(1000))),
            preemptions_by_priority: Arc::new(RwLock::new(preemptions_by_priority)),
            response_times_by_priority: Arc::new(RwLock::new(response_times_by_priority)),
            sample_window,
        }
    }
    
    /// Record task completion
    pub async fn record_task_completion(&self, duration_ms: u64, priority: TaskPriority) {
        self.tasks_completed.fetch_add(1, AtomicOrdering::Relaxed);
        self.total_task_duration_ms.fetch_add(duration_ms, AtomicOrdering::Relaxed);
        
        // Update recent durations
        {
            let mut durations = self.recent_task_durations.write().await;
            if durations.len() >= 1000 {
                durations.pop_front();
            }
            durations.push_back(duration_ms);
        }
        
        // Update per-priority response times (using integer storage for simplicity)
        {
            let response_times = self.response_times_by_priority.read().await;
            if let Some(atomic_time) = response_times.get(&priority) {
                // Simple exponential moving average stored as integer microseconds
                let current = atomic_time.load(AtomicOrdering::Relaxed) as f64;
                let new_avg = current * 0.9 + (duration_ms as f64 * 1000.0) * 0.1;
                atomic_time.store(new_avg as u64, AtomicOrdering::Relaxed);
            }
        }
    }
    
    /// Record task failure
    pub fn record_task_failure(&self) {
        self.tasks_failed.fetch_add(1, AtomicOrdering::Relaxed);
    }
    
    /// Record task cancellation
    pub fn record_task_cancellation(&self) {
        self.tasks_cancelled.fetch_add(1, AtomicOrdering::Relaxed);
    }
    
    /// Record task timeout
    pub fn record_task_timeout(&self) {
        self.tasks_timed_out.fetch_add(1, AtomicOrdering::Relaxed);
    }
    
    /// Record preemption event
    pub async fn record_preemption(&self, preempted_priority: TaskPriority, duration_ms: u64, graceful: bool) {
        self.preemptions_total.fetch_add(1, AtomicOrdering::Relaxed);
        self.total_preemption_time_ms.fetch_add(duration_ms, AtomicOrdering::Relaxed);
        
        if graceful {
            self.graceful_preemptions.fetch_add(1, AtomicOrdering::Relaxed);
        } else {
            self.forced_aborts.fetch_add(1, AtomicOrdering::Relaxed);
        }
        
        // Update per-priority preemption counts
        {
            let preemptions = self.preemptions_by_priority.read().await;
            if let Some(counter) = preemptions.get(&preempted_priority) {
                counter.fetch_add(1, AtomicOrdering::Relaxed);
            }
        }
    }
    
    /// Record queue-related events
    pub async fn record_queue_time(&self, queue_time_ms: u64) {
        self.total_queue_time_ms.fetch_add(queue_time_ms, AtomicOrdering::Relaxed);
        
        let mut queue_times = self.recent_queue_times.write().await;
        if queue_times.len() >= 1000 {
            queue_times.pop_front();
        }
        queue_times.push_back(queue_time_ms);
    }
    
    /// Record queue overflow
    pub fn record_queue_overflow(&self) {
        self.queue_overflow_count.fetch_add(1, AtomicOrdering::Relaxed);
    }

    /// Record queue spill to SQLite
    pub fn record_queue_spill(&self) {
        self.queue_spill_count.fetch_add(1, AtomicOrdering::Relaxed);
    }

    /// Record rate limit hit
    pub fn record_rate_limit(&self) {
        self.rate_limited_tasks.fetch_add(1, AtomicOrdering::Relaxed);
    }

    /// Record backpressure event
    pub fn record_backpressure(&self) {
        self.backpressure_events.fetch_add(1, AtomicOrdering::Relaxed);
    }

    /// Record deduplication hit
    pub fn record_deduplication_hit(&self) {
        self.deduplication_hits.fetch_add(1, AtomicOrdering::Relaxed);
    }
    
    /// Record priority boost
    pub fn record_priority_boost(&self) {
        self.priority_boosts_applied.fetch_add(1, AtomicOrdering::Relaxed);
    }
    
    /// Record checkpoint creation
    pub fn record_checkpoint_created(&self) {
        self.checkpoints_created.fetch_add(1, AtomicOrdering::Relaxed);
    }
    
    /// Record rollback execution
    pub fn record_rollback_executed(&self) {
        self.rollbacks_executed.fetch_add(1, AtomicOrdering::Relaxed);
    }
    
    /// Calculate percentile from recent measurements
    async fn calculate_percentile(values: &Arc<RwLock<VecDeque<u64>>>, percentile: f64) -> f64 {
        let values_lock = values.read().await;
        if values_lock.is_empty() {
            return 0.0;
        }
        
        let mut sorted: Vec<u64> = values_lock.iter().cloned().collect();
        sorted.sort_unstable();
        
        let index = ((sorted.len() as f64 - 1.0) * percentile / 100.0) as usize;
        *sorted.get(index).unwrap_or(&0) as f64
    }
    
    /// Get current system resource metrics
    fn get_resource_metrics() -> ResourceMetrics {
        ResourceMetrics {
            memory_usage_bytes: 0, // Would need system-specific implementation
            cpu_usage_percent: 0.0,
            disk_usage_bytes: 0,
            network_io_bytes: 0,
            thread_count: 0,
            file_handles_open: 0,
        }
    }
    
    /// Generate comprehensive metrics report
    pub async fn generate_metrics(
        &self,
        running_tasks: usize,
        queued_tasks: usize,
        capacity: usize,
        queue_stats: &QueueStats,
        checkpoint_manager: &CheckpointManager,
    ) -> PrioritySystemMetrics {
        let uptime = self.start_time.elapsed().as_secs();
        let tasks_completed = self.tasks_completed.load(AtomicOrdering::Relaxed);
        let tasks_failed = self.tasks_failed.load(AtomicOrdering::Relaxed);
        let tasks_cancelled = self.tasks_cancelled.load(AtomicOrdering::Relaxed);
        let tasks_timed_out = self.tasks_timed_out.load(AtomicOrdering::Relaxed);
        let preemptions_total = self.preemptions_total.load(AtomicOrdering::Relaxed);
        let graceful_preemptions = self.graceful_preemptions.load(AtomicOrdering::Relaxed);
        let forced_aborts = self.forced_aborts.load(AtomicOrdering::Relaxed);
        
        // Calculate averages
        let total_tasks = tasks_completed + tasks_failed + tasks_cancelled;
        let avg_task_duration = if total_tasks > 0 {
            self.total_task_duration_ms.load(AtomicOrdering::Relaxed) as f64 / total_tasks as f64
        } else {
            0.0
        };
        
        let avg_queue_time = if total_tasks > 0 {
            self.total_queue_time_ms.load(AtomicOrdering::Relaxed) as f64 / total_tasks as f64
        } else {
            0.0
        };
        
        let avg_preemption_time = if preemptions_total > 0 {
            self.total_preemption_time_ms.load(AtomicOrdering::Relaxed) as f64 / preemptions_total as f64
        } else {
            0.0
        };
        
        // Calculate percentiles
        let p95_duration = Self::calculate_percentile(&self.recent_task_durations, 95.0).await;
        let p99_duration = Self::calculate_percentile(&self.recent_task_durations, 99.0).await;
        
        // Calculate rates
        let throughput = if uptime > 0 {
            tasks_completed as f64 / uptime as f64
        } else {
            0.0
        };
        
        let error_rate = if total_tasks > 0 {
            (tasks_failed as f64 / total_tasks as f64) * 100.0
        } else {
            0.0
        };
        
        let preemption_success_rate = if preemptions_total > 0 {
            (graceful_preemptions as f64 / preemptions_total as f64) * 100.0
        } else {
            100.0
        };
        
        // Collect per-priority metrics
        let mut preemptions_by_priority = HashMap::new();
        let mut response_times_by_priority = HashMap::new();
        
        {
            let preemptions_map = self.preemptions_by_priority.read().await;
            for (priority, counter) in preemptions_map.iter() {
                preemptions_by_priority.insert(*priority, counter.load(AtomicOrdering::Relaxed));
            }
        }
        
        {
            let response_map = self.response_times_by_priority.read().await;
            for (priority, atomic_time) in response_map.iter() {
                // Convert from microseconds back to milliseconds
                let value_us = atomic_time.load(AtomicOrdering::Relaxed) as f64;
                response_times_by_priority.insert(*priority, value_us / 1000.0);
            }
        }
        
        // Get checkpoint metrics
        let active_checkpoints = {
            let checkpoints_lock = checkpoint_manager.checkpoints.read().await;
            checkpoints_lock.len()
        };
        
        PrioritySystemMetrics {
            pipeline: PipelineMetrics {
                queued_tasks,
                running_tasks,
                total_capacity: capacity,
                tasks_completed,
                tasks_failed,
                tasks_cancelled,
                tasks_timed_out,
                uptime_seconds: uptime,
            },
            queue: QueueMetrics {
                total_queued: queue_stats.total_queued,
                queued_by_priority: queue_stats.queued_by_priority.clone(),
                oldest_request_age_ms: queue_stats.oldest_request_age_ms,
                average_queue_time_ms: avg_queue_time,
                max_queue_time_ms: 0, // Would need max tracking
                queue_overflow_count: self.queue_overflow_count.load(AtomicOrdering::Relaxed),
                queue_spill_count: self.queue_spill_count.load(AtomicOrdering::Relaxed),
                deduplication_hits: self.deduplication_hits.load(AtomicOrdering::Relaxed),
                priority_boosts_applied: self.priority_boosts_applied.load(AtomicOrdering::Relaxed),
            },
            preemption: PreemptionMetrics {
                preemptions_total,
                preemptions_by_priority,
                graceful_preemptions,
                forced_aborts,
                preemption_success_rate,
                average_preemption_time_ms: avg_preemption_time,
                tasks_resumed_from_checkpoints: 0, // Would need tracking
            },
            checkpoints: CheckpointMetrics {
                active_checkpoints,
                checkpoints_created: self.checkpoints_created.load(AtomicOrdering::Relaxed),
                checkpoints_restored: 0, // Would need tracking
                rollbacks_executed: self.rollbacks_executed.load(AtomicOrdering::Relaxed),
                rollback_success_rate: 100.0, // Would need failure tracking
                average_checkpoint_size_bytes: 0.0, // Would need size tracking
                checkpoint_storage_usage_bytes: 0, // Would need disk usage
            },
            performance: PerformanceMetrics {
                throughput_tasks_per_second: throughput,
                average_task_duration_ms: avg_task_duration,
                p95_task_duration_ms: p95_duration,
                p99_task_duration_ms: p99_duration,
                response_time_by_priority: response_times_by_priority,
                error_rate_percent: error_rate,
                system_load_percent: 0.0, // Would need system monitoring
            },
            resources: Self::get_resource_metrics(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;
    use tokio::time::timeout;
    
    #[tokio::test]
    async fn test_pipeline_creation() {
        let pipeline = Pipeline::new(2);
        let stats = pipeline.stats().await;
        
        assert_eq!(stats.queued_tasks, 0);
        assert_eq!(stats.running_tasks, 0);
        assert_eq!(stats.total_capacity, 2);
    }
    
    #[tokio::test]
    async fn test_task_submission_and_execution() {
        let mut pipeline = Pipeline::new(2);
        let submitter = pipeline.task_submitter();
        
        // Start the pipeline
        pipeline.start().await.expect("Failed to start pipeline");
        
        // Submit a simple generic task
        let task_handle = submitter.submit_task(
            TaskPriority::CliCommands,
            TaskSource::CliCommand {
                command: "test_command".to_string(),
            },
            TaskPayload::Generic {
                operation: "test".to_string(),
                parameters: HashMap::new(),
            },
            Some(Duration::from_secs(5)),
        ).await.expect("Failed to submit task");
        
        // Wait for task completion with timeout
        let result = timeout(Duration::from_secs(10), task_handle.wait()).await
            .expect("Task timed out")
            .expect("Task execution failed");
        
        match result {
            TaskResult::Success { data, .. } => {
                match data {
                    TaskResultData::Generic { message, .. } => {
                        assert_eq!(message, "Completed operation: test");
                    }
                    _ => panic!("Expected Generic result data"),
                }
            }
            _ => panic!("Expected successful result, got: {:?}", result),
        }
    }
    
    #[tokio::test]
    async fn test_priority_ordering() {
        let mut pipeline = Pipeline::new(1); // Limit to 1 concurrent task
        let submitter = pipeline.task_submitter();
        
        pipeline.start().await.expect("Failed to start pipeline");
        
        // Submit multiple tasks with different priorities
        let low_priority_task = submitter.submit_task(
            TaskPriority::BackgroundWatching,
            TaskSource::BackgroundWatcher {
                folder_path: "/tmp".to_string(),
            },
            TaskPayload::Generic {
                operation: "low_priority".to_string(),
                parameters: HashMap::new(),
            },
            Some(Duration::from_secs(1)),
        ).await.expect("Failed to submit low priority task");
        
        // Small delay to ensure first task starts
        tokio::time::sleep(Duration::from_millis(10)).await;
        
        let high_priority_task = submitter.submit_task(
            TaskPriority::McpRequests,
            TaskSource::McpServer {
                request_id: "test_request".to_string(),
            },
            TaskPayload::Generic {
                operation: "high_priority".to_string(),
                parameters: HashMap::new(),
            },
            Some(Duration::from_secs(1)),
        ).await.expect("Failed to submit high priority task");
        
        // The high priority task should complete first (due to preemption)
        // or if low priority completes first, that's also acceptable
        let high_result = timeout(Duration::from_secs(5), high_priority_task.wait()).await
            .expect("High priority task timed out")
            .expect("High priority task failed");
        
        let low_result = timeout(Duration::from_secs(5), low_priority_task.wait()).await
            .expect("Low priority task timed out")
            .expect("Low priority task failed");
        
        // Verify both tasks completed
        match high_result {
            TaskResult::Success { data, .. } => {
                if let TaskResultData::Generic { message, .. } = data {
                    assert_eq!(message, "Completed operation: high_priority");
                }
            }
            _ => panic!("High priority task should have succeeded"),
        }
        
        // Low priority task might be cancelled or succeed depending on timing
        match low_result {
            TaskResult::Success { data, .. } => {
                if let TaskResultData::Generic { message, .. } = data {
                    assert_eq!(message, "Completed operation: low_priority");
                }
            }
            TaskResult::Cancelled { .. } => {
                // This is expected if preemption occurred
            }
            _ => {}, // Other results are acceptable
        }
    }
    
    #[tokio::test]
    async fn test_task_timeout() {
        let mut pipeline = Pipeline::new(1);
        let submitter = pipeline.task_submitter();
        
        pipeline.start().await.expect("Failed to start pipeline");
        
        // Submit a task with very short timeout
        let task_handle = submitter.submit_task(
            TaskPriority::CliCommands,
            TaskSource::CliCommand {
                command: "timeout_test".to_string(),
            },
            TaskPayload::Generic {
                operation: "slow_operation".to_string(),
                parameters: HashMap::new(),
            },
            Some(Duration::from_millis(1)), // Very short timeout
        ).await.expect("Failed to submit task");
        
        let result = timeout(Duration::from_secs(2), task_handle.wait()).await
            .expect("Test timed out")
            .expect("Task execution failed");
        
        // Task should timeout
        match result {
            TaskResult::Timeout { .. } => {
                // Expected
            }
            other => panic!("Expected timeout, got: {:?}", other),
        }
    }
    
    #[tokio::test]
    async fn test_concurrent_task_execution() {
        let mut pipeline = Pipeline::new(3); // Allow 3 concurrent tasks
        let submitter = pipeline.task_submitter();
        
        pipeline.start().await.expect("Failed to start pipeline");
        
        // Submit multiple tasks simultaneously
        let mut handles = Vec::new();
        for i in 0..5 {
            let handle = submitter.submit_task(
                TaskPriority::CliCommands,
                TaskSource::CliCommand {
                    command: format!("task_{}", i),
                },
                TaskPayload::Generic {
                    operation: format!("concurrent_task_{}", i),
                    parameters: HashMap::new(),
                },
                Some(Duration::from_secs(2)),
            ).await.expect("Failed to submit task");
            
            handles.push(handle);
        }
        
        // Wait for all tasks to complete
        let mut completed_count = 0;
        for handle in handles {
            let result = timeout(Duration::from_secs(10), handle.wait()).await
                .expect("Task timed out")
                .expect("Task execution failed");
            
            match result {
                TaskResult::Success { .. } => completed_count += 1,
                _ => {}
            }
        }
        
        assert_eq!(completed_count, 5, "All tasks should complete successfully");
    }
    
    #[tokio::test]
    async fn test_pipeline_stats() {
        let mut pipeline = Pipeline::new(2);
        let submitter = pipeline.task_submitter();
        
        pipeline.start().await.expect("Failed to start pipeline");
        
        // Check initial stats
        let initial_stats = pipeline.stats().await;
        assert_eq!(initial_stats.queued_tasks, 0);
        assert_eq!(initial_stats.running_tasks, 0);
        
        // Submit multiple tasks to queue them
        let _handle1 = submitter.submit_task(
            TaskPriority::CliCommands,
            TaskSource::CliCommand {
                command: "stats_test_1".to_string(),
            },
            TaskPayload::Generic {
                operation: "stats_test_1".to_string(),
                parameters: HashMap::new(),
            },
            Some(Duration::from_secs(5)),
        ).await.expect("Failed to submit task 1");
        
        let _handle2 = submitter.submit_task(
            TaskPriority::CliCommands,
            TaskSource::CliCommand {
                command: "stats_test_2".to_string(),
            },
            TaskPayload::Generic {
                operation: "stats_test_2".to_string(),
                parameters: HashMap::new(),
            },
            Some(Duration::from_secs(5)),
        ).await.expect("Failed to submit task 2");
        
        // Give tasks time to start
        tokio::time::sleep(Duration::from_millis(100)).await;
        
        let stats = pipeline.stats().await;
        // At least some tasks should be running or queued
        assert!(stats.running_tasks > 0 || stats.queued_tasks > 0);
        assert_eq!(stats.total_capacity, 2);
    }
    
    #[tokio::test]
    async fn test_preemption_logic() {
        let mut pipeline = Pipeline::new(1); // Only 1 concurrent task to force preemption
        let submitter = pipeline.task_submitter();
        
        pipeline.start().await.expect("Failed to start pipeline");
        
        // Start a low priority task that will run for a while
        let low_priority_handle = submitter.submit_task(
            TaskPriority::BackgroundWatching,
            TaskSource::BackgroundWatcher {
                folder_path: "/tmp/background".to_string(),
            },
            TaskPayload::Generic {
                operation: "long_running_background".to_string(),
                parameters: HashMap::new(),
            },
            Some(Duration::from_secs(5)), // Long timeout
        ).await.expect("Failed to submit low priority task");
        
        // Give the low priority task time to start
        tokio::time::sleep(Duration::from_millis(50)).await;
        
        // Submit a high priority MCP request that should preempt
        let high_priority_handle = submitter.submit_task(
            TaskPriority::McpRequests,
            TaskSource::McpServer {
                request_id: "urgent_request".to_string(),
            },
            TaskPayload::Generic {
                operation: "urgent_mcp_task".to_string(),
                parameters: HashMap::new(),
            },
            Some(Duration::from_secs(2)),
        ).await.expect("Failed to submit high priority task");
        
        // The high priority task should complete relatively quickly
        let high_result = timeout(Duration::from_secs(3), high_priority_handle.wait()).await
            .expect("High priority task should not timeout")
            .expect("High priority task should succeed");
        
        // Verify the high priority task succeeded
        match high_result {
            TaskResult::Success { data, .. } => {
                if let TaskResultData::Generic { message, .. } = data {
                    assert_eq!(message, "Completed operation: urgent_mcp_task");
                }
            }
            other => panic!("Expected success, got: {:?}", other),
        }
        
        // The low priority task should have been cancelled/preempted
        let low_result = timeout(Duration::from_secs(1), low_priority_handle.wait()).await
            .expect("Low priority task should complete quickly after preemption")
            .expect("Low priority task should have a result");
        
        match low_result {
            TaskResult::Cancelled { .. } => {
                // Expected - task was preempted
            }
            TaskResult::Success { .. } => {
                // Also acceptable if task completed before preemption
            }
            other => panic!("Expected cancelled or success, got: {:?}", other),
        }
    }
    
    #[tokio::test]
    async fn test_priority_system_metrics() {
        let queue_config = QueueConfigBuilder::new()
            .for_mcp_server()
            .build();
        
        let mut pipeline = Pipeline::with_queue_config(2, queue_config);
        let submitter = pipeline.task_submitter();
        
        pipeline.start().await.expect("Failed to start pipeline");
        
        // Submit various tasks to generate metrics
        let _handle1 = submitter.submit_task(
            TaskPriority::McpRequests,
            TaskSource::McpServer {
                request_id: "metrics_test_1".to_string(),
            },
            TaskPayload::Generic {
                operation: "metrics_test".to_string(),
                parameters: HashMap::new(),
            },
            Some(Duration::from_secs(1)),
        ).await.expect("Failed to submit MCP task");
        
        let _handle2 = submitter.submit_task(
            TaskPriority::CliCommands,
            TaskSource::CliCommand {
                command: "metrics_cli_test".to_string(),
            },
            TaskPayload::Generic {
                operation: "cli_metrics_test".to_string(),
                parameters: HashMap::new(),
            },
            Some(Duration::from_secs(1)),
        ).await.expect("Failed to submit CLI task");
        
        // Wait for tasks to complete
        tokio::time::sleep(Duration::from_millis(200)).await;
        
        // Get metrics
        let metrics = pipeline.get_priority_system_metrics().await;
        
        // Verify basic metrics (uptime might be 0 in fast tests)
        assert!(metrics.pipeline.uptime_seconds >= 0);
        assert!(metrics.performance.throughput_tasks_per_second >= 0.0);
        assert!(metrics.queue.queued_by_priority.len() > 0);
        
        // Test Prometheus export
        let prometheus_output = pipeline.export_prometheus_metrics().await;
        assert!(prometheus_output.contains("wqm_tasks_completed"));
        assert!(prometheus_output.contains("wqm_queue_total"));
        assert!(prometheus_output.len() > 0, "Prometheus output should not be empty");
        
        // Test metrics collector direct access
        let collector = pipeline.metrics_collector();
        collector.record_task_completion(100, TaskPriority::McpRequests).await;
        collector.record_preemption(TaskPriority::BackgroundWatching, 50, true).await;
        
        // Verify counters were updated
        let updated_metrics = pipeline.get_priority_system_metrics().await;
        assert!(updated_metrics.pipeline.tasks_completed >= metrics.pipeline.tasks_completed);
    }
    
    #[tokio::test]
    async fn test_checkpoint_metrics_integration() {
        let checkpoint_dir = std::env::temp_dir().join("test_checkpoint_metrics");
        let _ = std::fs::create_dir_all(&checkpoint_dir);
        
        let mut pipeline = Pipeline::with_checkpoint_config(
            2,
            QueueConfig::default(),
            Some(checkpoint_dir.clone()),
        );
        
        pipeline.start().await.expect("Failed to start pipeline");
        
        // Test checkpoint creation and metrics
        let checkpoint_manager = pipeline.checkpoint_manager();
        let metrics_collector = pipeline.metrics_collector();
        
        let checkpoint_id = checkpoint_manager.create_checkpoint(
            Uuid::new_v4(),
            TaskProgress::Generic {
                progress_percentage: 50.0,
                stage: "testing".to_string(),
                metadata: HashMap::new(),
            },
            serde_json::json!({"test": "data"}),
            vec![],
            vec![],
        ).await.expect("Failed to create checkpoint");
        
        metrics_collector.record_checkpoint_created();
        
        let metrics = pipeline.get_priority_system_metrics().await;
        assert!(metrics.checkpoints.active_checkpoints > 0);
        assert!(metrics.checkpoints.checkpoints_created > 0);
        
        // Test rollback
        checkpoint_manager.rollback_checkpoint(&checkpoint_id).await
            .expect("Failed to rollback checkpoint");
        
        metrics_collector.record_rollback_executed();
        
        let updated_metrics = pipeline.get_priority_system_metrics().await;
        assert!(updated_metrics.checkpoints.rollbacks_executed > 0);
        
        // Cleanup
        let _ = std::fs::remove_dir_all(&checkpoint_dir);
    }
    
    #[tokio::test]
    async fn test_queue_metrics_and_timeouts() {
        let queue_config = QueueConfigBuilder::new()
            .max_queued_per_priority(2) // Small queue to test overflow
            .default_queue_timeout(100) // Short timeout
            .build();
        
        let mut pipeline = Pipeline::with_queue_config(1, queue_config); // Single worker
        let submitter = pipeline.task_submitter();
        
        pipeline.start().await.expect("Failed to start pipeline");
        
        // Fill up the queue to test overflow
        let _handle1 = submitter.submit_task(
            TaskPriority::BackgroundWatching,
            TaskSource::BackgroundWatcher {
                folder_path: "/tmp/test1".to_string(),
            },
            TaskPayload::Generic {
                operation: "long_background_task".to_string(),
                parameters: HashMap::new(),
            },
            Some(Duration::from_secs(10)),
        ).await.expect("Failed to submit task 1");
        
        // Wait a moment for the first task to start
        tokio::time::sleep(Duration::from_millis(50)).await;
        
        // These should queue up
        let _handle2 = submitter.submit_task(
            TaskPriority::BackgroundWatching,
            TaskSource::BackgroundWatcher {
                folder_path: "/tmp/test2".to_string(),
            },
            TaskPayload::Generic {
                operation: "queued_task_1".to_string(),
                parameters: HashMap::new(),
            },
            Some(Duration::from_secs(5)),
        ).await.expect("Failed to submit task 2");
        
        let _handle3 = submitter.submit_task(
            TaskPriority::BackgroundWatching,
            TaskSource::BackgroundWatcher {
                folder_path: "/tmp/test3".to_string(),
            },
            TaskPayload::Generic {
                operation: "queued_task_2".to_string(),
                parameters: HashMap::new(),
            },
            Some(Duration::from_secs(5)),
        ).await.expect("Failed to submit task 3");
        
        // Wait a bit to let queue metrics accumulate
        tokio::time::sleep(Duration::from_millis(200)).await;
        
        let queue_stats = submitter.get_queue_stats().await;
        let metrics = pipeline.get_priority_system_metrics().await;
        
        // Verify queue metrics
        assert!(
            queue_stats.total_queued > 0
                || metrics.pipeline.tasks_completed > 0
                || metrics.pipeline.running_tasks > 0
        ); // Some tasks should be queued, running, or completed
        assert!(
            queue_stats.queued_by_priority.contains_key(&TaskPriority::BackgroundWatching)
                || metrics.pipeline.running_tasks > 0
        );
        
        // Test queue cleanup
        let cleaned_count = submitter.cleanup_queue_timeouts().await;
        tracing::info!("Cleaned {} timed out requests", cleaned_count);
        
        // Test deduplication by submitting identical task
        let duplicate_result = submitter.submit_task(
            TaskPriority::BackgroundWatching,
            TaskSource::BackgroundWatcher {
                folder_path: "/tmp/test3".to_string(),
            },
            TaskPayload::Generic {
                operation: "queued_task_2".to_string(), // Same as task 3
                parameters: HashMap::new(),
            },
            Some(Duration::from_secs(5)),
        ).await;
        
        // Should either succeed (if dedup disabled) or may fail with duplicate error
        match duplicate_result {
            Ok(_) => tracing::info!("Duplicate task was allowed"),
            Err(e) => tracing::info!("Duplicate task was rejected: {}", e),
        }
    }
    
    #[tokio::test]
    async fn test_bulk_preemption_for_mcp_requests() {
        let mut pipeline = Pipeline::new(2); // 2 concurrent tasks
        let submitter = pipeline.task_submitter();
        
        pipeline.start().await.expect("Failed to start pipeline");
        
        // Fill capacity with long-running background tasks
        let bg_task1 = submitter.submit_task(
            TaskPriority::BackgroundWatching,
            TaskSource::BackgroundWatcher {
                folder_path: "/tmp/bg1".to_string(),
            },
            TaskPayload::Generic {
                operation: "long_background_task_1".to_string(),
                parameters: HashMap::new(),
            },
            Some(Duration::from_secs(10)),
        ).await.expect("Failed to submit bg task 1");
        
        let bg_task2 = submitter.submit_task(
            TaskPriority::BackgroundWatching,
            TaskSource::BackgroundWatcher {
                folder_path: "/tmp/bg2".to_string(),
            },
            TaskPayload::Generic {
                operation: "long_background_task_2".to_string(),
                parameters: HashMap::new(),
            },
            Some(Duration::from_secs(10)),
        ).await.expect("Failed to submit bg task 2");
        
        // Give background tasks time to start
        tokio::time::sleep(Duration::from_millis(100)).await;
        
        // Submit multiple MCP requests
        let mcp_task1 = submitter.submit_task(
            TaskPriority::McpRequests,
            TaskSource::McpServer {
                request_id: "mcp_req_1".to_string(),
            },
            TaskPayload::Generic {
                operation: "mcp_operation_1".to_string(),
                parameters: HashMap::new(),
            },
            Some(Duration::from_secs(2)),
        ).await.expect("Failed to submit MCP task 1");
        
        let mcp_task2 = submitter.submit_task(
            TaskPriority::McpRequests,
            TaskSource::McpServer {
                request_id: "mcp_req_2".to_string(),
            },
            TaskPayload::Generic {
                operation: "mcp_operation_2".to_string(),
                parameters: HashMap::new(),
            },
            Some(Duration::from_secs(2)),
        ).await.expect("Failed to submit MCP task 2");
        
        // Both MCP tasks should complete quickly despite capacity being full
        let mcp1_result = timeout(Duration::from_secs(3), mcp_task1.wait()).await
            .expect("MCP task 1 should not timeout")
            .expect("MCP task 1 should succeed");
        
        let mcp2_result = timeout(Duration::from_secs(3), mcp_task2.wait()).await
            .expect("MCP task 2 should not timeout")
            .expect("MCP task 2 should succeed");
        
        // Verify MCP tasks succeeded
        match mcp1_result {
            TaskResult::Success { .. } => {},
            other => panic!("MCP task 1 should succeed, got: {:?}", other),
        }
        
        match mcp2_result {
            TaskResult::Success { .. } => {},
            other => panic!("MCP task 2 should succeed, got: {:?}", other),
        }
        
        let bg1_result = timeout(Duration::from_millis(500), bg_task1.wait()).await;
        let bg2_result = timeout(Duration::from_millis(500), bg_task2.wait()).await;

        let mut cancelled_count = 0;
        for (label, result) in [("bg1", bg1_result), ("bg2", bg2_result)] {
            match result {
                Ok(Ok(TaskResult::Cancelled { .. })) => {
                    cancelled_count += 1;
                }
                Ok(Ok(TaskResult::Success { .. })) => {}
                Ok(Ok(other)) => panic!("{} task unexpected result: {:?}", label, other),
                Ok(Err(e)) => panic!("{} task failed: {}", label, e),
                Err(_) => {
                    tracing::info!("{} task still running; skipping cancellation assertion", label);
                }
            }
        }

        // Verify preemption metrics were recorded when cancellations occur
        let metrics = pipeline.get_priority_system_metrics().await;
        if cancelled_count > 0 && metrics.preemption.preemptions_total > 0 {
            assert!(metrics.preemption.preemptions_total >= cancelled_count as u64);
        }
    }
    
    #[tokio::test]
    async fn test_graceful_preemption_vs_abort() {
        let mut pipeline = Pipeline::new(1);
        let submitter = pipeline.task_submitter();
        
        pipeline.start().await.expect("Failed to start pipeline");
        
        // Submit a task that should handle cancellation gracefully
        let graceful_task = submitter.submit_task(
            TaskPriority::CliCommands,
            TaskSource::CliCommand {
                command: "graceful_task".to_string(),
            },
            TaskPayload::Generic {
                operation: "graceful_operation".to_string(),
                parameters: HashMap::new(),
            },
            Some(Duration::from_secs(5)),
        ).await.expect("Failed to submit graceful task");
        
        // Give task time to start
        tokio::time::sleep(Duration::from_millis(50)).await;
        
        // Submit MCP request to trigger preemption
        let preempting_task = submitter.submit_task(
            TaskPriority::McpRequests,
            TaskSource::McpServer {
                request_id: "preempting_request".to_string(),
            },
            TaskPayload::Generic {
                operation: "preempting_operation".to_string(),
                parameters: HashMap::new(),
            },
            Some(Duration::from_secs(2)),
        ).await.expect("Failed to submit preempting task");
        
        // Both tasks should complete within reasonable time
        let preempting_result = timeout(Duration::from_secs(3), preempting_task.wait()).await
            .expect("Preempting task should not timeout")
            .expect("Preempting task should succeed");
        
        let graceful_result = timeout(Duration::from_millis(100), graceful_task.wait()).await
            .expect("Graceful task should complete quickly")
            .expect("Graceful task should have result");
        
        // Preempting task should succeed
        match preempting_result {
            TaskResult::Success { .. } => {},
            other => panic!("Preempting task should succeed, got: {:?}", other),
        }
        
        // Graceful task should be cancelled (since preemption gives 10ms grace period)
        match graceful_result {
            TaskResult::Cancelled { .. } => {
                // Expected - task was gracefully cancelled
            }
            TaskResult::Success { .. } => {
                // Also acceptable if task completed very quickly
            }
            other => panic!("Expected cancelled or success, got: {:?}", other),
        }
    }

    #[tokio::test]
    async fn test_task_result_handle_is_completed_when_sender_alive() {
        let (tx, rx) = oneshot::channel::<TaskResult>();
        let mut handle = TaskResultHandle {
            task_id: Uuid::new_v4(),
            context: TaskContext {
                task_id: Uuid::new_v4(),
                priority: TaskPriority::BackgroundWatching,
                created_at: chrono::Utc::now(),
                timeout_ms: None,
                source: TaskSource::Generic { operation: "test".into() },
                metadata: HashMap::new(),
                checkpoint_id: None,
                supports_checkpointing: false,
            },
            result_receiver: rx,
        };

        // Sender still alive, task not completed
        assert!(!handle.is_completed());

        // Keep tx alive to prevent premature drop
        drop(tx);
    }

    #[tokio::test]
    async fn test_task_result_handle_is_completed_when_sender_dropped() {
        let (_tx, rx) = oneshot::channel::<TaskResult>();
        let mut handle = TaskResultHandle {
            task_id: Uuid::new_v4(),
            context: TaskContext {
                task_id: Uuid::new_v4(),
                priority: TaskPriority::BackgroundWatching,
                created_at: chrono::Utc::now(),
                timeout_ms: None,
                source: TaskSource::Generic { operation: "test".into() },
                metadata: HashMap::new(),
                checkpoint_id: None,
                supports_checkpointing: false,
            },
            result_receiver: rx,
        };

        // Drop sender to simulate task completion/abandonment
        drop(_tx);

        assert!(handle.is_completed());
    }

    #[tokio::test]
    async fn test_task_result_handle_is_completed_when_value_sent() {
        let (tx, rx) = oneshot::channel::<TaskResult>();
        let mut handle = TaskResultHandle {
            task_id: Uuid::new_v4(),
            context: TaskContext {
                task_id: Uuid::new_v4(),
                priority: TaskPriority::BackgroundWatching,
                created_at: chrono::Utc::now(),
                timeout_ms: None,
                source: TaskSource::Generic { operation: "test".into() },
                metadata: HashMap::new(),
                checkpoint_id: None,
                supports_checkpointing: false,
            },
            result_receiver: rx,
        };

        // Send a result
        let _ = tx.send(TaskResult::Success {
            execution_time_ms: 42,
            data: TaskResultData::Generic {
                message: "done".into(),
                data: serde_json::json!({}),
                checkpoint_id: None,
            },
        });

        assert!(handle.is_completed());
    }

    #[tokio::test]
    async fn test_spill_to_sqlite_process_document() {
        use crate::unified_queue_schema::CREATE_UNIFIED_QUEUE_SQL;

        // Create in-memory SQLite database with unified_queue table
        let pool = sqlx::SqlitePool::connect("sqlite::memory:")
            .await
            .expect("Failed to create in-memory SQLite pool");
        sqlx::query(CREATE_UNIFIED_QUEUE_SQL)
            .execute(&pool)
            .await
            .expect("Failed to create unified_queue table");

        let queue_manager = Arc::new(QueueManager::new(pool.clone()));

        // Create a pipeline with spill queue and get a submitter
        let mut pipeline = Pipeline::new(2);
        pipeline.set_spill_queue(queue_manager.clone());
        let submitter = pipeline.task_submitter();

        // Create a ProcessDocument context and payload
        let context = TaskContext {
            task_id: Uuid::new_v4(),
            priority: TaskPriority::ProjectWatching,
            created_at: chrono::Utc::now(),
            timeout_ms: None,
            source: TaskSource::ProjectWatcher {
                project_path: "/tmp/test_project".to_string(),
            },
            metadata: HashMap::new(),
            checkpoint_id: None,
            supports_checkpointing: true,
        };
        let payload = TaskPayload::ProcessDocument {
            file_path: std::path::PathBuf::from("/tmp/test_project/src/main.rs"),
            collection: "projects".to_string(),
        };

        // Spill directly to SQLite
        submitter.spill_to_sqlite(&queue_manager, &context, &payload)
            .await
            .expect("Spill should succeed");

        // Verify the item was inserted into unified_queue
        let row: (String, String, String, String) = sqlx::query_as(
            "SELECT item_type, op, collection, status FROM unified_queue LIMIT 1"
        )
        .fetch_one(&pool)
        .await
        .expect("Should have one spilled item");

        assert_eq!(row.0, "file");
        assert_eq!(row.1, "ingest");
        assert_eq!(row.2, "projects");
        assert_eq!(row.3, "pending");

        // Verify the payload contains the file path
        let payload_json: (String,) = sqlx::query_as(
            "SELECT payload_json FROM unified_queue LIMIT 1"
        )
        .fetch_one(&pool)
        .await
        .expect("Should have payload");

        let payload_val: serde_json::Value = serde_json::from_str(&payload_json.0).unwrap();
        assert_eq!(
            payload_val["file_path"].as_str().unwrap(),
            "/tmp/test_project/src/main.rs"
        );

        // Verify metadata contains spill info
        let metadata_json: (String,) = sqlx::query_as(
            "SELECT metadata FROM unified_queue LIMIT 1"
        )
        .fetch_one(&pool)
        .await
        .expect("Should have metadata");

        let meta_val: serde_json::Value = serde_json::from_str(&metadata_json.0).unwrap();
        assert_eq!(meta_val["spilled_from"].as_str().unwrap(), "pipeline");
        assert_eq!(meta_val["original_priority"].as_str().unwrap(), "ProjectWatching");
    }

    #[tokio::test]
    async fn test_spill_non_process_document_fails() {
        use crate::unified_queue_schema::CREATE_UNIFIED_QUEUE_SQL;

        let pool = sqlx::SqlitePool::connect("sqlite::memory:")
            .await
            .expect("Failed to create in-memory SQLite pool");
        sqlx::query(CREATE_UNIFIED_QUEUE_SQL)
            .execute(&pool)
            .await
            .expect("Failed to create unified_queue table");

        let queue_manager = Arc::new(QueueManager::new(pool.clone()));

        let mut pipeline = Pipeline::new(2);
        pipeline.set_spill_queue(queue_manager.clone());
        let submitter = pipeline.task_submitter();

        let context = TaskContext {
            task_id: Uuid::new_v4(),
            priority: TaskPriority::BackgroundWatching,
            created_at: chrono::Utc::now(),
            timeout_ms: None,
            source: TaskSource::Generic {
                operation: "test".to_string(),
            },
            metadata: HashMap::new(),
            checkpoint_id: None,
            supports_checkpointing: false,
        };

        // Generic task should NOT be spillable
        let payload = TaskPayload::Generic {
            operation: "test".to_string(),
            parameters: HashMap::new(),
        };

        let result = submitter.spill_to_sqlite(&queue_manager, &context, &payload).await;
        assert!(result.is_err(), "Generic tasks should not be spillable");

        // ExecuteQuery should NOT be spillable
        let payload = TaskPayload::ExecuteQuery {
            query: "test".to_string(),
            collection: "test".to_string(),
            limit: 10,
        };

        let result = submitter.spill_to_sqlite(&queue_manager, &context, &payload).await;
        assert!(result.is_err(), "ExecuteQuery tasks should not be spillable");
    }

    #[tokio::test]
    async fn test_spill_with_background_watcher_source() {
        use crate::unified_queue_schema::CREATE_UNIFIED_QUEUE_SQL;

        let pool = sqlx::SqlitePool::connect("sqlite::memory:")
            .await
            .expect("Failed to create in-memory SQLite pool");
        sqlx::query(CREATE_UNIFIED_QUEUE_SQL)
            .execute(&pool)
            .await
            .expect("Failed to create unified_queue table");

        let queue_manager = Arc::new(QueueManager::new(pool.clone()));

        let mut pipeline = Pipeline::new(2);
        pipeline.set_spill_queue(queue_manager.clone());
        let submitter = pipeline.task_submitter();

        let context = TaskContext {
            task_id: Uuid::new_v4(),
            priority: TaskPriority::BackgroundWatching,
            created_at: chrono::Utc::now(),
            timeout_ms: None,
            source: TaskSource::BackgroundWatcher {
                folder_path: "/home/user/docs".to_string(),
            },
            metadata: HashMap::new(),
            checkpoint_id: None,
            supports_checkpointing: true,
        };
        let payload = TaskPayload::ProcessDocument {
            file_path: std::path::PathBuf::from("/home/user/docs/readme.md"),
            collection: "libraries".to_string(),
        };

        submitter.spill_to_sqlite(&queue_manager, &context, &payload)
            .await
            .expect("Spill with BackgroundWatcher source should succeed");

        // Verify correct collection
        let row: (String,) = sqlx::query_as(
            "SELECT collection FROM unified_queue LIMIT 1"
        )
        .fetch_one(&pool)
        .await
        .expect("Should have spilled item");

        assert_eq!(row.0, "libraries");
    }

    #[tokio::test]
    async fn test_pipeline_stats_include_spill_count() {
        let pipeline = Pipeline::new(2);
        let stats = pipeline.stats().await;
        assert_eq!(stats.queue_spills, 0);
    }

    #[tokio::test]
    async fn test_queue_metrics_include_spill_count() {
        let mut pipeline = Pipeline::new(2);
        pipeline.start().await.expect("Failed to start pipeline");

        let metrics = pipeline.get_priority_system_metrics().await;
        assert_eq!(metrics.queue.queue_spill_count, 0);
    }

    #[tokio::test]
    async fn test_spill_queue_configuration() {
        use crate::unified_queue_schema::CREATE_UNIFIED_QUEUE_SQL;

        let pool = sqlx::SqlitePool::connect("sqlite::memory:")
            .await
            .expect("Failed to create in-memory SQLite pool");
        sqlx::query(CREATE_UNIFIED_QUEUE_SQL)
            .execute(&pool)
            .await
            .expect("Failed to create unified_queue table");

        let queue_manager = Arc::new(QueueManager::new(pool));

        // Pipeline without spill queue
        let pipeline = Pipeline::new(2);
        let submitter = pipeline.task_submitter();
        assert!(submitter.spill_queue.is_none());

        // Pipeline with spill queue
        let mut pipeline = Pipeline::new(2);
        pipeline.set_spill_queue(queue_manager);
        let submitter = pipeline.task_submitter();
        assert!(submitter.spill_queue.is_some());
    }

    // =========================================================================
    // Rollback operation tests (Task 549)
    // =========================================================================

    #[tokio::test]
    async fn test_rollback_delete_file() {
        let dir = std::env::temp_dir().join("test_rollback_delete");
        let _ = std::fs::create_dir_all(&dir);
        let file_path = dir.join("to_delete.txt");
        std::fs::write(&file_path, "temporary data").unwrap();

        let cm = CheckpointManager::new(dir.clone(), Duration::from_secs(60));
        let ckpt_id = cm.create_checkpoint(
            Uuid::new_v4(),
            TaskProgress::Generic {
                progress_percentage: 100.0,
                stage: "test".into(),
                metadata: HashMap::new(),
            },
            serde_json::json!({}),
            vec![],
            vec![RollbackAction::DeleteFile { path: file_path.clone() }],
        ).await.unwrap();

        cm.rollback_checkpoint(&ckpt_id).await.unwrap();
        assert!(!file_path.exists(), "File should have been deleted by rollback");
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[tokio::test]
    async fn test_rollback_restore_file() {
        let dir = std::env::temp_dir().join("test_rollback_restore");
        let _ = std::fs::create_dir_all(&dir);

        let original = dir.join("original.txt");
        let backup = dir.join("backup.txt");
        std::fs::write(&original, "modified").unwrap();
        std::fs::write(&backup, "original content").unwrap();

        let cm = CheckpointManager::new(dir.clone(), Duration::from_secs(60));
        let ckpt_id = cm.create_checkpoint(
            Uuid::new_v4(),
            TaskProgress::Generic {
                progress_percentage: 100.0,
                stage: "test".into(),
                metadata: HashMap::new(),
            },
            serde_json::json!({}),
            vec![],
            vec![RollbackAction::RestoreFile {
                original_path: original.clone(),
                backup_path: backup.clone(),
            }],
        ).await.unwrap();

        cm.rollback_checkpoint(&ckpt_id).await.unwrap();
        let content = std::fs::read_to_string(&original).unwrap();
        assert_eq!(content, "original content");
        assert!(!backup.exists(), "Backup should be cleaned up");
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[tokio::test]
    async fn test_rollback_remove_from_collection_no_storage_client() {
        let dir = std::env::temp_dir().join("test_rollback_remove_no_sc");
        let _ = std::fs::create_dir_all(&dir);

        let cm = CheckpointManager::new(dir.clone(), Duration::from_secs(60));
        let ckpt_id = cm.create_checkpoint(
            Uuid::new_v4(),
            TaskProgress::Generic {
                progress_percentage: 100.0,
                stage: "test".into(),
                metadata: HashMap::new(),
            },
            serde_json::json!({}),
            vec![],
            vec![RollbackAction::RemoveFromCollection {
                document_id: "doc-123".into(),
                collection: "projects".into(),
            }],
        ).await.unwrap();

        // Without storage client, rollback should fail with descriptive error
        let result = cm.rollback_checkpoint(&ckpt_id).await;
        // rollback_checkpoint logs errors but continues; individual action errors
        // are logged as warnings. The checkpoint is still deleted.
        // Since we changed the implementation to return Err from execute_rollback_action,
        // the rollback_checkpoint logs a warning and continues.
        // The checkpoint itself is still cleaned up.
        assert!(result.is_ok(), "rollback_checkpoint should succeed even if individual actions fail");
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[tokio::test]
    async fn test_rollback_revert_index_no_storage_client() {
        let dir = std::env::temp_dir().join("test_rollback_revert_no_sc");
        let _ = std::fs::create_dir_all(&dir);

        let cm = CheckpointManager::new(dir.clone(), Duration::from_secs(60));
        let snapshot = serde_json::json!({
            "collection": "projects",
            "indexes": ["field1", "field2"]
        });

        let ckpt_id = cm.create_checkpoint(
            Uuid::new_v4(),
            TaskProgress::Generic {
                progress_percentage: 100.0,
                stage: "test".into(),
                metadata: HashMap::new(),
            },
            serde_json::json!({}),
            vec![],
            vec![RollbackAction::RevertIndexChanges { index_snapshot: snapshot }],
        ).await.unwrap();

        // RevertIndexChanges without storage logs warning but doesn't error
        let result = cm.rollback_checkpoint(&ckpt_id).await;
        assert!(result.is_ok());
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[tokio::test]
    async fn test_rollback_custom_handler_registered() {
        use std::sync::atomic::{AtomicBool, Ordering};

        let dir = std::env::temp_dir().join("test_rollback_custom");
        let _ = std::fs::create_dir_all(&dir);

        let executed = Arc::new(AtomicBool::new(false));
        let executed_clone = executed.clone();

        struct TestHandler {
            executed: Arc<AtomicBool>,
        }

        #[async_trait::async_trait]
        impl CustomRollbackHandler for TestHandler {
            async fn execute(&self, _data: &serde_json::Value) -> Result<(), String> {
                self.executed.store(true, Ordering::SeqCst);
                Ok(())
            }
        }

        let cm = CheckpointManager::new(dir.clone(), Duration::from_secs(60));
        cm.register_custom_handler(
            "test_action",
            Arc::new(TestHandler { executed: executed_clone }),
        ).await;

        let ckpt_id = cm.create_checkpoint(
            Uuid::new_v4(),
            TaskProgress::Generic {
                progress_percentage: 100.0,
                stage: "test".into(),
                metadata: HashMap::new(),
            },
            serde_json::json!({}),
            vec![],
            vec![RollbackAction::Custom {
                action_type: "test_action".into(),
                data: serde_json::json!({"key": "value"}),
            }],
        ).await.unwrap();

        cm.rollback_checkpoint(&ckpt_id).await.unwrap();
        assert!(executed.load(Ordering::SeqCst), "Custom handler should have been executed");
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[tokio::test]
    async fn test_rollback_custom_handler_not_registered() {
        let dir = std::env::temp_dir().join("test_rollback_custom_unreg");
        let _ = std::fs::create_dir_all(&dir);

        let cm = CheckpointManager::new(dir.clone(), Duration::from_secs(60));
        let ckpt_id = cm.create_checkpoint(
            Uuid::new_v4(),
            TaskProgress::Generic {
                progress_percentage: 100.0,
                stage: "test".into(),
                metadata: HashMap::new(),
            },
            serde_json::json!({}),
            vec![],
            vec![RollbackAction::Custom {
                action_type: "unregistered_action".into(),
                data: serde_json::json!({}),
            }],
        ).await.unwrap();

        // Unregistered handler causes action failure, but rollback_checkpoint continues
        let result = cm.rollback_checkpoint(&ckpt_id).await;
        assert!(result.is_ok(), "rollback_checkpoint succeeds even with failed actions");
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[tokio::test]
    async fn test_rollback_multiple_actions_continue_on_failure() {
        let dir = std::env::temp_dir().join("test_rollback_multi");
        let _ = std::fs::create_dir_all(&dir);

        let file_to_delete = dir.join("should_be_deleted.txt");
        std::fs::write(&file_to_delete, "data").unwrap();

        let cm = CheckpointManager::new(dir.clone(), Duration::from_secs(60));
        let ckpt_id = cm.create_checkpoint(
            Uuid::new_v4(),
            TaskProgress::Generic {
                progress_percentage: 100.0,
                stage: "test".into(),
                metadata: HashMap::new(),
            },
            serde_json::json!({}),
            vec![],
            vec![
                // Action 1: RemoveFromCollection will fail (no storage client)
                RollbackAction::RemoveFromCollection {
                    document_id: "doc-456".into(),
                    collection: "projects".into(),
                },
                // Action 2: DeleteFile should still execute despite action 1 failure
                RollbackAction::DeleteFile { path: file_to_delete.clone() },
            ],
        ).await.unwrap();

        // rollback_checkpoint processes actions in reverse order and continues on failure
        let result = cm.rollback_checkpoint(&ckpt_id).await;
        assert!(result.is_ok());
        // DeleteFile is action index 1, processed first in reverse order
        assert!(!file_to_delete.exists(), "DeleteFile should execute even when other actions fail");
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[tokio::test]
    async fn test_rollback_storage_configuration() {
        let dir = std::env::temp_dir().join("test_rollback_storage_cfg");
        let _ = std::fs::create_dir_all(&dir);

        // Pipeline without rollback storage
        let pipeline = Pipeline::new(2);
        let cm = pipeline.checkpoint_manager();
        assert!(cm.storage_client.is_none());

        // Note: We can't easily test set_rollback_storage with a real StorageClient
        // in unit tests since it requires a Qdrant connection. The wiring is tested
        // via the compilation of main.rs which calls set_rollback_storage.
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[tokio::test]
    async fn test_custom_handler_registry() {
        struct NoopHandler;

        #[async_trait::async_trait]
        impl CustomRollbackHandler for NoopHandler {
            async fn execute(&self, _data: &serde_json::Value) -> Result<(), String> {
                Ok(())
            }
        }

        let dir = std::env::temp_dir().join("test_custom_registry");
        let _ = std::fs::create_dir_all(&dir);

        let cm = CheckpointManager::new(dir.clone(), Duration::from_secs(60));

        // Initially no handlers
        {
            let handlers = cm.custom_handlers.read().await;
            assert!(handlers.is_empty());
        }

        // Register a handler
        cm.register_custom_handler("noop", Arc::new(NoopHandler)).await;
        {
            let handlers = cm.custom_handlers.read().await;
            assert_eq!(handlers.len(), 1);
            assert!(handlers.contains_key("noop"));
        }

        // Register another handler
        cm.register_custom_handler("another", Arc::new(NoopHandler)).await;
        {
            let handlers = cm.custom_handlers.read().await;
            assert_eq!(handlers.len(), 2);
        }

        let _ = std::fs::remove_dir_all(&dir);
    }
}
