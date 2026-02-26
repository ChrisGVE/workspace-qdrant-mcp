//! Priority-based document processing pipeline
//!
//! This module implements a priority-based task queuing system for responsive
//! MCP request handling with preemption capabilities.

mod checkpoint;
mod executor;
mod metrics;
mod pipeline;
mod request_queue;
mod submitter;
#[cfg(test)]
mod tests;

// Re-export all public types from submodules
pub use checkpoint::{
    CheckpointManager, CustomRollbackHandler, RollbackAction, TaskCheckpoint, TaskProgress,
};
pub use metrics::{
    CheckpointMetrics, MetricsCollector, PerformanceMetrics, PipelineMetrics, PreemptionMetrics,
    PrioritySystemMetrics, QueueMetrics, ResourceMetrics,
};
pub use pipeline::Pipeline;
pub use request_queue::{QueueConfig, QueueConfigBuilder, QueueStats, RequestQueue};
pub use submitter::{TaskResultHandle, TaskSubmitter};

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::time::Duration;
use thiserror::Error;
use uuid::Uuid;

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
    /// Pre-computed tenant_id from watch_folders (single source of truth).
    /// Populated from the queue item's tenant_id at task creation time.
    /// Used in spill_to_sqlite() to avoid re-deriving from filesystem.
    #[serde(default)]
    pub tenant_id: Option<String>,
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
    pub result_sender: tokio::sync::oneshot::Sender<TaskResult>,
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
        branch: String,
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
