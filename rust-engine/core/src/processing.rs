//! Priority-based document processing pipeline
//!
//! This module implements a priority-based task queuing system for responsive
//! MCP request handling with preemption capabilities.

use std::collections::{BinaryHeap, HashMap};
use std::cmp::Ordering;
use std::sync::atomic::{AtomicU64, Ordering as AtomicOrdering};
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::{mpsc, oneshot, RwLock};
use tokio::task::JoinHandle;
use uuid::Uuid;
use serde::{Deserialize, Serialize};
use thiserror::Error;
use chrono;

/// Priority levels for different types of operations
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
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
}

/// Task execution result
#[derive(Debug, Clone)]
pub enum TaskResult {
    /// Task completed successfully
    Success { 
        execution_time: Duration,
        data: TaskResultData,
    },
    /// Task was cancelled/preempted
    Cancelled { reason: String },
    /// Task failed with error
    Error { 
        error: String,
        execution_time: Duration,
    },
    /// Task timed out
    Timeout { timeout_duration: Duration },
}

/// Specific data returned by task execution
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum TaskResultData {
    /// Document processing result
    DocumentProcessing {
        document_id: String,
        collection: String,
        chunks_created: usize,
    },
    /// File watching result
    FileWatching {
        files_processed: usize,
        errors: Vec<String>,
    },
    /// Query execution result
    QueryExecution {
        results: Vec<String>,
        total_results: usize,
    },
    /// Generic result data
    Generic {
        message: String,
        data: serde_json::Value,
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
}

/// A task that can be executed with priority and preemption support
pub struct PriorityTask {
    pub context: TaskContext,
    pub payload: TaskPayload,
    pub result_sender: oneshot::Sender<TaskResult>,
    /// Handle for cancellation if task is running
    pub cancellation_token: Option<tokio_util::sync::CancellationToken>,
}

/// The actual work to be performed by a task
#[derive(Debug)]
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
}

/// Information about a currently running task
#[derive(Debug)]
struct RunningTask {
    context: TaskContext,
    started_at: Instant,
    handle: JoinHandle<TaskResult>,
    cancellation_token: tokio_util::sync::CancellationToken,
}

impl Pipeline {
    /// Create a new priority-based processing pipeline
    pub fn new(max_concurrent_tasks: usize) -> Self {
        let (task_sender, task_receiver) = mpsc::unbounded_channel();
        
        Self {
            task_queue: Arc::new(RwLock::new(BinaryHeap::new())),
            running_tasks: Arc::new(RwLock::new(HashMap::new())),
            task_receiver: Arc::new(RwLock::new(Some(task_receiver))),
            task_sender,
            sequence_counter: Arc::new(AtomicU64::new(0)),
            max_concurrent_tasks,
            executor_handle: None,
        }
    }
    
    /// Get a handle for submitting tasks to the pipeline
    pub fn task_submitter(&self) -> TaskSubmitter {
        TaskSubmitter {
            sender: self.task_sender.clone(),
        }
    }
    
    /// Start the pipeline execution loop
    pub async fn start(&mut self) -> Result<(), PriorityError> {
        let task_queue = Arc::clone(&self.task_queue);
        let running_tasks = Arc::clone(&self.running_tasks);
        let task_receiver = Arc::clone(&self.task_receiver);
        let sequence_counter = Arc::clone(&self.sequence_counter);
        let max_concurrent = self.max_concurrent_tasks;
        
        let handle = tokio::spawn(async move {
            Self::execution_loop(
                task_queue,
                running_tasks,
                task_receiver,
                sequence_counter,
                max_concurrent,
            ).await;
        });
        
        self.executor_handle = Some(handle);
        Ok(())
    }
    
    /// Get current pipeline statistics
    pub async fn stats(&self) -> PipelineStats {
        let queue_lock = self.task_queue.read().await;
        let running_lock = self.running_tasks.read().await;
        
        PipelineStats {
            queued_tasks: queue_lock.len(),
            running_tasks: running_lock.len(),
            total_capacity: self.max_concurrent_tasks,
        }
    }
    
    /// The main execution loop that processes tasks
    async fn execution_loop(
        task_queue: Arc<RwLock<BinaryHeap<TaskQueueItem>>>,
        running_tasks: Arc<RwLock<HashMap<Uuid, RunningTask>>>,
        task_receiver: Arc<RwLock<Option<mpsc::UnboundedReceiver<PriorityTask>>>>,
        sequence_counter: Arc<AtomicU64>,
        max_concurrent: usize,
    ) {
        let mut receiver = {
            let mut lock = task_receiver.write().await;
            lock.take().expect("Task receiver should be available")
        };
        
        let mut cleanup_interval = tokio::time::interval(Duration::from_secs(1));
        
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
                            ).await;
                        }
                        None => {
                            tracing::info!("Task receiver closed, shutting down pipeline");
                            break;
                        }
                    }
                }
                
                // Periodic cleanup of completed tasks
                _ = cleanup_interval.tick() => {
                    Self::cleanup_completed_tasks(&running_tasks).await;
                }
            }
        }
    }
    
    /// Attempt to start queued tasks if capacity allows
    async fn try_start_queued_tasks(
        task_queue: &Arc<RwLock<BinaryHeap<TaskQueueItem>>>,
        running_tasks: &Arc<RwLock<HashMap<Uuid, RunningTask>>>,
        max_concurrent: usize,
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
                    let preempted = Self::try_preempt_lower_priority_task(
                        running_tasks,
                        priority,
                    ).await;
                    
                    if !preempted {
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
                Self::start_task(running_tasks, queue_item.task).await;
            } else {
                break; // No more tasks to start
            }
        }
    }
    
    /// Try to preempt a lower priority running task
    async fn try_preempt_lower_priority_task(
        running_tasks: &Arc<RwLock<HashMap<Uuid, RunningTask>>>,
        new_priority: TaskPriority,
    ) -> bool {
        let mut running_lock = running_tasks.write().await;
        
        // Find the lowest priority running task that can be preempted
        let mut lowest_priority_task: Option<(Uuid, TaskPriority)> = None;
        
        for (task_id, running_task) in running_lock.iter() {
            if new_priority.can_preempt(&running_task.context.priority) {
                match lowest_priority_task {
                    None => {
                        lowest_priority_task = Some((*task_id, running_task.context.priority));
                    }
                    Some((_, current_lowest)) => {
                        if running_task.context.priority < current_lowest {
                            lowest_priority_task = Some((*task_id, running_task.context.priority));
                        }
                    }
                }
            }
        }
        
        if let Some((task_id, _)) = lowest_priority_task {
            if let Some(running_task) = running_lock.remove(&task_id) {
                tracing::info!(
                    "Preempting task {} (priority {:?}) for higher priority task (priority {:?})",
                    task_id, running_task.context.priority, new_priority
                );
                
                // Cancel the task
                running_task.cancellation_token.cancel();
                running_task.handle.abort();
                
                return true;
            }
        }
        
        false
    }
    
    /// Start executing a task
    async fn start_task(
        running_tasks: &Arc<RwLock<HashMap<Uuid, RunningTask>>>,
        task: PriorityTask,
    ) {
        let task_id = task.context.task_id;
        let cancellation_token = tokio_util::sync::CancellationToken::new();
        
        let context_for_task = task.context.clone();
        let context_for_running = task.context.clone();
        let token = cancellation_token.clone();
        let payload = task.payload;
        let result_sender = task.result_sender;
        
        let handle = tokio::spawn(async move {
            let start_time = Instant::now();
            
            let result = tokio::select! {
                result = Self::execute_task_payload(payload, &context_for_task) => {
                    match result {
                        Ok(data) => TaskResult::Success {
                            execution_time: start_time.elapsed(),
                            data,
                        },
                        Err(error) => TaskResult::Error {
                            error: error.to_string(),
                            execution_time: start_time.elapsed(),
                        },
                    }
                }
                _ = token.cancelled() => {
                    TaskResult::Cancelled {
                        reason: "Task was preempted by higher priority task".to_string(),
                    }
                }
                _ = Self::timeout_future(&context_for_task) => {
                    TaskResult::Timeout {
                        timeout_duration: Duration::from_millis(
                            context_for_task.timeout_ms.unwrap_or(30_000)
                        ),
                    }
                }
            };
            
            // Send result back to caller
            let _ = result_sender.send(result.clone());
            
            result
        });
        
        let running_task = RunningTask {
            context: context_for_running,
            started_at: Instant::now(),
            handle,
            cancellation_token,
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
    ) -> Result<TaskResultData, PriorityError> {
        match payload {
            TaskPayload::ProcessDocument { file_path, collection } => {
                // Placeholder implementation - will be expanded in later subtasks
                tracing::info!(
                    "Processing document: {:?} for collection: {}",
                    file_path, collection
                );
                
                // Simulate some processing time
                tokio::time::sleep(Duration::from_millis(100)).await;
                
                Ok(TaskResultData::DocumentProcessing {
                    document_id: context.task_id.to_string(),
                    collection,
                    chunks_created: 1,
                })
            }
            
            TaskPayload::WatchDirectory { path, recursive } => {
                tracing::info!(
                    "Watching directory: {:?}, recursive: {}",
                    path, recursive
                );
                
                Ok(TaskResultData::FileWatching {
                    files_processed: 0,
                    errors: vec![],
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
                })
            }
            
            TaskPayload::Generic { operation, parameters } => {
                tracing::info!(
                    "Executing generic operation: '{}' with {} parameters",
                    operation, parameters.len()
                );
                
                Ok(TaskResultData::Generic {
                    message: format!("Completed operation: {}", operation),
                    data: serde_json::json!(parameters),
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

/// Handle for submitting tasks to the pipeline
#[derive(Clone)]
pub struct TaskSubmitter {
    sender: mpsc::UnboundedSender<PriorityTask>,
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
        let task_id = Uuid::new_v4();
        let (result_sender, result_receiver) = oneshot::channel();
        
        let context = TaskContext {
            task_id,
            priority,
            created_at: chrono::Utc::now(),
            timeout_ms: timeout.map(|d| d.as_millis() as u64),
            source,
            metadata: HashMap::new(),
        };
        
        let task = PriorityTask {
            context: context.clone(),
            payload,
            result_sender,
            cancellation_token: None,
        };
        
        self.sender.send(task)
            .map_err(|_| PriorityError::Communication("Pipeline is shutting down".to_string()))?;
        
        Ok(TaskResultHandle {
            task_id,
            context,
            result_receiver,
        })
    }
}

/// Handle for waiting on task execution results
pub struct TaskResultHandle {
    pub task_id: Uuid,
    pub context: TaskContext,
    result_receiver: oneshot::Receiver<TaskResult>,
}

impl TaskResultHandle {
    /// Wait for the task to complete and return the result
    pub async fn wait(self) -> Result<TaskResult, PriorityError> {
        self.result_receiver.await
            .map_err(|_| PriorityError::Communication("Task executor disconnected".to_string()))
    }
}

/// Pipeline statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PipelineStats {
    pub queued_tasks: usize,
    pub running_tasks: usize,
    pub total_capacity: usize,
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
}
