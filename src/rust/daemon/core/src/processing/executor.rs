//! Task execution loop with preemption and concurrent task management
//!
//! Contains the core execution loop that processes tasks from the priority
//! queue, manages concurrent task slots, handles preemption of lower-priority
//! tasks, and executes task payloads.

use std::cmp::Ordering;
use std::collections::{BinaryHeap, HashMap};
use std::sync::atomic::{AtomicU64, Ordering as AtomicOrdering};
use std::sync::Arc;
use std::time::{Duration, Instant};

use tokio::sync::{mpsc, RwLock};
use tokio::task::JoinHandle;
use uuid::Uuid;

use super::checkpoint::{CheckpointManager, TaskCheckpoint};
use super::{
    PriorityError, PriorityTask, TaskContext, TaskPayload, TaskPriority, TaskResult,
    TaskResultData,
};

/// Priority queue implementation for tasks
/// Uses reverse ordering so highest priority comes first
pub(crate) struct TaskQueueItem {
    pub(crate) task: PriorityTask,
    pub(crate) sequence: u64,
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

/// Information about a currently running task
#[derive(Debug)]
pub(crate) struct RunningTask {
    pub(crate) context: TaskContext,
    pub(crate) started_at: Instant,
    pub(crate) handle: JoinHandle<TaskResult>,
    pub(crate) cancellation_token: tokio_util::sync::CancellationToken,
    /// Current checkpoint if task supports checkpointing
    pub(crate) current_checkpoint: Option<TaskCheckpoint>,
    /// Whether the task is in a preemptible state
    pub(crate) is_preemptible: bool,
    /// Progress tracking for consistency checks
    #[allow(dead_code)]
    pub(crate) last_progress_update: Instant,
}

/// The main execution loop that processes tasks
pub(crate) async fn execution_loop(
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
    let mut checkpoint_cleanup_interval = tokio::time::interval(Duration::from_secs(300));

    loop {
        tokio::select! {
            task = receiver.recv() => {
                match task {
                    Some(task) => {
                        let sequence = sequence_counter.fetch_add(1, AtomicOrdering::Relaxed);
                        let queue_item = TaskQueueItem { task, sequence };

                        let mut queue_lock = task_queue.write().await;
                        queue_lock.push(queue_item);
                        drop(queue_lock);

                        try_start_queued_tasks(
                            &task_queue, &running_tasks, max_concurrent, &ingestion_engine,
                        ).await;
                    }
                    None => {
                        tracing::info!("Task receiver closed, shutting down pipeline");
                        break;
                    }
                }
            }

            _ = cleanup_interval.tick() => {
                cleanup_completed_tasks(&running_tasks).await;
                try_start_queued_tasks(
                    &task_queue, &running_tasks, max_concurrent, &ingestion_engine,
                ).await;
            }

            _ = queue_cleanup_interval.tick() => {
                // Placeholder: queue timeout cleanup handled externally
            }

            _ = checkpoint_cleanup_interval.tick() => {
                // Placeholder: checkpoint cleanup handled externally
            }
        }
    }
}

/// Attempt to start queued tasks if capacity allows
pub(crate) async fn try_start_queued_tasks(
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
            let next_task_priority = {
                let queue_lock = task_queue.read().await;
                queue_lock.peek().map(|item| item.task.context.priority)
            };

            if let Some(priority) = next_task_priority {
                let slots_needed = if allows_bulk_preemption(priority) {
                    let queue_lock = task_queue.read().await;
                    let mcp_tasks_queued = queue_lock.iter()
                        .filter(|item| {
                            matches!(item.task.context.priority, TaskPriority::McpRequests)
                        })
                        .count()
                        .min(max_concurrent - 1);
                    mcp_tasks_queued.max(1)
                } else {
                    1
                };

                let preempted_count = try_preempt_multiple_tasks(
                    running_tasks, priority, slots_needed,
                ).await;

                if preempted_count == 0 {
                    break;
                }
            } else {
                break;
            }
        }

        let task_item = {
            let mut queue_lock = task_queue.write().await;
            queue_lock.pop()
        };

        if let Some(queue_item) = task_item {
            start_task(running_tasks, queue_item.task, ingestion_engine.clone()).await;
        } else {
            break;
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

    let best_candidate = find_best_preemption_candidate(&running_lock, new_priority);

    if let Some((task_id, old_priority, _, supports_checkpointing, is_preemptible)) =
        best_candidate
    {
        if let Some(running_task) = running_lock.remove(&task_id) {
            tracing::info!(
                "Preempting task {} (priority {:?}) for higher priority task (priority {:?})",
                task_id, old_priority, new_priority
            );

            handle_preemption_checkpoint(
                &running_task, task_id, supports_checkpointing, is_preemptible,
            );

            // Gracefully cancel the task
            running_task.cancellation_token.cancel();

            let grace_period = if supports_checkpointing { 100 } else { 10 };
            drop(running_lock);
            tokio::time::sleep(Duration::from_millis(grace_period)).await;

            // Force abort if still running after grace period
            if !running_task.handle.is_finished() {
                tracing::warn!("Task {} did not respond to cancellation, aborting", task_id);
                running_task.handle.abort();

                if supports_checkpointing {
                    if let Some(checkpoint) = &running_task.current_checkpoint {
                        tracing::warn!(
                            "Force aborting checkpointed task {}, attempting rollback",
                            task_id
                        );
                        if let Err(e) = checkpoint_manager
                            .rollback_checkpoint(&checkpoint.checkpoint_id)
                            .await
                        {
                            tracing::error!(
                                "Failed to rollback checkpoint for aborted task {}: {}",
                                task_id, e
                            );
                        }
                    }
                }
            }

            return true;
        }
    }

    false
}

/// Find the best candidate for preemption among running tasks
fn find_best_preemption_candidate(
    running_lock: &HashMap<Uuid, RunningTask>,
    new_priority: TaskPriority,
) -> Option<(Uuid, TaskPriority, Instant, bool, bool)> {
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

            if should_replace_candidate(&best_candidate, &candidate) {
                best_candidate = Some(candidate);
            }
        }
    }

    best_candidate
}

/// Determine if a new candidate should replace the current best
fn should_replace_candidate(
    current: &Option<(Uuid, TaskPriority, Instant, bool, bool)>,
    new: &(Uuid, TaskPriority, Instant, bool, bool),
) -> bool {
    match current {
        None => true,
        Some((_, cur_prio, cur_start, cur_ckpt, cur_preempt)) => {
            let (_, new_prio, new_start, new_ckpt, new_preempt) = new;
            *new_prio < *cur_prio
                || (*new_prio == *cur_prio
                    && ((*new_preempt && !cur_preempt)
                        || (*new_preempt == *cur_preempt
                            && ((*new_ckpt && !cur_ckpt)
                                || (*new_ckpt == *cur_ckpt && *new_start > *cur_start)))))
        }
    }
}

/// Log checkpoint status during preemption
fn handle_preemption_checkpoint(
    running_task: &RunningTask,
    task_id: Uuid,
    supports_checkpointing: bool,
    is_preemptible: bool,
) {
    if supports_checkpointing && is_preemptible {
        if let Some(checkpoint) = &running_task.current_checkpoint {
            tracing::info!(
                "Task {} has existing checkpoint: {}",
                task_id, checkpoint.checkpoint_id
            );
        } else {
            tracing::debug!("Creating checkpoint for preempted task {}", task_id);
        }
    }
}

/// Try to preempt multiple lower priority tasks if needed
async fn try_preempt_multiple_tasks(
    running_tasks: &Arc<RwLock<HashMap<Uuid, RunningTask>>>,
    new_priority: TaskPriority,
    slots_needed: usize,
) -> usize {
    let mut preempted_count = 0;

    for _ in 0..slots_needed {
        let temp_cm = Arc::new(CheckpointManager::new(
            std::env::temp_dir().join("temp_checkpoints"),
            Duration::from_secs(3600),
        ));
        if try_preempt_lower_priority_task(running_tasks, &temp_cm, new_priority).await {
            preempted_count += 1;
        } else {
            break;
        }
    }

    preempted_count
}

/// Check if a task priority allows bulk preemption (MCP requests only)
fn allows_bulk_preemption(priority: TaskPriority) -> bool {
    matches!(priority, TaskPriority::McpRequests)
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
            result = execute_task_payload(payload, &context_for_task, &ingestion_engine) => {
                handle_execution_result(result, &context_for_task, start_time)
            }
            _ = token.cancelled() => {
                handle_cancellation(&context_for_task)
            }
            _ = timeout_future(&context_for_task) => {
                handle_timeout(&context_for_task)
            }
        };

        let _ = result_sender.send(result.clone());

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
        is_preemptible: true,
        last_progress_update: Instant::now(),
    };

    let mut running_lock = running_tasks.write().await;
    running_lock.insert(task_id, running_task);
}

/// Handle the result of task payload execution
fn handle_execution_result(
    result: Result<TaskResultData, PriorityError>,
    context: &TaskContext,
    start_time: Instant,
) -> TaskResult {
    match result {
        Ok(data) => TaskResult::Success {
            execution_time_ms: start_time.elapsed().as_millis() as u64,
            data,
        },
        Err(error) => {
            let checkpoint_id = if context.supports_checkpointing {
                Some(format!(
                    "error_{}_{}",
                    context.task_id,
                    chrono::Utc::now().timestamp()
                ))
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

/// Handle task cancellation
fn handle_cancellation(context: &TaskContext) -> TaskResult {
    let checkpoint_id = if context.supports_checkpointing {
        Some(format!(
            "cancelled_{}_{}",
            context.task_id,
            chrono::Utc::now().timestamp()
        ))
    } else {
        None
    };

    TaskResult::Cancelled {
        reason: "Task was preempted by higher priority task".to_string(),
        checkpoint_id,
        partial_data: None,
    }
}

/// Handle task timeout
fn handle_timeout(context: &TaskContext) -> TaskResult {
    let checkpoint_id = if context.supports_checkpointing {
        Some(format!(
            "timeout_{}_{}",
            context.task_id,
            chrono::Utc::now().timestamp()
        ))
    } else {
        None
    };

    TaskResult::Timeout {
        timeout_duration_ms: context.timeout_ms.unwrap_or(30_000),
        checkpoint_id,
    }
}

/// Create a timeout future if task has timeout configured
async fn timeout_future(context: &TaskContext) {
    if let Some(timeout_ms) = context.timeout_ms {
        tokio::time::sleep(Duration::from_millis(timeout_ms)).await;
    } else {
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
        TaskPayload::ProcessDocument { file_path, collection, branch } => {
            execute_process_document(
                &file_path, &collection, &branch, context, ingestion_engine,
            ).await
        }
        TaskPayload::WatchDirectory { path, recursive } => {
            tracing::info!("Watching directory: {:?}, recursive: {}", path, recursive);
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
            execute_generic_task(&operation, &parameters, context).await
        }
    }
}

/// Execute a ProcessDocument task payload
async fn execute_process_document(
    file_path: &std::path::Path,
    collection: &str,
    branch: &str,
    context: &TaskContext,
    ingestion_engine: &Option<Arc<crate::IngestionEngine>>,
) -> Result<TaskResultData, PriorityError> {
    if let Some(engine) = ingestion_engine {
        tracing::info!(
            file = %file_path.display(),
            collection = %collection,
            "Processing document with ingestion engine"
        );

        let result = engine
            .process_document(file_path, collection, branch)
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
        tracing::info!(
            "Processing document (stub): {:?} for collection: {}",
            file_path, collection
        );
        tokio::time::sleep(Duration::from_millis(100)).await;

        Ok(TaskResultData::DocumentProcessing {
            document_id: context.task_id.to_string(),
            collection: collection.to_string(),
            chunks_created: 1,
            checkpoint_id: context.checkpoint_id.clone(),
        })
    }
}

/// Execute a Generic task payload
async fn execute_generic_task(
    operation: &str,
    parameters: &HashMap<String, serde_json::Value>,
    context: &TaskContext,
) -> Result<TaskResultData, PriorityError> {
    tracing::info!(
        "Executing generic operation: '{}' with {} parameters",
        operation, parameters.len()
    );

    let sleep_duration = if operation.starts_with("long_") {
        Duration::from_millis(2000)
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

/// Clean up completed running tasks
pub(crate) async fn cleanup_completed_tasks(
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
