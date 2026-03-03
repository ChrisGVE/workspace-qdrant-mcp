//! Task execution loop with preemption and concurrent task management
//!
//! Contains the core execution loop that processes tasks from the priority
//! queue, manages concurrent task slots, handles preemption of lower-priority
//! tasks, and executes task payloads.

mod payload;
mod preemption;
mod queue;
mod running;

pub(crate) use queue::TaskQueueItem;
pub(crate) use running::{cleanup_completed_tasks, RunningTask};

use std::collections::{BinaryHeap, HashMap};
use std::sync::atomic::{AtomicU64, Ordering as AtomicOrdering};
use std::sync::Arc;
use std::time::{Duration, Instant};

use tokio::sync::{mpsc, RwLock};
use uuid::Uuid;

use super::{
    PriorityError, PriorityTask, TaskContext, TaskPriority, TaskResult, TaskResultData,
};
use payload::execute_task_payload;
use preemption::{allows_bulk_preemption, try_preempt_multiple_tasks};

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
