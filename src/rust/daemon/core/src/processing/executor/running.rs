//! Running task tracking struct

use std::time::Instant;

use tokio::task::JoinHandle;
use uuid::Uuid;

use super::super::checkpoint::TaskCheckpoint;
use super::super::{TaskContext, TaskResult};

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

/// Clean up completed running tasks
pub(crate) async fn cleanup_completed_tasks(
    running_tasks: &std::sync::Arc<
        tokio::sync::RwLock<std::collections::HashMap<Uuid, RunningTask>>,
    >,
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
