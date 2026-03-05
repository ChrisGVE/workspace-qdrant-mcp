//! Task preemption logic: finding candidates and performing preemption

use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, Instant};

use tokio::sync::RwLock;
use uuid::Uuid;

use super::super::checkpoint::CheckpointManager;
use super::super::TaskPriority;
use super::running::RunningTask;

/// Check if a task priority allows bulk preemption (MCP requests only)
pub(crate) fn allows_bulk_preemption(priority: TaskPriority) -> bool {
    matches!(priority, TaskPriority::McpRequests)
}

/// Find the best candidate for preemption among running tasks
pub(crate) fn find_best_preemption_candidate(
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
pub(crate) fn handle_preemption_checkpoint(
    running_task: &RunningTask,
    task_id: Uuid,
    supports_checkpointing: bool,
    is_preemptible: bool,
) {
    if supports_checkpointing && is_preemptible {
        if let Some(checkpoint) = &running_task.current_checkpoint {
            tracing::info!(
                "Task {} has existing checkpoint: {}",
                task_id,
                checkpoint.checkpoint_id
            );
        } else {
            tracing::debug!("Creating checkpoint for preempted task {}", task_id);
        }
    }
}

/// Try to preempt a lower priority running task with consistency checks
pub(crate) async fn try_preempt_lower_priority_task(
    running_tasks: &Arc<RwLock<HashMap<Uuid, RunningTask>>>,
    checkpoint_manager: &Arc<CheckpointManager>,
    new_priority: TaskPriority,
) -> bool {
    let mut running_lock = running_tasks.write().await;

    let best_candidate = find_best_preemption_candidate(&running_lock, new_priority);

    if let Some((task_id, old_priority, _, supports_checkpointing, is_preemptible)) = best_candidate
    {
        if let Some(running_task) = running_lock.remove(&task_id) {
            tracing::info!(
                "Preempting task {} (priority {:?}) for higher priority task (priority {:?})",
                task_id,
                old_priority,
                new_priority
            );

            handle_preemption_checkpoint(
                &running_task,
                task_id,
                supports_checkpointing,
                is_preemptible,
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
                                task_id,
                                e
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

/// Try to preempt multiple lower priority tasks if needed
pub(crate) async fn try_preempt_multiple_tasks(
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
