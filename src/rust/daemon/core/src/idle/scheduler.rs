//! Maintenance scheduler — runs eligible tasks during idle periods.

use std::time::Instant;

use tokio_util::sync::CancellationToken;
use tracing::{debug, info};

use super::task::{MaintenanceContext, MaintenanceResult, MaintenanceTask};
use super::IdleState;

/// Execution state for a registered task.
struct RegisteredTask {
    task: Box<dyn MaintenanceTask>,
    last_run: Option<Instant>,
    in_progress: bool,
}

/// Status snapshot of a registered maintenance task (for CLI/gRPC).
#[derive(Debug, Clone)]
pub struct MaintenanceTaskStatus {
    pub name: String,
    pub in_progress: bool,
    pub last_run_secs_ago: Option<u64>,
}

/// Scheduler that manages and executes maintenance tasks during idle periods.
///
/// Call `tick()` each processing-loop iteration when idle. The scheduler picks
/// the first eligible task, runs one batch, and returns. When real work arrives,
/// call `cancel_active()` so the running batch can yield.
pub struct MaintenanceScheduler {
    tasks: Vec<RegisteredTask>,
    active_task_idx: Option<usize>,
    current_cancel: Option<CancellationToken>,
}

impl MaintenanceScheduler {
    pub fn new() -> Self {
        Self {
            tasks: Vec::new(),
            active_task_idx: None,
            current_cancel: None,
        }
    }

    /// Register a maintenance task.
    pub fn register(&mut self, task: Box<dyn MaintenanceTask>) {
        info!("Registered maintenance task: {}", task.name());
        self.tasks.push(RegisteredTask {
            task,
            last_run: None,
            in_progress: false,
        });
    }

    /// Number of registered tasks.
    pub fn task_count(&self) -> usize {
        self.tasks.len()
    }

    /// Cancel the currently running batch so it yields promptly.
    pub fn cancel_active(&mut self) {
        if let Some(token) = self.current_cancel.take() {
            token.cancel();
        }
    }

    /// Run one tick of the scheduler. Returns `true` if a task batch ran.
    ///
    /// - Resumes an in-progress (yielded) task before considering new ones.
    /// - Picks the first eligible task based on idle state, delay, and cooldown.
    /// - Runs exactly one `run_batch()` call, then returns.
    pub async fn tick(
        &mut self,
        idle_state: IdleState,
        idle_elapsed_secs: u64,
        ctx: &MaintenanceContext<'_>,
    ) -> bool {
        // Resume in-progress task first
        if let Some(idx) = self.active_task_idx {
            let reg = &self.tasks[idx];
            if reg.task.can_run_in(idle_state) {
                return self.run_task_batch(idx, ctx).await;
            }
            // Idle state changed and task can no longer run — park it
            debug!(
                "Maintenance task '{}' parked: idle state changed to {:?}",
                reg.task.name(),
                idle_state
            );
            return false;
        }

        // Find next eligible task
        let now = Instant::now();
        for idx in 0..self.tasks.len() {
            let reg = &self.tasks[idx];
            if !reg.task.can_run_in(idle_state) {
                continue;
            }
            if idle_elapsed_secs < reg.task.idle_delay_secs() {
                continue;
            }
            if let Some(last) = reg.last_run {
                if now.duration_since(last).as_secs() < reg.task.cooldown_secs() {
                    continue;
                }
            }
            return self.run_task_batch(idx, ctx).await;
        }
        false
    }

    /// Run a single batch of the task at `idx`.
    async fn run_task_batch(&mut self, idx: usize, ctx: &MaintenanceContext<'_>) -> bool {
        let cancel = CancellationToken::new();
        self.current_cancel = Some(cancel.clone());

        let reg = &mut self.tasks[idx];
        if !reg.in_progress {
            info!("Starting maintenance: {}", reg.task.name());
            reg.in_progress = true;
            reg.task.reset();
        }

        let result = reg.task.run_batch(ctx, &cancel).await;
        self.current_cancel = None;

        match result {
            MaintenanceResult::Continue => {
                self.active_task_idx = Some(idx);
            }
            MaintenanceResult::Done => {
                info!("Maintenance complete: {}", reg.task.name());
                reg.in_progress = false;
                reg.last_run = Some(Instant::now());
                self.active_task_idx = None;
            }
            MaintenanceResult::Yielded => {
                debug!("Maintenance yielded: {} (will resume)", reg.task.name());
                self.active_task_idx = Some(idx);
            }
        }
        true
    }

    /// Snapshot of all registered tasks for observability.
    pub fn status(&self) -> Vec<MaintenanceTaskStatus> {
        let now = Instant::now();
        self.tasks
            .iter()
            .map(|reg| MaintenanceTaskStatus {
                name: reg.task.name().to_string(),
                in_progress: reg.in_progress,
                last_run_secs_ago: reg.last_run.map(|t| now.duration_since(t).as_secs()),
            })
            .collect()
    }
}
