//! `LoopState` groups all mutable variables that persist across loop iterations
//! in the unified queue processing loop.

use std::collections::VecDeque;

use crate::circuit_breaker::CircuitBreaker;
use crate::queue_health::probes::hard_state::PollOutcome;
use crate::unified_queue_processor::config::UnifiedProcessorConfig;

use super::circuit_breakers::new_sqlite_breaker;

pub(super) struct LoopState {
    /// Last time periodic metrics were logged.
    pub last_metrics_log: chrono::DateTime<chrono::Utc>,
    /// Current adaptive semaphore permit target (None = use config default).
    pub adaptive_target_permits: Option<usize>,
    /// Whether the warmup-complete log line has already been emitted.
    pub warmup_logged: bool,
    /// When the last resurrection pass ran.
    pub last_resurrection: std::time::Instant,
    /// Configuration for the metadata uplift pass (carries current generation).
    pub uplift_config: crate::metadata_uplift::UpliftConfig,
    /// When the last uplift pass was attempted.
    pub last_uplift_attempt: std::time::Instant,
    /// When the last triage pass ran.
    pub last_triage: std::time::Instant,
    /// When the queue first became idle (reset on non-empty batch).
    pub idle_since: Option<std::time::Instant>,
    /// When the last grammar idle-update check ran.
    pub last_grammar_check: std::time::Instant,
    /// When the last DLQ purge ran.
    pub last_dlq_purge: std::time::Instant,
    /// SQLite circuit breaker for queue operations.
    pub sqlite_breaker: CircuitBreaker,
    /// Remaining recovery ramp cycles after Qdrant comes back.
    pub recovery_ramp_remaining: usize,
    /// Maintenance task scheduler.
    pub maintenance_scheduler: crate::idle::MaintenanceScheduler,
    /// Previous absolute DLQ count, for the A3 per-poll delta-rate. `None` until
    /// the first poll seeds it from a live count (DOM-09 — never starts at 0).
    pub prev_dlq: Option<u64>,
    /// Number of DLQ delta samples fed so far (gates A3's ≥2-sample rule).
    pub dlq_samples_seen: u64,
    /// Poll-local outcome ring for the B4 all-items-failing predicate (kept
    /// local, never shared, so B4 needs no shared `Mutex` — PERF-08).
    pub outcome_ring: VecDeque<PollOutcome>,
    /// Whether the previous poll actually dispatched a non-empty batch. Read by
    /// the next poll's `update_health_probes` (which runs pre-dequeue, so it
    /// carries a one-poll lag): when the last poll dispatched nothing yet a
    /// backlog remains, throughput is emitted as 0 so the slow lane decays
    /// instead of freezing at a stale healthy rate during a circuit-breaker
    /// stall (#144). Starts `true` so a fresh start emits no spurious zero before
    /// the first poll has run.
    pub last_poll_dispatched: bool,
}

impl LoopState {
    /// Initialise loop state from processor configuration.
    ///
    /// `baseline_ttl_secs` is the runtime-configured `control_baseline` prune
    /// horizon (`QueueHealthConfig::baseline_ttl_secs`), threaded in so the
    /// persist task honors the tuned value rather than a hardcoded const (#143).
    /// It is separate from `UnifiedProcessorConfig`, which does not carry
    /// queue-health settings.
    pub(super) fn new(config: &UnifiedProcessorConfig, baseline_ttl_secs: u64) -> Self {
        let uplift_config = crate::metadata_uplift::UpliftConfig::default();

        let last_resurrection = std::time::Instant::now()
            .checked_sub(std::time::Duration::from_secs(
                config.failed_resurrection_interval_secs,
            ))
            .unwrap_or_else(std::time::Instant::now);

        let last_uplift_attempt = std::time::Instant::now()
            .checked_sub(std::time::Duration::from_secs(
                uplift_config.min_interval_secs,
            ))
            .unwrap_or_else(std::time::Instant::now);

        let last_triage = std::time::Instant::now()
            .checked_sub(std::time::Duration::from_secs(config.triage_interval_secs))
            .unwrap_or_else(std::time::Instant::now);

        let last_grammar_check = std::time::Instant::now()
            .checked_sub(std::time::Duration::from_secs(3600))
            .unwrap_or_else(std::time::Instant::now);

        let mut maintenance_scheduler = crate::idle::MaintenanceScheduler::new();
        maintenance_scheduler
            .register(Box::new(crate::idle::tasks::FilesystemReconcileTask::new()));
        maintenance_scheduler.register(Box::new(crate::idle::tasks::OrphanCleanupTask::new()));
        maintenance_scheduler.register(Box::new(crate::idle::tasks::QueueCleanupTask::new()));
        maintenance_scheduler.register(Box::new(
            crate::idle::tasks::StaleProjectDeactivationTask::new(),
        ));
        maintenance_scheduler.register(Box::new(crate::idle::tasks::GroupingSchedulerTask::new()));
        maintenance_scheduler.register(Box::new(
            crate::idle::tasks::ElaboratesMaintenanceTask::new(),
        ));
        maintenance_scheduler.register(Box::new(
            crate::switchboard::persist_task::ControlBaselinePersistTask::with_ttl_secs(
                baseline_ttl_secs,
            ),
        ));

        Self {
            last_metrics_log: chrono::Utc::now(),
            adaptive_target_permits: None,
            warmup_logged: false,
            last_resurrection,
            uplift_config,
            last_uplift_attempt,
            last_triage,
            idle_since: None,
            last_grammar_check,
            last_dlq_purge: std::time::Instant::now(),
            sqlite_breaker: new_sqlite_breaker(config),
            recovery_ramp_remaining: 0,
            maintenance_scheduler,
            prev_dlq: None,
            dlq_samples_seen: 0,
            outcome_ring: VecDeque::new(),
            last_poll_dispatched: true,
        }
    }
}
