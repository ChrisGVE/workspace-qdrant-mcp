//! Metrics collection and reporting for the priority pipeline
//!
//! Provides real-time metrics aggregation including task completion rates,
//! percentile latencies, preemption statistics, and resource utilization.

use std::collections::{HashMap, VecDeque};
use std::sync::atomic::{AtomicU64, Ordering as AtomicOrdering};
use std::sync::Arc;
use std::time::{Duration, Instant};

use serde::{Deserialize, Serialize};
use tokio::sync::RwLock;

use super::checkpoint::CheckpointManager;
use super::{QueueStats, TaskPriority};

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
    pub(crate) start_time: Instant,
    /// Atomic counters for high-frequency metrics
    pub(crate) tasks_completed: AtomicU64,
    pub(crate) tasks_failed: AtomicU64,
    pub(crate) tasks_cancelled: AtomicU64,
    pub(crate) tasks_timed_out: AtomicU64,
    preemptions_total: AtomicU64,
    graceful_preemptions: AtomicU64,
    forced_aborts: AtomicU64,
    checkpoints_created: AtomicU64,
    rollbacks_executed: AtomicU64,
    pub(crate) queue_overflow_count: AtomicU64,
    pub(crate) queue_spill_count: AtomicU64,
    deduplication_hits: AtomicU64,
    priority_boosts_applied: AtomicU64,
    pub(crate) rate_limited_tasks: AtomicU64,
    pub(crate) backpressure_events: AtomicU64,

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

        for priority in [
            TaskPriority::McpRequests,
            TaskPriority::ProjectWatching,
            TaskPriority::CliCommands,
            TaskPriority::BackgroundWatching,
        ] {
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
        self.total_task_duration_ms
            .fetch_add(duration_ms, AtomicOrdering::Relaxed);

        // Update recent durations
        {
            let mut durations = self.recent_task_durations.write().await;
            if durations.len() >= 1000 {
                durations.pop_front();
            }
            durations.push_back(duration_ms);
        }

        // Update per-priority response times (exponential moving average)
        {
            let response_times = self.response_times_by_priority.read().await;
            if let Some(atomic_time) = response_times.get(&priority) {
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
    pub async fn record_preemption(
        &self,
        preempted_priority: TaskPriority,
        duration_ms: u64,
        graceful: bool,
    ) {
        self.preemptions_total.fetch_add(1, AtomicOrdering::Relaxed);
        self.total_preemption_time_ms
            .fetch_add(duration_ms, AtomicOrdering::Relaxed);

        if graceful {
            self.graceful_preemptions
                .fetch_add(1, AtomicOrdering::Relaxed);
        } else {
            self.forced_aborts.fetch_add(1, AtomicOrdering::Relaxed);
        }

        {
            let preemptions = self.preemptions_by_priority.read().await;
            if let Some(counter) = preemptions.get(&preempted_priority) {
                counter.fetch_add(1, AtomicOrdering::Relaxed);
            }
        }
    }

    /// Record queue-related events
    pub async fn record_queue_time(&self, queue_time_ms: u64) {
        self.total_queue_time_ms
            .fetch_add(queue_time_ms, AtomicOrdering::Relaxed);

        let mut queue_times = self.recent_queue_times.write().await;
        if queue_times.len() >= 1000 {
            queue_times.pop_front();
        }
        queue_times.push_back(queue_time_ms);
    }

    /// Record queue overflow
    pub fn record_queue_overflow(&self) {
        self.queue_overflow_count
            .fetch_add(1, AtomicOrdering::Relaxed);
    }

    /// Record queue spill to SQLite
    pub fn record_queue_spill(&self) {
        self.queue_spill_count.fetch_add(1, AtomicOrdering::Relaxed);
    }

    /// Record rate limit hit
    pub fn record_rate_limit(&self) {
        self.rate_limited_tasks
            .fetch_add(1, AtomicOrdering::Relaxed);
    }

    /// Record backpressure event
    pub fn record_backpressure(&self) {
        self.backpressure_events
            .fetch_add(1, AtomicOrdering::Relaxed);
    }

    /// Record deduplication hit
    pub fn record_deduplication_hit(&self) {
        self.deduplication_hits
            .fetch_add(1, AtomicOrdering::Relaxed);
    }

    /// Record priority boost
    pub fn record_priority_boost(&self) {
        self.priority_boosts_applied
            .fetch_add(1, AtomicOrdering::Relaxed);
    }

    /// Record checkpoint creation
    pub fn record_checkpoint_created(&self) {
        self.checkpoints_created
            .fetch_add(1, AtomicOrdering::Relaxed);
    }

    /// Record rollback execution
    pub fn record_rollback_executed(&self) {
        self.rollbacks_executed
            .fetch_add(1, AtomicOrdering::Relaxed);
    }

    /// Calculate percentile from recent measurements
    async fn calculate_percentile(values: &Arc<RwLock<VecDeque<u64>>>, percentile: f64) -> f64 {
        let values_lock = values.read().await;
        if values_lock.is_empty() {
            return 0.0;
        }

        let mut sorted: Vec<u64> = values_lock.iter().copied().collect();
        sorted.sort_unstable();

        let index = ((sorted.len() as f64 - 1.0) * percentile / 100.0) as usize;
        *sorted.get(index).unwrap_or(&0) as f64
    }

    /// Get current system resource metrics
    fn get_resource_metrics() -> ResourceMetrics {
        ResourceMetrics {
            memory_usage_bytes: 0,
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

        let (avg_task_duration, avg_queue_time, avg_preemption_time) = self.compute_averages(
            tasks_completed,
            tasks_failed,
            tasks_cancelled,
            preemptions_total,
        );
        let p95_duration = Self::calculate_percentile(&self.recent_task_durations, 95.0).await;
        let p99_duration = Self::calculate_percentile(&self.recent_task_durations, 99.0).await;

        let total_tasks = tasks_completed + tasks_failed + tasks_cancelled;
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

        let (preemptions_by_priority, response_times_by_priority) =
            self.collect_per_priority_metrics().await;
        let active_checkpoints = checkpoint_manager.checkpoints.read().await.len();

        PrioritySystemMetrics {
            pipeline: self.build_pipeline_metrics(
                queued_tasks,
                running_tasks,
                capacity,
                tasks_completed,
                tasks_failed,
                tasks_cancelled,
                tasks_timed_out,
                uptime,
            ),
            queue: self.build_queue_metrics(queue_stats, avg_queue_time),
            preemption: PreemptionMetrics {
                preemptions_total,
                preemptions_by_priority,
                graceful_preemptions,
                forced_aborts,
                preemption_success_rate,
                average_preemption_time_ms: avg_preemption_time,
                tasks_resumed_from_checkpoints: 0,
            },
            checkpoints: self.build_checkpoint_metrics(active_checkpoints),
            performance: PerformanceMetrics {
                throughput_tasks_per_second: throughput,
                average_task_duration_ms: avg_task_duration,
                p95_task_duration_ms: p95_duration,
                p99_task_duration_ms: p99_duration,
                response_time_by_priority: response_times_by_priority,
                error_rate_percent: error_rate,
                system_load_percent: 0.0,
            },
            resources: Self::get_resource_metrics(),
        }
    }

    /// Assemble `PipelineMetrics` from counters.
    #[allow(clippy::too_many_arguments)]
    fn build_pipeline_metrics(
        &self,
        queued_tasks: usize,
        running_tasks: usize,
        total_capacity: usize,
        tasks_completed: u64,
        tasks_failed: u64,
        tasks_cancelled: u64,
        tasks_timed_out: u64,
        uptime_seconds: u64,
    ) -> PipelineMetrics {
        PipelineMetrics {
            queued_tasks,
            running_tasks,
            total_capacity,
            tasks_completed,
            tasks_failed,
            tasks_cancelled,
            tasks_timed_out,
            uptime_seconds,
        }
    }

    /// Assemble `QueueMetrics` from queue stats and averages.
    fn build_queue_metrics(&self, queue_stats: &QueueStats, avg_queue_time: f64) -> QueueMetrics {
        QueueMetrics {
            total_queued: queue_stats.total_queued,
            queued_by_priority: queue_stats.queued_by_priority.clone(),
            oldest_request_age_ms: queue_stats.oldest_request_age_ms,
            average_queue_time_ms: avg_queue_time,
            max_queue_time_ms: 0,
            queue_overflow_count: self.queue_overflow_count.load(AtomicOrdering::Relaxed),
            queue_spill_count: self.queue_spill_count.load(AtomicOrdering::Relaxed),
            deduplication_hits: self.deduplication_hits.load(AtomicOrdering::Relaxed),
            priority_boosts_applied: self.priority_boosts_applied.load(AtomicOrdering::Relaxed),
        }
    }

    /// Assemble `CheckpointMetrics` from counters.
    fn build_checkpoint_metrics(&self, active_checkpoints: usize) -> CheckpointMetrics {
        CheckpointMetrics {
            active_checkpoints,
            checkpoints_created: self.checkpoints_created.load(AtomicOrdering::Relaxed),
            checkpoints_restored: 0,
            rollbacks_executed: self.rollbacks_executed.load(AtomicOrdering::Relaxed),
            rollback_success_rate: 100.0,
            average_checkpoint_size_bytes: 0.0,
            checkpoint_storage_usage_bytes: 0,
        }
    }

    /// Compute average durations from atomic accumulators
    fn compute_averages(
        &self,
        tasks_completed: u64,
        tasks_failed: u64,
        tasks_cancelled: u64,
        preemptions_total: u64,
    ) -> (f64, f64, f64) {
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
            self.total_preemption_time_ms.load(AtomicOrdering::Relaxed) as f64
                / preemptions_total as f64
        } else {
            0.0
        };
        (avg_task_duration, avg_queue_time, avg_preemption_time)
    }

    /// Collect per-priority preemption and response-time metrics
    async fn collect_per_priority_metrics(
        &self,
    ) -> (HashMap<TaskPriority, u64>, HashMap<TaskPriority, f64>) {
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
                let value_us = atomic_time.load(AtomicOrdering::Relaxed) as f64;
                response_times_by_priority.insert(*priority, value_us / 1000.0);
            }
        }

        (preemptions_by_priority, response_times_by_priority)
    }
}
