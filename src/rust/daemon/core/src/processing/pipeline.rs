//! Pipeline struct with constructors, configuration, and public API
//!
//! The `Pipeline` is the main entry point for the priority-based processing
//! system. It owns the task queue, running tasks map, checkpoint manager,
//! metrics collector, and provides methods for lifecycle management.

use std::collections::{BinaryHeap, HashMap};
use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering as AtomicOrdering};
use std::sync::Arc;
use std::time::Duration;

use tokio::sync::{mpsc, oneshot, RwLock};
use tokio::task::JoinHandle;
use uuid::Uuid;

use crate::queue_operations::QueueManager;
use crate::storage::StorageClient;

use super::checkpoint::CheckpointManager;
use super::executor::{self, RunningTask, TaskQueueItem};
use super::metrics::MetricsCollector;
use super::request_queue::{QueueConfig, RequestQueue};
use super::submitter::{TaskResultHandle, TaskSubmitter};
use super::{
    PipelineStats, PriorityError, PriorityTask, PrioritySystemMetrics,
    TaskContext, TaskPayload, TaskPriority, TaskSource,
};

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

        let checkpoint_path = checkpoint_dir.unwrap_or_else(|| {
            std::env::temp_dir().join("wqm_checkpoints")
        });

        let checkpoint_manager = Arc::new(CheckpointManager::new(
            checkpoint_path,
            Duration::from_secs(3600),
        ));

        let metrics_collector = Arc::new(MetricsCollector::new(
            Duration::from_secs(60),
        ));

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
        let rate_limiter = TaskSubmitter::create_rate_limiter(
            self.request_queue.config.enable_rate_limiting,
            self.request_queue.config.max_tasks_per_second,
        );

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
            executor::execution_loop(
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
            .ok_or_else(|| PriorityError::Checkpoint(
                format!("Checkpoint {checkpoint_id} not found"),
            ))?;

        let task_id = checkpoint.task_id;
        let (result_sender, result_receiver) = oneshot::channel();

        let payload = TaskPayload::Generic {
            operation: "resume_from_checkpoint".to_string(),
            parameters: {
                let mut params = HashMap::new();
                params.insert(
                    "checkpoint_id".to_string(),
                    serde_json::Value::String(checkpoint_id.to_string()),
                );
                params.insert(
                    "original_task_id".to_string(),
                    serde_json::Value::String(task_id.to_string()),
                );
                params.insert("state_data".to_string(), checkpoint.state_data);
                params
            },
        };

        let context = TaskContext {
            task_id: Uuid::new_v4(),
            priority: new_priority.unwrap_or(TaskPriority::ProjectWatching),
            created_at: chrono::Utc::now(),
            timeout_ms: Some(60_000),
            source: TaskSource::Generic {
                operation: "checkpoint_resume".to_string(),
            },
            metadata: {
                let mut metadata = HashMap::new();
                metadata.insert("resumed_from".to_string(), checkpoint_id.to_string());
                metadata.insert("original_task_id".to_string(), task_id.to_string());
                metadata
            },
            checkpoint_id: Some(checkpoint_id.to_string()),
            supports_checkpointing: true,
            tenant_id: None,
        };

        let task = PriorityTask {
            context: context.clone(),
            payload,
            result_sender,
            cancellation_token: None,
        };

        self.task_sender.send(task)
            .map_err(|_| PriorityError::Communication(
                "Pipeline is shutting down".to_string(),
            ))?;

        tracing::info!(
            "Resumed task from checkpoint {} with new task ID {}",
            checkpoint_id, context.task_id
        );

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

        output.push_str("# HELP wqm_tasks_total Total number of tasks processed\n");
        output.push_str("# TYPE wqm_tasks_total counter\n");
        output.push_str(&format!("wqm_tasks_completed {{}} {}\n", metrics.pipeline.tasks_completed));
        output.push_str(&format!("wqm_tasks_failed {{}} {}\n", metrics.pipeline.tasks_failed));
        output.push_str(&format!("wqm_tasks_cancelled {{}} {}\n", metrics.pipeline.tasks_cancelled));
        output.push_str(&format!("wqm_tasks_timed_out {{}} {}\n", metrics.pipeline.tasks_timed_out));

        output.push_str("# HELP wqm_queue_size Current queue size\n");
        output.push_str("# TYPE wqm_queue_size gauge\n");
        output.push_str(&format!("wqm_queue_total {{}} {}\n", metrics.queue.total_queued));

        for (priority, count) in &metrics.queue.queued_by_priority {
            output.push_str(&format!(
                "wqm_queue_by_priority{{priority=\"{:?}\"}} {}\n",
                priority, count
            ));
        }

        output.push_str("# HELP wqm_task_duration_seconds Task execution duration\n");
        output.push_str("# TYPE wqm_task_duration_seconds histogram\n");
        output.push_str(&format!(
            "wqm_task_duration_average {{}} {}\n",
            metrics.performance.average_task_duration_ms / 1000.0
        ));
        output.push_str(&format!(
            "wqm_task_duration_p95 {{}} {}\n",
            metrics.performance.p95_task_duration_ms / 1000.0
        ));
        output.push_str(&format!(
            "wqm_task_duration_p99 {{}} {}\n",
            metrics.performance.p99_task_duration_ms / 1000.0
        ));

        output.push_str("# HELP wqm_preemptions_total Total preemptions\n");
        output.push_str("# TYPE wqm_preemptions_total counter\n");
        output.push_str(&format!(
            "wqm_preemptions_total {{}} {}\n",
            metrics.preemption.preemptions_total
        ));
        output.push_str(&format!(
            "wqm_preemptions_graceful {{}} {}\n",
            metrics.preemption.graceful_preemptions
        ));
        output.push_str(&format!(
            "wqm_preemptions_forced {{}} {}\n",
            metrics.preemption.forced_aborts
        ));

        output.push_str("# HELP wqm_checkpoints_active Active checkpoints\n");
        output.push_str("# TYPE wqm_checkpoints_active gauge\n");
        output.push_str(&format!(
            "wqm_checkpoints_active {{}} {}\n",
            metrics.checkpoints.active_checkpoints
        ));
        output.push_str(&format!(
            "wqm_rollbacks_total {{}} {}\n",
            metrics.checkpoints.rollbacks_executed
        ));

        output
    }
}

impl Default for Pipeline {
    fn default() -> Self {
        Self::new(4)
    }
}
