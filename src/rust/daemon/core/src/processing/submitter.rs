//! Task submission with rate limiting, backpressure, and spill-to-SQLite
//!
//! Provides the `TaskSubmitter` handle for enqueuing tasks into the pipeline
//! with retry logic, exponential backoff, and overflow spill to the SQLite
//! unified queue for processing by the `UnifiedQueueProcessor`.

use std::collections::HashMap;
use std::num::NonZeroU32;
use std::sync::Arc;
use std::time::Duration;

use governor::{Quota, RateLimiter, clock::DefaultClock, state::{InMemoryState, NotKeyed}};
use tokio::sync::{mpsc, oneshot};
use uuid::Uuid;

use crate::queue_operations::QueueManager;
use super::{
    PriorityError, PriorityTask, TaskContext, TaskPayload, TaskPriority, TaskResult,
    TaskResultData, TaskSource,
};
use super::metrics::MetricsCollector;
use super::request_queue::RequestQueue;

/// Handle for submitting tasks to the pipeline
#[derive(Clone)]
pub struct TaskSubmitter {
    pub(crate) sender: mpsc::UnboundedSender<PriorityTask>,
    pub(crate) request_queue: Arc<RequestQueue>,
    pub(crate) metrics_collector: Arc<MetricsCollector>,
    pub(crate) rate_limiter: Option<Arc<RateLimiter<NotKeyed, InMemoryState, DefaultClock>>>,
    /// Optional SQLite queue manager for spill-to-disk on overflow
    pub(crate) spill_queue: Option<Arc<QueueManager>>,
}

impl TaskSubmitter {
    /// Create a rate limiter from queue config if rate limiting is enabled
    pub(crate) fn create_rate_limiter(
        enable: bool,
        max_tps: Option<u64>,
    ) -> Option<Arc<RateLimiter<NotKeyed, InMemoryState, DefaultClock>>> {
        if enable {
            if let Some(max_tps) = max_tps {
                let quota = Quota::per_second(NonZeroU32::new(max_tps as u32).unwrap());
                Some(Arc::new(RateLimiter::direct(quota)))
            } else {
                None
            }
        } else {
            None
        }
    }

    /// Submit a task for execution and get a future for the result
    pub async fn submit_task(
        &self,
        priority: TaskPriority,
        source: TaskSource,
        payload: TaskPayload,
        timeout: Option<Duration>,
    ) -> Result<TaskResultHandle, PriorityError> {
        self.submit_task_with_queue_timeout(priority, source, payload, timeout, None).await
    }

    /// Submit a task with separate queue timeout
    pub async fn submit_task_with_queue_timeout(
        &self,
        priority: TaskPriority,
        source: TaskSource,
        payload: TaskPayload,
        execution_timeout: Option<Duration>,
        queue_timeout: Option<Duration>,
    ) -> Result<TaskResultHandle, PriorityError> {
        let task_id = Uuid::new_v4();
        let (result_sender, result_receiver) = oneshot::channel();

        let supports_checkpointing = match &payload {
            TaskPayload::ProcessDocument { .. } => true,
            TaskPayload::WatchDirectory { .. } => true,
            TaskPayload::ExecuteQuery { .. } => false,
            TaskPayload::Generic { .. } => true,
        };

        let context = TaskContext {
            task_id,
            priority,
            created_at: chrono::Utc::now(),
            timeout_ms: execution_timeout.map(|d| d.as_millis() as u64),
            source,
            metadata: HashMap::new(),
            checkpoint_id: None,
            supports_checkpointing,
            tenant_id: None,
        };

        let task = PriorityTask {
            context: context.clone(),
            payload: payload.clone(),
            result_sender,
            cancellation_token: None,
        };

        // Check rate limit
        if let Some(ref limiter) = self.rate_limiter {
            if limiter.check().is_err() {
                self.metrics_collector.record_rate_limit();
                tracing::warn!("Rate limit exceeded, rejecting task {}", task_id);
                return Err(PriorityError::Communication(
                    "Rate limit exceeded".to_string(),
                ));
            }
        }

        // Check for backpressure
        self.check_backpressure(task_id);

        // Retry with exponential backoff when queue is full
        if let Some(spill_handle) =
            self.retry_and_enqueue(task_id, &context, &payload, task, queue_timeout).await?
        {
            return Ok(spill_handle);
        }

        Ok(TaskResultHandle {
            task_id,
            context,
            result_receiver,
        })
    }

    /// Check and log backpressure condition
    fn check_backpressure(&self, task_id: Uuid) {
        let utilization = self.request_queue.get_utilization();
        if self.request_queue.config.enable_backpressure
            && utilization >= self.request_queue.config.backpressure_threshold
        {
            self.metrics_collector.record_backpressure();
            tracing::warn!(
                "Backpressure detected: queue at {:.1}% capacity (threshold: {:.1}%), \
                 task {} queued with potential delays",
                utilization * 100.0,
                self.request_queue.config.backpressure_threshold * 100.0,
                task_id
            );
        }
    }

    /// Retry enqueue with exponential backoff.
    /// Returns `Ok(Some(handle))` if the task was spilled to SQLite,
    /// `Ok(None)` if the task was successfully enqueued to the in-memory pipeline,
    /// or `Err` on failure.
    async fn retry_and_enqueue(
        &self,
        task_id: Uuid,
        context: &TaskContext,
        payload: &TaskPayload,
        task: PriorityTask,
        queue_timeout: Option<Duration>,
    ) -> Result<Option<TaskResultHandle>, PriorityError> {
        let max_retries = 5;
        let mut retry_delay_ms = 100u64;

        for attempt in 0..max_retries {
            if !self.request_queue.has_capacity() {
                self.metrics_collector.record_queue_overflow();
                if attempt > 0 {
                    tracing::warn!(
                        "Queue overflow: retry attempt {}/{} for task {} after {}ms delay",
                        attempt, max_retries, task_id, retry_delay_ms
                    );
                } else {
                    tracing::error!(
                        "Request queue FULL, task {} will be retried with backpressure \
                         - file watching BLOCKED until queue drains",
                        task_id
                    );
                }

                if attempt < max_retries - 1 {
                    tokio::time::sleep(Duration::from_millis(retry_delay_ms)).await;
                    retry_delay_ms *= 2;
                    continue;
                } else {
                    // All retries exhausted - attempt spill to SQLite
                    return self.handle_spill_or_reject(task_id, context, payload).await;
                }
            } else {
                if attempt > 0 {
                    tracing::info!(
                        "Task {} successfully queued after {} retry attempts",
                        task_id, attempt
                    );
                }
                break;
            }
        }

        // Enqueue to request queue, then dequeue and send to pipeline
        match self.request_queue.enqueue(task, queue_timeout).await {
            Ok(()) => {
                if let Some(queued_task) = self.request_queue.dequeue().await {
                    self.sender.send(queued_task)
                        .map_err(|_| PriorityError::Communication(
                            "Pipeline is shutting down".to_string(),
                        ))?;
                } else {
                    return Err(PriorityError::Communication(
                        "Task was queued but could not be dequeued".to_string(),
                    ));
                }
            }
            Err(e) => return Err(e),
        }

        Ok(None)
    }

    /// Attempt to spill to SQLite or reject the task
    async fn handle_spill_or_reject(
        &self,
        task_id: Uuid,
        context: &TaskContext,
        payload: &TaskPayload,
    ) -> Result<Option<TaskResultHandle>, PriorityError> {
        if let Some(ref spill_queue) = self.spill_queue {
            match self.spill_to_sqlite(spill_queue, context, payload).await {
                Ok(()) => {
                    self.metrics_collector.record_queue_spill();
                    tracing::warn!(
                        "Task {} spilled to SQLite after retry attempts \
                         - will be processed by queue processor",
                        task_id
                    );
                    let (result_sender, result_receiver) = oneshot::channel();
                    let _ = result_sender.send(TaskResult::Success {
                        execution_time_ms: 0,
                        data: TaskResultData::Generic {
                            message: "Task spilled to SQLite unified_queue".to_string(),
                            data: serde_json::json!({"spilled_to_sqlite": true}),
                            checkpoint_id: None,
                        },
                    });
                    return Ok(Some(TaskResultHandle {
                        task_id,
                        context: context.clone(),
                        result_receiver,
                    }));
                }
                Err(spill_err) => {
                    tracing::error!(
                        "Task {} REJECTED after retry attempts and SQLite spill failed: {}",
                        task_id, spill_err
                    );
                    return Err(PriorityError::QueueCapacityExceeded {
                        current: self.request_queue.size(),
                        max: self.request_queue.capacity(),
                    });
                }
            }
        } else {
            tracing::error!(
                "Task {} REJECTED after retry attempts - queue still full \
                 (no spill target configured)",
                task_id
            );
            return Err(PriorityError::QueueCapacityExceeded {
                current: self.request_queue.size(),
                max: self.request_queue.capacity(),
            });
        }
    }

    /// Submit a high priority task that bypasses the queue
    pub async fn submit_urgent_task(
        &self,
        source: TaskSource,
        payload: TaskPayload,
        timeout: Option<Duration>,
    ) -> Result<TaskResultHandle, PriorityError> {
        let task_id = Uuid::new_v4();
        let (result_sender, result_receiver) = oneshot::channel();

        let context = TaskContext {
            task_id,
            priority: TaskPriority::McpRequests,
            created_at: chrono::Utc::now(),
            timeout_ms: timeout.map(|d| d.as_millis() as u64),
            source,
            metadata: {
                let mut metadata = HashMap::new();
                metadata.insert("urgent".to_string(), "true".to_string());
                metadata
            },
            checkpoint_id: None,
            supports_checkpointing: false,
            tenant_id: None,
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

    /// Get queue statistics
    pub async fn get_queue_stats(&self) -> super::QueueStats {
        self.request_queue.get_stats().await
    }

    /// Clean up timed out requests from queue
    pub async fn cleanup_queue_timeouts(&self) -> usize {
        self.request_queue.cleanup_timeouts().await
    }

    /// Spill a task to SQLite unified_queue when in-memory queue is full
    ///
    /// Maps Pipeline TaskPayload to unified_queue format and persists it.
    /// The UnifiedQueueProcessor will pick it up on its next poll cycle.
    #[cfg_attr(test, allow(dead_code))]
    pub(crate) async fn spill_to_sqlite(
        &self,
        queue_manager: &QueueManager,
        context: &TaskContext,
        payload: &TaskPayload,
    ) -> Result<(), PriorityError> {
        use crate::unified_queue_schema::{
            ItemType, QueueOperation as UnifiedOp, FilePayload as UqFilePayload,
        };

        match payload {
            TaskPayload::ProcessDocument { file_path, collection, branch } => {
                let tenant_id = self.resolve_tenant_id(context, file_path);

                let file_payload = UqFilePayload {
                    file_path: file_path.to_string_lossy().to_string(),
                    file_type: None,
                    file_hash: None,
                    size_bytes: None,
                    old_path: None,
                };

                let payload_json = serde_json::to_string(&file_payload)
                    .map_err(|e| PriorityError::Communication(
                        format!("Failed to serialize spill payload: {}", e),
                    ))?;

                let metadata = serde_json::json!({
                    "spilled_from": "pipeline",
                    "original_priority": format!("{:?}", context.priority),
                    "task_id": context.task_id.to_string(),
                }).to_string();

                queue_manager.enqueue_unified(
                    ItemType::File,
                    UnifiedOp::Add,
                    &tenant_id,
                    collection,
                    &payload_json,
                    Some(branch),
                    Some(&metadata),
                ).await.map_err(|e| PriorityError::Communication(
                    format!("SQLite spill failed: {}", e),
                ))?;

                tracing::info!(
                    file_path = %file_path.display(),
                    collection = %collection,
                    tenant_id = %tenant_id,
                    "Task spilled to SQLite unified_queue"
                );

                Ok(())
            }
            _ => {
                Err(PriorityError::Communication(
                    format!(
                        "Cannot spill {:?} task to SQLite - only ProcessDocument is supported",
                        std::mem::discriminant(payload)
                    ),
                ))
            }
        }
    }

    /// Resolve tenant_id from context or derive from filesystem paths
    fn resolve_tenant_id(
        &self,
        context: &TaskContext,
        file_path: &std::path::Path,
    ) -> String {
        if let Some(ref tid) = context.tenant_id {
            return tid.clone();
        }

        match &context.source {
            TaskSource::ProjectWatcher { project_path } => {
                wqm_common::project_id::calculate_tenant_id(
                    std::path::Path::new(project_path),
                )
            }
            TaskSource::BackgroundWatcher { folder_path } => {
                wqm_common::project_id::calculate_tenant_id(
                    std::path::Path::new(folder_path),
                )
            }
            _ => {
                let parent = file_path.parent()
                    .unwrap_or_else(|| std::path::Path::new("/"));
                wqm_common::project_id::calculate_tenant_id(parent)
            }
        }
    }
}

/// Handle for waiting on task execution results
pub struct TaskResultHandle {
    pub task_id: Uuid,
    pub context: TaskContext,
    pub(crate) result_receiver: oneshot::Receiver<TaskResult>,
}

impl TaskResultHandle {
    /// Create a TaskResultHandle for testing purposes.
    #[cfg(test)]
    pub(crate) fn new_for_test(
        task_id: Uuid,
        context: TaskContext,
        result_receiver: oneshot::Receiver<TaskResult>,
    ) -> Self {
        Self { task_id, context, result_receiver }
    }

    /// Wait for the task to complete and return the result
    pub async fn wait(self) -> Result<TaskResult, PriorityError> {
        self.result_receiver.await
            .map_err(|_| PriorityError::Communication(
                "Task executor disconnected".to_string(),
            ))
    }

    /// Check if the task result sender has been dropped (task completed or abandoned).
    ///
    /// Returns true when the oneshot sender is no longer alive, meaning the task
    /// executor has either sent a result or been dropped. Used by IPC cleanup to
    /// identify completed tasks that can be removed from the active tasks map.
    pub fn is_completed(&mut self) -> bool {
        match self.result_receiver.try_recv() {
            Ok(_) => true,
            Err(oneshot::error::TryRecvError::Closed) => true,
            Err(oneshot::error::TryRecvError::Empty) => false,
        }
    }
}
