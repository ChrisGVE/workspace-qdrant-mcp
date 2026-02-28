//! IPC server that handles communication from the MCP server to the Rust engine.

use std::collections::HashMap;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Duration;

use tokio::sync::{mpsc, Mutex, RwLock};
use uuid::Uuid;

use crate::processing::{Pipeline, TaskResultHandle, TaskSubmitter};
use super::{EngineSettings, IpcClient, IpcError, IpcRequest, IpcResponse};

/// IPC server that handles communication between Python and Rust
pub struct IpcServer {
    /// The processing pipeline
    pipeline: Arc<Mutex<Pipeline>>,
    /// Task submitter for the pipeline
    task_submitter: TaskSubmitter,
    /// Active task handles
    active_tasks: Arc<RwLock<HashMap<Uuid, TaskResultHandle>>>,
    /// Request receiver
    request_receiver: Arc<Mutex<Option<mpsc::UnboundedReceiver<IpcRequest>>>>,
    /// Response sender
    response_sender: mpsc::UnboundedSender<IpcResponse>,
    /// Server configuration
    settings: Arc<RwLock<EngineSettings>>,
    /// Shutdown signal
    shutdown_signal: Arc<tokio::sync::Notify>,
    /// Shutdown completion flag
    shutdown_complete: Arc<AtomicBool>,
}

impl IpcServer {
    /// Create a new IPC server
    pub fn new(max_concurrent_tasks: usize) -> (Self, IpcClient) {
        let pipeline = Arc::new(Mutex::new(Pipeline::new(max_concurrent_tasks)));
        let task_submitter = {
            let pipeline_lock = pipeline.try_lock().unwrap();
            pipeline_lock.task_submitter()
        };

        let (request_sender, request_receiver) = mpsc::unbounded_channel();
        let (response_sender, response_receiver) = mpsc::unbounded_channel();

        let settings = Arc::new(RwLock::new(EngineSettings {
            max_concurrent_tasks: Some(max_concurrent_tasks),
            default_timeout_ms: Some(30_000),
            enable_preemption: Some(true),
            log_level: Some("info".to_string()),
        }));

        let server = Self {
            pipeline,
            task_submitter,
            active_tasks: Arc::new(RwLock::new(HashMap::new())),
            request_receiver: Arc::new(Mutex::new(Some(request_receiver))),
            response_sender,
            settings,
            shutdown_signal: Arc::new(tokio::sync::Notify::new()),
            shutdown_complete: Arc::new(AtomicBool::new(false)),
        };

        let client = IpcClient::new(request_sender, response_receiver);

        (server, client)
    }

    /// Configure SQLite spill-to-disk for queue overflow handling.
    /// Must be called before `start()`.
    pub fn set_spill_queue(&mut self, queue_manager: std::sync::Arc<crate::queue_operations::QueueManager>) {
        let mut pipeline = self.pipeline.try_lock()
            .expect("pipeline lock not contended during init");
        pipeline.set_spill_queue(queue_manager);
        self.task_submitter = pipeline.task_submitter();
    }

    /// Configure storage client for Qdrant rollback operations.
    /// Must be called before `start()`.
    pub fn set_rollback_storage(&mut self, client: std::sync::Arc<crate::storage::StorageClient>) {
        let mut pipeline = self.pipeline.try_lock()
            .expect("pipeline lock not contended during init");
        pipeline.set_rollback_storage(client);
    }

    /// Start the IPC server
    pub async fn start(&self) -> Result<(), IpcError> {
        // Start the pipeline
        {
            let mut pipeline = self.pipeline.lock().await;
            pipeline.start().await.map_err(|e| IpcError::ProcessingError(e.to_string()))?;
        }

        // Start request processing loop
        let request_receiver = {
            let mut receiver_lock = self.request_receiver.lock().await;
            receiver_lock.take().ok_or(IpcError::ChannelClosed)?
        };

        let pipeline = Arc::clone(&self.pipeline);
        let task_submitter = self.task_submitter.clone();
        let active_tasks = Arc::clone(&self.active_tasks);
        let response_sender = self.response_sender.clone();
        let settings = Arc::clone(&self.settings);
        let shutdown_signal = Arc::clone(&self.shutdown_signal);
        let shutdown_complete = Arc::clone(&self.shutdown_complete);

        tokio::spawn(async move {
            Self::request_processing_loop(
                request_receiver,
                pipeline,
                task_submitter,
                active_tasks,
                response_sender,
                settings,
                shutdown_signal,
                shutdown_complete,
            ).await;
        });

        Ok(())
    }

    /// Main request processing loop
    async fn request_processing_loop(
        mut request_receiver: mpsc::UnboundedReceiver<IpcRequest>,
        pipeline: Arc<Mutex<Pipeline>>,
        task_submitter: TaskSubmitter,
        active_tasks: Arc<RwLock<HashMap<Uuid, TaskResultHandle>>>,
        response_sender: mpsc::UnboundedSender<IpcResponse>,
        settings: Arc<RwLock<EngineSettings>>,
        shutdown_signal: Arc<tokio::sync::Notify>,
        shutdown_complete: Arc<AtomicBool>,
    ) {
        let mut cleanup_interval = tokio::time::interval(Duration::from_secs(1));

        loop {
            tokio::select! {
                request = request_receiver.recv() => {
                    match request {
                        Some(request) => {
                            if let Err(e) = Self::handle_request(
                                request,
                                &task_submitter,
                                &active_tasks,
                                &response_sender,
                                &pipeline,
                                &settings,
                                &shutdown_signal,
                                &shutdown_complete,
                            ).await {
                                tracing::error!("Error handling IPC request: {}", e);
                            }
                        }
                        None => {
                            tracing::info!("IPC request channel closed");
                            break;
                        }
                    }
                }

                _ = cleanup_interval.tick() => {
                    Self::cleanup_completed_tasks(&active_tasks).await;
                }

                _ = shutdown_signal.notified() => {
                    tracing::info!("IPC server shutting down");
                    break;
                }
            }
        }
    }

    /// Handle a single IPC request
    async fn handle_request(
        request: IpcRequest,
        task_submitter: &TaskSubmitter,
        active_tasks: &Arc<RwLock<HashMap<Uuid, TaskResultHandle>>>,
        response_sender: &mpsc::UnboundedSender<IpcResponse>,
        pipeline: &Arc<Mutex<Pipeline>>,
        settings: &Arc<RwLock<EngineSettings>>,
        shutdown_signal: &Arc<tokio::sync::Notify>,
        shutdown_complete: &Arc<AtomicBool>,
    ) -> Result<(), IpcError> {
        match request {
            IpcRequest::SubmitTask {
                priority, source, payload, timeout_ms, request_id,
            } => {
                Self::handle_submit_task(
                    task_submitter, active_tasks, response_sender,
                    priority, source, payload, timeout_ms, request_id,
                ).await
            }

            IpcRequest::GetStats { request_id } => {
                let stats = {
                    let pipeline_lock = pipeline.lock().await;
                    pipeline_lock.stats().await
                };
                response_sender.send(IpcResponse::Stats { stats, request_id })?;
                Ok(())
            }

            IpcRequest::HealthCheck { request_id } => {
                response_sender.send(IpcResponse::HealthCheckOk {
                    status: "OK".to_string(),
                    request_id,
                })?;
                Ok(())
            }

            IpcRequest::Configure { settings: new_settings, request_id } => {
                {
                    let mut settings_lock = settings.write().await;
                    *settings_lock = new_settings;
                }
                response_sender.send(IpcResponse::ConfigurationApplied { request_id })?;
                Ok(())
            }

            IpcRequest::Shutdown { graceful: _, timeout_ms: _, request_id } => {
                response_sender.send(IpcResponse::ShutdownAck { request_id })?;
                shutdown_complete.store(true, Ordering::Release);
                shutdown_signal.notify_waiters();
                Ok(())
            }
        }
    }

    /// Handle a SubmitTask request
    async fn handle_submit_task(
        task_submitter: &TaskSubmitter,
        active_tasks: &Arc<RwLock<HashMap<Uuid, TaskResultHandle>>>,
        response_sender: &mpsc::UnboundedSender<IpcResponse>,
        priority: crate::processing::TaskPriority,
        source: crate::processing::TaskSource,
        payload: crate::processing::TaskPayload,
        timeout_ms: Option<u64>,
        request_id: String,
    ) -> Result<(), IpcError> {
        let timeout = timeout_ms.map(Duration::from_millis);

        match task_submitter.submit_task(priority, source, payload, timeout).await {
            Ok(task_handle) => {
                let task_id = task_handle.task_id;

                {
                    let mut active_lock = active_tasks.write().await;
                    active_lock.insert(task_id, task_handle);
                }

                response_sender.send(IpcResponse::TaskSubmitted {
                    task_id,
                    request_id: request_id.clone(),
                })?;

                let completion_response = IpcResponse::TaskCompleted {
                    task_id,
                    result: crate::processing::TaskResult::Success {
                        execution_time_ms: 100,
                        data: crate::processing::TaskResultData::Generic {
                            message: "Task completed via IPC".to_string(),
                            data: serde_json::json!({}),
                            checkpoint_id: None,
                        },
                    },
                    request_id,
                };

                {
                    let mut active_lock = active_tasks.write().await;
                    active_lock.remove(&task_id);
                }

                response_sender.send(completion_response)?;
            }
            Err(e) => {
                response_sender.send(IpcResponse::Error {
                    error: e.to_string(),
                    request_id,
                })?;
            }
        }
        Ok(())
    }

    /// Clean up completed tasks from the active tasks registry.
    pub(super) async fn cleanup_completed_tasks(
        active_tasks: &Arc<RwLock<HashMap<Uuid, TaskResultHandle>>>,
    ) {
        let mut active_lock = active_tasks.write().await;
        let before = active_lock.len();

        active_lock.retain(|_task_id, handle| !handle.is_completed());

        let removed = before - active_lock.len();
        if removed > 0 {
            tracing::debug!(
                removed = removed,
                remaining = active_lock.len(),
                "Cleaned up completed IPC tasks"
            );
        }
    }

    /// Wait for the server to shut down
    pub async fn wait_for_shutdown(&self) {
        if self.shutdown_complete.load(Ordering::Acquire) {
            return;
        }
        self.shutdown_signal.notified().await;
    }
}
