//! Inter-process communication between Python MCP server and Rust engine
//!
//! This module provides various IPC mechanisms for communication between
//! the Python MCP server and the Rust priority processing engine.

use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Duration;

use serde::{Deserialize, Serialize};
use thiserror::Error;
use tokio::sync::{mpsc, Mutex, RwLock};
use uuid::Uuid;

use crate::processing::{
    Pipeline, TaskPriority, TaskSource, TaskPayload, TaskResult, TaskResultHandle, TaskSubmitter,
    PipelineStats,
};

/// IPC communication errors
#[derive(Error, Debug)]
pub enum IpcError {
    #[error("Channel closed")]
    ChannelClosed,
    
    #[error("Request timeout")]
    Timeout,
    
    #[error("Invalid request format: {0}")]
    InvalidRequest(String),
    
    #[error("Processing error: {0}")]
    ProcessingError(String),
    
    #[error("Serialization error: {0}")]
    SerializationError(#[from] serde_json::Error),
    
    #[error("IO error: {0}")]
    IoError(#[from] std::io::Error),
    
    #[error("Send error: {0}")]
    SendError(String),
}

impl<T> From<mpsc::error::SendError<T>> for IpcError {
    fn from(err: mpsc::error::SendError<T>) -> Self {
        IpcError::SendError(err.to_string())
    }
}

/// Request types that can be sent from Python to Rust engine
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", content = "data")]
pub enum IpcRequest {
    /// Submit a task for processing
    SubmitTask {
        priority: TaskPriority,
        source: TaskSource,
        payload: TaskPayload,
        timeout_ms: Option<u64>,
        request_id: String,
    },
    
    /// Get pipeline statistics
    GetStats {
        request_id: String,
    },
    
    /// Health check
    HealthCheck {
        request_id: String,
    },
    
    /// Shutdown the engine
    Shutdown {
        graceful: bool,
        timeout_ms: Option<u64>,
        request_id: String,
    },
    
    /// Configure engine settings
    Configure {
        settings: EngineSettings,
        request_id: String,
    },
}

/// Response types sent from Rust engine to Python
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", content = "data")]
pub enum IpcResponse {
    /// Task submitted successfully
    TaskSubmitted {
        task_id: Uuid,
        request_id: String,
    },
    
    /// Task completed
    TaskCompleted {
        task_id: Uuid,
        result: TaskResult,
        request_id: String,
    },
    
    /// Pipeline statistics
    Stats {
        stats: PipelineStats,
        request_id: String,
    },
    
    /// Health check response
    HealthCheckOk {
        status: String,
        request_id: String,
    },
    
    /// Engine shutdown acknowledgment
    ShutdownAck {
        request_id: String,
    },
    
    /// Configuration applied
    ConfigurationApplied {
        request_id: String,
    },
    
    /// Error response
    Error {
        error: String,
        request_id: String,
    },
}

/// Engine configuration settings
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EngineSettings {
    pub max_concurrent_tasks: Option<usize>,
    pub default_timeout_ms: Option<u64>,
    pub enable_preemption: Option<bool>,
    pub log_level: Option<String>,
}

/// IPC communication channel types
#[derive(Debug, Clone)]
pub enum IpcChannelType {
    /// In-memory channels (fastest, same process)
    InMemory,
    /// Unix domain sockets (cross-process, Unix only)
    #[cfg(unix)]
    UnixSocket { path: PathBuf },
    /// Named pipes (cross-process, Windows)
    #[cfg(windows)]
    NamedPipe { name: String },
    /// TCP sockets (cross-process, network capable)
    TcpSocket { host: String, port: u16 },
    /// Shared memory (fastest cross-process)
    SharedMemory { segment_name: String, size: usize },
}

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
    pub fn new(
        max_concurrent_tasks: usize,
    ) -> (Self, IpcClient) {
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
        
        let client = IpcClient {
            request_sender,
            response_receiver: Arc::new(Mutex::new(Some(response_receiver))),
        };
        
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
                // Handle incoming requests
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
                
                // Periodic cleanup
                _ = cleanup_interval.tick() => {
                    Self::cleanup_completed_tasks(&active_tasks).await;
                }
                
                // Shutdown signal
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
                priority, 
                source, 
                payload, 
                timeout_ms,
                request_id 
            } => {
                let timeout = timeout_ms.map(Duration::from_millis);
                
                match task_submitter.submit_task(priority, source, payload, timeout).await {
                    Ok(task_handle) => {
                        let task_id = task_handle.task_id;
                        
                        // Store the task handle
                        {
                            let mut active_lock = active_tasks.write().await;
                            active_lock.insert(task_id, task_handle);
                        }
                        
                        // Send immediate response
                        let response = IpcResponse::TaskSubmitted {
                            task_id,
                            request_id: request_id.clone(),
                        };
                        response_sender.send(response)?;
                        
                        // For now, we immediately send a completion response
                        // In a full implementation, we'd wait for actual task completion
                        let completion_response = IpcResponse::TaskCompleted {
                            task_id,
                            result: TaskResult::Success {
                                execution_time_ms: 100,
                                data: crate::processing::TaskResultData::Generic {
                                    message: "Task completed via IPC".to_string(),
                                    data: serde_json::json!({}),
                                    checkpoint_id: None,
                                },
                            },
                            request_id,
                        };
                        
                        // Remove from active tasks
                        {
                            let mut active_lock = active_tasks.write().await;
                            active_lock.remove(&task_id);
                        }
                        
                        response_sender.send(completion_response)?;
                    }
                    Err(e) => {
                        let response = IpcResponse::Error {
                            error: e.to_string(),
                            request_id,
                        };
                        response_sender.send(response)?;
                    }
                }
            }
            
            IpcRequest::GetStats { request_id } => {
                let stats = {
                    let pipeline_lock = pipeline.lock().await;
                    pipeline_lock.stats().await
                };
                
                let response = IpcResponse::Stats {
                    stats,
                    request_id,
                };
                response_sender.send(response)?;
            }
            
            IpcRequest::HealthCheck { request_id } => {
                let response = IpcResponse::HealthCheckOk {
                    status: "OK".to_string(),
                    request_id,
                };
                response_sender.send(response)?;
            }
            
            IpcRequest::Configure { settings: new_settings, request_id } => {
                {
                    let mut settings_lock = settings.write().await;
                    *settings_lock = new_settings;
                }
                
                let response = IpcResponse::ConfigurationApplied {
                    request_id,
                };
                response_sender.send(response)?;
            }
            
            IpcRequest::Shutdown { graceful: _, timeout_ms: _, request_id } => {
                let response = IpcResponse::ShutdownAck {
                    request_id,
                };
                response_sender.send(response)?;
                shutdown_complete.store(true, Ordering::Release);
                shutdown_signal.notify_waiters();
            }
        }
        
        Ok(())
    }
    
    /// Clean up completed tasks from the active tasks registry.
    ///
    /// Checks each task handle to see if its result sender has been dropped
    /// (indicating the task executor finished or was abandoned). Completed
    /// tasks are removed to free resources and prevent unbounded growth.
    async fn cleanup_completed_tasks(
        active_tasks: &Arc<RwLock<HashMap<Uuid, TaskResultHandle>>>,
    ) {
        let mut active_lock = active_tasks.write().await;
        let before = active_lock.len();

        // Retain only tasks whose result receiver is still waiting
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

/// IPC client for sending requests to the Rust engine
pub struct IpcClient {
    request_sender: mpsc::UnboundedSender<IpcRequest>,
    response_receiver: Arc<Mutex<Option<mpsc::UnboundedReceiver<IpcResponse>>>>,
}

impl IpcClient {
    /// Submit a task for processing
    pub async fn submit_task(
        &self,
        priority: TaskPriority,
        source: TaskSource,
        payload: TaskPayload,
        timeout_ms: Option<u64>,
    ) -> Result<String, IpcError> {
        let request_id = Uuid::new_v4().to_string();
        
        let request = IpcRequest::SubmitTask {
            priority,
            source,
            payload,
            timeout_ms,
            request_id: request_id.clone(),
        };
        
        self.request_sender.send(request)
            .map_err(|_| IpcError::ChannelClosed)?;
        
        Ok(request_id)
    }
    
    /// Get pipeline statistics
    pub async fn get_stats(&self) -> Result<String, IpcError> {
        let request_id = Uuid::new_v4().to_string();
        
        let request = IpcRequest::GetStats {
            request_id: request_id.clone(),
        };
        
        self.request_sender.send(request)
            .map_err(|_| IpcError::ChannelClosed)?;
        
        Ok(request_id)
    }
    
    /// Perform health check
    pub async fn health_check(&self) -> Result<String, IpcError> {
        let request_id = Uuid::new_v4().to_string();
        
        let request = IpcRequest::HealthCheck {
            request_id: request_id.clone(),
        };
        
        self.request_sender.send(request)
            .map_err(|_| IpcError::ChannelClosed)?;
        
        Ok(request_id)
    }
    
    /// Configure the engine
    pub async fn configure(&self, settings: EngineSettings) -> Result<String, IpcError> {
        let request_id = Uuid::new_v4().to_string();
        
        let request = IpcRequest::Configure {
            settings,
            request_id: request_id.clone(),
        };
        
        self.request_sender.send(request)
            .map_err(|_| IpcError::ChannelClosed)?;
        
        Ok(request_id)
    }
    
    /// Shutdown the engine
    pub async fn shutdown(&self, graceful: bool, timeout_ms: Option<u64>) -> Result<String, IpcError> {
        let request_id = Uuid::new_v4().to_string();
        
        let request = IpcRequest::Shutdown {
            graceful,
            timeout_ms,
            request_id: request_id.clone(),
        };
        
        self.request_sender.send(request)
            .map_err(|_| IpcError::ChannelClosed)?;
        
        Ok(request_id)
    }
    
    /// Get next response (non-blocking)
    pub async fn try_recv_response(&self) -> Result<Option<IpcResponse>, IpcError> {
        let mut receiver_lock = self.response_receiver.lock().await;
        if let Some(ref mut receiver) = receiver_lock.as_mut() {
            match receiver.try_recv() {
                Ok(response) => Ok(Some(response)),
                Err(mpsc::error::TryRecvError::Empty) => Ok(None),
                Err(mpsc::error::TryRecvError::Disconnected) => Err(IpcError::ChannelClosed),
            }
        } else {
            Err(IpcError::ChannelClosed)
        }
    }
    
    /// Wait for next response
    pub async fn recv_response(&self) -> Result<IpcResponse, IpcError> {
        let mut receiver_lock = self.response_receiver.lock().await;
        if let Some(ref mut receiver) = receiver_lock.as_mut() {
            receiver.recv().await.ok_or(IpcError::ChannelClosed)
        } else {
            Err(IpcError::ChannelClosed)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;
    use crate::processing::{TaskPriority, TaskSource, TaskPayload};
    
    #[tokio::test]
    async fn test_ipc_server_client_communication() {
        let (server, client) = IpcServer::new(2);
        
        // Start the server
        server.start().await.expect("Failed to start IPC server");
        
        // Test health check
        let request_id = client.health_check().await.expect("Failed to send health check");
        
        // Wait for response
        let response = tokio::time::timeout(
            Duration::from_secs(1),
            client.recv_response()
        ).await.expect("Health check timed out").expect("Failed to receive response");
        
        match response {
            IpcResponse::HealthCheckOk { request_id: resp_id, status } => {
                assert_eq!(resp_id, request_id);
                assert_eq!(status, "OK");
            }
            other => panic!("Expected HealthCheckOk, got: {:?}", other),
        }
    }
    
    #[tokio::test]
    async fn test_ipc_task_submission() {
        let (server, client) = IpcServer::new(2);
        
        server.start().await.expect("Failed to start IPC server");
        
        // Submit a task
        let request_id = client.submit_task(
            TaskPriority::McpRequests,
            TaskSource::McpServer {
                request_id: "test_request".to_string(),
            },
            TaskPayload::Generic {
                operation: "test_operation".to_string(),
                parameters: HashMap::new(),
            },
            Some(5000),
        ).await.expect("Failed to submit task");
        
        // Wait for task submitted response
        let response = tokio::time::timeout(
            Duration::from_secs(1),
            client.recv_response()
        ).await.expect("Task submission timed out").expect("Failed to receive response");
        
        match response {
            IpcResponse::TaskSubmitted { task_id: _, request_id: resp_id } => {
                assert_eq!(resp_id, request_id);
            }
            other => panic!("Expected TaskSubmitted, got: {:?}", other),
        }
    }
    
    #[tokio::test]
    async fn test_ipc_get_stats() {
        let (server, client) = IpcServer::new(3);
        
        server.start().await.expect("Failed to start IPC server");
        
        // Get stats
        let request_id = client.get_stats().await.expect("Failed to get stats");
        
        // Wait for response
        let response = tokio::time::timeout(
            Duration::from_secs(1),
            client.recv_response()
        ).await.expect("Get stats timed out").expect("Failed to receive response");
        
        match response {
            IpcResponse::Stats { stats, request_id: resp_id } => {
                assert_eq!(resp_id, request_id);
                assert_eq!(stats.total_capacity, 3);
            }
            other => panic!("Expected Stats, got: {:?}", other),
        }
    }
    
    #[tokio::test]
    async fn test_ipc_configuration() {
        let (server, client) = IpcServer::new(2);
        
        server.start().await.expect("Failed to start IPC server");
        
        // Configure engine
        let settings = EngineSettings {
            max_concurrent_tasks: Some(5),
            default_timeout_ms: Some(10000),
            enable_preemption: Some(false),
            log_level: Some("debug".to_string()),
        };
        
        let request_id = client.configure(settings).await.expect("Failed to configure");
        
        // Wait for response
        let response = tokio::time::timeout(
            Duration::from_secs(1),
            client.recv_response()
        ).await.expect("Configuration timed out").expect("Failed to receive response");
        
        match response {
            IpcResponse::ConfigurationApplied { request_id: resp_id } => {
                assert_eq!(resp_id, request_id);
            }
            other => panic!("Expected ConfigurationApplied, got: {:?}", other),
        }
    }
    
    /// Helper to create a test TaskResultHandle
    fn make_test_handle(
        id: Uuid,
        rx: tokio::sync::oneshot::Receiver<crate::processing::TaskResult>,
    ) -> crate::processing::TaskResultHandle {
        use crate::processing::TaskContext;
        use chrono::Utc;

        let ctx = TaskContext {
            task_id: id,
            priority: TaskPriority::BackgroundWatching,
            created_at: Utc::now(),
            timeout_ms: None,
            source: TaskSource::Generic { operation: "test".into() },
            metadata: HashMap::new(),
            checkpoint_id: None,
            supports_checkpointing: false,
        };

        crate::processing::TaskResultHandle::new_for_test(id, ctx, rx)
    }

    #[tokio::test]
    async fn test_cleanup_completed_tasks_removes_finished() {
        use crate::processing::{TaskResult, TaskResultHandle};
        use tokio::sync::oneshot;

        let active_tasks: Arc<RwLock<HashMap<Uuid, TaskResultHandle>>> =
            Arc::new(RwLock::new(HashMap::new()));

        // Create a "completed" task (sender dropped)
        let id1 = Uuid::new_v4();
        let (_tx1, rx1) = oneshot::channel::<TaskResult>();
        drop(_tx1); // simulate completion
        let handle1 = make_test_handle(id1, rx1);

        // Create an "active" task (sender still alive)
        let id2 = Uuid::new_v4();
        let (tx2, rx2) = oneshot::channel::<TaskResult>();
        let handle2 = make_test_handle(id2, rx2);

        {
            let mut lock = active_tasks.write().await;
            lock.insert(id1, handle1);
            lock.insert(id2, handle2);
        }

        assert_eq!(active_tasks.read().await.len(), 2);

        IpcServer::cleanup_completed_tasks(&active_tasks).await;

        // Only the active task should remain
        let remaining = active_tasks.read().await;
        assert_eq!(remaining.len(), 1);
        assert!(remaining.contains_key(&id2));

        // Keep tx2 alive for the duration of the test
        drop(tx2);
    }

    #[tokio::test]
    async fn test_cleanup_completed_tasks_noop_when_all_active() {
        use crate::processing::{TaskResult, TaskResultHandle};
        use tokio::sync::oneshot;

        let active_tasks: Arc<RwLock<HashMap<Uuid, TaskResultHandle>>> =
            Arc::new(RwLock::new(HashMap::new()));

        let id1 = Uuid::new_v4();
        let (tx1, rx1) = oneshot::channel::<TaskResult>();
        let handle1 = make_test_handle(id1, rx1);

        {
            let mut lock = active_tasks.write().await;
            lock.insert(id1, handle1);
        }

        IpcServer::cleanup_completed_tasks(&active_tasks).await;

        // Task should still be there - it's active
        assert_eq!(active_tasks.read().await.len(), 1);

        drop(tx1);
    }

    #[tokio::test]
    async fn test_ipc_shutdown() {
        let (server, client) = IpcServer::new(2);
        
        server.start().await.expect("Failed to start IPC server");
        
        // Send shutdown
        let request_id = client.shutdown(true, Some(1000)).await.expect("Failed to shutdown");
        
        // Wait for shutdown ack
        let response = tokio::time::timeout(
            Duration::from_secs(1),
            client.recv_response()
        ).await.expect("Shutdown timed out").expect("Failed to receive response");
        
        match response {
            IpcResponse::ShutdownAck { request_id: resp_id } => {
                assert_eq!(resp_id, request_id);
            }
            other => panic!("Expected ShutdownAck, got: {:?}", other),
        }
        
        // Server should shut down
        tokio::time::timeout(
            Duration::from_secs(2),
            server.wait_for_shutdown()
        ).await.expect("Server did not shut down in time");
    }
}
