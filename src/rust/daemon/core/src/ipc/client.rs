//! IPC client for sending requests to the Rust engine.

use std::sync::Arc;

use tokio::sync::{mpsc, Mutex};
use uuid::Uuid;

use crate::processing::{TaskPayload, TaskPriority, TaskSource};
use super::{EngineSettings, IpcError, IpcRequest, IpcResponse};

/// IPC client for sending requests to the Rust engine
pub struct IpcClient {
    request_sender: mpsc::UnboundedSender<IpcRequest>,
    response_receiver: Arc<Mutex<Option<mpsc::UnboundedReceiver<IpcResponse>>>>,
}

impl IpcClient {
    /// Create a new IPC client (called internally by IpcServer::new)
    pub(super) fn new(
        request_sender: mpsc::UnboundedSender<IpcRequest>,
        response_receiver: mpsc::UnboundedReceiver<IpcResponse>,
    ) -> Self {
        Self {
            request_sender,
            response_receiver: Arc::new(Mutex::new(Some(response_receiver))),
        }
    }

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
