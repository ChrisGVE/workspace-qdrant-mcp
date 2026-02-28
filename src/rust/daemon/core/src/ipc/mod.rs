//! Inter-process communication between Python MCP server and Rust engine
//!
//! This module provides IPC mechanisms for communication between
//! the MCP server and the Rust priority processing engine.

mod client;
mod server;

pub use client::IpcClient;
pub use server::IpcServer;

use std::path::PathBuf;

use serde::{Deserialize, Serialize};
use thiserror::Error;
use tokio::sync::mpsc;

use crate::processing::{
    PipelineStats, TaskPayload, TaskPriority, TaskResult, TaskSource,
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
    GetStats { request_id: String },

    /// Health check
    HealthCheck { request_id: String },

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
    TaskSubmitted { task_id: uuid::Uuid, request_id: String },

    /// Task completed
    TaskCompleted {
        task_id: uuid::Uuid,
        result: TaskResult,
        request_id: String,
    },

    /// Pipeline statistics
    Stats {
        stats: PipelineStats,
        request_id: String,
    },

    /// Health check response
    HealthCheckOk { status: String, request_id: String },

    /// Engine shutdown acknowledgment
    ShutdownAck { request_id: String },

    /// Configuration applied
    ConfigurationApplied { request_id: String },

    /// Error response
    Error { error: String, request_id: String },
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

#[cfg(test)]
mod tests;
