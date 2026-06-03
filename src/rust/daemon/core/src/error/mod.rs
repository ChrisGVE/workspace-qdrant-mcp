//! Comprehensive error handling for workspace-qdrant-mcp
//!
//! This module provides structured error types with detailed context,
//! recovery strategies, and circuit breaker patterns for reliable operations.

mod circuit_breaker;
mod monitor;
mod recovery;

pub use circuit_breaker::{CircuitBreaker, CircuitBreakerStatus};
pub use monitor::{DefaultErrorMonitor, ErrorMonitor, ErrorStats};
pub use recovery::{ErrorRecovery, ErrorRecoveryStrategy};

use std::collections::HashMap;
use std::fmt;
use std::time::Duration;
use thiserror::Error;

/// Main error type for the workspace-qdrant-mcp system
#[derive(Error, Debug)]
pub enum WorkspaceError {
    #[error("Configuration error: {message}")]
    Configuration {
        message: String,
        #[source]
        source: Option<Box<dyn std::error::Error + Send + Sync>>,
    },

    #[error("Network error: {message} (attempt {attempt}/{max_attempts})")]
    Network {
        message: String,
        attempt: u32,
        max_attempts: u32,
        #[source]
        source: Option<Box<dyn std::error::Error + Send + Sync>>,
    },

    #[error("Qdrant connection error: {message}")]
    QdrantConnection {
        message: String,
        url: String,
        #[source]
        source: Option<Box<dyn std::error::Error + Send + Sync>>,
    },

    #[error("File system error: {message} (path: {path})")]
    FileSystem {
        message: String,
        path: String,
        operation: String,
        #[source]
        source: Option<Box<dyn std::error::Error + Send + Sync>>,
    },

    #[error("Document processing error: {message} (file: {file_path})")]
    DocumentProcessing {
        message: String,
        file_path: String,
        document_type: String,
        #[source]
        source: Option<Box<dyn std::error::Error + Send + Sync>>,
    },

    #[error("Embedding generation error: {message} (model: {model})")]
    Embedding {
        message: String,
        model: String,
        retry_count: u32,
        #[source]
        source: Option<Box<dyn std::error::Error + Send + Sync>>,
    },

    #[error("IPC communication error: {message} (endpoint: {endpoint})")]
    IpcCommunication {
        message: String,
        endpoint: String,
        #[source]
        source: Option<Box<dyn std::error::Error + Send + Sync>>,
    },

    #[error("Task processing error: {message} (task_id: {task_id})")]
    TaskProcessing {
        message: String,
        task_id: String,
        priority: String,
        #[source]
        source: Option<Box<dyn std::error::Error + Send + Sync>>,
    },

    #[error("Validation error: {message}")]
    Validation {
        message: String,
        field: Option<String>,
    },

    #[error("Timeout error: {message} (duration: {duration_ms}ms)")]
    Timeout {
        message: String,
        duration_ms: u64,
        operation: String,
    },

    #[error("Resource exhaustion: {message}")]
    ResourceExhaustion {
        message: String,
        resource_type: String,
        current_usage: Option<u64>,
        limit: Option<u64>,
    },

    #[error("Circuit breaker open: {service} is unavailable")]
    CircuitBreakerOpen {
        service: String,
        failure_count: u32,
        last_failure: String,
    },

    #[error("Authentication error: {message}")]
    Authentication { message: String, service: String },

    #[error("Internal error: {message}")]
    Internal {
        message: String,
        component: String,
        #[source]
        source: Option<Box<dyn std::error::Error + Send + Sync>>,
    },
}

/// Daemon-specific errors for internal operations.
#[derive(Error, Debug)]
pub enum DaemonError {
    /// Parse error during source code analysis.
    #[error("Parse error: {0}")]
    ParseError(String),

    /// IO error.
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    /// Generic error message.
    #[error("{0}")]
    Other(String),
}

impl WorkspaceError {
    /// Create a configuration error
    pub fn configuration(message: impl Into<String>) -> Self {
        Self::Configuration {
            message: message.into(),
            source: None,
        }
    }

    /// Create a configuration error with source
    pub fn configuration_with_source(
        message: impl Into<String>,
        source: impl std::error::Error + Send + Sync + 'static,
    ) -> Self {
        Self::Configuration {
            message: message.into(),
            source: Some(Box::new(source)),
        }
    }

    /// Create a network error
    pub fn network(message: impl Into<String>, attempt: u32, max_attempts: u32) -> Self {
        Self::Network {
            message: message.into(),
            attempt,
            max_attempts,
            source: None,
        }
    }

    /// Create a Qdrant connection error
    pub fn qdrant_connection(message: impl Into<String>, url: impl Into<String>) -> Self {
        Self::QdrantConnection {
            message: message.into(),
            url: url.into(),
            source: None,
        }
    }

    /// Create a file system error
    pub fn file_system(
        message: impl Into<String>,
        path: impl Into<String>,
        operation: impl Into<String>,
    ) -> Self {
        Self::FileSystem {
            message: message.into(),
            path: path.into(),
            operation: operation.into(),
            source: None,
        }
    }

    /// Create a document processing error
    pub fn document_processing(
        message: impl Into<String>,
        file_path: impl Into<String>,
        document_type: impl Into<String>,
    ) -> Self {
        Self::DocumentProcessing {
            message: message.into(),
            file_path: file_path.into(),
            document_type: document_type.into(),
            source: None,
        }
    }

    /// Create a timeout error
    pub fn timeout(
        message: impl Into<String>,
        duration: Duration,
        operation: impl Into<String>,
    ) -> Self {
        Self::Timeout {
            message: message.into(),
            duration_ms: duration.as_millis() as u64,
            operation: operation.into(),
        }
    }

    /// Create a circuit breaker error
    pub fn circuit_breaker_open(
        service: impl Into<String>,
        failure_count: u32,
        last_failure: impl Into<String>,
    ) -> Self {
        Self::CircuitBreakerOpen {
            service: service.into(),
            failure_count,
            last_failure: last_failure.into(),
        }
    }

    /// Check if this error is retryable
    pub fn is_retryable(&self) -> bool {
        match self {
            Self::Network { .. } => true,
            Self::QdrantConnection { .. } => true,
            Self::Timeout { .. } => true,
            Self::ResourceExhaustion { .. } => true,
            Self::IpcCommunication { .. } => true,
            Self::Embedding { .. } => true,
            Self::Configuration { .. } => false,
            Self::Validation { .. } => false,
            Self::Authentication { .. } => false,
            Self::CircuitBreakerOpen { .. } => false,
            Self::FileSystem { operation, .. } => operation != "delete",
            Self::DocumentProcessing { .. } => false,
            Self::TaskProcessing { .. } => false,
            Self::Internal { .. } => false,
        }
    }

    /// Get suggested retry delay
    pub fn retry_delay(&self) -> Option<Duration> {
        match self {
            Self::Network { attempt, .. } => {
                Some(Duration::from_millis(1000 * (2_u64.pow(*attempt - 1))))
            }
            Self::QdrantConnection { .. } => Some(Duration::from_secs(2)),
            Self::Timeout { .. } => Some(Duration::from_millis(500)),
            Self::ResourceExhaustion { .. } => Some(Duration::from_secs(5)),
            Self::IpcCommunication { .. } => Some(Duration::from_millis(100)),
            Self::Embedding { retry_count, .. } => {
                Some(Duration::from_millis(500 * (2_u64.pow(*retry_count))))
            }
            Self::FileSystem { .. } => Some(Duration::from_millis(100)),
            _ => None,
        }
    }

    /// Get error category for monitoring
    pub fn category(&self) -> &'static str {
        match self {
            Self::Configuration { .. } => "configuration",
            Self::Network { .. } => "network",
            Self::QdrantConnection { .. } => "database",
            Self::FileSystem { .. } => "filesystem",
            Self::DocumentProcessing { .. } => "processing",
            Self::Embedding { .. } => "embedding",
            Self::IpcCommunication { .. } => "ipc",
            Self::TaskProcessing { .. } => "task",
            Self::Validation { .. } => "validation",
            Self::Timeout { .. } => "timeout",
            Self::ResourceExhaustion { .. } => "resource",
            Self::CircuitBreakerOpen { .. } => "circuit_breaker",
            Self::Authentication { .. } => "auth",
            Self::Internal { .. } => "internal",
        }
    }

    /// Get severity level
    pub fn severity(&self) -> ErrorSeverity {
        match self {
            Self::Configuration { .. } => ErrorSeverity::High,
            Self::Authentication { .. } => ErrorSeverity::High,
            Self::CircuitBreakerOpen { .. } => ErrorSeverity::High,
            Self::Internal { .. } => ErrorSeverity::High,
            Self::Network { .. } => ErrorSeverity::Medium,
            Self::QdrantConnection { .. } => ErrorSeverity::Medium,
            Self::ResourceExhaustion { .. } => ErrorSeverity::Medium,
            Self::Timeout { .. } => ErrorSeverity::Medium,
            Self::FileSystem { .. } => ErrorSeverity::Low,
            Self::DocumentProcessing { .. } => ErrorSeverity::Low,
            Self::Embedding { .. } => ErrorSeverity::Low,
            Self::IpcCommunication { .. } => ErrorSeverity::Low,
            Self::TaskProcessing { .. } => ErrorSeverity::Low,
            Self::Validation { .. } => ErrorSeverity::Low,
        }
    }

    /// Convert error to dictionary for structured logging
    pub fn to_dict(&self) -> HashMap<String, String> {
        let mut dict = HashMap::new();

        dict.insert("error_type".to_string(), self.category().to_string());
        dict.insert("severity".to_string(), self.severity().to_string());
        dict.insert("retryable".to_string(), self.is_retryable().to_string());
        dict.insert("message".to_string(), self.to_string());

        self.insert_variant_fields(&mut dict);

        dict
    }

    fn insert_variant_fields(&self, dict: &mut HashMap<String, String>) {
        self.insert_io_variant_fields(dict);
        self.insert_ops_variant_fields(dict);
    }

    fn insert_io_variant_fields(&self, d: &mut HashMap<String, String>) {
        match self {
            Self::Configuration { message, .. } => {
                d.insert("config_message".into(), message.clone());
            }
            Self::Network {
                message,
                attempt,
                max_attempts,
                ..
            } => {
                d.insert("network_message".into(), message.clone());
                d.insert("attempt".into(), attempt.to_string());
                d.insert("max_attempts".into(), max_attempts.to_string());
            }
            Self::QdrantConnection { message, url, .. } => {
                d.insert("qdrant_message".into(), message.clone());
                d.insert("url".into(), url.clone());
            }
            Self::FileSystem {
                message,
                path,
                operation,
                ..
            } => {
                d.insert("fs_message".into(), message.clone());
                d.insert("path".into(), path.clone());
                d.insert("operation".into(), operation.clone());
            }
            Self::DocumentProcessing {
                message,
                file_path,
                document_type,
                ..
            } => {
                d.insert("doc_message".into(), message.clone());
                d.insert("file_path".into(), file_path.clone());
                d.insert("document_type".into(), document_type.clone());
            }
            Self::Embedding {
                message,
                model,
                retry_count,
                ..
            } => {
                d.insert("emb_message".into(), message.clone());
                d.insert("model".into(), model.clone());
                d.insert("retry_count".into(), retry_count.to_string());
            }
            Self::IpcCommunication {
                message, endpoint, ..
            } => {
                d.insert("ipc_message".into(), message.clone());
                d.insert("endpoint".into(), endpoint.clone());
            }
            _ => {}
        }
    }

    fn insert_ops_variant_fields(&self, d: &mut HashMap<String, String>) {
        match self {
            Self::TaskProcessing {
                message,
                task_id,
                priority,
                ..
            } => {
                d.insert("task_message".into(), message.clone());
                d.insert("task_id".into(), task_id.clone());
                d.insert("priority".into(), priority.clone());
            }
            Self::Validation { message, field } => {
                d.insert("val_message".into(), message.clone());
                if let Some(f) = field {
                    d.insert("field".into(), f.clone());
                }
            }
            Self::Timeout {
                message,
                duration_ms,
                operation,
            } => {
                d.insert("timeout_message".into(), message.clone());
                d.insert("duration_ms".into(), duration_ms.to_string());
                d.insert("operation".into(), operation.clone());
            }
            Self::ResourceExhaustion {
                message,
                resource_type,
                current_usage,
                limit,
            } => {
                d.insert("resource_message".into(), message.clone());
                d.insert("resource_type".into(), resource_type.clone());
                if let Some(u) = current_usage {
                    d.insert("current_usage".into(), u.to_string());
                }
                if let Some(l) = limit {
                    d.insert("limit".into(), l.to_string());
                }
            }
            Self::CircuitBreakerOpen {
                service,
                failure_count,
                last_failure,
            } => {
                d.insert("service".into(), service.clone());
                d.insert("failure_count".into(), failure_count.to_string());
                d.insert("last_failure".into(), last_failure.clone());
            }
            Self::Authentication { message, service } => {
                d.insert("auth_message".into(), message.clone());
                d.insert("service".into(), service.clone());
            }
            Self::Internal {
                message, component, ..
            } => {
                d.insert("internal_message".into(), message.clone());
                d.insert("component".into(), component.clone());
            }
            _ => {}
        }
    }
}

/// Error severity levels for monitoring and alerting
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ErrorSeverity {
    Low,
    Medium,
    High,
    Critical,
}

impl fmt::Display for ErrorSeverity {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Low => write!(f, "low"),
            Self::Medium => write!(f, "medium"),
            Self::High => write!(f, "high"),
            Self::Critical => write!(f, "critical"),
        }
    }
}

// Conversion from the daemon config loader error (WI-a2).
impl From<crate::config::ConfigError> for WorkspaceError {
    fn from(error: crate::config::ConfigError) -> Self {
        WorkspaceError::Configuration {
            message: error.to_string(),
            source: Some(Box::new(error)),
        }
    }
}

// Type alias for common result type
pub type Result<T> = std::result::Result<T, WorkspaceError>;

#[cfg(test)]
mod tests;
