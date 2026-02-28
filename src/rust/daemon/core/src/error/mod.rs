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
    Authentication {
        message: String,
        service: String,
    },

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
    pub fn to_dict(&self) -> std::collections::HashMap<String, String> {
        let mut dict = std::collections::HashMap::new();

        dict.insert("error_type".to_string(), self.category().to_string());
        dict.insert("severity".to_string(), self.severity().to_string());
        dict.insert("retryable".to_string(), self.is_retryable().to_string());
        dict.insert("message".to_string(), self.to_string());

        match self {
            Self::Configuration { message, .. } => {
                dict.insert("config_message".to_string(), message.clone());
            }
            Self::Network { message, attempt, max_attempts, .. } => {
                dict.insert("network_message".to_string(), message.clone());
                dict.insert("attempt".to_string(), attempt.to_string());
                dict.insert("max_attempts".to_string(), max_attempts.to_string());
            }
            Self::QdrantConnection { message, url, .. } => {
                dict.insert("qdrant_message".to_string(), message.clone());
                dict.insert("url".to_string(), url.clone());
            }
            Self::FileSystem { message, path, operation, .. } => {
                dict.insert("fs_message".to_string(), message.clone());
                dict.insert("path".to_string(), path.clone());
                dict.insert("operation".to_string(), operation.clone());
            }
            Self::DocumentProcessing { message, file_path, document_type, .. } => {
                dict.insert("doc_message".to_string(), message.clone());
                dict.insert("file_path".to_string(), file_path.clone());
                dict.insert("document_type".to_string(), document_type.clone());
            }
            Self::Embedding { message, model, retry_count, .. } => {
                dict.insert("emb_message".to_string(), message.clone());
                dict.insert("model".to_string(), model.clone());
                dict.insert("retry_count".to_string(), retry_count.to_string());
            }
            Self::IpcCommunication { message, endpoint, .. } => {
                dict.insert("ipc_message".to_string(), message.clone());
                dict.insert("endpoint".to_string(), endpoint.clone());
            }
            Self::TaskProcessing { message, task_id, priority, .. } => {
                dict.insert("task_message".to_string(), message.clone());
                dict.insert("task_id".to_string(), task_id.clone());
                dict.insert("priority".to_string(), priority.clone());
            }
            Self::Validation { message, field } => {
                dict.insert("val_message".to_string(), message.clone());
                if let Some(field) = field {
                    dict.insert("field".to_string(), field.clone());
                }
            }
            Self::Timeout { message, duration_ms, operation } => {
                dict.insert("timeout_message".to_string(), message.clone());
                dict.insert("duration_ms".to_string(), duration_ms.to_string());
                dict.insert("operation".to_string(), operation.clone());
            }
            Self::ResourceExhaustion { message, resource_type, current_usage, limit } => {
                dict.insert("resource_message".to_string(), message.clone());
                dict.insert("resource_type".to_string(), resource_type.clone());
                if let Some(usage) = current_usage {
                    dict.insert("current_usage".to_string(), usage.to_string());
                }
                if let Some(limit) = limit {
                    dict.insert("limit".to_string(), limit.to_string());
                }
            }
            Self::CircuitBreakerOpen { service, failure_count, last_failure } => {
                dict.insert("service".to_string(), service.clone());
                dict.insert("failure_count".to_string(), failure_count.to_string());
                dict.insert("last_failure".to_string(), last_failure.clone());
            }
            Self::Authentication { message, service } => {
                dict.insert("auth_message".to_string(), message.clone());
                dict.insert("service".to_string(), service.clone());
            }
            Self::Internal { message, component, .. } => {
                dict.insert("internal_message".to_string(), message.clone());
                dict.insert("component".to_string(), component.clone());
            }
        }

        dict
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

// Conversion from UnifiedConfigError
impl From<crate::unified_config::UnifiedConfigError> for WorkspaceError {
    fn from(error: crate::unified_config::UnifiedConfigError) -> Self {
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
