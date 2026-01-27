//! CLI error types module
//!
//! Custom error types for CLI operations using thiserror for structured
//! error handling across all commands.
//!
//! Note: Error types and helpers are infrastructure for future CLI commands.

#![allow(dead_code)]

use thiserror::Error;

/// CLI error types
#[derive(Error, Debug)]
pub enum CliError {
    /// Connection to memexd daemon failed
    #[error("Failed to connect to daemon at {address}: {message}")]
    DaemonConnection {
        address: String,
        message: String,
    },

    /// gRPC call failed
    #[error("gRPC error: {message}")]
    Grpc {
        message: String,
        #[source]
        source: Option<tonic::Status>,
    },

    /// Configuration is invalid
    #[error("Configuration error: {0}")]
    Config(String),

    /// Output formatting failed
    #[error("Output formatting error: {0}")]
    OutputFormat(String),

    /// User input validation failed
    #[error("Invalid argument: {0}")]
    InvalidArgument(String),

    /// File system operation failed
    #[error("File system error: {message}")]
    FileSystem {
        message: String,
        #[source]
        source: Option<std::io::Error>,
    },

    /// Service operation failed (start/stop/restart)
    #[error("Service operation failed: {0}")]
    ServiceOperation(String),

    /// Daemon not running
    #[error("Daemon not running. Start with: wqm service start")]
    DaemonNotRunning,

    /// Operation timeout
    #[error("Operation timed out after {seconds}s")]
    Timeout { seconds: u64 },

    /// Unknown or unexpected error
    #[error("{0}")]
    Other(String),
}

impl CliError {
    /// Create a daemon connection error
    pub fn daemon_connection(address: impl Into<String>, message: impl Into<String>) -> Self {
        Self::DaemonConnection {
            address: address.into(),
            message: message.into(),
        }
    }

    /// Create a gRPC error from tonic status
    pub fn grpc(status: tonic::Status) -> Self {
        Self::Grpc {
            message: status.message().to_string(),
            source: Some(status),
        }
    }

    /// Create a gRPC error with custom message
    pub fn grpc_message(message: impl Into<String>) -> Self {
        Self::Grpc {
            message: message.into(),
            source: None,
        }
    }

    /// Create a file system error
    pub fn filesystem(message: impl Into<String>, source: std::io::Error) -> Self {
        Self::FileSystem {
            message: message.into(),
            source: Some(source),
        }
    }
}

/// Convert tonic::Status to CliError automatically
impl From<tonic::Status> for CliError {
    fn from(status: tonic::Status) -> Self {
        Self::grpc(status)
    }
}

/// Convert std::io::Error to CliError
impl From<std::io::Error> for CliError {
    fn from(err: std::io::Error) -> Self {
        Self::FileSystem {
            message: err.to_string(),
            source: Some(err),
        }
    }
}

/// Result type alias for CLI operations
pub type CliResult<T> = std::result::Result<T, CliError>;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_daemon_connection_error() {
        let err = CliError::daemon_connection("http://localhost:50051", "connection refused");
        assert!(err.to_string().contains("localhost:50051"));
        assert!(err.to_string().contains("connection refused"));
    }

    #[test]
    fn test_config_error() {
        let err = CliError::Config("invalid port".to_string());
        assert!(err.to_string().contains("invalid port"));
    }

    #[test]
    fn test_daemon_not_running() {
        let err = CliError::DaemonNotRunning;
        assert!(err.to_string().contains("service start"));
    }
}
