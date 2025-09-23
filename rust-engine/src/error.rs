//! Error types for the Workspace Qdrant Daemon

use thiserror::Error;
use tonic::{Code, Status};

/// Main error type for daemon operations
#[derive(Error, Debug)]
pub enum DaemonError {
    #[error("Configuration error: {0}")]
    Config(#[from] config::ConfigError),

    #[error("Database error: {0}")]
    Database(#[from] sqlx::Error),

    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),

    #[error("gRPC error: {0}")]
    Grpc(#[from] tonic::transport::Error),

    #[error("Serialization error: {0}")]
    Serialization(#[from] serde_json::Error),

    #[error("File watching error: {0}")]
    FileWatcher(#[from] notify::Error),

    #[error("Git error: {0}")]
    Git(#[from] git2::Error),

    #[error("HTTP client error: {0}")]
    Http(#[from] reqwest::Error),

    #[error("Document processing error: {message}")]
    DocumentProcessing { message: String },

    #[error("Search error: {message}")]
    Search { message: String },

    #[error("Memory management error: {message}")]
    Memory { message: String },

    #[error("System error: {message}")]
    System { message: String },

    #[error("Project detection error: {message}")]
    ProjectDetection { message: String },

    #[error("Connection pool error: {message}")]
    ConnectionPool { message: String },

    #[error("Timeout error: operation timed out after {seconds}s")]
    Timeout { seconds: u64 },

    #[error("Resource not found: {resource}")]
    NotFound { resource: String },

    #[error("Invalid input: {message}")]
    InvalidInput { message: String },

    #[error("Internal error: {message}")]
    Internal { message: String },
}

impl From<DaemonError> for Status {
    fn from(err: DaemonError) -> Self {
        match err {
            DaemonError::Config(_) | DaemonError::InvalidInput { .. } => {
                Status::new(Code::InvalidArgument, err.to_string())
            },
            DaemonError::NotFound { .. } => {
                Status::new(Code::NotFound, err.to_string())
            },
            DaemonError::Timeout { .. } => {
                Status::new(Code::DeadlineExceeded, err.to_string())
            },
            DaemonError::Database(_) | DaemonError::Io(_) | DaemonError::FileWatcher(_) => {
                Status::new(Code::Internal, "Internal server error")
            },
            _ => Status::new(Code::Internal, "Internal server error"),
        }
    }
}

/// Result type alias for daemon operations
pub type DaemonResult<T> = Result<T, DaemonError>;

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::{Error, ErrorKind};

    #[test]
    fn test_daemon_error_display() {
        let err = DaemonError::DocumentProcessing {
            message: "Test error".to_string(),
        };
        assert_eq!(format!("{}", err), "Document processing error: Test error");

        let err = DaemonError::Search {
            message: "Search failed".to_string(),
        };
        assert_eq!(format!("{}", err), "Search error: Search failed");

        let err = DaemonError::Memory {
            message: "Memory allocation failed".to_string(),
        };
        assert_eq!(format!("{}", err), "Memory management error: Memory allocation failed");

        let err = DaemonError::System {
            message: "System call failed".to_string(),
        };
        assert_eq!(format!("{}", err), "System error: System call failed");

        let err = DaemonError::ProjectDetection {
            message: "No git repository found".to_string(),
        };
        assert_eq!(format!("{}", err), "Project detection error: No git repository found");

        let err = DaemonError::ConnectionPool {
            message: "Pool exhausted".to_string(),
        };
        assert_eq!(format!("{}", err), "Connection pool error: Pool exhausted");

        let err = DaemonError::Timeout { seconds: 30 };
        assert_eq!(format!("{}", err), "Timeout error: operation timed out after 30s");

        let err = DaemonError::NotFound {
            resource: "document".to_string(),
        };
        assert_eq!(format!("{}", err), "Resource not found: document");

        let err = DaemonError::InvalidInput {
            message: "Invalid format".to_string(),
        };
        assert_eq!(format!("{}", err), "Invalid input: Invalid format");

        let err = DaemonError::Internal {
            message: "Unexpected state".to_string(),
        };
        assert_eq!(format!("{}", err), "Internal error: Unexpected state");
    }

    #[test]
    fn test_daemon_error_debug() {
        let err = DaemonError::DocumentProcessing {
            message: "Test error".to_string(),
        };
        let debug_str = format!("{:?}", err);
        assert!(debug_str.contains("DocumentProcessing"));
        assert!(debug_str.contains("Test error"));
    }

    #[test]
    fn test_daemon_error_from_io_error() {
        let io_error = Error::new(ErrorKind::NotFound, "File not found");
        let daemon_error: DaemonError = io_error.into();

        match daemon_error {
            DaemonError::Io(ref e) => {
                assert_eq!(e.kind(), ErrorKind::NotFound);
                assert_eq!(e.to_string(), "File not found");
            },
            _ => panic!("Expected DaemonError::Io"),
        }
    }

    #[test]
    fn test_daemon_error_from_serde_json_error() {
        let json_error = serde_json::from_str::<serde_json::Value>("invalid json")
            .unwrap_err();
        let daemon_error: DaemonError = json_error.into();

        match daemon_error {
            DaemonError::Serialization(_) => {
                // Success - correct variant
            },
            _ => panic!("Expected DaemonError::Serialization"),
        }
    }

    #[test]
    fn test_status_conversion_invalid_argument() {
        let err = DaemonError::InvalidInput {
            message: "Bad request".to_string(),
        };
        let status: Status = err.into();
        assert_eq!(status.code(), Code::InvalidArgument);
        assert!(status.message().contains("Invalid input: Bad request"));

        let err = DaemonError::Config(config::ConfigError::Message("Config parse error".to_string()));
        let status: Status = err.into();
        assert_eq!(status.code(), Code::InvalidArgument);
    }

    #[test]
    fn test_status_conversion_not_found() {
        let err = DaemonError::NotFound {
            resource: "user".to_string(),
        };
        let status: Status = err.into();
        assert_eq!(status.code(), Code::NotFound);
        assert!(status.message().contains("Resource not found: user"));
    }

    #[test]
    fn test_status_conversion_deadline_exceeded() {
        let err = DaemonError::Timeout { seconds: 60 };
        let status: Status = err.into();
        assert_eq!(status.code(), Code::DeadlineExceeded);
        assert!(status.message().contains("operation timed out after 60s"));
    }

    #[test]
    fn test_status_conversion_internal_error() {
        let io_error = Error::new(ErrorKind::PermissionDenied, "Access denied");
        let err = DaemonError::Io(io_error);
        let status: Status = err.into();
        assert_eq!(status.code(), Code::Internal);
        assert_eq!(status.message(), "Internal server error");

        let err = DaemonError::Database(sqlx::Error::PoolClosed);
        let status: Status = err.into();
        assert_eq!(status.code(), Code::Internal);
        assert_eq!(status.message(), "Internal server error");

        let err = DaemonError::DocumentProcessing {
            message: "Processing failed".to_string(),
        };
        let status: Status = err.into();
        assert_eq!(status.code(), Code::Internal);
        assert_eq!(status.message(), "Internal server error");
    }

    #[test]
    fn test_daemon_result_type() {
        let success: DaemonResult<i32> = Ok(42);
        assert_eq!(success.unwrap(), 42);

        let error: DaemonResult<i32> = Err(DaemonError::Internal {
            message: "Test error".to_string(),
        });
        assert!(error.is_err());
    }

    #[test]
    fn test_error_chain() {
        let io_error = Error::new(ErrorKind::NotFound, "File not found");
        let daemon_error = DaemonError::Io(io_error);

        let error_string = format!("{}", daemon_error);
        assert!(error_string.contains("I/O error:"));
        assert!(error_string.contains("File not found"));
    }

    #[test]
    fn test_all_error_variants_are_testable() {
        // Test that we can create all error variants
        let errors = vec![
            DaemonError::DocumentProcessing { message: "test".to_string() },
            DaemonError::Search { message: "test".to_string() },
            DaemonError::Memory { message: "test".to_string() },
            DaemonError::System { message: "test".to_string() },
            DaemonError::ProjectDetection { message: "test".to_string() },
            DaemonError::ConnectionPool { message: "test".to_string() },
            DaemonError::Timeout { seconds: 10 },
            DaemonError::NotFound { resource: "test".to_string() },
            DaemonError::InvalidInput { message: "test".to_string() },
            DaemonError::Internal { message: "test".to_string() },
        ];

        for error in errors {
            // Each error should be debuggable
            let debug_str = format!("{:?}", error);
            assert!(!debug_str.is_empty());

            // Each error should be displayable
            let display_str = format!("{}", error);
            assert!(!display_str.is_empty());

            // Each error should convert to Status
            let _status: Status = error.into();
        }
    }

    #[test]
    fn test_error_source_chain() {
        let io_error = Error::new(ErrorKind::NotFound, "File not found");
        let daemon_error = DaemonError::Io(io_error);

        // Test that the source chain is preserved
        let source = std::error::Error::source(&daemon_error);
        assert!(source.is_some());

        if let Some(source) = source {
            assert_eq!(source.to_string(), "File not found");
        }
    }

    #[test]
    fn test_error_conversions_comprehensive() {
        // Test git2 error conversion
        let git_error = git2::Error::from_str("test git error");
        let daemon_error: DaemonError = git_error.into();
        match daemon_error {
            DaemonError::Git(_) => {},
            _ => panic!("Expected Git error"),
        }

        // Test notify error conversion
        let notify_error = notify::Error::generic("test notify error");
        let daemon_error: DaemonError = notify_error.into();
        match daemon_error {
            DaemonError::FileWatcher(_) => {},
            _ => panic!("Expected FileWatcher error"),
        }

        // Test that we can handle tonic transport errors in the From trait
        // Note: We can't easily create tonic::transport::Error instances in tests
        // due to private constructors, but the From trait implementation exists
    }

    #[test]
    fn test_timeout_error_edge_cases() {
        // Test various timeout values
        let timeouts = [0, 1, 60, 3600, u64::MAX];
        for seconds in timeouts {
            let error = DaemonError::Timeout { seconds };
            let status: Status = error.into();
            assert_eq!(status.code(), Code::DeadlineExceeded);
            assert!(status.message().contains(&seconds.to_string()));
        }
    }

    #[test]
    fn test_daemon_error_complex_chaining() {
        // Test complex error chain with multiple conversions
        let io_error = Error::new(ErrorKind::PermissionDenied, "Access denied");
        let daemon_error = DaemonError::Io(io_error);

        // Convert to Status and back
        let status: Status = daemon_error.into();
        assert_eq!(status.code(), Code::Internal);
        assert_eq!(status.message(), "Internal server error");
    }

    #[test]
    fn test_all_error_variants_status_conversion() {
        let errors = vec![
            DaemonError::DocumentProcessing { message: "test".to_string() },
            DaemonError::Search { message: "test".to_string() },
            DaemonError::Memory { message: "test".to_string() },
            DaemonError::System { message: "test".to_string() },
            DaemonError::ProjectDetection { message: "test".to_string() },
            DaemonError::ConnectionPool { message: "test".to_string() },
            DaemonError::Internal { message: "test".to_string() },
        ];

        for error in errors {
            let status: Status = error.into();
            assert_eq!(status.code(), Code::Internal);
            assert_eq!(status.message(), "Internal server error");
        }
    }
}