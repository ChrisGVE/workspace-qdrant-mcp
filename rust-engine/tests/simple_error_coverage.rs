//! Simple error handling coverage tests

use workspace_qdrant_daemon::error::*;
use std::io::{Error as IoError, ErrorKind};

#[test]
fn test_daemon_error_creation_and_display() {
    // Test Config error
    let config_error = DaemonError::Config(
        config::ConfigError::Message("Invalid configuration".to_string())
    );
    let config_msg = format!("{}", config_error);
    assert!(config_msg.contains("Configuration error"));
    assert!(config_msg.contains("Invalid configuration"));

    // Test IO error
    let io_error = DaemonError::Io(IoError::new(ErrorKind::NotFound, "File not found"));
    let io_msg = format!("{}", io_error);
    assert!(io_msg.contains("I/O error"));
    assert!(io_msg.contains("File not found"));

    // Test Database error
    let db_error = DaemonError::Database(sqlx::Error::PoolClosed);
    let db_msg = format!("{}", db_error);
    assert!(db_msg.contains("Database error"));

    // Test Document Processing error
    let proc_error = DaemonError::DocumentProcessing {
        message: "Failed to process document".to_string()
    };
    let proc_msg = format!("{}", proc_error);
    assert!(proc_msg.contains("Document processing error"));
    assert!(proc_msg.contains("Failed to process document"));

    // Test Search error
    let search_error = DaemonError::Search {
        message: "Search failed".to_string()
    };
    let search_msg = format!("{}", search_error);
    assert!(search_msg.contains("Search error"));
    assert!(search_msg.contains("Search failed"));

    // Test Memory error
    let memory_error = DaemonError::Memory {
        message: "Memory operation failed".to_string()
    };
    let memory_msg = format!("{}", memory_error);
    assert!(memory_msg.contains("Memory management error"));
    assert!(memory_msg.contains("Memory operation failed"));

    // Test System error
    let system_error = DaemonError::System {
        message: "System call failed".to_string()
    };
    let system_msg = format!("{}", system_error);
    assert!(system_msg.contains("System error"));
    assert!(system_msg.contains("System call failed"));
}

#[test]
fn test_daemon_error_debug() {
    let error = DaemonError::DocumentProcessing {
        message: "Test error".to_string()
    };
    let debug_str = format!("{:?}", error);
    assert!(debug_str.contains("DocumentProcessing"));
    assert!(debug_str.contains("Test error"));
}

#[test]
fn test_daemon_error_from_conversions() {
    // Test From<std::io::Error>
    let io_error = IoError::new(ErrorKind::PermissionDenied, "Permission denied");
    let daemon_error: DaemonError = io_error.into();
    match daemon_error {
        DaemonError::Io(e) => {
            assert_eq!(e.kind(), ErrorKind::PermissionDenied);
        },
        _ => panic!("Expected IO error conversion"),
    }

    // Test From<config::ConfigError>
    let config_error = config::ConfigError::Message("Invalid config".to_string());
    let daemon_error: DaemonError = config_error.into();
    match daemon_error {
        DaemonError::Config(_) => {
            // Expected conversion
        },
        _ => panic!("Expected Config error conversion"),
    }

    // Test From<sqlx::Error>
    let sql_error = sqlx::Error::RowNotFound;
    let daemon_error: DaemonError = sql_error.into();
    match daemon_error {
        DaemonError::Database(_) => {
            // Expected conversion
        },
        _ => panic!("Expected Database error conversion"),
    }
}

#[test]
fn test_daemon_error_std_error_trait() {
    let error = DaemonError::DocumentProcessing {
        message: "Test processing error".to_string()
    };

    // Test that it implements std::error::Error
    let std_error: &dyn std::error::Error = &error;
    assert!(std_error.to_string().contains("Document processing error"));
}

#[test]
fn test_daemon_error_chain() {
    // Create a chain of errors to test error handling
    let io_error = IoError::new(ErrorKind::NotFound, "Config file not found");
    let daemon_error = DaemonError::Io(io_error);

    let error_string = format!("{}", daemon_error);
    assert!(error_string.contains("I/O error"));
    assert!(error_string.contains("Config file not found"));
}

#[test]
fn test_daemon_result_type() {
    // Test successful result
    let success: DaemonResult<i32> = Ok(42);
    assert_eq!(success.unwrap(), 42);

    // Test error result
    let error: DaemonResult<i32> = Err(DaemonError::DocumentProcessing {
        message: "Test error".to_string()
    });
    assert!(error.is_err());

    match error {
        Err(DaemonError::DocumentProcessing { message }) => {
            assert_eq!(message, "Test error");
        },
        _ => panic!("Expected document processing error"),
    }
}

#[test]
fn test_daemon_error_send_sync() {
    fn assert_send<T: Send>() {}
    fn assert_sync<T: Sync>() {}

    assert_send::<DaemonError>();
    assert_sync::<DaemonError>();
}

#[test]
fn test_all_structured_error_variants() {
    // Test all the structured error variants
    let errors = vec![
        DaemonError::DocumentProcessing { message: "test".to_string() },
        DaemonError::Search { message: "test".to_string() },
        DaemonError::Memory { message: "test".to_string() },
        DaemonError::System { message: "test".to_string() },
        DaemonError::ProjectDetection { message: "test".to_string() },
        DaemonError::ConnectionPool { message: "test".to_string() },
        DaemonError::Timeout { seconds: 30 },
        DaemonError::NotFound { resource: "document".to_string() },
        DaemonError::InvalidInput { message: "invalid data".to_string() },
        DaemonError::Internal { message: "internal error".to_string() },
    ];

    for error in errors {
        // Each error should be debuggable
        let debug_str = format!("{:?}", error);
        assert!(!debug_str.is_empty());

        // Each error should be displayable
        let display_str = format!("{}", error);
        assert!(!display_str.is_empty());

        // Each error should convert to Status
        let _status: tonic::Status = error.into();
    }
}

#[test]
fn test_timeout_error() {
    let timeout_error = DaemonError::Timeout { seconds: 60 };
    let msg = format!("{}", timeout_error);
    assert!(msg.contains("Timeout error"));
    assert!(msg.contains("60s"));
}

#[test]
fn test_not_found_error() {
    let not_found_error = DaemonError::NotFound {
        resource: "collection".to_string()
    };
    let msg = format!("{}", not_found_error);
    assert!(msg.contains("Resource not found"));
    assert!(msg.contains("collection"));
}

#[test]
fn test_error_to_status_conversion() {
    // Test InvalidInput -> InvalidArgument
    let invalid_input = DaemonError::InvalidInput {
        message: "Bad input".to_string()
    };
    let status: tonic::Status = invalid_input.into();
    assert_eq!(status.code(), tonic::Code::InvalidArgument);

    // Test NotFound -> NotFound
    let not_found = DaemonError::NotFound {
        resource: "user".to_string()
    };
    let status: tonic::Status = not_found.into();
    assert_eq!(status.code(), tonic::Code::NotFound);

    // Test Timeout -> DeadlineExceeded
    let timeout = DaemonError::Timeout { seconds: 30 };
    let status: tonic::Status = timeout.into();
    assert_eq!(status.code(), tonic::Code::DeadlineExceeded);

    // Test IO -> Internal
    let io_error = DaemonError::Io(IoError::new(ErrorKind::NotFound, "File not found"));
    let status: tonic::Status = io_error.into();
    assert_eq!(status.code(), tonic::Code::Internal);
}

#[test]
fn test_error_equality_and_comparison() {
    // Test that we can compare error messages
    let error1 = DaemonError::DocumentProcessing {
        message: "Same message".to_string()
    };
    let error2 = DaemonError::DocumentProcessing {
        message: "Same message".to_string()
    };
    let error3 = DaemonError::DocumentProcessing {
        message: "Different message".to_string()
    };

    // Test string representation equality
    assert_eq!(format!("{}", error1), format!("{}", error2));
    assert_ne!(format!("{}", error1), format!("{}", error3));

    // Test that different error types produce different strings
    let search_error = DaemonError::Search {
        message: "Same message".to_string()
    };
    assert_ne!(format!("{}", error1), format!("{}", search_error));
}