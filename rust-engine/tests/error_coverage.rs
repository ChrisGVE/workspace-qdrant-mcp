//! Error handling coverage tests

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
    assert!(io_msg.contains("IO error"));
    assert!(io_msg.contains("File not found"));

    // Test Database error
    let db_error = DaemonError::Database(sqlx::Error::Configuration("DB config error".into()));
    let db_msg = format!("{}", db_error);
    assert!(db_msg.contains("Database error"));

    // Test gRPC error
    let grpc_error = DaemonError::Grpc(tonic::Status::internal("Internal server error"));
    let grpc_msg = format!("{}", grpc_error);
    assert!(grpc_msg.contains("gRPC error"));
    assert!(grpc_msg.contains("Internal server error"));

    // Test Processing error
    let proc_error = DaemonError::Processing("Failed to process document".to_string());
    let proc_msg = format!("{}", proc_error);
    assert!(proc_msg.contains("Processing error"));
    assert!(proc_msg.contains("Failed to process document"));

    // Test Watcher error
    let watch_error = DaemonError::Watcher("File watcher failed".to_string());
    let watch_msg = format!("{}", watch_error);
    assert!(watch_msg.contains("File watcher error"));
    assert!(watch_msg.contains("File watcher failed"));

    // Test Qdrant error
    let qdrant_error = DaemonError::Qdrant("Connection to Qdrant failed".to_string());
    let qdrant_msg = format!("{}", qdrant_error);
    assert!(qdrant_msg.contains("Qdrant error"));
    assert!(qdrant_msg.contains("Connection to Qdrant failed"));
}

#[test]
fn test_daemon_error_debug() {
    let error = DaemonError::Processing("Test error".to_string());
    let debug_str = format!("{:?}", error);
    assert!(debug_str.contains("Processing"));
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

    // Test From<tonic::Status>
    let grpc_status = tonic::Status::unavailable("Service unavailable");
    let daemon_error: DaemonError = grpc_status.into();
    match daemon_error {
        DaemonError::Grpc(status) => {
            assert_eq!(status.code(), tonic::Code::Unavailable);
        },
        _ => panic!("Expected gRPC error conversion"),
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
    let error = DaemonError::Processing("Test processing error".to_string());

    // Test that it implements std::error::Error
    let std_error: &dyn std::error::Error = &error;
    assert!(std_error.to_string().contains("Processing error"));

    // Test source (should be None for our custom errors)
    assert!(std_error.source().is_none());
}

#[test]
fn test_daemon_error_chain() {
    // Create a chain of errors to test error handling
    let io_error = IoError::new(ErrorKind::NotFound, "Config file not found");
    let daemon_error = DaemonError::Io(io_error);

    let error_string = format!("{}", daemon_error);
    assert!(error_string.contains("IO error"));
    assert!(error_string.contains("Config file not found"));
}

#[test]
fn test_daemon_result_type() {
    // Test successful result
    let success: DaemonResult<i32> = Ok(42);
    assert_eq!(success.unwrap(), 42);

    // Test error result
    let error: DaemonResult<i32> = Err(DaemonError::Processing("Test error".to_string()));
    assert!(error.is_err());

    match error {
        Err(DaemonError::Processing(msg)) => {
            assert_eq!(msg, "Test error");
        },
        _ => panic!("Expected processing error"),
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
fn test_config_error_types() {
    use config::ConfigError;

    // Test Message error
    let msg_error = ConfigError::Message("Invalid value".to_string());
    let daemon_error = DaemonError::Config(msg_error);
    let error_str = format!("{}", daemon_error);
    assert!(error_str.contains("Invalid value"));

    // Test that ConfigError can be formatted
    let config_error = ConfigError::Message("Test config message".to_string());
    let config_str = format!("{:?}", config_error);
    assert!(config_str.contains("Message"));
    assert!(config_str.contains("Test config message"));
}

#[test]
fn test_error_equality_and_comparison() {
    // Test that we can compare error messages
    let error1 = DaemonError::Processing("Same message".to_string());
    let error2 = DaemonError::Processing("Same message".to_string());
    let error3 = DaemonError::Processing("Different message".to_string());

    // Test string representation equality
    assert_eq!(format!("{}", error1), format!("{}", error2));
    assert_ne!(format!("{}", error1), format!("{}", error3));

    // Test that different error types produce different strings
    let watcher_error = DaemonError::Watcher("Same message".to_string());
    assert_ne!(format!("{}", error1), format!("{}", watcher_error));
}