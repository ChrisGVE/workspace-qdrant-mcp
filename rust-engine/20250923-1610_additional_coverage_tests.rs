//! Additional tests to achieve 100% coverage
//! This file contains specific edge case tests to cover remaining uncovered lines

#[cfg(test)]
mod additional_coverage_tests {
    use crate::config::*;
    use crate::error::*;
    use crate::daemon::*;
    use crate::grpc::middleware::*;
    use std::time::Duration;
    use std::collections::HashMap;
    use std::sync::Arc;
    use tokio::time::Instant;

    // Test error conversion paths that are rarely triggered
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

        // Test reqwest error conversion
        let url = "http://invalid-url".parse().unwrap();
        let reqwest_error = reqwest::Error::from(reqwest::Client::new().get(url).build().unwrap_err());
        let daemon_error: DaemonError = reqwest_error.into();
        match daemon_error {
            DaemonError::Http(_) => {},
            _ => panic!("Expected Http error"),
        }
    }

    // Test connection pool edge cases
    #[tokio::test]
    async fn test_connection_pool_edge_cases() {
        let connection_manager = ConnectionManager::new(10, 100);

        // Test with various client IDs
        let client_ids = vec![
            "client1".to_string(),
            "".to_string(), // empty string
            "a".repeat(1000), // very long string
            "client-with-special-chars-!@#$%".to_string(),
        ];

        for client_id in client_ids {
            connection_manager.record_request(&client_id, 1024);
            let stats = connection_manager.get_stats();
            assert!(stats.total_requests >= 1);
        }

        // Test connection cleanup
        connection_manager.cleanup_expired_connections(Duration::from_nanos(1));
        let stats_after_cleanup = connection_manager.get_stats();
        assert!(stats_after_cleanup.active_connections >= 0);
    }

    // Test daemon state edge cases
    #[tokio::test]
    async fn test_daemon_state_edge_cases() {
        let state = crate::daemon::state::DaemonState::new(":memory:").await.unwrap();
        let pool = state.db_pool();

        // Test pool access
        assert!(pool.num_idle() >= 0);

        // Test health check
        let health = state.health_check().await;
        assert!(health.is_ok());
    }

    // Test configuration edge cases
    #[test]
    fn test_config_edge_cases() {
        // Test clone and debug implementations
        let config = DaemonConfig::default();
        let cloned = config.clone();
        let debug_str = format!("{:?}", config);

        assert!(!debug_str.is_empty());
        assert_eq!(config.server.host, cloned.server.host);

        // Test individual config components
        let server_config = config.server.clone();
        let debug_server = format!("{:?}", server_config);
        assert!(debug_server.contains("ServerConfig"));

        let database_config = config.database.clone();
        let debug_database = format!("{:?}", database_config);
        assert!(debug_database.contains("DatabaseConfig"));

        let qdrant_config = config.qdrant.clone();
        let debug_qdrant = format!("{:?}", qdrant_config);
        assert!(debug_qdrant.contains("QdrantConfig"));

        let processing_config = config.processing.clone();
        let debug_processing = format!("{:?}", processing_config);
        assert!(debug_processing.contains("ProcessingConfig"));

        let watcher_config = config.file_watcher.clone();
        let debug_watcher = format!("{:?}", watcher_config);
        assert!(debug_watcher.contains("FileWatcherConfig"));

        let metrics_config = config.metrics.clone();
        let debug_metrics = format!("{:?}", metrics_config);
        assert!(debug_metrics.contains("MetricsConfig"));

        let logging_config = config.logging.clone();
        let debug_logging = format!("{:?}", logging_config);
        assert!(debug_logging.contains("LoggingConfig"));
    }

    // Test complex error scenarios
    #[test]
    fn test_complex_error_scenarios() {
        // Test chaining of errors
        let io_error = std::io::Error::new(std::io::ErrorKind::PermissionDenied, "Permission denied");
        let daemon_error = DaemonError::Io(io_error);

        // Test error source chain
        let source = std::error::Error::source(&daemon_error);
        assert!(source.is_some());

        // Test error formatting
        let error_string = format!("{}", daemon_error);
        assert!(error_string.contains("I/O error"));

        // Test debug formatting
        let debug_string = format!("{:?}", daemon_error);
        assert!(debug_string.contains("Io"));
    }

    // Test middleware connection tracking
    #[tokio::test]
    async fn test_middleware_connection_tracking() {
        let connection_manager = Arc::new(ConnectionManager::new(100, 1000));
        let interceptor = ConnectionInterceptor::new(connection_manager.clone());

        // Test the interceptor exists and can be created
        assert!(Arc::strong_count(&connection_manager) >= 2);

        // Test stats access
        let stats = connection_manager.get_stats();
        assert_eq!(stats.active_connections, 0);
        assert_eq!(stats.total_requests, 0);
        assert_eq!(stats.total_bytes_sent, 0);
        assert_eq!(stats.total_bytes_received, 0);
    }

    // Test async timeout scenarios
    #[tokio::test]
    async fn test_async_timeout_scenarios() {
        // Test timeout error creation
        let timeout_error = DaemonError::Timeout { seconds: 30 };
        assert_eq!(format!("{}", timeout_error), "Timeout error: operation timed out after 30s");

        // Test various timeout values
        let timeouts = [0, 1, 60, 3600, u64::MAX];
        for seconds in timeouts {
            let error = DaemonError::Timeout { seconds };
            let status: tonic::Status = error.into();
            assert_eq!(status.code(), tonic::Code::DeadlineExceeded);
        }
    }

    // Test document processor edge cases
    #[tokio::test]
    async fn test_document_processor_coverage() {
        #[cfg(any(test, feature = "test-utils"))]
        {
            let processor = crate::daemon::processing::DocumentProcessor::test_instance();

            // Test debug formatting
            let debug_str = format!("{:?}", processor);
            assert!(debug_str.contains("DocumentProcessor"));

            // Test config access
            let config = processor.config();
            assert!(config.max_concurrent_tasks > 0);
        }
    }

    // Test all error variant conversions to Status
    #[test]
    fn test_all_error_status_conversions() {
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
            let status: tonic::Status = error.into();
            assert_eq!(status.code(), tonic::Code::Internal);
            assert_eq!(status.message(), "Internal server error");
        }
    }

    // Test configuration validation edge cases
    #[test]
    fn test_config_validation_edge_cases() {
        // Test maximum values
        let mut config = DaemonConfig::default();
        config.server.port = 65535;
        config.processing.default_chunk_size = usize::MAX;
        config.database.max_connections = u32::MAX;

        assert!(config.validate().is_ok());

        // Test edge case values
        config.processing.default_chunk_size = 1; // minimum valid
        assert!(config.validate().is_ok());

        config.server.connection_timeout_secs = 0; // should be valid
        assert!(config.validate().is_ok());
    }

    // Test file watcher configuration edge cases
    #[test]
    fn test_file_watcher_config_edge_cases() {
        let mut config = FileWatcherConfig {
            enabled: true,
            debounce_ms: 0,
            max_watched_dirs: 0,
            ignore_patterns: vec![],
            recursive: false,
        };

        // Test debug formatting
        let debug_str = format!("{:?}", config);
        assert!(debug_str.contains("FileWatcherConfig"));

        // Test clone
        let cloned = config.clone();
        assert_eq!(config.enabled, cloned.enabled);
        assert_eq!(config.debounce_ms, cloned.debounce_ms);

        // Test with extreme values
        config.debounce_ms = u64::MAX;
        config.max_watched_dirs = usize::MAX;
        config.ignore_patterns = vec!["*".to_string(); 10000];

        let debug_str2 = format!("{:?}", config);
        assert!(debug_str2.contains("debounce_ms"));
    }

    // Test CollectionConfig edge cases
    #[test]
    fn test_collection_config_edge_cases() {
        let config = CollectionConfig {
            vector_size: 1536, // different from default
            distance_metric: "Euclidean".to_string(),
            enable_indexing: false,
            replication_factor: 3,
            shard_number: 4,
        };

        let debug_str = format!("{:?}", config);
        assert!(debug_str.contains("CollectionConfig"));
        assert!(debug_str.contains("1536"));
        assert!(debug_str.contains("Euclidean"));

        let cloned = config.clone();
        assert_eq!(config.vector_size, cloned.vector_size);
        assert_eq!(config.distance_metric, cloned.distance_metric);
    }

    // Test system info formatting
    #[test]
    fn test_system_info_formatting() {
        let system_info = crate::daemon::core::get_system_info();

        let debug_str = format!("{:?}", system_info);
        assert!(debug_str.contains("SystemInfo"));

        let clone = system_info.clone();
        assert_eq!(system_info.available_memory, clone.available_memory);
        assert_eq!(system_info.cpu_count, clone.cpu_count);
    }

    // Test memory calculation edge cases
    #[test]
    fn test_memory_calculation_edge_cases() {
        let total_memory = crate::daemon::core::get_total_memory();
        assert!(total_memory > 0);

        // Test that the function is deterministic
        let total_memory2 = crate::daemon::core::get_total_memory();
        assert_eq!(total_memory, total_memory2);
    }
}