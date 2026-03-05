//! Tests for logging_config and logging_structured

#[cfg(test)]
mod tests {
    use crate::error::WorkspaceError;
    use crate::monitoring::logging_config::*;
    use crate::monitoring::logging_perf::*;
    use crate::monitoring::logging_structured::*;
    use serde_json::Value;
    use serial_test::serial;
    use std::env;
    use std::path::PathBuf;
    use tempfile::NamedTempFile;

    #[serial]
    #[test]
    fn test_logging_config_from_environment() {
        let keys = [
            "RUST_LOG",
            "WQM_LOG_LEVEL",
            "WQM_LOG_JSON",
            "WQM_LOG_CONSOLE",
            "WQM_LOG_FILE",
            "WQM_LOG_FILE_PATH",
            "WQM_LOG_METRICS",
            "WQM_LOG_ERROR_TRACKING",
            "WQM_LOG_FIELD_TEAM",
        ];

        let previous: Vec<Option<String>> = keys.iter().map(|key| env::var(key).ok()).collect();

        env::remove_var("WQM_LOG_LEVEL");
        env::set_var("RUST_LOG", "debug");
        env::set_var("WQM_LOG_JSON", "true");
        env::set_var("WQM_LOG_CONSOLE", "0");
        env::set_var("WQM_LOG_FILE", "1");
        env::set_var("WQM_LOG_FILE_PATH", "/tmp/wqm.log");
        env::set_var("WQM_LOG_METRICS", "false");
        env::set_var("WQM_LOG_ERROR_TRACKING", "false");
        env::set_var("WQM_LOG_FIELD_TEAM", "core");

        let config = LoggingConfig::from_environment();

        assert_eq!(config.level, tracing::Level::DEBUG);
        assert!(config.json_format);
        assert!(!config.console_output);
        assert!(config.file_logging);
        assert_eq!(
            config.log_file_path.as_ref().map(PathBuf::from),
            Some(PathBuf::from("/tmp/wqm.log"))
        );
        assert!(!config.performance_metrics);
        assert!(!config.error_tracking);
        assert_eq!(config.global_fields.get("team"), Some(&"core".to_string()));

        for (key, value) in keys.iter().zip(previous) {
            match value {
                Some(v) => env::set_var(key, v),
                None => env::remove_var(key),
            }
        }
    }

    #[serial]
    #[test]
    fn test_wqm_log_level_takes_precedence_over_rust_log() {
        let prev_wqm = env::var("WQM_LOG_LEVEL").ok();
        let prev_rust = env::var("RUST_LOG").ok();

        env::set_var("WQM_LOG_LEVEL", "WARN");
        env::set_var("RUST_LOG", "TRACE");

        let config = LoggingConfig::from_environment();
        assert_eq!(config.level, tracing::Level::WARN);

        match prev_wqm {
            Some(v) => env::set_var("WQM_LOG_LEVEL", v),
            None => env::remove_var("WQM_LOG_LEVEL"),
        }
        match prev_rust {
            Some(v) => env::set_var("RUST_LOG", v),
            None => env::remove_var("RUST_LOG"),
        }
    }

    #[serial]
    #[test]
    fn test_wqm_log_dir_overrides_default() {
        let prev = env::var("WQM_LOG_DIR").ok();

        env::set_var("WQM_LOG_DIR", "/custom/log/path");
        let dir = get_canonical_log_dir();
        assert_eq!(dir, PathBuf::from("/custom/log/path"));

        env::remove_var("WQM_LOG_DIR");
        let dir = get_canonical_log_dir();
        assert_ne!(dir, PathBuf::from("/custom/log/path"));

        match prev {
            Some(v) => env::set_var("WQM_LOG_DIR", v),
            None => env::remove_var("WQM_LOG_DIR"),
        }
    }

    #[serial]
    #[test]
    fn test_rotation_config_from_environment() {
        let keys = [
            "WQM_LOG_ROTATION_SIZE_MB",
            "WQM_LOG_ROTATION_COUNT",
            "WQM_LOG_ROTATION_COMPRESS",
        ];
        let previous: Vec<Option<String>> = keys.iter().map(|k| env::var(k).ok()).collect();

        env::set_var("WQM_LOG_ROTATION_SIZE_MB", "100");
        env::set_var("WQM_LOG_ROTATION_COUNT", "10");
        env::set_var("WQM_LOG_ROTATION_COMPRESS", "false");

        let config = LoggingConfig::from_environment();
        assert_eq!(config.rotation_size_mb, 100);
        assert_eq!(config.rotation_count, 10);
        assert!(!config.compress_rotated);

        for (key, value) in keys.iter().zip(previous) {
            match value {
                Some(v) => env::set_var(key, v),
                None => env::remove_var(key),
            }
        }
    }

    #[test]
    fn test_rotation_defaults() {
        let config = LoggingConfig::default();
        assert_eq!(config.rotation_size_mb, 50);
        assert_eq!(config.rotation_count, 5);
        assert!(config.compress_rotated);
    }

    #[test]
    fn test_performance_metrics() {
        let mut metrics = PerformanceMetrics::default();

        metrics.record_operation("test_op", 100.0);
        metrics.record_operation("test_op", 200.0);
        metrics.record_operation("index", 50.0);
        metrics.record_error("test_error");

        let summary = metrics.get_summary();

        let counts = summary
            .get("operation_counts")
            .and_then(Value::as_object)
            .expect("operation counts available");
        assert_eq!(counts.get("test_op").and_then(Value::as_u64), Some(2));
        assert_eq!(counts.get("index").and_then(Value::as_u64), Some(1));

        let stats = summary
            .get("operation_stats")
            .and_then(Value::as_object)
            .expect("operation stats available");
        let ingest_stats = stats
            .get("test_op")
            .and_then(Value::as_object)
            .expect("test_op stats available");

        let avg = ingest_stats.get("avg_ms").and_then(Value::as_f64).unwrap();
        assert!((avg - 150.0).abs() < 1e-6);
        assert_eq!(ingest_stats.get("count").and_then(Value::as_f64), Some(2.0));

        let error_counts = summary
            .get("error_counts")
            .and_then(Value::as_object)
            .expect("error counts available");
        assert_eq!(
            error_counts.get("test_error").and_then(Value::as_u64),
            Some(1)
        );
    }

    #[tokio::test]
    async fn test_track_async_operation() {
        let result = track_async_operation("test_op", async { Ok::<_, &str>("success") }).await;
        assert!(result.is_ok());

        let result = track_async_operation("test_op", async { Err::<&str, _>("error") }).await;
        assert!(result.is_err());
    }

    #[test]
    fn test_initialize_logging_console_only() {
        let _config = LoggingConfig {
            console_output: true,
            file_logging: false,
            json_format: false,
            ..Default::default()
        };
    }

    #[test]
    fn test_initialize_logging_with_file() {
        let temp_file = NamedTempFile::new().unwrap();
        let _config = LoggingConfig {
            console_output: false,
            file_logging: true,
            log_file_path: Some(temp_file.path().to_path_buf()),
            json_format: true,
            ..Default::default()
        };
    }

    #[test]
    fn test_error_severity_logging() {
        let error = WorkspaceError::network("Test error", 1, 3);
        log_error_with_context(&error, "test_context");
    }
}
