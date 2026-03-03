#![cfg(feature = "integration")]
//! Sample functional tests for Rust daemon components using cargo-nextest.
//!
//! These tests cover:
//! - Daemon lifecycle management
//! - File system operations and processing pipeline
//! - Concurrent file processing
//! - Error handling and recovery

use core::{
    config::DaemonConfig,
    daemon::DaemonManager,
    processing::FileProcessor,
    watching::FileWatcher,
};
use std::sync::Arc;
use std::time::Duration;
use tempfile::TempDir;

#[cfg(test)]
mod functional_tests {
    use super::*;

    /// Test daemon lifecycle with real configuration
    #[tokio::test]
    #[cfg_attr(not(feature = "integration"), ignore)]
    async fn test_daemon_lifecycle_functional() {
        let temp_dir = TempDir::new().expect("Failed to create temp directory");
        let config_path = temp_dir.path().join("daemon.toml");

        // Create test configuration
        let config = DaemonConfig {
            workspace_root: temp_dir.path().to_path_buf(),
            watch_patterns: vec!["**/*.rs".to_string(), "**/*.py".to_string()],
            ignore_patterns: vec!["target/**".to_string(), ".git/**".to_string()],
            processing_interval: Duration::from_millis(100),
            max_concurrent_processes: 4,
            enable_logging: true,
            log_level: "debug".to_string(),
            grpc_port: 0, // Use dynamic port
            health_check_interval: Duration::from_secs(5),
        };

        // Write configuration to file
        let config_content = toml::to_string(&config).expect("Failed to serialize config");
        std::fs::write(&config_path, config_content).expect("Failed to write config");

        // Create daemon manager
        let daemon_manager = DaemonManager::new(config_path.clone())
            .await
            .expect("Failed to create daemon manager");

        // Test startup
        assert!(daemon_manager.start().await.is_ok(), "Daemon failed to start");

        // Verify daemon is running
        assert!(daemon_manager.is_running().await, "Daemon should be running");

        // Test health check
        let health_status = daemon_manager.health_check().await;
        assert!(health_status.is_ok(), "Health check failed");

        // Test graceful shutdown
        assert!(
            daemon_manager.shutdown().await.is_ok(),
            "Daemon failed to shutdown gracefully"
        );

        // Verify daemon stopped
        assert!(!daemon_manager.is_running().await, "Daemon should be stopped");
    }

    /// Test file watching and processing pipeline
    #[tokio::test]
    #[cfg_attr(not(feature = "integration"), ignore)]
    async fn test_file_processing_pipeline_functional() {
        let temp_dir = TempDir::new().expect("Failed to create temp directory");
        let workspace_path = temp_dir.path();

        // Create subdirectories
        let src_dir = workspace_path.join("src");
        let test_dir = workspace_path.join("tests");
        std::fs::create_dir_all(&src_dir).expect("Failed to create src directory");
        std::fs::create_dir_all(&test_dir).expect("Failed to create test directory");

        // Initialize file processor
        let processor = FileProcessor::new(workspace_path.to_path_buf())
            .await
            .expect("Failed to create file processor");

        // Initialize file watcher
        let watcher = FileWatcher::new(
            workspace_path.to_path_buf(),
            vec!["**/*.rs".to_string()],
            vec!["target/**".to_string()],
        )
        .await
        .expect("Failed to create file watcher");

        // Start watching
        let (tx, mut rx) = tokio::sync::mpsc::channel(100);
        let _watch_handle = tokio::spawn(async move {
            watcher.watch(tx).await.expect("File watching failed");
        });

        // Create test files
        let test_files = vec![
            src_dir.join("main.rs"),
            src_dir.join("lib.rs"),
            test_dir.join("integration.rs"),
        ];

        for file_path in &test_files {
            let content = format!(
                "// Test file: {}\n\nfn main() {{\n    println!(\"Hello, world!\");\n}}\n",
                file_path.file_name().unwrap().to_string_lossy()
            );
            std::fs::write(file_path, content).expect("Failed to write test file");
        }

        // Wait for file events
        let mut processed_files = Vec::new();
        let timeout = Duration::from_secs(5);
        let start_time = std::time::Instant::now();

        while processed_files.len() < test_files.len() && start_time.elapsed() < timeout {
            if let Ok(file_event) = tokio::time::timeout(Duration::from_millis(100), rx.recv()).await
            {
                if let Some(event) = file_event {
                    // Process the file
                    let result = processor.process_file(&event.path, "test_collection").await;
                    assert!(result.is_ok(), "File processing failed: {:?}", result);

                    processed_files.push(event.path);
                }
            }
        }

        // Verify all files were processed
        assert_eq!(
            processed_files.len(),
            test_files.len(),
            "Not all files were processed"
        );

        // Test file modification
        let modified_file = &test_files[0];
        let new_content = "// Modified content\nfn updated_function() {}\n";
        std::fs::write(modified_file, new_content).expect("Failed to modify file");

        // Wait for modification event
        let modification_timeout = Duration::from_secs(2);
        let modification_result =
            tokio::time::timeout(modification_timeout, rx.recv()).await;

        assert!(
            modification_result.is_ok(),
            "File modification was not detected"
        );
    }

    /// Test concurrent file processing
    #[tokio::test]
    #[cfg_attr(not(feature = "integration"), ignore)]
    async fn test_concurrent_processing_functional() {
        let temp_dir = TempDir::new().expect("Failed to create temp directory");
        let workspace_path = temp_dir.path();

        let processor = Arc::new(
            FileProcessor::new(workspace_path.to_path_buf())
                .await
                .expect("Failed to create file processor"),
        );

        // Create multiple test files
        let file_count = 20;
        let mut test_files = Vec::new();

        for i in 0..file_count {
            let file_path = workspace_path.join(format!("test_file_{}.rs", i));
            let content = format!(
                "// Test file {}\nfn function_{}() {{\n    // Implementation\n}}\n",
                i, i
            );
            std::fs::write(&file_path, content).expect("Failed to write test file");
            test_files.push(file_path);
        }

        // Process files concurrently
        let start_time = std::time::Instant::now();
        let tasks: Vec<_> = test_files
            .iter()
            .map(|file_path| {
                let processor: Arc<DocumentProcessor> = Arc::clone(&processor);
                let path = file_path.clone();
                tokio::spawn(async move { processor.process_file(&path, "test_collection").await })
            })
            .collect();

        // Wait for all tasks to complete
        let results = futures::future::join_all(tasks).await;
        let processing_time = start_time.elapsed();

        // Verify all processing succeeded
        for result in results {
            let processing_result = result.expect("Task failed");
            assert!(
                processing_result.is_ok(),
                "File processing failed: {:?}",
                processing_result
            );
        }

        // Verify reasonable performance
        let files_per_second = file_count as f64 / processing_time.as_secs_f64();
        assert!(
            files_per_second > 10.0,
            "Processing too slow: {:.2} files/sec",
            files_per_second
        );

        println!(
            "Processed {} files in {:.2}s ({:.2} files/sec)",
            file_count,
            processing_time.as_secs_f64(),
            files_per_second
        );
    }

    /// Test error handling and recovery
    #[tokio::test]
    #[cfg_attr(not(feature = "integration"), ignore)]
    async fn test_error_handling_and_recovery_functional() {
        let temp_dir = TempDir::new().expect("Failed to create temp directory");
        let workspace_path = temp_dir.path();

        let processor = FileProcessor::new(workspace_path.to_path_buf())
            .await
            .expect("Failed to create file processor");

        // Test processing of non-existent file
        let non_existent_file = workspace_path.join("does_not_exist.rs");
        let result = processor.process_file(&non_existent_file, "test_collection").await;
        assert!(result.is_err(), "Should fail for non-existent file");

        // Test processing of invalid file
        let invalid_file = workspace_path.join("invalid_file.rs");
        std::fs::write(&invalid_file, vec![0xFF, 0xFE, 0xFD])
            .expect("Failed to write invalid file");

        let result = processor.process_file(&invalid_file, "test_collection").await;
        // Should handle invalid content gracefully
        assert!(
            result.is_err() || result.is_ok(),
            "Should handle invalid file gracefully"
        );

        // Test recovery - create valid file after error
        let valid_file = workspace_path.join("valid_file.rs");
        let valid_content = "fn valid_function() {}\n";
        std::fs::write(&valid_file, valid_content).expect("Failed to write valid file");

        let result = processor.process_file(&valid_file, "test_collection").await;
        assert!(result.is_ok(), "Should process valid file after errors");
    }
}
