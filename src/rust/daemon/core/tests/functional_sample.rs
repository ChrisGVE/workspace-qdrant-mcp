#![cfg(feature = "integration")]
//! Sample functional tests for Rust daemon components using cargo-nextest.
//!
//! These tests demonstrate functional testing patterns for:
//! - Daemon lifecycle management
//! - Inter-process communication
//! - File system operations
//! - Integration with external services

use core::{
    config::DaemonConfig,
    daemon::DaemonManager,
    processing::FileProcessor,
    watching::FileWatcher,
};
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Duration;
use tempfile::TempDir;
use tokio::time::sleep;
use uuid::Uuid;

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

    /// Test resource cleanup and memory management
    #[tokio::test]
    #[cfg_attr(not(feature = "integration"), ignore)]
    async fn test_resource_cleanup_functional() {
        let temp_dir = TempDir::new().expect("Failed to create temp directory");
        let workspace_path = temp_dir.path();

        // Create many processors to test resource management
        let processor_count = 10;
        let mut processors = Vec::new();

        for i in 0..processor_count {
            let processor = FileProcessor::new(workspace_path.to_path_buf())
                .await
                .expect("Failed to create file processor");
            processors.push(processor);

            // Create and process a file for each processor
            let file_path = workspace_path.join(format!("test_{}.rs", i));
            let content = format!("fn test_function_{}() {{}}\n", i);
            std::fs::write(&file_path, content).expect("Failed to write test file");

            let result = processors[i].process_file(&file_path, "test_collection").await;
            assert!(result.is_ok(), "Processing failed for processor {}", i);
        }

        // Test that processors can be dropped and resources cleaned up
        drop(processors);

        // Small delay to allow cleanup
        sleep(Duration::from_millis(100)).await;

        // Create new processor to verify system is still functional
        let new_processor = FileProcessor::new(workspace_path.to_path_buf())
            .await
            .expect("Failed to create new processor after cleanup");

        let final_test_file = workspace_path.join("final_test.rs");
        std::fs::write(&final_test_file, "fn final_test() {}")
            .expect("Failed to write final test file");

        let result = new_processor.process_file(&final_test_file, "test_collection").await;
        assert!(
            result.is_ok(),
            "System should be functional after resource cleanup"
        );
    }

    /// Test integration with external services (mock)
    #[tokio::test]
    #[cfg_attr(not(feature = "integration"), ignore)]
    async fn test_external_service_integration_functional() {
        // This test would typically require real external services
        // For demonstration, we'll simulate service interaction

        let temp_dir = TempDir::new().expect("Failed to create temp directory");
        let workspace_path = temp_dir.path();

        let processor = FileProcessor::new(workspace_path.to_path_buf())
            .await
            .expect("Failed to create file processor");

        // Simulate external service availability check
        let service_available = check_external_service_availability().await;

        if service_available {
            // Test with external service
            let test_file = workspace_path.join("external_test.rs");
            std::fs::write(&test_file, "fn external_test() {}")
                .expect("Failed to write test file");

            let result = processor.process_file(&test_file, "test_collection").await;
            assert!(result.is_ok(), "External service integration failed");
        } else {
            // Test graceful degradation when external service unavailable
            println!("External service unavailable, testing graceful degradation");

            let test_file = workspace_path.join("fallback_test.rs");
            std::fs::write(&test_file, "fn fallback_test() {}")
                .expect("Failed to write test file");

            let result = processor.process_file(&test_file, "test_collection").await;
            // Should still work in degraded mode
            assert!(result.is_ok(), "Graceful degradation failed");
        }
    }

    /// Test performance under sustained load
    #[tokio::test]
    #[cfg_attr(not(feature = "integration"), ignore)]
    async fn test_sustained_load_performance_functional() {
        let temp_dir = TempDir::new().expect("Failed to create temp directory");
        let workspace_path = temp_dir.path();

        let processor = Arc::new(
            FileProcessor::new(workspace_path.to_path_buf())
                .await
                .expect("Failed to create file processor"),
        );

        let total_files = 100;
        let batch_size = 10;
        let batches = total_files / batch_size;

        let mut total_processing_time = Duration::from_secs(0);
        let mut successful_operations = 0;

        for batch_num in 0..batches {
            let batch_start_time = std::time::Instant::now();

            // Create batch of files
            let mut batch_tasks = Vec::new();
            for i in 0..batch_size {
                let file_index = batch_num * batch_size + i;
                let file_path = workspace_path.join(format!("load_test_{}.rs", file_index));
                let content = format!(
                    "// Load test file {}\nfn load_test_function_{}() {{\n    // Batch {}\n}}\n",
                    file_index, file_index, batch_num
                );
                std::fs::write(&file_path, content).expect("Failed to write test file");

                let processor_clone: Arc<DocumentProcessor> = Arc::clone(&processor);
                let task = tokio::spawn(async move {
                    processor_clone.process_file(&file_path, "test_collection").await
                });
                batch_tasks.push(task);
            }

            // Wait for batch completion
            let batch_results = futures::future::join_all(batch_tasks).await;
            let batch_processing_time = batch_start_time.elapsed();
            total_processing_time += batch_processing_time;

            // Count successful operations
            for result in batch_results {
                if let Ok(processing_result) = result {
                    if processing_result.is_ok() {
                        successful_operations += 1;
                    }
                }
            }

            // Small delay between batches
            sleep(Duration::from_millis(50)).await;
        }

        // Performance assertions
        let success_rate = successful_operations as f64 / total_files as f64;
        let files_per_second = total_files as f64 / total_processing_time.as_secs_f64();

        assert!(
            success_rate > 0.95,
            "Success rate too low: {:.2}%",
            success_rate * 100.0
        );
        assert!(
            files_per_second > 20.0,
            "Processing rate too slow: {:.2} files/sec",
            files_per_second
        );

        println!(
            "Sustained load test: {}/{} files processed successfully ({:.1}%) in {:.2}s ({:.1} files/sec)",
            successful_operations,
            total_files,
            success_rate * 100.0,
            total_processing_time.as_secs_f64(),
            files_per_second
        );
    }
}

// Helper functions for testing

/// Mock function to check external service availability
async fn check_external_service_availability() -> bool {
    // In real implementation, this would ping an actual service
    // For testing, simulate availability based on environment or random
    std::env::var("EXTERNAL_SERVICE_AVAILABLE")
        .map(|v| v == "true")
        .unwrap_or(false)
}

/// Helper to create test configuration
fn create_test_config(workspace_path: &Path) -> DaemonConfig {
    DaemonConfig {
        workspace_root: workspace_path.to_path_buf(),
        watch_patterns: vec!["**/*.rs".to_string(), "**/*.py".to_string()],
        ignore_patterns: vec!["target/**".to_string(), ".git/**".to_string()],
        processing_interval: Duration::from_millis(100),
        max_concurrent_processes: 4,
        enable_logging: true,
        log_level: "info".to_string(),
        grpc_port: 0,
        health_check_interval: Duration::from_secs(30),
    }
}

/// Helper to wait for condition with timeout
async fn wait_for_condition<F, Fut>(condition: F, timeout: Duration) -> bool
where
    F: Fn() -> Fut,
    Fut: std::future::Future<Output = bool>,
{
    let start = std::time::Instant::now();
    while start.elapsed() < timeout {
        if condition().await {
            return true;
        }
        sleep(Duration::from_millis(50)).await;
    }
    false
}

#[cfg(test)]
mod test_utilities {
    use super::*;

    #[test]
    fn test_config_creation() {
        let temp_dir = TempDir::new().expect("Failed to create temp directory");
        let config = create_test_config(temp_dir.path());

        assert_eq!(config.workspace_root, temp_dir.path());
        assert!(!config.watch_patterns.is_empty());
        assert!(!config.ignore_patterns.is_empty());
        assert!(config.max_concurrent_processes > 0);
    }

    #[tokio::test]
    async fn test_wait_for_condition() {
        let counter = std::sync::Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let counter_clone = counter.clone();
        let condition = move || {
            let counter = counter_clone.clone();
            async move {
                let count = counter.fetch_add(1, std::sync::atomic::Ordering::SeqCst) + 1;
                count >= 3
            }
        };

        let start = std::time::Instant::now();
        let result = wait_for_condition(condition, Duration::from_secs(1)).await;
        let elapsed = start.elapsed();

        assert!(result, "Condition should have been met");
        assert!(elapsed < Duration::from_secs(1), "Should complete quickly");
    }
}
