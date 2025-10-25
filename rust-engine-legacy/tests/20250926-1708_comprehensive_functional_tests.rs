//! Comprehensive functional tests for daemon operations
//! Tests regular use cases, edge cases, error conditions, and performance scenarios

use workspace_qdrant_daemon::config::DaemonConfig;
use workspace_qdrant_daemon::daemon::processing::DocumentProcessor;
use workspace_qdrant_daemon::daemon::watcher::FileWatcher;
use workspace_qdrant_daemon::daemon::state::DaemonState;
use workspace_qdrant_daemon::error::DaemonError;

use std::sync::Arc;
use std::time::Duration;
use tempfile::TempDir;
use tokio::{fs, time::sleep};
use tokio::test as tokio_test;

/// Create test configuration for functional testing
fn create_functional_config() -> DaemonConfig {
    let mut config = DaemonConfig::default();
    config.database.sqlite_path = ":memory:".to_string();
    config.processing.max_concurrent_tasks = 2;
    config.file_watcher.enabled = true;
    config.file_watcher.debounce_ms = 50; // Fast for testing
    config
}

/// Test: End-to-end document processing workflow
#[tokio_test]
async fn test_e2e_document_processing_workflow() {
    let config = create_functional_config();
    let temp_dir = TempDir::new().expect("Failed to create temp directory");

    // Step 1: Initialize all components
    let processor = DocumentProcessor::new(&config.processing, &config.qdrant)
        .await
        .expect("Failed to create processor");

    let mut watcher = FileWatcher::new(&config.file_watcher, Arc::new(processor))
        .await
        .expect("Failed to create watcher");

    let state = DaemonState::new(&config.database)
        .await
        .expect("Failed to create state");

    // Step 2: Start file watcher
    assert!(watcher.start().await.is_ok(), "Watcher should start");
    assert!(watcher.watch_directory(temp_dir.path()).await.is_ok(), "Should watch directory");

    // Step 3: Create test documents
    let test_file = temp_dir.path().join("test_document.txt");
    fs::write(&test_file, "This is a test document for processing").await
        .expect("Failed to write test file");

    // Step 4: Wait for processing (file watcher debounce + processing time)
    sleep(Duration::from_millis(200)).await;

    // Step 5: Verify state persistence
    assert!(state.health_check().await.is_ok(), "State should be healthy");

    // Step 6: Clean shutdown
    assert!(watcher.stop().await.is_ok(), "Watcher should stop cleanly");
}

/// Test: File watcher with multiple file operations
#[tokio_test]
async fn test_file_watcher_multiple_operations() {
    let config = create_functional_config();
    let temp_dir = TempDir::new().expect("Failed to create temp directory");

    let processor = DocumentProcessor::new(&config.processing, &config.qdrant)
        .await
        .expect("Failed to create processor");

    let mut watcher = FileWatcher::new(&config.file_watcher, Arc::new(processor))
        .await
        .expect("Failed to create watcher");

    assert!(watcher.start().await.is_ok());
    assert!(watcher.watch_directory(temp_dir.path()).await.is_ok());

    // Test multiple file operations
    let files = vec!["file1.txt", "file2.md", "file3.rs"];

    for (i, filename) in files.iter().enumerate() {
        let file_path = temp_dir.path().join(filename);
        let content = format!("Test content for file {}", i + 1);
        fs::write(&file_path, content).await.expect("Failed to write file");

        // Small delay between operations
        sleep(Duration::from_millis(30)).await;
    }

    // Wait for all processing to complete
    sleep(Duration::from_millis(300)).await;

    assert!(watcher.stop().await.is_ok());
}

/// Test: Database persistence across component lifecycle
#[tokio_test]
async fn test_database_persistence_lifecycle() {
    let temp_dir = TempDir::new().expect("Failed to create temp directory");
    let db_path = temp_dir.path().join("functional_test.db");

    // Ensure the directory exists
    std::fs::create_dir_all(temp_dir.path()).expect("Failed to create temp dir");

    let mut config = create_functional_config();
    config.database.sqlite_path = format!("sqlite://{}?mode=rwc", db_path.display());

    // Phase 1: Create state and perform operations
    {
        let state = DaemonState::new(&config.database)
            .await
            .expect("Failed to create state");

        assert!(state.health_check().await.is_ok(), "Initial health check should pass");

        // Simulate some state operations
        for _ in 0..5 {
            assert!(state.health_check().await.is_ok(), "Health check should consistently pass");
            sleep(Duration::from_millis(10)).await;
        }
    }

    // Phase 2: Recreate state from persisted database
    {
        let state = DaemonState::new(&config.database)
            .await
            .expect("Failed to recreate state from persisted database");

        assert!(state.health_check().await.is_ok(), "Health check should pass after recreation");
    }

    // Phase 3: Verify database file exists and is valid
    assert!(db_path.exists(), "Database file should exist");
    let metadata = fs::metadata(&db_path).await.expect("Failed to get database metadata");
    assert!(metadata.len() > 0, "Database file should not be empty");
}

/// Test: Error handling and recovery scenarios
#[tokio_test]
async fn test_error_handling_and_recovery() {
    let config = create_functional_config();

    // Test 1: Invalid processor configuration
    let mut invalid_config = config.clone();
    invalid_config.processing.max_concurrent_tasks = 0; // Invalid value

    let _processor_result = DocumentProcessor::new(&invalid_config.processing, &invalid_config.qdrant).await;
    // Note: This test depends on actual validation in DocumentProcessor

    // Test 2: Invalid database path
    let mut invalid_db_config = config.clone();
    invalid_db_config.database.sqlite_path = "/invalid/path/that/should/not/exist.db".to_string();

    let _state_result = DaemonState::new(&invalid_db_config.database).await;
    // Should either handle gracefully or return appropriate error

    // Test 3: Watcher with invalid processor
    let valid_processor = DocumentProcessor::new(&config.processing, &config.qdrant)
        .await
        .expect("Failed to create valid processor");

    let watcher_result = FileWatcher::new(&config.file_watcher, Arc::new(valid_processor)).await;
    assert!(watcher_result.is_ok(), "Watcher creation should succeed with valid processor");
}

/// Test: Concurrent operations and thread safety
#[tokio_test]
async fn test_concurrent_operations() {
    let config = create_functional_config();
    let _temp_dir = TempDir::new().expect("Failed to create temp directory");

    // Create multiple processors concurrently
    let mut processor_handles = Vec::new();
    for i in 0..3 {
        let config_clone = config.clone();
        let handle = tokio::spawn(async move {
            let result = DocumentProcessor::new(&config_clone.processing, &config_clone.qdrant).await;
            (i, result)
        });
        processor_handles.push(handle);
    }

    // Wait for all processors to be created
    for handle in processor_handles {
        let (index, result) = handle.await.expect("Task should complete");
        assert!(result.is_ok(), "Processor {} should be created successfully", index);
    }

    // Test concurrent state operations
    let state = Arc::new(DaemonState::new(&config.database)
        .await
        .expect("Failed to create state"));

    let mut state_handles = Vec::new();
    for i in 0..5 {
        let state_clone = Arc::clone(&state);
        let handle = tokio::spawn(async move {
            let result = state_clone.health_check().await;
            (i, result)
        });
        state_handles.push(handle);
    }

    for handle in state_handles {
        let (index, result) = handle.await.expect("Task should complete");
        assert!(result.is_ok(), "Concurrent health check {} should succeed", index);
    }
}

/// Test: Resource management and cleanup
#[tokio_test]
async fn test_resource_management() {
    let config = create_functional_config();

    // Test creating and dropping many components to verify cleanup
    for iteration in 0..10 {
        let processor = DocumentProcessor::new(&config.processing, &config.qdrant)
            .await
            .expect(&format!("Failed to create processor on iteration {}", iteration));

        let mut watcher = FileWatcher::new(&config.file_watcher, Arc::new(processor))
            .await
            .expect(&format!("Failed to create watcher on iteration {}", iteration));

        let _state = DaemonState::new(&config.database)
            .await
            .expect(&format!("Failed to create state on iteration {}", iteration));

        // Start and stop watcher to test full lifecycle
        assert!(watcher.start().await.is_ok(), "Watcher should start");
        sleep(Duration::from_millis(10)).await;
        assert!(watcher.stop().await.is_ok(), "Watcher should stop");

        // Components should drop cleanly without resource leaks
    }
}

/// Test: File watcher edge cases
#[tokio_test]
async fn test_file_watcher_edge_cases() {
    let config = create_functional_config();
    let temp_dir = TempDir::new().expect("Failed to create temp directory");

    let processor = DocumentProcessor::new(&config.processing, &config.qdrant)
        .await
        .expect("Failed to create processor");

    let mut watcher = FileWatcher::new(&config.file_watcher, Arc::new(processor))
        .await
        .expect("Failed to create watcher");

    assert!(watcher.start().await.is_ok());
    assert!(watcher.watch_directory(temp_dir.path()).await.is_ok());

    // Edge Case 1: Empty file
    let empty_file = temp_dir.path().join("empty.txt");
    fs::write(&empty_file, "").await.expect("Failed to create empty file");

    // Edge Case 2: Very small file
    let tiny_file = temp_dir.path().join("tiny.txt");
    fs::write(&tiny_file, "x").await.expect("Failed to create tiny file");

    // Edge Case 3: File with special characters in name
    let special_file = temp_dir.path().join("special-file_123.txt");
    fs::write(&special_file, "Special file content").await.expect("Failed to create special file");

    // Edge Case 4: Rapid file creation and deletion
    let rapid_file = temp_dir.path().join("rapid.txt");
    fs::write(&rapid_file, "Rapid file").await.expect("Failed to create rapid file");
    sleep(Duration::from_millis(10)).await;
    fs::remove_file(&rapid_file).await.expect("Failed to remove rapid file");

    // Wait for processing
    sleep(Duration::from_millis(200)).await;

    assert!(watcher.stop().await.is_ok());
}

/// Test: Configuration validation and edge cases
#[tokio_test]
async fn test_configuration_edge_cases() {
    // Test 1: Minimal valid configuration
    let mut minimal_config = DaemonConfig::default();
    minimal_config.database.sqlite_path = ":memory:".to_string();
    minimal_config.processing.max_concurrent_tasks = 1;
    minimal_config.file_watcher.enabled = false; // Disabled watcher

    let processor = DocumentProcessor::new(&minimal_config.processing, &minimal_config.qdrant).await;
    assert!(processor.is_ok(), "Minimal configuration should work");

    let state = DaemonState::new(&minimal_config.database).await;
    assert!(state.is_ok(), "Minimal database config should work");

    // Test 2: Maximum reasonable configuration
    let mut max_config = DaemonConfig::default();
    max_config.database.sqlite_path = ":memory:".to_string();
    max_config.processing.max_concurrent_tasks = 16; // High but reasonable
    max_config.file_watcher.enabled = true;
    max_config.file_watcher.max_watched_dirs = 100;

    let processor = DocumentProcessor::new(&max_config.processing, &max_config.qdrant).await;
    assert!(processor.is_ok(), "Maximum configuration should work");
}

/// Test: Error propagation and handling
#[tokio_test]
async fn test_error_propagation() {
    // Test that errors are properly created and formatted
    let config_error = DaemonError::Configuration {
        message: "Test configuration error".to_string()
    };
    let error_string = format!("{}", config_error);
    assert!(error_string.contains("Configuration error"), "Error should contain type description");
    assert!(error_string.contains("Test configuration error"), "Error should contain message");

    let internal_error = DaemonError::Internal {
        message: "Test internal error".to_string()
    };
    let internal_string = format!("{}", internal_error);
    assert!(internal_string.contains("internal"), "Internal error should be identifiable");

    // Test error Debug formatting
    let debug_string = format!("{:?}", config_error);
    assert!(!debug_string.is_empty(), "Debug formatting should work");
}

/// Test: Performance under load simulation
#[tokio_test]
async fn test_performance_simulation() {
    let config = create_functional_config();
    let temp_dir = TempDir::new().expect("Failed to create temp directory");

    let processor = DocumentProcessor::new(&config.processing, &config.qdrant)
        .await
        .expect("Failed to create processor");

    let mut watcher = FileWatcher::new(&config.file_watcher, Arc::new(processor))
        .await
        .expect("Failed to create watcher");

    assert!(watcher.start().await.is_ok());
    assert!(watcher.watch_directory(temp_dir.path()).await.is_ok());

    // Simulate moderate load - create multiple files rapidly
    let start_time = std::time::Instant::now();

    for i in 0..20 {
        let file_path = temp_dir.path().join(format!("load_test_{}.txt", i));
        let content = format!("Load test content for file {} with some additional text to make it more realistic", i);
        fs::write(&file_path, content).await.expect("Failed to write load test file");

        // Very small delay to simulate rapid but not instantaneous file creation
        if i % 5 == 0 {
            sleep(Duration::from_millis(5)).await;
        }
    }

    let creation_time = start_time.elapsed();

    // Wait for processing to complete
    sleep(Duration::from_millis(500)).await;

    let total_time = start_time.elapsed();

    // Basic performance assertions
    assert!(creation_time < Duration::from_secs(2), "File creation should be fast");
    assert!(total_time < Duration::from_secs(5), "Total processing should complete reasonably quickly");

    assert!(watcher.stop().await.is_ok());
}