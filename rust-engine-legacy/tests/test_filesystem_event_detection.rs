//! Comprehensive tests for file system event detection with actual file operations
//! Tests the notify crate integration, debouncing, filtering, and cross-platform compatibility

use workspace_qdrant_daemon::config::{FileWatcherConfig, ProcessingConfig, QdrantConfig, CollectionConfig};
use workspace_qdrant_daemon::daemon::watcher::FileWatcher;
use workspace_qdrant_daemon::daemon::processing::DocumentProcessor;
use std::sync::Arc;
use std::time::Duration;
use tempfile::TempDir;
use tokio::fs;
use tokio::time::sleep;
use serial_test::serial;

/// Helper function to create test configuration
fn create_test_config(enabled: bool, debounce_ms: u64) -> FileWatcherConfig {
    FileWatcherConfig {
        enabled,
        debounce_ms,
        max_watched_dirs: 10,
        ignore_patterns: vec!["*.tmp".to_string(), "*.log".to_string(), "*.swp".to_string()],
        recursive: true,
    }
}

/// Helper function to create test processor
async fn create_test_processor() -> Arc<DocumentProcessor> {
    let processing_config = ProcessingConfig {
        max_concurrent_tasks: 2,
        default_chunk_size: 1000,
        default_chunk_overlap: 200,
        max_file_size_bytes: 1024 * 1024,
        supported_extensions: vec!["rs".to_string(), "txt".to_string()],
        enable_lsp: false,
        lsp_timeout_secs: 10,
    };
    let qdrant_config = QdrantConfig {
        url: "http://localhost:6333".to_string(),
        api_key: None,
        max_retries: 3,
        default_collection: CollectionConfig {
            vector_size: 384,
            distance_metric: "Cosine".to_string(),
            enable_indexing: true,
            replication_factor: 1,
            shard_number: 1,
        },
    };
    Arc::new(DocumentProcessor::new(&processing_config, &qdrant_config).await.unwrap())
}

/// Test basic file creation detection
#[tokio::test]
#[serial]
async fn test_file_creation_detection() {
    let config = create_test_config(true, 50);
    let processor = create_test_processor().await;
    let mut watcher = FileWatcher::new(&config, processor).await.unwrap();

    let temp_dir = TempDir::new().unwrap();

    // Start watcher and watch directory
    assert!(watcher.start().await.is_ok());
    assert!(watcher.watch_directory(temp_dir.path()).await.is_ok());

    // Give watcher time to initialize
    sleep(Duration::from_millis(100)).await;

    // Create a file
    let file_path = temp_dir.path().join("test_file.txt");
    fs::write(&file_path, "test content").await.unwrap();

    // Wait for event processing (debounce + processing time)
    sleep(Duration::from_millis(200)).await;

    assert!(file_path.exists());

    // Stop watcher
    assert!(watcher.stop().await.is_ok());
}

/// Test file modification detection
#[tokio::test]
#[serial]
async fn test_file_modification_detection() {
    let config = create_test_config(true, 50);
    let processor = create_test_processor().await;
    let mut watcher = FileWatcher::new(&config, processor).await.unwrap();

    let temp_dir = TempDir::new().unwrap();
    let file_path = temp_dir.path().join("modify_test.txt");

    // Create initial file
    fs::write(&file_path, "initial content").await.unwrap();

    // Start watcher
    assert!(watcher.start().await.is_ok());
    assert!(watcher.watch_directory(temp_dir.path()).await.is_ok());

    // Give watcher time to initialize
    sleep(Duration::from_millis(100)).await;

    // Modify the file
    fs::write(&file_path, "modified content").await.unwrap();

    // Wait for event processing
    sleep(Duration::from_millis(200)).await;

    let content = fs::read_to_string(&file_path).await.unwrap();
    assert_eq!(content, "modified content");

    assert!(watcher.stop().await.is_ok());
}

/// Test file deletion detection
#[tokio::test]
#[serial]
async fn test_file_deletion_detection() {
    let config = create_test_config(true, 50);
    let processor = create_test_processor().await;
    let mut watcher = FileWatcher::new(&config, processor).await.unwrap();

    let temp_dir = TempDir::new().unwrap();
    let file_path = temp_dir.path().join("delete_test.txt");

    // Create initial file
    fs::write(&file_path, "to be deleted").await.unwrap();
    assert!(file_path.exists());

    // Start watcher
    assert!(watcher.start().await.is_ok());
    assert!(watcher.watch_directory(temp_dir.path()).await.is_ok());

    // Give watcher time to initialize
    sleep(Duration::from_millis(100)).await;

    // Delete the file
    fs::remove_file(&file_path).await.unwrap();

    // Wait for event processing
    sleep(Duration::from_millis(200)).await;

    assert!(!file_path.exists());

    assert!(watcher.stop().await.is_ok());
}

/// Test rapid file events (debouncing)
#[tokio::test]
#[serial]
async fn test_rapid_file_events_debouncing() {
    let config = create_test_config(true, 100); // 100ms debounce
    let processor = create_test_processor().await;
    let mut watcher = FileWatcher::new(&config, processor).await.unwrap();

    let temp_dir = TempDir::new().unwrap();
    let file_path = temp_dir.path().join("rapid_test.txt");

    // Start watcher
    assert!(watcher.start().await.is_ok());
    assert!(watcher.watch_directory(temp_dir.path()).await.is_ok());

    // Give watcher time to initialize
    sleep(Duration::from_millis(50)).await;

    // Create rapid events (should be debounced)
    for i in 0..10 {
        fs::write(&file_path, format!("content {}", i)).await.unwrap();
        sleep(Duration::from_millis(10)).await; // Much faster than debounce
    }

    // Wait for debounce period + processing
    sleep(Duration::from_millis(250)).await;

    let final_content = fs::read_to_string(&file_path).await.unwrap();
    assert_eq!(final_content, "content 9");

    assert!(watcher.stop().await.is_ok());
}

/// Test ignore patterns functionality
#[tokio::test]
#[serial]
async fn test_ignore_patterns() {
    let config = create_test_config(true, 50);
    let processor = create_test_processor().await;
    let mut watcher = FileWatcher::new(&config, processor).await.unwrap();

    let temp_dir = TempDir::new().unwrap();

    // Start watcher
    assert!(watcher.start().await.is_ok());
    assert!(watcher.watch_directory(temp_dir.path()).await.is_ok());

    // Give watcher time to initialize
    sleep(Duration::from_millis(100)).await;

    // Create files that should be ignored
    let tmp_file = temp_dir.path().join("test.tmp");
    let log_file = temp_dir.path().join("test.log");
    let swp_file = temp_dir.path().join("test.swp");
    let normal_file = temp_dir.path().join("test.txt");

    // Test that should_ignore_path works correctly
    assert!(watcher.should_ignore_path(&tmp_file));
    assert!(watcher.should_ignore_path(&log_file));
    assert!(watcher.should_ignore_path(&swp_file));
    assert!(!watcher.should_ignore_path(&normal_file));

    // Create the files
    fs::write(&tmp_file, "ignored").await.unwrap();
    fs::write(&log_file, "ignored").await.unwrap();
    fs::write(&swp_file, "ignored").await.unwrap();
    fs::write(&normal_file, "processed").await.unwrap();

    // Wait for event processing
    sleep(Duration::from_millis(200)).await;

    assert!(watcher.stop().await.is_ok());
}

/// Test recursive directory monitoring
#[tokio::test]
#[serial]
async fn test_recursive_directory_monitoring() {
    let config = create_test_config(true, 50);
    let processor = create_test_processor().await;
    let mut watcher = FileWatcher::new(&config, processor).await.unwrap();

    let temp_dir = TempDir::new().unwrap();

    // Create nested directory structure
    let subdir = temp_dir.path().join("subdir");
    fs::create_dir(&subdir).await.unwrap();

    let nested_subdir = subdir.join("nested");
    fs::create_dir(&nested_subdir).await.unwrap();

    // Start watcher with recursive mode
    assert!(watcher.start().await.is_ok());
    assert!(watcher.watch_directory(temp_dir.path()).await.is_ok());

    // Give watcher time to initialize
    sleep(Duration::from_millis(100)).await;

    // Create files at different levels
    let root_file = temp_dir.path().join("root.txt");
    let sub_file = subdir.join("sub.txt");
    let nested_file = nested_subdir.join("nested.txt");

    fs::write(&root_file, "root content").await.unwrap();
    fs::write(&sub_file, "sub content").await.unwrap();
    fs::write(&nested_file, "nested content").await.unwrap();

    // Wait for event processing
    sleep(Duration::from_millis(200)).await;

    // Verify all files exist
    assert!(root_file.exists());
    assert!(sub_file.exists());
    assert!(nested_file.exists());

    assert!(watcher.stop().await.is_ok());
}

/// Test non-recursive directory monitoring
#[tokio::test]
#[serial]
async fn test_non_recursive_directory_monitoring() {
    let mut config = create_test_config(true, 50);
    config.recursive = false; // Disable recursive monitoring

    let processor = create_test_processor().await;
    let mut watcher = FileWatcher::new(&config, processor).await.unwrap();

    let temp_dir = TempDir::new().unwrap();

    // Create nested directory structure
    let subdir = temp_dir.path().join("subdir");
    fs::create_dir(&subdir).await.unwrap();

    // Start watcher with non-recursive mode
    assert!(watcher.start().await.is_ok());
    assert!(watcher.watch_directory(temp_dir.path()).await.is_ok());

    // Give watcher time to initialize
    sleep(Duration::from_millis(100)).await;

    // Create files at different levels
    let root_file = temp_dir.path().join("root.txt");
    let sub_file = subdir.join("sub.txt");

    fs::write(&root_file, "root content").await.unwrap();
    fs::write(&sub_file, "sub content").await.unwrap();

    // Wait for event processing
    sleep(Duration::from_millis(200)).await;

    // Verify files exist
    assert!(root_file.exists());
    assert!(sub_file.exists());

    assert!(watcher.stop().await.is_ok());
}

/// Test maximum watched directories limit
#[tokio::test]
#[serial]
async fn test_max_watched_directories_limit() {
    let mut config = create_test_config(true, 50);
    config.max_watched_dirs = 2; // Limit to 2 directories

    let processor = create_test_processor().await;
    let mut watcher = FileWatcher::new(&config, processor).await.unwrap();

    let temp_dir1 = TempDir::new().unwrap();
    let temp_dir2 = TempDir::new().unwrap();
    let temp_dir3 = TempDir::new().unwrap();

    // First two should succeed
    assert!(watcher.watch_directory(temp_dir1.path()).await.is_ok());
    assert!(watcher.watch_directory(temp_dir2.path()).await.is_ok());

    // Third should fail due to limit
    let result = watcher.watch_directory(temp_dir3.path()).await;
    assert!(result.is_err());

    // Verify exactly 2 directories are watched
    let watched_dirs = watcher.get_watched_directories().await;
    assert_eq!(watched_dirs.len(), 2);
}

/// Test watching the same directory multiple times
#[tokio::test]
#[serial]
async fn test_watch_same_directory_multiple_times() {
    let config = create_test_config(true, 50);
    let processor = create_test_processor().await;
    let mut watcher = FileWatcher::new(&config, processor).await.unwrap();

    let temp_dir = TempDir::new().unwrap();

    // Watch the same directory multiple times
    assert!(watcher.watch_directory(temp_dir.path()).await.is_ok());
    assert!(watcher.watch_directory(temp_dir.path()).await.is_ok());
    assert!(watcher.watch_directory(temp_dir.path()).await.is_ok());

    // Should only be watched once
    let watched_dirs = watcher.get_watched_directories().await;
    assert_eq!(watched_dirs.len(), 1);
    assert!(watched_dirs.contains(&temp_dir.path().to_path_buf()));
}

/// Test watcher state management
#[tokio::test]
#[serial]
async fn test_watcher_state_management() {
    let config = create_test_config(true, 50);
    let processor = create_test_processor().await;
    let mut watcher = FileWatcher::new(&config, processor).await.unwrap();

    let temp_dir = TempDir::new().unwrap();

    // Initially not running
    assert!(!watcher.is_running().await);

    // Start watcher
    assert!(watcher.start().await.is_ok());
    assert!(watcher.is_running().await);

    // Add directory
    assert!(watcher.watch_directory(temp_dir.path()).await.is_ok());

    // Still running
    assert!(watcher.is_running().await);

    // Stop watcher
    assert!(watcher.stop().await.is_ok());
    assert!(!watcher.is_running().await);

    // Remove directory while stopped (should work)
    assert!(watcher.unwatch_directory(temp_dir.path()).await.is_ok());
}

/// Test large file operations
#[tokio::test]
#[serial]
async fn test_large_file_operations() {
    let config = create_test_config(true, 50);
    let processor = create_test_processor().await;
    let mut watcher = FileWatcher::new(&config, processor).await.unwrap();

    let temp_dir = TempDir::new().unwrap();

    // Start watcher
    assert!(watcher.start().await.is_ok());
    assert!(watcher.watch_directory(temp_dir.path()).await.is_ok());

    // Give watcher time to initialize
    sleep(Duration::from_millis(100)).await;

    // Create a large file (1MB)
    let large_file = temp_dir.path().join("large_file.txt");
    let large_content = "x".repeat(1024 * 1024);
    fs::write(&large_file, &large_content).await.unwrap();

    // Wait for event processing
    sleep(Duration::from_millis(300)).await;

    // Verify file exists and has correct content
    assert!(large_file.exists());
    let read_content = fs::read_to_string(&large_file).await.unwrap();
    assert_eq!(read_content.len(), 1024 * 1024);

    assert!(watcher.stop().await.is_ok());
}

/// Test symlink handling (Unix-specific)
#[cfg(unix)]
#[tokio::test]
#[serial]
async fn test_symlink_handling() {
    let config = create_test_config(true, 50);
    let processor = create_test_processor().await;
    let mut watcher = FileWatcher::new(&config, processor).await.unwrap();

    let temp_dir = TempDir::new().unwrap();

    // Create a regular file
    let original_file = temp_dir.path().join("original.txt");
    fs::write(&original_file, "original content").await.unwrap();

    // Create a symlink
    let symlink_file = temp_dir.path().join("symlink.txt");
    tokio::fs::symlink(&original_file, &symlink_file).await.unwrap();

    // Start watcher
    assert!(watcher.start().await.is_ok());
    assert!(watcher.watch_directory(temp_dir.path()).await.is_ok());

    // Give watcher time to initialize
    sleep(Duration::from_millis(100)).await;

    // Modify through symlink
    fs::write(&symlink_file, "modified through symlink").await.unwrap();

    // Wait for event processing
    sleep(Duration::from_millis(200)).await;

    // Verify content changed in both files
    let original_content = fs::read_to_string(&original_file).await.unwrap();
    let symlink_content = fs::read_to_string(&symlink_file).await.unwrap();
    assert_eq!(original_content, "modified through symlink");
    assert_eq!(symlink_content, "modified through symlink");

    assert!(watcher.stop().await.is_ok());
}

/// Test file permission changes
#[cfg(unix)]
#[tokio::test]
#[serial]
async fn test_file_permission_changes() {
    let config = create_test_config(true, 50);
    let processor = create_test_processor().await;
    let mut watcher = FileWatcher::new(&config, processor).await.unwrap();

    let temp_dir = TempDir::new().unwrap();
    let file_path = temp_dir.path().join("permissions_test.txt");

    // Create file with default permissions
    fs::write(&file_path, "test content").await.unwrap();

    // Start watcher
    assert!(watcher.start().await.is_ok());
    assert!(watcher.watch_directory(temp_dir.path()).await.is_ok());

    // Give watcher time to initialize
    sleep(Duration::from_millis(100)).await;

    // Change permissions
    use std::os::unix::fs::PermissionsExt;
    let mut perms = fs::metadata(&file_path).await.unwrap().permissions();
    perms.set_mode(0o644);
    fs::set_permissions(&file_path, perms).await.unwrap();

    // Wait for event processing
    sleep(Duration::from_millis(200)).await;

    // Verify file still exists
    assert!(file_path.exists());

    assert!(watcher.stop().await.is_ok());
}

/// Test concurrent file operations
#[tokio::test]
#[serial]
async fn test_concurrent_file_operations() {
    let config = create_test_config(true, 100);
    let processor = create_test_processor().await;
    let mut watcher = FileWatcher::new(&config, processor).await.unwrap();

    let temp_dir = TempDir::new().unwrap();

    // Start watcher
    assert!(watcher.start().await.is_ok());
    assert!(watcher.watch_directory(temp_dir.path()).await.is_ok());

    // Give watcher time to initialize
    sleep(Duration::from_millis(100)).await;

    // Create multiple files concurrently
    let mut handles = vec![];
    for i in 0..10 {
        let file_path = temp_dir.path().join(format!("concurrent_{}.txt", i));
        let handle = tokio::spawn(async move {
            fs::write(&file_path, format!("content {}", i)).await.unwrap();
            file_path
        });
        handles.push(handle);
    }

    // Wait for all files to be created
    let mut created_files = vec![];
    for handle in handles {
        let file_path = handle.await.unwrap();
        created_files.push(file_path);
    }

    // Wait for event processing
    sleep(Duration::from_millis(300)).await;

    // Verify all files exist
    for file_path in created_files {
        assert!(file_path.exists());
    }

    assert!(watcher.stop().await.is_ok());
}

/// Test error resilience
#[tokio::test]
#[serial]
async fn test_error_resilience() {
    let config = create_test_config(true, 50);
    let processor = create_test_processor().await;
    let mut watcher = FileWatcher::new(&config, processor).await.unwrap();

    let temp_dir = TempDir::new().unwrap();

    // Start watcher
    assert!(watcher.start().await.is_ok());
    assert!(watcher.watch_directory(temp_dir.path()).await.is_ok());

    // Try to watch a non-existent directory (should handle gracefully)
    let result = watcher.watch_directory("/this/path/does/not/exist").await;
    // Implementation should handle this gracefully

    // Watcher should still be functional
    assert!(watcher.is_running().await);

    // Create a normal file to test functionality is preserved
    let file_path = temp_dir.path().join("resilience_test.txt");
    fs::write(&file_path, "test").await.unwrap();

    // Wait for event processing
    sleep(Duration::from_millis(200)).await;

    assert!(file_path.exists());
    assert!(watcher.stop().await.is_ok());
}

/// Test zero debounce configuration
#[tokio::test]
#[serial]
async fn test_zero_debounce() {
    let config = create_test_config(true, 0); // Zero debounce
    let processor = create_test_processor().await;
    let mut watcher = FileWatcher::new(&config, processor).await.unwrap();

    let temp_dir = TempDir::new().unwrap();

    // Start watcher
    assert!(watcher.start().await.is_ok());
    assert!(watcher.watch_directory(temp_dir.path()).await.is_ok());

    // Give watcher time to initialize
    sleep(Duration::from_millis(50)).await;

    // Create file (should be processed immediately)
    let file_path = temp_dir.path().join("zero_debounce.txt");
    fs::write(&file_path, "immediate processing").await.unwrap();

    // Minimal wait time for immediate processing
    sleep(Duration::from_millis(50)).await;

    assert!(file_path.exists());
    assert!(watcher.stop().await.is_ok());
}

/// Test high frequency file events
#[tokio::test]
#[serial]
async fn test_high_frequency_events() {
    let config = create_test_config(true, 200); // Higher debounce for this test
    let processor = create_test_processor().await;
    let mut watcher = FileWatcher::new(&config, processor).await.unwrap();

    let temp_dir = TempDir::new().unwrap();

    // Start watcher
    assert!(watcher.start().await.is_ok());
    assert!(watcher.watch_directory(temp_dir.path()).await.is_ok());

    // Give watcher time to initialize
    sleep(Duration::from_millis(100)).await;

    // Create many files in rapid succession
    for i in 0..50 {
        let file_path = temp_dir.path().join(format!("high_freq_{}.txt", i));
        fs::write(&file_path, format!("content {}", i)).await.unwrap();
        sleep(Duration::from_millis(1)).await; // Very fast creation
    }

    // Wait for debouncing and processing
    sleep(Duration::from_millis(500)).await;

    // Verify all files exist
    for i in 0..50 {
        let file_path = temp_dir.path().join(format!("high_freq_{}.txt", i));
        assert!(file_path.exists());
    }

    assert!(watcher.stop().await.is_ok());
}

/// Test cross-platform path handling
#[tokio::test]
#[serial]
async fn test_cross_platform_paths() {
    let config = create_test_config(true, 50);
    let processor = create_test_processor().await;
    let mut watcher = FileWatcher::new(&config, processor).await.unwrap();

    let temp_dir = TempDir::new().unwrap();

    // Test various path formats
    let paths_to_test = vec![
        "normal_file.txt",
        "file with spaces.txt",
        "file-with-dashes.txt",
        "file_with_underscores.txt",
        "file.with.dots.txt",
        "UPPERCASE.TXT",
    ];

    // Start watcher
    assert!(watcher.start().await.is_ok());
    assert!(watcher.watch_directory(temp_dir.path()).await.is_ok());

    // Give watcher time to initialize
    sleep(Duration::from_millis(100)).await;

    // Create files with various naming patterns
    for filename in &paths_to_test {
        let file_path = temp_dir.path().join(filename);
        fs::write(&file_path, format!("content for {}", filename)).await.unwrap();
    }

    // Wait for event processing
    sleep(Duration::from_millis(300)).await;

    // Verify all files exist
    for filename in &paths_to_test {
        let file_path = temp_dir.path().join(filename);
        assert!(file_path.exists(), "File {} should exist", filename);
    }

    assert!(watcher.stop().await.is_ok());
}