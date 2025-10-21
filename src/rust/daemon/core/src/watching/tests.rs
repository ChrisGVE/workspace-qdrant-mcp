//! Comprehensive tests for file watching functionality

use super::*;
use shared_test_utils::*;
use std::fs;
use std::time::Duration;
use tempfile::TempDir;
use crate::processing::Pipeline;

/// Create a test configuration for watching
fn test_watcher_config() -> WatcherConfig {
    WatcherConfig {
        include_patterns: vec!["*.txt".to_string(), "*.md".to_string(), "*.rs".to_string()],
        exclude_patterns: vec!["*.tmp".to_string(), "*.swp".to_string()],
        recursive: true,
        max_depth: 3,
        debounce_ms: 100, // Short debounce for faster tests
        polling_interval_ms: 100,
        min_polling_interval_ms: 50,
        max_polling_interval_ms: 5000,
        max_queue_size: 1000,
        task_priority: TaskPriority::BackgroundWatching,
        default_collection: "test_collection".to_string(),
        process_existing: false,
        max_file_size: Some(1024 * 1024), // 1MB
        use_polling: false,
        batch_processing: BatchConfig {
            enabled: true,
            max_batch_size: 5,
            max_batch_wait_ms: 500,
            group_by_type: true,
        },
        max_debouncer_capacity: 10000,
        max_batcher_capacity: 5000,
        telemetry: TelemetryConfig {
            enabled: false, // Disable telemetry in tests
            history_retention: 10,
            collection_interval_secs: 60,
            cpu_usage: false,
            memory_usage: false,
            latency: false,
            queue_depth: false,
            throughput: false,
        },
    }
}

/// Create a test task submitter
async fn create_test_task_submitter() -> TaskSubmitter {
    let pipeline = Pipeline::new(4);
    pipeline.task_submitter()
}

#[cfg(test)]
mod single_folder_watch_tests {
    use super::*;

    #[tokio::test]
    async fn test_watch_single_folder_creation() -> TestResult {
        init_test_tracing();

        let temp_dir = TempDir::new()?;
        let config = test_watcher_config();
        let task_submitter = create_test_task_submitter().await;

        let watcher = FileWatcher::new(config, task_submitter)?;

        // Start watching the temp directory
        watcher.watch_path(temp_dir.path()).await?;

        // Verify watcher is active
        let stats = watcher.stats().await;
        assert_eq!(stats.watched_paths, 1);

        let watched_paths = watcher.watched_paths().await;
        assert_eq!(watched_paths.len(), 1);
        assert_eq!(watched_paths[0], temp_dir.path());

        watcher.stop_watching().await?;
        Ok(())
    }

    #[tokio::test]
    async fn test_detect_file_creation() -> TestResult {
        init_test_tracing();

        let temp_dir = TempDir::new()?;
        let config = test_watcher_config();
        let task_submitter = create_test_task_submitter().await;

        let watcher = FileWatcher::new(config, task_submitter)?;
        watcher.watch_path(temp_dir.path()).await?;

        // Give the watcher time to start
        tokio::time::sleep(Duration::from_millis(200)).await;

        // Create a new file
        let test_file = temp_dir.path().join("test.txt");
        fs::write(&test_file, "Hello, world!")?;

        // Wait for event processing (debounce + processing time)
        tokio::time::sleep(Duration::from_millis(500)).await;

        // Verify event was detected
        let stats = watcher.stats().await;
        assert!(stats.events_received > 0, "Expected at least one event received");

        watcher.stop_watching().await?;
        Ok(())
    }

    #[tokio::test]
    async fn test_detect_file_modification() -> TestResult {
        init_test_tracing();

        let temp_dir = TempDir::new()?;
        let test_file = temp_dir.path().join("test.txt");
        fs::write(&test_file, "Initial content")?;

        let config = test_watcher_config();
        let task_submitter = create_test_task_submitter().await;

        let watcher = FileWatcher::new(config, task_submitter)?;
        watcher.watch_path(temp_dir.path()).await?;

        tokio::time::sleep(Duration::from_millis(200)).await;

        // Reset stats
        let initial_stats = watcher.stats().await;

        // Modify the file
        fs::write(&test_file, "Modified content")?;

        // Wait for event processing
        tokio::time::sleep(Duration::from_millis(500)).await;

        let final_stats = watcher.stats().await;
        assert!(
            final_stats.events_received > initial_stats.events_received,
            "Expected modification event to be detected"
        );

        watcher.stop_watching().await?;
        Ok(())
    }

    #[tokio::test]
    async fn test_detect_file_deletion() -> TestResult {
        init_test_tracing();

        let temp_dir = TempDir::new()?;
        let test_file = temp_dir.path().join("test.txt");
        fs::write(&test_file, "Content to delete")?;

        let config = test_watcher_config();
        let task_submitter = create_test_task_submitter().await;

        let watcher = FileWatcher::new(config, task_submitter)?;
        watcher.watch_path(temp_dir.path()).await?;

        tokio::time::sleep(Duration::from_millis(200)).await;

        let initial_stats = watcher.stats().await;

        // Delete the file
        fs::remove_file(&test_file)?;

        // Wait for event processing
        tokio::time::sleep(Duration::from_millis(500)).await;

        let final_stats = watcher.stats().await;
        assert!(
            final_stats.events_received > initial_stats.events_received,
            "Expected deletion event to be detected"
        );

        watcher.stop_watching().await?;
        Ok(())
    }

    #[tokio::test]
    async fn test_detect_file_rename() -> TestResult {
        init_test_tracing();

        let temp_dir = TempDir::new()?;
        let old_file = temp_dir.path().join("old.txt");
        let new_file = temp_dir.path().join("new.txt");
        fs::write(&old_file, "Content to rename")?;

        let config = test_watcher_config();
        let task_submitter = create_test_task_submitter().await;

        let watcher = FileWatcher::new(config, task_submitter)?;
        watcher.watch_path(temp_dir.path()).await?;

        tokio::time::sleep(Duration::from_millis(200)).await;

        let initial_stats = watcher.stats().await;

        // Rename the file
        fs::rename(&old_file, &new_file)?;

        // Wait for event processing
        tokio::time::sleep(Duration::from_millis(500)).await;

        let final_stats = watcher.stats().await;
        assert!(
            final_stats.events_received > initial_stats.events_received,
            "Expected rename event to be detected"
        );

        watcher.stop_watching().await?;
        Ok(())
    }

    #[tokio::test]
    async fn test_multiple_file_operations() -> TestResult {
        init_test_tracing();

        let temp_dir = TempDir::new()?;
        let config = test_watcher_config();
        let task_submitter = create_test_task_submitter().await;

        let watcher = FileWatcher::new(config, task_submitter)?;
        watcher.watch_path(temp_dir.path()).await?;

        tokio::time::sleep(Duration::from_millis(200)).await;

        // Perform multiple operations
        let file1 = temp_dir.path().join("file1.txt");
        let file2 = temp_dir.path().join("file2.md");
        let file3 = temp_dir.path().join("file3.rs");

        fs::write(&file1, "File 1")?;
        fs::write(&file2, "File 2")?;
        fs::write(&file3, "File 3")?;

        // Wait for event processing
        tokio::time::sleep(Duration::from_millis(800)).await;

        let stats = watcher.stats().await;
        assert!(stats.events_received >= 3, "Expected at least 3 events received");

        watcher.stop_watching().await?;
        Ok(())
    }

    #[tokio::test]
    async fn test_event_detection_within_2_seconds() -> TestResult {
        init_test_tracing();

        let temp_dir = TempDir::new()?;
        let config = test_watcher_config();
        let task_submitter = create_test_task_submitter().await;

        let watcher = FileWatcher::new(config, task_submitter)?;
        watcher.watch_path(temp_dir.path()).await?;

        tokio::time::sleep(Duration::from_millis(200)).await;

        let initial_stats = watcher.stats().await;
        let start_time = Instant::now();

        // Create a file
        let test_file = temp_dir.path().join("timing_test.txt");
        fs::write(&test_file, "Timing test")?;

        // Wait for event to be detected (poll every 50ms)
        let detected = wait_for_condition(
            || {
                let rt = tokio::runtime::Handle::current();
                let stats_future = watcher.stats();
                let stats = rt.block_on(stats_future);
                stats.events_received > initial_stats.events_received
            },
            Duration::from_secs(2),
            Duration::from_millis(50)
        ).await;

        let elapsed = start_time.elapsed();

        assert!(detected.is_ok(), "Event should be detected within 2 seconds");
        assert!(elapsed < Duration::from_secs(2), "Detection took too long: {:?}", elapsed);

        watcher.stop_watching().await?;
        Ok(())
    }

    #[tokio::test]
    async fn test_pattern_filtering() -> TestResult {
        init_test_tracing();

        let temp_dir = TempDir::new()?;
        let config = test_watcher_config();
        let task_submitter = create_test_task_submitter().await;

        let watcher = FileWatcher::new(config, task_submitter)?;
        watcher.watch_path(temp_dir.path()).await?;

        tokio::time::sleep(Duration::from_millis(200)).await;

        // Create files with different extensions
        let included_file = temp_dir.path().join("included.txt");
        let excluded_file = temp_dir.path().join("excluded.tmp");

        fs::write(&included_file, "Should be included")?;
        fs::write(&excluded_file, "Should be excluded")?;

        tokio::time::sleep(Duration::from_millis(500)).await;

        let stats = watcher.stats().await;
        // Excluded file should increase filtered count
        assert!(stats.events_filtered > 0, "Expected some events to be filtered");

        watcher.stop_watching().await?;
        Ok(())
    }

    #[tokio::test]
    async fn test_various_file_types() -> TestResult {
        init_test_tracing();

        let temp_dir = TempDir::new()?;
        let config = test_watcher_config();
        let task_submitter = create_test_task_submitter().await;

        let watcher = FileWatcher::new(config, task_submitter)?;
        watcher.watch_path(temp_dir.path()).await?;

        tokio::time::sleep(Duration::from_millis(200)).await;

        // Create files of different types and sizes
        let small_file = temp_dir.path().join("small.txt");
        let medium_file = temp_dir.path().join("medium.md");
        let large_file = temp_dir.path().join("large.rs");

        fs::write(&small_file, "Small")?;
        fs::write(&medium_file, "Medium ".repeat(100))?;
        fs::write(&large_file, "Large ".repeat(1000))?;

        tokio::time::sleep(Duration::from_millis(800)).await;

        let stats = watcher.stats().await;
        assert!(stats.events_received >= 3, "Expected events for all file types");

        watcher.stop_watching().await?;
        Ok(())
    }

    #[tokio::test]
    async fn test_file_size_limits() -> TestResult {
        init_test_tracing();

        let temp_dir = TempDir::new()?;
        let mut config = test_watcher_config();
        config.max_file_size = Some(100); // 100 bytes limit

        let task_submitter = create_test_task_submitter().await;

        let watcher = FileWatcher::new(config, task_submitter)?;
        watcher.watch_path(temp_dir.path()).await?;

        tokio::time::sleep(Duration::from_millis(200)).await;

        // Create files below and above the size limit
        let small_file = temp_dir.path().join("small.txt");
        let large_file = temp_dir.path().join("large.txt");

        fs::write(&small_file, "Small")?; // ~5 bytes
        fs::write(&large_file, "X".repeat(200))?; // 200 bytes

        tokio::time::sleep(Duration::from_millis(500)).await;

        let stats = watcher.stats().await;
        // Large file should be filtered
        assert!(stats.events_filtered > 0, "Expected large file to be filtered");

        watcher.stop_watching().await?;
        Ok(())
    }
}

#[cfg(test)]
mod edge_cases_tests {
    use super::*;

    #[tokio::test]
    async fn test_watch_nonexistent_path() -> TestResult {
        init_test_tracing();

        let config = test_watcher_config();
        let task_submitter = create_test_task_submitter().await;

        let watcher = FileWatcher::new(config, task_submitter)?;

        let result = watcher.watch_path(Path::new("/nonexistent/path")).await;
        assert!(result.is_err(), "Should fail to watch nonexistent path");

        match result {
            Err(WatchingError::Config { .. }) => Ok(()),
            _ => Err("Expected Config error".into()),
        }
    }

    #[tokio::test]
    async fn test_watch_stop_watch_cycle() -> TestResult {
        init_test_tracing();

        let temp_dir = TempDir::new()?;
        let config = test_watcher_config();
        let task_submitter = create_test_task_submitter().await;

        let watcher = FileWatcher::new(config, task_submitter)?;

        // Start watching
        watcher.watch_path(temp_dir.path()).await?;
        assert_eq!(watcher.watched_paths().await.len(), 1);

        // Stop watching
        watcher.stop_watching().await?;
        assert_eq!(watcher.watched_paths().await.len(), 0);

        // Start watching again
        watcher.watch_path(temp_dir.path()).await?;
        assert_eq!(watcher.watched_paths().await.len(), 1);

        watcher.stop_watching().await?;
        Ok(())
    }

    #[tokio::test]
    async fn test_empty_directory() -> TestResult {
        init_test_tracing();

        let temp_dir = TempDir::new()?;
        let config = test_watcher_config();
        let task_submitter = create_test_task_submitter().await;

        let watcher = FileWatcher::new(config, task_submitter)?;
        watcher.watch_path(temp_dir.path()).await?;

        tokio::time::sleep(Duration::from_millis(300)).await;

        let stats = watcher.stats().await;
        // No files, so no events
        assert_eq!(stats.events_received, 0);

        watcher.stop_watching().await?;
        Ok(())
    }

    #[tokio::test]
    async fn test_subdirectory_operations() -> TestResult {
        init_test_tracing();

        let temp_dir = TempDir::new()?;
        let subdir = temp_dir.path().join("subdir");
        fs::create_dir(&subdir)?;

        let config = test_watcher_config();
        let task_submitter = create_test_task_submitter().await;

        let watcher = FileWatcher::new(config, task_submitter)?;
        watcher.watch_path(temp_dir.path()).await?;

        tokio::time::sleep(Duration::from_millis(200)).await;

        // Create file in subdirectory
        let subfile = subdir.join("test.txt");
        fs::write(&subfile, "Subdirectory file")?;

        tokio::time::sleep(Duration::from_millis(500)).await;

        let stats = watcher.stats().await;
        assert!(stats.events_received > 0, "Should detect events in subdirectories");

        watcher.stop_watching().await?;
        Ok(())
    }
}

#[cfg(test)]
mod project_auto_watch_tests {
    use super::*;

    // Note: These tests are designed for the project auto-watch feature which would
    // require integration with SQLiteStateManager to detect projects at startup.
    // The current FileWatcher implementation doesn't have this functionality yet,
    // so these tests are placeholders for the future implementation.

    #[tokio::test]
    async fn test_watcher_can_be_configured_for_startup() -> TestResult {
        init_test_tracing();

        let temp_dir = TempDir::new()?;
        let config = test_watcher_config();
        let task_submitter = create_test_task_submitter().await;

        // Create a file watcher that could be used at startup
        let watcher = FileWatcher::new(config, task_submitter)?;

        // Verify watcher is in initialized state
        let stats = watcher.stats().await;
        assert_eq!(stats.watched_paths, 0);

        // Could be started with multiple paths
        watcher.watch_path(temp_dir.path()).await?;
        assert_eq!(watcher.watched_paths().await.len(), 1);

        watcher.stop_watching().await?;
        Ok(())
    }

    #[tokio::test]
    async fn test_watcher_handles_multiple_project_paths() -> TestResult {
        init_test_tracing();

        let temp_dir1 = TempDir::new()?;
        let temp_dir2 = TempDir::new()?;
        let temp_dir3 = TempDir::new()?;

        let config = test_watcher_config();
        let task_submitter = create_test_task_submitter().await;

        let watcher = FileWatcher::new(config, task_submitter)?;

        // Simulate startup by watching multiple project directories
        // Note: Current implementation would replace the watcher each time
        // Future implementation should support multiple concurrent watches
        watcher.watch_path(temp_dir1.path()).await?;

        // For now, we can at least verify it accepts the path
        let watched = watcher.watched_paths().await;
        assert!(watched.contains(&temp_dir1.path().to_path_buf()));

        watcher.stop_watching().await?;
        Ok(())
    }

    #[tokio::test]
    async fn test_watcher_survives_restart_cycle() -> TestResult {
        init_test_tracing();

        let temp_dir = TempDir::new()?;
        let config = test_watcher_config();
        let task_submitter = create_test_task_submitter().await;

        let watcher = FileWatcher::new(config.clone(), task_submitter.clone())?;

        // Start watching
        watcher.watch_path(temp_dir.path()).await?;
        assert_eq!(watcher.watched_paths().await.len(), 1);

        // Stop (simulating daemon restart)
        watcher.stop_watching().await?;
        assert_eq!(watcher.watched_paths().await.len(), 0);

        // Create new watcher (simulating daemon restart)
        let watcher2 = FileWatcher::new(config, task_submitter)?;

        // In a real implementation, this would reload from SQLite state
        // For now, just verify we can restart
        watcher2.watch_path(temp_dir.path()).await?;
        assert_eq!(watcher2.watched_paths().await.len(), 1);

        watcher2.stop_watching().await?;
        Ok(())
    }

    #[tokio::test]
    async fn test_watcher_handles_nested_project_structures() -> TestResult {
        init_test_tracing();

        let temp_dir = TempDir::new()?;
        let project_root = temp_dir.path();

        // Create nested project structure
        let src_dir = project_root.join("src");
        let tests_dir = project_root.join("tests");
        fs::create_dir_all(&src_dir)?;
        fs::create_dir_all(&tests_dir)?;

        let config = test_watcher_config();
        let task_submitter = create_test_task_submitter().await;

        let watcher = FileWatcher::new(config, task_submitter)?;

        // Watch the root project directory
        watcher.watch_path(project_root).await?;

        tokio::time::sleep(Duration::from_millis(200)).await;

        // Create files in nested directories
        fs::write(src_dir.join("main.rs"), "fn main() {}")?;
        fs::write(tests_dir.join("test.rs"), "#[test] fn it_works() {}")?;

        tokio::time::sleep(Duration::from_millis(500)).await;

        let stats = watcher.stats().await;
        assert!(stats.events_received > 0, "Should detect events in nested structures");

        watcher.stop_watching().await?;
        Ok(())
    }

    #[tokio::test]
    async fn test_process_existing_files_on_startup() -> TestResult {
        init_test_tracing();

        let temp_dir = TempDir::new()?;

        // Create files before watcher starts
        fs::write(temp_dir.path().join("existing1.txt"), "File 1")?;
        fs::write(temp_dir.path().join("existing2.md"), "File 2")?;
        fs::write(temp_dir.path().join("existing3.rs"), "File 3")?;

        // Configure with process_existing enabled
        let mut config = test_watcher_config();
        config.process_existing = true;

        let task_submitter = create_test_task_submitter().await;
        let watcher = FileWatcher::new(config, task_submitter)?;

        // When watching starts, it should process existing files
        watcher.watch_path(temp_dir.path()).await?;

        // Give time for initial scan to complete
        tokio::time::sleep(Duration::from_millis(1000)).await;

        let stats = watcher.stats().await;
        // Should have processed the 3 existing files
        assert!(stats.tasks_submitted >= 3, "Should process existing files on startup");

        watcher.stop_watching().await?;
        Ok(())
    }

    #[tokio::test]
    async fn test_startup_with_invalid_paths() -> TestResult {
        init_test_tracing();

        let config = test_watcher_config();
        let task_submitter = create_test_task_submitter().await;

        let watcher = FileWatcher::new(config, task_submitter)?;

        // Try to watch nonexistent path
        let result = watcher.watch_path(Path::new("/nonexistent/project")).await;
        assert!(result.is_err(), "Should fail gracefully with invalid path");

        // Watcher should still be in clean state
        assert_eq!(watcher.watched_paths().await.len(), 0);

        Ok(())
    }
}

