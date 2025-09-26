//! Comprehensive unit tests for daemon/watcher.rs
//! Tests file system monitoring capabilities, error handling, and edge cases

use workspace_qdrant_daemon::config::{FileWatcherConfig, ProcessingConfig, QdrantConfig, CollectionConfig};
use workspace_qdrant_daemon::daemon::watcher::FileWatcher;
use workspace_qdrant_daemon::daemon::processing::DocumentProcessor;
use std::sync::Arc;
use tempfile::{TempDir, NamedTempFile};

#[cfg(test)]
mod watcher_tests {
    use super::*;

    // Test configuration factory functions
    fn create_test_config(enabled: bool) -> FileWatcherConfig {
        FileWatcherConfig {
            enabled,
            debounce_ms: 100,
            max_watched_dirs: 10,
            ignore_patterns: vec!["*.tmp".to_string(), "*.log".to_string()],
            recursive: true,
        }
    }

    fn create_disabled_config() -> FileWatcherConfig {
        FileWatcherConfig {
            enabled: false,
            debounce_ms: 500,
            max_watched_dirs: 5,
            ignore_patterns: vec!["*.bak".to_string()],
            recursive: false,
        }
    }

    fn create_custom_config(debounce: u64, max_dirs: usize, recursive: bool) -> FileWatcherConfig {
        FileWatcherConfig {
            enabled: true,
            debounce_ms: debounce,
            max_watched_dirs: max_dirs,
            ignore_patterns: vec!["*.swp".to_string(), "node_modules/*".to_string()],
            recursive,
        }
    }

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
            timeout_secs: 30,
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

    // Basic construction and configuration tests
    #[tokio::test]
    async fn test_file_watcher_new_with_valid_config() {
        let config = create_test_config(true);
        let processor = create_test_processor();

        let result = FileWatcher::new(&config, processor).await;
        assert!(result.is_ok());

        let watcher = result.unwrap();
        assert_eq!(watcher.config().enabled, true);
        assert_eq!(watcher.config().debounce_ms, 100);
        assert_eq!(watcher.config().max_watched_dirs, 10);
        assert_eq!(watcher.config().recursive, true);
        assert_eq!(watcher.config().ignore_patterns.len(), 2);
    }

    #[tokio::test]
    async fn test_file_watcher_new_disabled() {
        let config = create_disabled_config();
        let processor = create_test_processor();

        let result = FileWatcher::new(&config, processor).await;
        assert!(result.is_ok());

        let watcher = result.unwrap();
        assert_eq!(watcher.config().enabled, false);
        assert_eq!(watcher.config().debounce_ms, 500);
        assert_eq!(watcher.config().max_watched_dirs, 5);
        assert_eq!(watcher.config().recursive, false);
    }

    #[tokio::test]
    async fn test_file_watcher_new_custom_config() {
        let config = create_custom_config(2000, 50, false);
        let processor = create_test_processor();

        let result = FileWatcher::new(&config, processor).await;
        assert!(result.is_ok());

        let watcher = result.unwrap();
        assert_eq!(watcher.config().debounce_ms, 2000);
        assert_eq!(watcher.config().max_watched_dirs, 50);
        assert_eq!(watcher.config().recursive, false);
        assert!(watcher.config().ignore_patterns.contains(&"*.swp".to_string()));
        assert!(watcher.config().ignore_patterns.contains(&"node_modules/*".to_string()));
    }

    #[tokio::test]
    async fn test_file_watcher_debug_implementation() {
        let config = create_test_config(true);
        let processor = create_test_processor();

        let watcher = FileWatcher::new(&config, processor).await.unwrap();
        let debug_output = format!("{:?}", watcher);

        assert!(debug_output.contains("FileWatcher"));
        assert!(debug_output.contains("config"));
        assert!(debug_output.contains("processor"));
        assert!(debug_output.contains("watcher"));
    }

    // Start/Stop lifecycle tests
    #[tokio::test]
    async fn test_file_watcher_start_when_enabled() {
        let config = create_test_config(true);
        let processor = create_test_processor();

        let watcher = FileWatcher::new(&config, processor).await.unwrap();
        let result = watcher.start().await;

        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn test_file_watcher_start_when_disabled() {
        let config = create_disabled_config();
        let processor = create_test_processor();

        let watcher = FileWatcher::new(&config, processor).await.unwrap();
        let result = watcher.start().await;

        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn test_file_watcher_stop_after_start() {
        let config = create_test_config(true);
        let processor = create_test_processor();

        let watcher = FileWatcher::new(&config, processor).await.unwrap();

        assert!(watcher.start().await.is_ok());
        assert!(watcher.stop().await.is_ok());
    }

    #[tokio::test]
    async fn test_file_watcher_stop_without_start() {
        let config = create_test_config(true);
        let processor = create_test_processor();

        let watcher = FileWatcher::new(&config, processor).await.unwrap();
        let result = watcher.stop().await;

        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn test_file_watcher_multiple_start_stop_cycles() {
        let config = create_test_config(true);
        let processor = create_test_processor();

        let watcher = FileWatcher::new(&config, processor).await.unwrap();

        // Multiple start/stop cycles should work
        for i in 0..5 {
            assert!(watcher.start().await.is_ok(), "Start cycle {} failed", i);
            assert!(watcher.stop().await.is_ok(), "Stop cycle {} failed", i);
        }
    }

    #[tokio::test]
    async fn test_file_watcher_concurrent_start_stop() {
        let config = create_test_config(true);
        let processor = create_test_processor();

        let watcher = Arc::new(FileWatcher::new(&config, processor).await.unwrap());

        // Spawn concurrent start/stop operations
        let watcher1: Arc<FileWatcher> = Arc::clone(&watcher);
        let watcher2: Arc<FileWatcher> = Arc::clone(&watcher);
        let watcher3: Arc<FileWatcher> = Arc::clone(&watcher);

        let start_task = tokio::spawn(async move { watcher1.start().await });
        let stop_task = tokio::spawn(async move { watcher2.stop().await });
        let start_task2 = tokio::spawn(async move { watcher3.start().await });

        let (start_result, stop_result, start_result2) = tokio::join!(start_task, stop_task, start_task2);

        assert!(start_result.unwrap().is_ok());
        assert!(stop_result.unwrap().is_ok());
        assert!(start_result2.unwrap().is_ok());
    }

    // Directory watching tests
    #[tokio::test]
    async fn test_file_watcher_watch_valid_directory() {
        let config = create_test_config(true);
        let processor = create_test_processor();

        let mut watcher = FileWatcher::new(&config, processor).await.unwrap();
        let temp_dir = TempDir::new().unwrap();

        let result = watcher.watch_directory(temp_dir.path()).await;
        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn test_file_watcher_watch_directory_as_string() {
        let config = create_test_config(true);
        let processor = create_test_processor();

        let mut watcher = FileWatcher::new(&config, processor).await.unwrap();

        let result = watcher.watch_directory("/tmp").await;
        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn test_file_watcher_watch_directory_as_pathbuf() {
        let config = create_test_config(true);
        let processor = create_test_processor();

        let mut watcher = FileWatcher::new(&config, processor).await.unwrap();
        let temp_dir = TempDir::new().unwrap();
        let path_buf = temp_dir.path().to_path_buf();

        let result = watcher.watch_directory(path_buf).await;
        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn test_file_watcher_watch_nonexistent_directory() {
        let config = create_test_config(true);
        let processor = create_test_processor();

        let mut watcher = FileWatcher::new(&config, processor).await.unwrap();
        let nonexistent_path = "/this/path/does/not/exist/anywhere";

        let result = watcher.watch_directory(nonexistent_path).await;
        // Current implementation doesn't validate paths, so this passes
        // In a real implementation, this might return an error
        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn test_file_watcher_watch_file_instead_of_directory() {
        let config = create_test_config(true);
        let processor = create_test_processor();

        let mut watcher = FileWatcher::new(&config, processor).await.unwrap();
        let temp_file = NamedTempFile::new().unwrap();

        let result = watcher.watch_directory(temp_file.path()).await;
        // Current implementation doesn't validate that path is a directory
        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn test_file_watcher_watch_multiple_directories() {
        let config = create_test_config(true);
        let processor = create_test_processor();

        let mut watcher = FileWatcher::new(&config, processor).await.unwrap();
        let temp_dir1 = TempDir::new().unwrap();
        let temp_dir2 = TempDir::new().unwrap();
        let temp_dir3 = TempDir::new().unwrap();

        assert!(watcher.watch_directory(temp_dir1.path()).await.is_ok());
        assert!(watcher.watch_directory(temp_dir2.path()).await.is_ok());
        assert!(watcher.watch_directory(temp_dir3.path()).await.is_ok());
    }

    #[tokio::test]
    async fn test_file_watcher_watch_same_directory_multiple_times() {
        let config = create_test_config(true);
        let processor = create_test_processor();

        let mut watcher = FileWatcher::new(&config, processor).await.unwrap();
        let temp_dir = TempDir::new().unwrap();

        // Watching the same directory multiple times should work
        assert!(watcher.watch_directory(temp_dir.path()).await.is_ok());
        assert!(watcher.watch_directory(temp_dir.path()).await.is_ok());
        assert!(watcher.watch_directory(temp_dir.path()).await.is_ok());
    }

    // Directory unwatching tests
    #[tokio::test]
    async fn test_file_watcher_unwatch_valid_directory() {
        let config = create_test_config(true);
        let processor = create_test_processor();

        let mut watcher = FileWatcher::new(&config, processor).await.unwrap();
        let temp_dir = TempDir::new().unwrap();

        assert!(watcher.watch_directory(temp_dir.path()).await.is_ok());
        assert!(watcher.unwatch_directory(temp_dir.path()).await.is_ok());
    }

    #[tokio::test]
    async fn test_file_watcher_unwatch_directory_as_string() {
        let config = create_test_config(true);
        let processor = create_test_processor();

        let mut watcher = FileWatcher::new(&config, processor).await.unwrap();

        assert!(watcher.watch_directory("/tmp").await.is_ok());
        assert!(watcher.unwatch_directory("/tmp").await.is_ok());
    }

    #[tokio::test]
    async fn test_file_watcher_unwatch_directory_as_pathbuf() {
        let config = create_test_config(true);
        let processor = create_test_processor();

        let mut watcher = FileWatcher::new(&config, processor).await.unwrap();
        let temp_dir = TempDir::new().unwrap();
        let path_buf = temp_dir.path().to_path_buf();

        assert!(watcher.watch_directory(&path_buf).await.is_ok());
        assert!(watcher.unwatch_directory(path_buf).await.is_ok());
    }

    #[tokio::test]
    async fn test_file_watcher_unwatch_without_watch() {
        let config = create_test_config(true);
        let processor = create_test_processor();

        let mut watcher = FileWatcher::new(&config, processor).await.unwrap();
        let temp_dir = TempDir::new().unwrap();

        // Unwatching a directory that was never watched should work
        let result = watcher.unwatch_directory(temp_dir.path()).await;
        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn test_file_watcher_unwatch_nonexistent_directory() {
        let config = create_test_config(true);
        let processor = create_test_processor();

        let mut watcher = FileWatcher::new(&config, processor).await.unwrap();
        let nonexistent_path = "/this/path/does/not/exist";

        let result = watcher.unwatch_directory(nonexistent_path).await;
        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn test_file_watcher_unwatch_multiple_directories() {
        let config = create_test_config(true);
        let processor = create_test_processor();

        let mut watcher = FileWatcher::new(&config, processor).await.unwrap();
        let temp_dir1 = TempDir::new().unwrap();
        let temp_dir2 = TempDir::new().unwrap();
        let temp_dir3 = TempDir::new().unwrap();

        // Watch multiple directories
        assert!(watcher.watch_directory(temp_dir1.path()).await.is_ok());
        assert!(watcher.watch_directory(temp_dir2.path()).await.is_ok());
        assert!(watcher.watch_directory(temp_dir3.path()).await.is_ok());

        // Unwatch them in different order
        assert!(watcher.unwatch_directory(temp_dir2.path()).await.is_ok());
        assert!(watcher.unwatch_directory(temp_dir1.path()).await.is_ok());
        assert!(watcher.unwatch_directory(temp_dir3.path()).await.is_ok());
    }

    #[tokio::test]
    async fn test_file_watcher_unwatch_same_directory_multiple_times() {
        let config = create_test_config(true);
        let processor = create_test_processor();

        let mut watcher = FileWatcher::new(&config, processor).await.unwrap();
        let temp_dir = TempDir::new().unwrap();

        assert!(watcher.watch_directory(temp_dir.path()).await.is_ok());

        // Unwatching the same directory multiple times should work
        assert!(watcher.unwatch_directory(temp_dir.path()).await.is_ok());
        assert!(watcher.unwatch_directory(temp_dir.path()).await.is_ok());
        assert!(watcher.unwatch_directory(temp_dir.path()).await.is_ok());
    }

    // Complex workflow tests
    #[tokio::test]
    async fn test_file_watcher_full_lifecycle() {
        let config = create_test_config(true);
        let processor = create_test_processor();

        let mut watcher = FileWatcher::new(&config, processor).await.unwrap();
        let temp_dir1 = TempDir::new().unwrap();
        let temp_dir2 = TempDir::new().unwrap();

        // Full lifecycle: start -> watch -> unwatch -> stop
        assert!(watcher.start().await.is_ok());
        assert!(watcher.watch_directory(temp_dir1.path()).await.is_ok());
        assert!(watcher.watch_directory(temp_dir2.path()).await.is_ok());
        assert!(watcher.unwatch_directory(temp_dir1.path()).await.is_ok());
        assert!(watcher.unwatch_directory(temp_dir2.path()).await.is_ok());
        assert!(watcher.stop().await.is_ok());
    }

    #[tokio::test]
    async fn test_file_watcher_watch_operations_without_start() {
        let config = create_test_config(true);
        let processor = create_test_processor();

        let mut watcher = FileWatcher::new(&config, processor).await.unwrap();
        let temp_dir = TempDir::new().unwrap();

        // Watch/unwatch operations should work even if start() was never called
        assert!(watcher.watch_directory(temp_dir.path()).await.is_ok());
        assert!(watcher.unwatch_directory(temp_dir.path()).await.is_ok());
    }

    #[tokio::test]
    async fn test_file_watcher_operations_after_stop() {
        let config = create_test_config(true);
        let processor = create_test_processor();

        let mut watcher = FileWatcher::new(&config, processor).await.unwrap();
        let temp_dir = TempDir::new().unwrap();

        assert!(watcher.start().await.is_ok());
        assert!(watcher.stop().await.is_ok());

        // Operations after stop should still work (in current implementation)
        assert!(watcher.watch_directory(temp_dir.path()).await.is_ok());
        assert!(watcher.unwatch_directory(temp_dir.path()).await.is_ok());
    }

    // Configuration edge cases
    #[tokio::test]
    async fn test_file_watcher_extreme_config_values() {
        let config = FileWatcherConfig {
            enabled: true,
            debounce_ms: 0,
            max_watched_dirs: 0,
            ignore_patterns: vec![],
            recursive: true,
        };
        let processor = create_test_processor();

        let result = FileWatcher::new(&config, processor).await;
        assert!(result.is_ok());

        let watcher = result.unwrap();
        assert_eq!(watcher.config().debounce_ms, 0);
        assert_eq!(watcher.config().max_watched_dirs, 0);
        assert_eq!(watcher.config().ignore_patterns.len(), 0);
    }

    #[tokio::test]
    async fn test_file_watcher_large_config_values() {
        let config = FileWatcherConfig {
            enabled: true,
            debounce_ms: u64::MAX,
            max_watched_dirs: usize::MAX,
            ignore_patterns: vec!["*".to_string(); 1000], // Large ignore list
            recursive: false,
        };
        let processor = create_test_processor();

        let result = FileWatcher::new(&config, processor).await;
        assert!(result.is_ok());

        let watcher = result.unwrap();
        assert_eq!(watcher.config().debounce_ms, u64::MAX);
        assert_eq!(watcher.config().max_watched_dirs, usize::MAX);
        assert_eq!(watcher.config().ignore_patterns.len(), 1000);
    }

    // Processor Arc sharing tests
    #[tokio::test]
    async fn test_file_watcher_processor_arc_sharing() {
        let config = create_test_config(true);
        let processor = create_test_processor();
        let initial_strong_count = Arc::strong_count(&processor);

        let _watcher = FileWatcher::new(&config, Arc::clone(&processor)).await.unwrap();

        // The watcher should hold a reference to the processor
        assert!(Arc::strong_count(&processor) > initial_strong_count);
    }

    #[tokio::test]
    async fn test_file_watcher_multiple_instances_shared_processor() {
        let config = create_test_config(true);
        let processor = create_test_processor();
        let initial_strong_count = Arc::strong_count(&processor);

        let _watcher1 = FileWatcher::new(&config, Arc::clone(&processor)).await.unwrap();
        let _watcher2 = FileWatcher::new(&config, Arc::clone(&processor)).await.unwrap();
        let _watcher3 = FileWatcher::new(&config, Arc::clone(&processor)).await.unwrap();

        // Multiple watchers should all share the same processor
        assert_eq!(Arc::strong_count(&processor), initial_strong_count + 3);
    }

    // Path handling edge cases
    #[tokio::test]
    async fn test_file_watcher_empty_path() {
        let config = create_test_config(true);
        let processor = create_test_processor();

        let mut watcher = FileWatcher::new(&config, processor).await.unwrap();

        let result = watcher.watch_directory("").await;
        assert!(result.is_ok());

        let result = watcher.unwatch_directory("").await;
        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn test_file_watcher_root_path() {
        let config = create_test_config(true);
        let processor = create_test_processor();

        let mut watcher = FileWatcher::new(&config, processor).await.unwrap();

        let result = watcher.watch_directory("/").await;
        assert!(result.is_ok());

        let result = watcher.unwatch_directory("/").await;
        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn test_file_watcher_relative_path() {
        let config = create_test_config(true);
        let processor = create_test_processor();

        let mut watcher = FileWatcher::new(&config, processor).await.unwrap();

        let result = watcher.watch_directory("./").await;
        assert!(result.is_ok());

        let result = watcher.unwatch_directory("../").await;
        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn test_file_watcher_path_with_special_characters() {
        let config = create_test_config(true);
        let processor = create_test_processor();

        let mut watcher = FileWatcher::new(&config, processor).await.unwrap();

        // Paths with spaces, unicode, and special characters
        let special_paths = vec![
            "/path with spaces",
            "/path-with-dashes",
            "/path_with_underscore",
            "/path.with.dots",
            "/path/with/unicode/文档",
            "/path/with/émoji/🚀",
        ];

        for path in special_paths {
            assert!(watcher.watch_directory(path).await.is_ok(), "Failed to watch path: {}", path);
            assert!(watcher.unwatch_directory(path).await.is_ok(), "Failed to unwatch path: {}", path);
        }
    }

    // Thread safety tests
    #[test]
    fn test_file_watcher_send_trait() {
        fn assert_send<T: Send>() {}
        assert_send::<FileWatcher>();
    }

    #[test]
    fn test_file_watcher_sync_trait() {
        fn assert_sync<T: Sync>() {}
        assert_sync::<FileWatcher>();
    }

    #[tokio::test]
    async fn test_file_watcher_concurrent_operations() {
        let config = create_test_config(true);
        let processor = create_test_processor();

        let watcher = Arc::new(tokio::sync::Mutex::new(
            FileWatcher::new(&config, processor).await.unwrap()
        ));

        let temp_dirs: Vec<_> = (0..10).map(|_| TempDir::new().unwrap()).collect();

        // Create multiple concurrent tasks that perform watch/unwatch operations
        let mut tasks = vec![];

        for (i, temp_dir) in temp_dirs.iter().enumerate() {
            let watcher_clone: Arc<tokio::sync::Mutex<FileWatcher>> = Arc::clone(&watcher);
            let path = temp_dir.path().to_path_buf();

            let task = tokio::spawn(async move {
                let mut guard = watcher_clone.lock().await;
                guard.watch_directory(&path).await?;
                tokio::time::sleep(std::time::Duration::from_millis(10)).await;
                guard.unwatch_directory(&path).await
            });

            tasks.push(task);
        }

        // Wait for all tasks to complete
        for (i, task) in tasks.into_iter().enumerate() {
            let result = task.await;
            assert!(result.is_ok(), "Task {} failed", i);
            assert!(result.unwrap().is_ok(), "Watch/unwatch operation {} failed", i);
        }
    }

    // Configuration cloning and comparison tests
    #[tokio::test]
    async fn test_file_watcher_config_independence() {
        let mut config1 = create_test_config(true);
        let config2 = config1.clone();

        // Modify original config
        config1.enabled = false;
        config1.debounce_ms = 999;

        // Cloned config should be independent
        assert_eq!(config2.enabled, true);
        assert_eq!(config2.debounce_ms, 100);
    }

    #[tokio::test]
    async fn test_file_watcher_with_cloned_config() {
        let config = create_test_config(true);
        let config_clone = config.clone();
        let processor = create_test_processor();

        let watcher1 = FileWatcher::new(&config, Arc::clone(&processor)).await.unwrap();
        let watcher2 = FileWatcher::new(&config_clone, processor).await.unwrap();

        // Both watchers should have the same configuration values
        assert_eq!(watcher1.config().enabled, watcher2.config().enabled);
        assert_eq!(watcher1.config().debounce_ms, watcher2.config().debounce_ms);
        assert_eq!(watcher1.config().max_watched_dirs, watcher2.config().max_watched_dirs);
        assert_eq!(watcher1.config().recursive, watcher2.config().recursive);
    }

    // Error resilience tests (testing placeholder implementation behavior)
    #[tokio::test]
    async fn test_file_watcher_operations_always_succeed() {
        let config = create_test_config(true);
        let processor = create_test_processor();

        let mut watcher = FileWatcher::new(&config, processor).await.unwrap();

        // In the current placeholder implementation, all operations should succeed
        // regardless of input validity
        let problematic_paths = vec![
            "",
            "/",
            "/dev/null",
            "/proc/cpuinfo",
            "/this/absolutely/does/not/exist/anywhere/at/all",
            "/root", // Potentially restricted
        ];

        for path in problematic_paths {
            assert!(watcher.start().await.is_ok());
            assert!(watcher.watch_directory(path).await.is_ok());
            assert!(watcher.unwatch_directory(path).await.is_ok());
            assert!(watcher.stop().await.is_ok());
        }
    }

    // Performance and stress tests
    #[tokio::test]
    async fn test_file_watcher_many_directories() {
        let config = create_custom_config(50, 1000, true);
        let processor = create_test_processor();

        let mut watcher = FileWatcher::new(&config, processor).await.unwrap();

        // Create many temporary directories
        let temp_dirs: Vec<_> = (0..100).map(|_| TempDir::new().unwrap()).collect();

        // Watch all directories
        for temp_dir in &temp_dirs {
            assert!(watcher.watch_directory(temp_dir.path()).await.is_ok());
        }

        // Unwatch all directories
        for temp_dir in &temp_dirs {
            assert!(watcher.unwatch_directory(temp_dir.path()).await.is_ok());
        }
    }

    #[tokio::test]
    async fn test_file_watcher_rapid_operations() {
        let config = create_test_config(true);
        let processor = create_test_processor();

        let mut watcher = FileWatcher::new(&config, processor).await.unwrap();
        let temp_dir = TempDir::new().unwrap();

        // Perform many rapid operations
        for _ in 0..50 {
            assert!(watcher.start().await.is_ok());
            assert!(watcher.watch_directory(temp_dir.path()).await.is_ok());
            assert!(watcher.unwatch_directory(temp_dir.path()).await.is_ok());
            assert!(watcher.stop().await.is_ok());
        }
    }
}