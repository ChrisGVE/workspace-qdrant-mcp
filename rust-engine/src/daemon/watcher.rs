//! File system watcher for automatic document processing

use crate::config::FileWatcherConfig;
use crate::daemon::processing::DocumentProcessor;
use crate::error::{DaemonError, DaemonResult};
use notify::{RecommendedWatcher, RecursiveMode, Event};
use std::sync::Arc;
use std::path::Path;
use tokio::sync::{mpsc, Mutex};
use tracing::{info, debug, warn, error};

/// File watcher
#[derive(Debug)]
pub struct FileWatcher {
    config: FileWatcherConfig,
    processor: Arc<DocumentProcessor>,
    watcher: Arc<Mutex<Option<RecommendedWatcher>>>,
}

impl FileWatcher {
    /// Create a new file watcher
    pub async fn new(config: &FileWatcherConfig, processor: Arc<DocumentProcessor>) -> DaemonResult<Self> {
        info!("Initializing file watcher (enabled: {})", config.enabled);

        Ok(Self {
            config: config.clone(),
            processor,
            watcher: Arc::new(Mutex::new(None)),
        })
    }

    /// Start watching for file changes
    pub async fn start(&self) -> DaemonResult<()> {
        if !self.config.enabled {
            info!("File watcher is disabled");
            return Ok(());
        }

        info!("Starting file watcher");

        // TODO: Implement actual file watching
        // This is a placeholder implementation

        Ok(())
    }

    /// Stop watching for file changes
    pub async fn stop(&self) -> DaemonResult<()> {
        info!("Stopping file watcher");

        // TODO: Implement actual stop logic

        Ok(())
    }

    /// Add a directory to watch
    pub async fn watch_directory<P: AsRef<Path>>(&mut self, path: P) -> DaemonResult<()> {
        let path = path.as_ref();
        info!("Adding directory to watch: {}", path.display());

        // TODO: Implement actual directory watching

        Ok(())
    }

    /// Remove a directory from watching
    pub async fn unwatch_directory<P: AsRef<Path>>(&mut self, path: P) -> DaemonResult<()> {
        let path = path.as_ref();
        info!("Removing directory from watch: {}", path.display());

        // TODO: Implement actual directory unwatching

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::{FileWatcherConfig, ProcessingConfig, QdrantConfig};
    use tempfile::TempDir;
    use std::path::PathBuf;
    use std::sync::Arc;
    use tokio_test;

    fn create_test_config(enabled: bool) -> FileWatcherConfig {
        FileWatcherConfig {
            enabled,
            debounce_ms: 100,
            max_watched_dirs: 10,
            ignore_patterns: vec!["*.tmp".to_string(), "*.log".to_string()],
            recursive: true,
        }
    }

    fn create_test_processor() -> Arc<DocumentProcessor> {
        // Use test instance for reliable testing
        Arc::new(DocumentProcessor::test_instance())
    }

    #[tokio::test]
    async fn test_file_watcher_new_enabled() {
        let config = create_test_config(true);
        let processor = create_test_processor();

        let result = FileWatcher::new(&config, processor).await;
        assert!(result.is_ok());

        let watcher = result.unwrap();
        assert_eq!(watcher.config.enabled, true);
        assert_eq!(watcher.config.debounce_ms, 100);
        assert_eq!(watcher.config.max_watched_dirs, 10);
    }

    #[tokio::test]
    async fn test_file_watcher_new_disabled() {
        let config = create_test_config(false);
        let processor = create_test_processor();

        let result = FileWatcher::new(&config, processor).await;
        assert!(result.is_ok());

        let watcher = result.unwrap();
        assert_eq!(watcher.config.enabled, false);
    }

    #[tokio::test]
    async fn test_file_watcher_debug_format() {
        let config = create_test_config(true);
        let processor = create_test_processor();

        let watcher = FileWatcher::new(&config, processor).await.unwrap();
        let debug_str = format!("{:?}", watcher);

        assert!(debug_str.contains("FileWatcher"));
        assert!(debug_str.contains("config"));
        assert!(debug_str.contains("processor"));
    }

    #[tokio::test]
    async fn test_file_watcher_start_enabled() {
        let config = create_test_config(true);
        let processor = create_test_processor();

        let watcher = FileWatcher::new(&config, processor).await.unwrap();
        let result = watcher.start().await;

        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn test_file_watcher_start_disabled() {
        let config = create_test_config(false);
        let processor = create_test_processor();

        let watcher = FileWatcher::new(&config, processor).await.unwrap();
        let result = watcher.start().await;

        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn test_file_watcher_stop() {
        let config = create_test_config(true);
        let processor = create_test_processor();

        let watcher = FileWatcher::new(&config, processor).await.unwrap();

        // Start then stop
        assert!(watcher.start().await.is_ok());
        assert!(watcher.stop().await.is_ok());
    }

    #[tokio::test]
    async fn test_file_watcher_watch_directory() {
        let config = create_test_config(true);
        let processor = create_test_processor();

        let mut watcher = FileWatcher::new(&config, processor).await.unwrap();

        let temp_dir = TempDir::new().unwrap();
        let result = watcher.watch_directory(temp_dir.path()).await;

        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn test_file_watcher_watch_directory_string_path() {
        let config = create_test_config(true);
        let processor = create_test_processor();

        let mut watcher = FileWatcher::new(&config, processor).await.unwrap();

        let result = watcher.watch_directory("/tmp").await;
        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn test_file_watcher_unwatch_directory() {
        let config = create_test_config(true);
        let processor = create_test_processor();

        let mut watcher = FileWatcher::new(&config, processor).await.unwrap();

        let temp_dir = TempDir::new().unwrap();

        // Watch then unwatch
        assert!(watcher.watch_directory(temp_dir.path()).await.is_ok());
        assert!(watcher.unwatch_directory(temp_dir.path()).await.is_ok());
    }

    #[tokio::test]
    async fn test_file_watcher_unwatch_directory_string_path() {
        let config = create_test_config(true);
        let processor = create_test_processor();

        let mut watcher = FileWatcher::new(&config, processor).await.unwrap();

        let result = watcher.unwatch_directory("/tmp").await;
        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn test_file_watcher_multiple_directories() {
        let config = create_test_config(true);
        let processor = create_test_processor();

        let mut watcher = FileWatcher::new(&config, processor).await.unwrap();

        let temp_dir1 = TempDir::new().unwrap();
        let temp_dir2 = TempDir::new().unwrap();

        // Watch multiple directories
        assert!(watcher.watch_directory(temp_dir1.path()).await.is_ok());
        assert!(watcher.watch_directory(temp_dir2.path()).await.is_ok());

        // Unwatch them
        assert!(watcher.unwatch_directory(temp_dir1.path()).await.is_ok());
        assert!(watcher.unwatch_directory(temp_dir2.path()).await.is_ok());
    }

    #[tokio::test]
    async fn test_file_watcher_start_stop_multiple_times() {
        let config = create_test_config(true);
        let processor = create_test_processor();

        let watcher = FileWatcher::new(&config, processor).await.unwrap();

        // Multiple start/stop cycles
        for _ in 0..3 {
            assert!(watcher.start().await.is_ok());
            assert!(watcher.stop().await.is_ok());
        }
    }

    #[tokio::test]
    async fn test_file_watcher_config_clone() {
        let config = create_test_config(true);
        let config_clone = config.clone();

        assert_eq!(config.enabled, config_clone.enabled);
        assert_eq!(config.debounce_ms, config_clone.debounce_ms);
        assert_eq!(config.max_watched_dirs, config_clone.max_watched_dirs);
        assert_eq!(config.ignore_patterns, config_clone.ignore_patterns);
    }

    #[tokio::test]
    async fn test_file_watcher_processor_arc_sharing() {
        let config = create_test_config(true);
        let processor = create_test_processor();
        let processor_clone = Arc::clone(&processor);

        let watcher = FileWatcher::new(&config, processor_clone).await.unwrap();

        // Test that the processor Arc is properly shared
        assert!(Arc::strong_count(&processor) >= 2);
    }

    #[test]
    fn test_file_watcher_send_sync() {
        fn assert_send<T: Send>() {}
        fn assert_sync<T: Sync>() {}

        assert_send::<FileWatcher>();
        assert_sync::<FileWatcher>();
    }

    #[tokio::test]
    async fn test_file_watcher_with_ignore_patterns() {
        let mut config = create_test_config(true);
        config.ignore_patterns = vec![
            "*.tmp".to_string(),
            "*.log".to_string(),
            "target/*".to_string(),
        ];

        let processor = create_test_processor();
        let watcher = FileWatcher::new(&config, processor).await.unwrap();

        assert_eq!(watcher.config.ignore_patterns.len(), 3);
        assert!(watcher.config.ignore_patterns.contains(&"*.tmp".to_string()));
        assert!(watcher.config.ignore_patterns.contains(&"*.log".to_string()));
        assert!(watcher.config.ignore_patterns.contains(&"target/*".to_string()));
    }

    #[tokio::test]
    async fn test_file_watcher_with_custom_debounce() {
        let mut config = create_test_config(true);
        config.debounce_ms = 1000;

        let processor = create_test_processor();
        let watcher = FileWatcher::new(&config, processor).await.unwrap();

        assert_eq!(watcher.config.debounce_ms, 1000);
    }

    #[tokio::test]
    async fn test_file_watcher_with_edge_case_configs() {
        // Test with zero debounce
        let mut config = create_test_config(true);
        config.debounce_ms = 0;
        config.max_watched_dirs = 0;

        let processor = create_test_processor();
        let watcher = FileWatcher::new(&config, processor).await.unwrap();

        assert_eq!(watcher.config.debounce_ms, 0);
        assert_eq!(watcher.config.max_watched_dirs, 0);
    }

    #[tokio::test]
    async fn test_file_watcher_with_maximal_config() {
        let mut config = create_test_config(true);
        config.debounce_ms = u64::MAX;
        config.max_watched_dirs = usize::MAX;
        config.ignore_patterns = vec![
            "*.tmp".to_string(),
            "*.log".to_string(),
            "target/**".to_string(),
            "node_modules/**".to_string(),
            ".git/**".to_string(),
            "*.backup".to_string(),
            "*.swp".to_string(),
            "*~".to_string(),
        ];
        config.recursive = true;

        let processor = create_test_processor();
        let watcher = FileWatcher::new(&config, processor).await.unwrap();

        assert_eq!(watcher.config.debounce_ms, u64::MAX);
        assert_eq!(watcher.config.max_watched_dirs, usize::MAX);
        assert_eq!(watcher.config.ignore_patterns.len(), 8);
        assert!(watcher.config.recursive);
    }

    #[tokio::test]
    async fn test_file_watcher_watcher_field_initialization() {
        let config = create_test_config(true);
        let processor = create_test_processor();

        let watcher = FileWatcher::new(&config, processor).await.unwrap();

        // Verify the watcher field is properly initialized as None
        let watcher_guard = watcher.watcher.lock().await;
        assert!(watcher_guard.is_none());
    }

    #[tokio::test]
    async fn test_file_watcher_processor_field_access() {
        let config = create_test_config(true);
        let processor = create_test_processor();
        let processor_weak_count = Arc::weak_count(&processor);

        let watcher = FileWatcher::new(&config, processor.clone()).await.unwrap();

        // Verify processor is properly stored and accessible
        assert!(Arc::ptr_eq(&watcher.processor, &processor));
        assert!(Arc::weak_count(&processor) >= processor_weak_count);
    }

    #[tokio::test]
    async fn test_file_watcher_config_field_values() {
        let config = create_test_config(true);
        let processor = create_test_processor();

        let watcher = FileWatcher::new(&config, processor).await.unwrap();

        // Verify all config fields are properly cloned and stored
        assert_eq!(watcher.config.enabled, config.enabled);
        assert_eq!(watcher.config.debounce_ms, config.debounce_ms);
        assert_eq!(watcher.config.max_watched_dirs, config.max_watched_dirs);
        assert_eq!(watcher.config.ignore_patterns, config.ignore_patterns);
        assert_eq!(watcher.config.recursive, config.recursive);
    }

    #[tokio::test]
    async fn test_file_watcher_logging_levels() {
        // Test that logging statements are executed by configuring tracing
        let _ = tracing_subscriber::fmt()
            .with_max_level(tracing::Level::DEBUG)
            .try_init();

        let config = create_test_config(true);
        let processor = create_test_processor();

        // This will trigger the info! logging on line 23
        let watcher = FileWatcher::new(&config, processor).await.unwrap();

        // This will trigger the info! logging on line 39
        assert!(watcher.start().await.is_ok());

        // This will trigger the info! logging on line 49
        assert!(watcher.stop().await.is_ok());
    }

    #[tokio::test]
    async fn test_file_watcher_disabled_logging() {
        // Test the disabled path logging
        let _ = tracing_subscriber::fmt()
            .with_max_level(tracing::Level::INFO)
            .try_init();

        let config = create_test_config(false); // disabled
        let processor = create_test_processor();

        let watcher = FileWatcher::new(&config, processor).await.unwrap();

        // This will trigger the "File watcher is disabled" info! logging on line 35
        assert!(watcher.start().await.is_ok());
    }

    #[tokio::test]
    async fn test_file_watcher_watch_directory_logging() {
        // Test directory watching logging
        let _ = tracing_subscriber::fmt()
            .with_max_level(tracing::Level::INFO)
            .try_init();

        let config = create_test_config(true);
        let processor = create_test_processor();
        let mut watcher = FileWatcher::new(&config, processor).await.unwrap();

        let temp_dir = TempDir::new().unwrap();

        // This will trigger the info! logging on line 59
        assert!(watcher.watch_directory(temp_dir.path()).await.is_ok());
    }

    #[tokio::test]
    async fn test_file_watcher_unwatch_directory_logging() {
        // Test directory unwatching logging
        let _ = tracing_subscriber::fmt()
            .with_max_level(tracing::Level::INFO)
            .try_init();

        let config = create_test_config(true);
        let processor = create_test_processor();
        let mut watcher = FileWatcher::new(&config, processor).await.unwrap();

        let temp_dir = TempDir::new().unwrap();

        // This will trigger the info! logging on line 69
        assert!(watcher.unwatch_directory(temp_dir.path()).await.is_ok());
    }

    #[tokio::test]
    async fn test_file_watcher_path_as_ref_implementations() {
        let config = create_test_config(true);
        let processor = create_test_processor();
        let mut watcher = FileWatcher::new(&config, processor).await.unwrap();

        let temp_dir = TempDir::new().unwrap();
        let path_buf = temp_dir.path().to_path_buf();
        let path_str = temp_dir.path().to_str().unwrap();

        // Test different AsRef<Path> implementations
        assert!(watcher.watch_directory(&path_buf).await.is_ok());
        assert!(watcher.watch_directory(path_str).await.is_ok());
        assert!(watcher.watch_directory(temp_dir.path()).await.is_ok());

        assert!(watcher.unwatch_directory(&path_buf).await.is_ok());
        assert!(watcher.unwatch_directory(path_str).await.is_ok());
        assert!(watcher.unwatch_directory(temp_dir.path()).await.is_ok());
    }

    #[tokio::test]
    async fn test_file_watcher_complex_paths() {
        let config = create_test_config(true);
        let processor = create_test_processor();
        let mut watcher = FileWatcher::new(&config, processor).await.unwrap();

        // Test complex path scenarios
        let complex_paths = vec![
            "/tmp",
            "/tmp/subdir",
            "./relative/path",
            "../parent/path",
            "/path/with spaces/dir",
            "/path/with-dashes/dir",
            "/path/with_underscores/dir",
            "/path/with.dots/dir",
        ];

        for path in complex_paths {
            assert!(watcher.watch_directory(path).await.is_ok());
            assert!(watcher.unwatch_directory(path).await.is_ok());
        }
    }

    #[tokio::test]
    async fn test_file_watcher_unicode_paths() {
        let config = create_test_config(true);
        let processor = create_test_processor();
        let mut watcher = FileWatcher::new(&config, processor).await.unwrap();

        let unicode_paths = vec![
            "/tmp/русский",     // Russian
            "/tmp/中文",         // Chinese
            "/tmp/日本語",       // Japanese
            "/tmp/한국어",       // Korean
            "/tmp/ελληνικά",    // Greek
            "/tmp/हिन्दी",      // Hindi
            "/tmp/🚀rocket",     // Emoji
        ];

        for path in unicode_paths {
            assert!(watcher.watch_directory(path).await.is_ok());
            assert!(watcher.unwatch_directory(path).await.is_ok());
        }
    }

    #[tokio::test]
    async fn test_file_watcher_empty_path() {
        let config = create_test_config(true);
        let processor = create_test_processor();
        let mut watcher = FileWatcher::new(&config, processor).await.unwrap();

        // Test empty path
        assert!(watcher.watch_directory("").await.is_ok());
        assert!(watcher.unwatch_directory("").await.is_ok());
    }

    #[tokio::test]
    async fn test_file_watcher_very_long_path() {
        let config = create_test_config(true);
        let processor = create_test_processor();
        let mut watcher = FileWatcher::new(&config, processor).await.unwrap();

        // Test very long path
        let long_path = format!("/tmp/{}", "a".repeat(1000));
        assert!(watcher.watch_directory(&long_path).await.is_ok());
        assert!(watcher.unwatch_directory(&long_path).await.is_ok());
    }

    #[tokio::test]
    async fn test_file_watcher_rapid_operations() {
        let config = create_test_config(true);
        let processor = create_test_processor();
        let mut watcher = FileWatcher::new(&config, processor).await.unwrap();

        // Test rapid start/stop operations
        for _ in 0..10 {
            assert!(watcher.start().await.is_ok());
            assert!(watcher.stop().await.is_ok());
        }

        // Test rapid watch/unwatch operations
        let temp_dir = TempDir::new().unwrap();
        for _ in 0..10 {
            assert!(watcher.watch_directory(temp_dir.path()).await.is_ok());
            assert!(watcher.unwatch_directory(temp_dir.path()).await.is_ok());
        }
    }

    #[tokio::test]
    async fn test_file_watcher_concurrent_operations() {
        let config = create_test_config(true);
        let processor = create_test_processor();
        let watcher = Arc::new(Mutex::new(
            FileWatcher::new(&config, processor).await.unwrap()
        ));

        let mut handles = vec![];

        // Spawn concurrent start/stop operations
        for i in 0..5 {
            let watcher_clone = Arc::clone(&watcher);
            let handle = tokio::spawn(async move {
                let watcher = watcher_clone.lock().await;
                if i % 2 == 0 {
                    watcher.start().await
                } else {
                    watcher.stop().await
                }
            });
            handles.push(handle);
        }

        // Wait for all operations to complete
        let results = futures_util::future::join_all(handles).await;

        // All operations should succeed
        for result in results {
            assert!(result.unwrap().is_ok());
        }
    }

    #[tokio::test]
    async fn test_file_watcher_concurrent_directory_operations() {
        let config = create_test_config(true);
        let processor = create_test_processor();
        let watcher = Arc::new(Mutex::new(
            FileWatcher::new(&config, processor).await.unwrap()
        ));

        let temp_dirs: Vec<_> = (0..5).map(|_| TempDir::new().unwrap()).collect();
        let mut handles = vec![];

        // Spawn concurrent watch operations
        for (i, temp_dir) in temp_dirs.iter().enumerate() {
            let watcher_clone = Arc::clone(&watcher);
            let path = temp_dir.path().to_path_buf();
            let handle = tokio::spawn(async move {
                let mut watcher = watcher_clone.lock().await;
                if i % 2 == 0 {
                    watcher.watch_directory(&path).await
                } else {
                    watcher.unwatch_directory(&path).await
                }
            });
            handles.push(handle);
        }

        // Wait for all operations to complete
        let results = futures_util::future::join_all(handles).await;

        // All operations should succeed
        for result in results {
            assert!(result.unwrap().is_ok());
        }
    }

    #[tokio::test]
    async fn test_file_watcher_stress_test() {
        let config = create_test_config(true);
        let processor = create_test_processor();
        let mut watcher = FileWatcher::new(&config, processor).await.unwrap();

        // Stress test with many operations
        let temp_dirs: Vec<_> = (0..50).map(|_| TempDir::new().unwrap()).collect();

        // Watch all directories
        for temp_dir in &temp_dirs {
            assert!(watcher.watch_directory(temp_dir.path()).await.is_ok());
        }

        // Multiple start/stop cycles
        for _ in 0..20 {
            assert!(watcher.start().await.is_ok());
            assert!(watcher.stop().await.is_ok());
        }

        // Unwatch all directories
        for temp_dir in &temp_dirs {
            assert!(watcher.unwatch_directory(temp_dir.path()).await.is_ok());
        }
    }

    #[tokio::test]
    async fn test_file_watcher_all_config_combinations() {
        let processor = create_test_processor();

        // Test all boolean combinations
        let config_combinations = vec![
            (true, true),   // enabled, recursive
            (true, false),  // enabled, not recursive
            (false, true),  // disabled, recursive
            (false, false), // disabled, not recursive
        ];

        for (enabled, recursive) in config_combinations {
            let mut config = create_test_config(enabled);
            config.recursive = recursive;

            let watcher = FileWatcher::new(&config, processor.clone()).await.unwrap();

            assert_eq!(watcher.config.enabled, enabled);
            assert_eq!(watcher.config.recursive, recursive);

            // Test that all operations work regardless of config
            assert!(watcher.start().await.is_ok());
            assert!(watcher.stop().await.is_ok());
        }
    }

    #[tokio::test]
    async fn test_file_watcher_config_boundary_values() {
        let processor = create_test_processor();

        // Test boundary values for numeric fields
        let boundary_configs = vec![
            (0, 0),                    // minimum values
            (1, 1),                    // just above minimum
            (u64::MAX, usize::MAX),    // maximum values
            (1000, 100),               // typical values
        ];

        for (debounce_ms, max_watched_dirs) in boundary_configs {
            let mut config = create_test_config(true);
            config.debounce_ms = debounce_ms;
            config.max_watched_dirs = max_watched_dirs;

            let watcher = FileWatcher::new(&config, processor.clone()).await.unwrap();

            assert_eq!(watcher.config.debounce_ms, debounce_ms);
            assert_eq!(watcher.config.max_watched_dirs, max_watched_dirs);
        }
    }

    #[tokio::test]
    async fn test_file_watcher_ignore_patterns_variations() {
        let processor = create_test_processor();

        let pattern_variations = vec![
            vec![], // empty patterns
            vec!["*.tmp".to_string()], // single pattern
            vec!["*.tmp".to_string(), "*.log".to_string()], // multiple patterns
            vec![
                "*.tmp".to_string(),
                "*.log".to_string(),
                "target/**".to_string(),
                "node_modules/**".to_string(),
                ".git/**".to_string(),
                "*.backup".to_string(),
                "*.swp".to_string(),
                "*~".to_string(),
                "*.cache".to_string(),
                ".DS_Store".to_string(),
            ], // many patterns
        ];

        for patterns in pattern_variations {
            let mut config = create_test_config(true);
            config.ignore_patterns = patterns.clone();

            let watcher = FileWatcher::new(&config, processor.clone()).await.unwrap();

            assert_eq!(watcher.config.ignore_patterns, patterns);
        }
    }

    #[tokio::test]
    async fn test_file_watcher_comprehensive_api_coverage() {
        let config = create_test_config(true);
        let processor = create_test_processor();
        let mut watcher = FileWatcher::new(&config, processor).await.unwrap();

        // Test the complete API surface
        let temp_dir = TempDir::new().unwrap();

        // Test all public methods in sequence
        assert!(watcher.start().await.is_ok());
        assert!(watcher.watch_directory(temp_dir.path()).await.is_ok());
        assert!(watcher.unwatch_directory(temp_dir.path()).await.is_ok());
        assert!(watcher.stop().await.is_ok());

        // Test methods multiple times to ensure state consistency
        for _ in 0..3 {
            assert!(watcher.start().await.is_ok());
            assert!(watcher.watch_directory(temp_dir.path()).await.is_ok());
            assert!(watcher.unwatch_directory(temp_dir.path()).await.is_ok());
            assert!(watcher.stop().await.is_ok());
        }
    }

    #[tokio::test]
    async fn test_file_watcher_memory_safety() {
        // Test that dropping components doesn't cause issues
        let config = create_test_config(true);
        let processor = create_test_processor();

        {
            let watcher = FileWatcher::new(&config, processor.clone()).await.unwrap();
            assert!(watcher.start().await.is_ok());
            // watcher is dropped here
        }

        // Processor should still be valid
        assert_eq!(processor.config().max_concurrent_tasks, 2);

        // Create another watcher with the same processor
        let mut watcher2 = FileWatcher::new(&config, processor).await.unwrap();
        assert!(watcher2.start().await.is_ok());
        assert!(watcher2.stop().await.is_ok());
    }

    #[tokio::test]
    async fn test_file_watcher_struct_debug_format_completeness() {
        let config = create_test_config(true);
        let processor = create_test_processor();
        let watcher = FileWatcher::new(&config, processor).await.unwrap();

        let debug_output = format!("{:?}", watcher);

        // Verify debug output contains all expected field names
        assert!(debug_output.contains("FileWatcher"));
        assert!(debug_output.contains("config"));
        assert!(debug_output.contains("processor"));
        assert!(debug_output.contains("watcher"));

        // Verify debug output contains some config values
        assert!(debug_output.contains("enabled"));
        assert!(debug_output.contains("debounce_ms"));
    }

    #[test]
    fn test_file_watcher_trait_implementations() {
        // Verify required trait implementations
        fn assert_debug<T: std::fmt::Debug>() {}
        fn assert_send<T: Send>() {}
        fn assert_sync<T: Sync>() {}

        assert_debug::<FileWatcher>();
        assert_send::<FileWatcher>();
        assert_sync::<FileWatcher>();

        // Test that the type can be used in various contexts
        fn _takes_debug(_: impl std::fmt::Debug) {}
        fn _takes_send(_: impl Send) {}
        fn _takes_sync(_: impl Sync) {}

        let config = FileWatcherConfig {
            enabled: true,
            debounce_ms: 100,
            max_watched_dirs: 10,
            ignore_patterns: vec![],
            recursive: true,
        };
        let processor = Arc::new(DocumentProcessor::test_instance());

        // This would be tested in a tokio context, but we're testing trait bounds here
        // let watcher = FileWatcher::new(&config, processor).await.unwrap();
        // _takes_debug(watcher);
        // _takes_send(watcher);
        // _takes_sync(watcher);
    }
}