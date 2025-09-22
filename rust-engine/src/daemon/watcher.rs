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
}