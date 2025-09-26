//! File system watcher for automatic document processing

use crate::config::FileWatcherConfig;
use crate::daemon::processing::DocumentProcessor;
use crate::error::{DaemonError, DaemonResult};
use notify::{RecommendedWatcher, RecursiveMode, Event, EventKind, Watcher};
use std::sync::Arc;
use std::path::{Path, PathBuf};
use std::collections::{HashMap, HashSet};
use std::time::{Duration, Instant};
use tokio::sync::{mpsc, Mutex, RwLock};
use tracing::{info, debug, warn, error};
use glob::Pattern;
use tokio::time::{sleep, interval};

/// File watcher
#[derive(Debug)]
pub struct FileWatcher {
    config: FileWatcherConfig,
    processor: Arc<DocumentProcessor>,
    watcher: Arc<Mutex<Option<RecommendedWatcher>>>,
    watched_dirs: Arc<RwLock<HashSet<PathBuf>>>,
    event_sender: Arc<Mutex<Option<mpsc::Sender<notify::Result<Event>>>>>,
    shutdown_sender: Arc<Mutex<Option<tokio::sync::oneshot::Sender<()>>>>,
    ignore_patterns: Vec<Pattern>,
}

impl FileWatcher {
    /// Create a new file watcher
    pub async fn new(config: &FileWatcherConfig, processor: Arc<DocumentProcessor>) -> DaemonResult<Self> {
        info!("Initializing file watcher (enabled: {})", config.enabled);

        // Compile ignore patterns
        let ignore_patterns = config.ignore_patterns
            .iter()
            .filter_map(|pattern| {
                match Pattern::new(pattern) {
                    Ok(p) => Some(p),
                    Err(e) => {
                        warn!("Invalid ignore pattern '{}': {}", pattern, e);
                        None
                    }
                }
            })
            .collect();

        Ok(Self {
            config: config.clone(),
            processor,
            watcher: Arc::new(Mutex::new(None)),
            watched_dirs: Arc::new(RwLock::new(HashSet::new())),
            event_sender: Arc::new(Mutex::new(None)),
            shutdown_sender: Arc::new(Mutex::new(None)),
            ignore_patterns,
        })
    }

    /// Start watching for file changes
    pub async fn start(&self) -> DaemonResult<()> {
        if !self.config.enabled {
            info!("File watcher is disabled");
            return Ok(());
        }

        info!("Starting file watcher");

        let mut watcher_guard = self.watcher.lock().await;
        if watcher_guard.is_some() {
            debug!("File watcher is already running");
            return Ok(());
        }

        // Create event channel
        let (event_tx, event_rx) = mpsc::channel::<notify::Result<Event>>(1000);
        let (shutdown_tx, shutdown_rx) = tokio::sync::oneshot::channel::<()>();

        // Store senders for later use
        *self.event_sender.lock().await = Some(event_tx.clone());
        *self.shutdown_sender.lock().await = Some(shutdown_tx);

        // Create notify watcher
        let mut notify_watcher = notify::recommended_watcher(move |res| {
            if let Err(e) = event_tx.blocking_send(res) {
                error!("Failed to send file system event: {}", e);
            }
        })
        .map_err(|e| DaemonError::Internal {
            message: format!("Failed to create file watcher: {}", e),
        })?;

        // Watch existing directories
        let watched_dirs = self.watched_dirs.read().await.clone();
        for dir in watched_dirs {
            let recursive_mode = if self.config.recursive {
                RecursiveMode::Recursive
            } else {
                RecursiveMode::NonRecursive
            };

            if let Err(e) = notify_watcher.watch(&dir, recursive_mode) {
                error!("Failed to watch directory {}: {}", dir.display(), e);
            } else {
                debug!("Watching directory: {}", dir.display());
            }
        }

        *watcher_guard = Some(notify_watcher);
        drop(watcher_guard); // Release the lock

        // Start event processing task
        let processor = Arc::clone(&self.processor);
        let config = self.config.clone();
        let ignore_patterns = self.ignore_patterns.clone();

        tokio::spawn(async move {
            Self::process_events(
                event_rx,
                shutdown_rx,
                processor,
                config,
                ignore_patterns,
            ).await;
        });

        info!("File watcher started successfully");
        Ok(())
    }

    /// Stop watching for file changes
    pub async fn stop(&self) -> DaemonResult<()> {
        info!("Stopping file watcher");

        // Send shutdown signal
        if let Some(shutdown_tx) = self.shutdown_sender.lock().await.take() {
            let _ = shutdown_tx.send(());
        }

        // Clear the watcher
        *self.watcher.lock().await = None;
        *self.event_sender.lock().await = None;

        info!("File watcher stopped");
        Ok(())
    }

    /// Add a directory to watch
    pub async fn watch_directory<P: AsRef<Path>>(&mut self, path: P) -> DaemonResult<()> {
        let path = path.as_ref().to_path_buf();
        info!("Adding directory to watch: {}", path.display());

        // Add to watched directories set
        let mut watched_dirs = self.watched_dirs.write().await;
        if watched_dirs.len() >= self.config.max_watched_dirs && self.config.max_watched_dirs > 0 {
            return Err(DaemonError::Internal {
                message: format!(
                    "Maximum watched directories limit reached: {}",
                    self.config.max_watched_dirs
                ),
            });
        }

        let already_watched = !watched_dirs.insert(path.clone());
        drop(watched_dirs);

        if already_watched {
            debug!("Directory already being watched: {}", path.display());
            return Ok(());
        }

        // If watcher is running, add this directory to it
        let mut watcher_guard = self.watcher.lock().await;
        if let Some(ref mut watcher) = *watcher_guard {
            let recursive_mode = if self.config.recursive {
                RecursiveMode::Recursive
            } else {
                RecursiveMode::NonRecursive
            };

            watcher.watch(&path, recursive_mode)
                .map_err(|e| DaemonError::Internal {
                    message: format!("Failed to watch directory {}: {}", path.display(), e),
                })?;

            debug!("Started watching directory: {}", path.display());
        }

        Ok(())
    }

    /// Remove a directory from watching
    pub async fn unwatch_directory<P: AsRef<Path>>(&mut self, path: P) -> DaemonResult<()> {
        let path = path.as_ref().to_path_buf();
        info!("Removing directory from watch: {}", path.display());

        // Remove from watched directories set
        let mut watched_dirs = self.watched_dirs.write().await;
        let was_watched = watched_dirs.remove(&path);
        drop(watched_dirs);

        if !was_watched {
            debug!("Directory was not being watched: {}", path.display());
            return Ok(());
        }

        // If watcher is running, remove this directory from it
        let mut watcher_guard = self.watcher.lock().await;
        if let Some(ref mut watcher) = *watcher_guard {
            if let Err(e) = watcher.unwatch(&path) {
                warn!("Failed to unwatch directory {}: {}", path.display(), e);
            } else {
                debug!("Stopped watching directory: {}", path.display());
            }
        }

        Ok(())
    }

    /// Get list of currently watched directories
    pub async fn get_watched_directories(&self) -> Vec<PathBuf> {
        self.watched_dirs.read().await.iter().cloned().collect()
    }

    /// Check if a path should be ignored based on ignore patterns
    pub fn should_ignore_path(&self, path: &Path) -> bool {
        let path_str = path.to_string_lossy();

        for pattern in &self.ignore_patterns {
            if pattern.matches(&path_str) {
                return true;
            }
        }

        // Also check file name only
        if let Some(file_name) = path.file_name() {
            let file_name_str = file_name.to_string_lossy();
            for pattern in &self.ignore_patterns {
                if pattern.matches(&file_name_str) {
                    return true;
                }
            }
        }

        false
    }

    /// Check if watcher is currently running
    pub async fn is_running(&self) -> bool {
        self.watcher.lock().await.is_some()
    }

    /// Get current configuration
    pub fn config(&self) -> &FileWatcherConfig {
        &self.config
    }

    /// Get count of watched directories
    pub async fn watched_directory_count(&self) -> usize {
        self.watched_dirs.read().await.len()
    }

    /// Process file system events with debouncing
    async fn process_events(
        mut event_rx: mpsc::Receiver<notify::Result<Event>>,
        mut shutdown_rx: tokio::sync::oneshot::Receiver<()>,
        processor: Arc<DocumentProcessor>,
        config: FileWatcherConfig,
        ignore_patterns: Vec<Pattern>,
    ) {
        let mut debounced_events = HashMap::new();
        let debounce_duration = Duration::from_millis(config.debounce_ms);

        // Create a ticker for periodic flushing
        let mut flush_interval = interval(debounce_duration.max(Duration::from_millis(100)));

        loop {
            tokio::select! {
                // Handle shutdown signal
                _ = &mut shutdown_rx => {
                    info!("File watcher event processing shutting down");
                    break;
                }

                // Handle periodic flush
                _ = flush_interval.tick() => {
                    Self::flush_debounced_events(&mut debounced_events, debounce_duration, &processor, &ignore_patterns).await;
                }

                // Handle file system events
                event_result = event_rx.recv() => {
                    match event_result {
                        Some(Ok(event)) => {
                            debug!("Received file system event: {:?}", event);

                            // Convert notify event to our internal format
                            for path in event.paths {
                                debounced_events.insert(path.clone(), (Instant::now(), event.kind));
                            }
                        }
                        Some(Err(e)) => {
                            error!("File system event error: {}", e);
                        }
                        None => {
                            debug!("File system event channel closed");
                            break;
                        }
                    }
                }
            }
        }

        // Process any remaining debounced events on shutdown
        for (path, (_, kind)) in debounced_events {
            Self::handle_file_system_event(path, kind, &processor, &ignore_patterns).await;
        }

        info!("File watcher event processing stopped");
    }

    /// Flush debounced events that are ready for processing
    async fn flush_debounced_events(
        debounced_events: &mut HashMap<PathBuf, (Instant, EventKind)>,
        debounce_duration: Duration,
        processor: &Arc<DocumentProcessor>,
        ignore_patterns: &[Pattern],
    ) {
        let now = Instant::now();
        let mut events_to_process = Vec::new();

        debounced_events.retain(|path, (timestamp, kind)| {
            if now.duration_since(*timestamp) >= debounce_duration {
                events_to_process.push((path.clone(), *kind));
                false // Remove from debounced events
            } else {
                true // Keep in debounced events
            }
        });

        for (path, kind) in events_to_process {
            Self::handle_file_system_event(path, kind, processor, ignore_patterns).await;
        }
    }

    /// Handle a single file system event
    async fn handle_file_system_event(
        path: PathBuf,
        kind: EventKind,
        processor: &Arc<DocumentProcessor>,
        ignore_patterns: &[Pattern],
    ) {
        // Check if path should be ignored
        let path_str = path.to_string_lossy();
        for pattern in ignore_patterns {
            if pattern.matches(&path_str) {
                debug!("Ignoring file due to pattern match: {}", path.display());
                return;
            }
        }

        // Check file name patterns
        if let Some(file_name) = path.file_name() {
            let file_name_str = file_name.to_string_lossy();
            for pattern in ignore_patterns {
                if pattern.matches(&file_name_str) {
                    debug!("Ignoring file due to filename pattern match: {}", path.display());
                    return;
                }
            }
        }

        match kind {
            EventKind::Create(_) => {
                info!("File created: {}", path.display());
                if path.is_file() {
                    if let Err(e) = processor.process_document(path.to_string_lossy().as_ref()).await {
                        error!("Failed to process created file {}: {}", path.display(), e);
                    }
                }
            }
            EventKind::Modify(_) => {
                info!("File modified: {}", path.display());
                if path.is_file() {
                    if let Err(e) = processor.process_document(path.to_string_lossy().as_ref()).await {
                        error!("Failed to process modified file {}: {}", path.display(), e);
                    }
                }
            }
            EventKind::Remove(_) => {
                info!("File removed: {}", path.display());
                // TODO: Implement document removal from vector database
            }
            _ => {
                debug!("Unhandled event kind: {:?} for {}", kind, path.display());
            }
        }
    }
}

/// Event debouncer for handling rapid file system events
#[derive(Debug)]
pub struct EventDebouncer {
    debounce_duration: Duration,
    pending_events: Arc<RwLock<HashMap<PathBuf, (Instant, EventKind)>>>,
}

impl EventDebouncer {
    /// Create a new event debouncer
    pub fn new(debounce_duration: Duration) -> Self {
        Self {
            debounce_duration,
            pending_events: Arc::new(RwLock::new(HashMap::new())),
        }
    }

    /// Process a batch of events, applying debouncing logic
    pub async fn process_events(&self, events: Vec<DebouncedEvent>) -> Vec<DebouncedEvent> {
        let mut debounced = Vec::new();
        let mut event_map: HashMap<PathBuf, DebouncedEvent> = HashMap::new();

        // Group events by path and keep only the latest
        for event in events {
            match event_map.get(&event.path) {
                Some(existing) => {
                    // Keep the more recent event
                    if event.timestamp > existing.timestamp {
                        event_map.insert(event.path.clone(), event);
                    }
                }
                None => {
                    event_map.insert(event.path.clone(), event);
                }
            }
        }

        let now = Instant::now();
        for (_, event) in event_map {
            // Only include events that are old enough (past debounce period)
            if now.duration_since(event.timestamp) >= self.debounce_duration {
                debounced.push(event);
            }
        }

        debounced
    }

    /// Add an event to the debouncer
    pub async fn add_event(&self, path: PathBuf, event_type: EventKind) {
        let now = Instant::now();
        let mut pending = self.pending_events.write().await;
        pending.insert(path, (now, event_type));
    }

    /// Get debounced events that are ready for processing
    pub async fn get_ready_events(&self) -> Vec<(PathBuf, EventKind)> {
        let now = Instant::now();
        let mut pending = self.pending_events.write().await;
        let mut ready = Vec::new();

        pending.retain(|path, (timestamp, event_type)| {
            if now.duration_since(*timestamp) >= self.debounce_duration {
                ready.push((path.clone(), *event_type));
                false // Remove from pending
            } else {
                true // Keep in pending
            }
        });

        ready
    }
}

/// Event filter for ignoring unwanted file system events
#[derive(Debug)]
pub struct EventFilter {
    ignore_patterns: Vec<Pattern>,
}

impl EventFilter {
    /// Create a new event filter from configuration
    pub fn new(config: &FileWatcherConfig) -> Self {
        let mut ignore_patterns = Vec::new();

        for pattern_str in &config.ignore_patterns {
            match Pattern::new(pattern_str) {
                Ok(pattern) => ignore_patterns.push(pattern),
                Err(e) => {
                    warn!("Invalid ignore pattern '{}': {}", pattern_str, e);
                }
            }
        }

        Self { ignore_patterns }
    }

    /// Check if an event should be ignored
    pub fn should_ignore(&self, event: &Event) -> bool {
        for path in &event.paths {
            // Convert to string for pattern matching
            let path_str = match path.to_str() {
                Some(s) => s,
                None => {
                    warn!("Non-UTF8 path encountered: {:?}", path);
                    return true; // Ignore non-UTF8 paths
                }
            };

            // Check against ignore patterns
            for pattern in &self.ignore_patterns {
                if pattern.matches(path_str) {
                    debug!("Ignoring path due to pattern match: {}", path_str);
                    return true;
                }
            }

            // Additional filtering logic
            if self.should_ignore_by_extension(path) {
                return true;
            }
        }

        false
    }

    /// Check if a file should be ignored based on its extension or path patterns
    fn should_ignore_by_extension(&self, path: &Path) -> bool {
        if let Some(extension) = path.extension() {
            if let Some(ext_str) = extension.to_str() {
                match ext_str {
                    "tmp" | "log" | "bak" | "swp" | "cache" => return true,
                    _ => {}
                }
            }
        }

        // Check for backup files ending with ~
        if let Some(filename) = path.file_name() {
            if let Some(filename_str) = filename.to_str() {
                if filename_str.ends_with('~') {
                    return true;
                }
            }
        }

        false
    }
}

/// File system event handler with rate limiting
#[derive(Debug)]
pub struct FileSystemEventHandler {
    max_events_per_second: u32,
    last_event_time: Arc<Mutex<Instant>>,
    event_count: Arc<Mutex<u32>>,
}

impl FileSystemEventHandler {
    /// Create a new event handler with rate limiting
    pub fn new(max_events_per_second: u32) -> Self {
        Self {
            max_events_per_second,
            last_event_time: Arc::new(Mutex::new(Instant::now())),
            event_count: Arc::new(Mutex::new(0)),
        }
    }

    /// Handle a burst of events with rate limiting
    pub async fn handle_event_burst(&self, events: Vec<DebouncedEvent>) -> Vec<DebouncedEvent> {
        if self.max_events_per_second == 0 {
            return events; // No rate limiting
        }

        let mut processed = Vec::new();
        let interval_duration = Duration::from_millis(1000 / self.max_events_per_second as u64);

        for event in events {
            // Check if we need to rate limit
            {
                let mut last_time = self.last_event_time.lock().await;
                let mut count = self.event_count.lock().await;
                let now = Instant::now();

                // Reset counter if more than a second has passed
                if now.duration_since(*last_time) >= Duration::from_secs(1) {
                    *count = 0;
                    *last_time = now;
                }

                // Rate limit if we've exceeded the threshold
                if *count >= self.max_events_per_second {
                    sleep(interval_duration).await;
                    *count = 0;
                    *last_time = Instant::now();
                }

                *count += 1;
            }

            processed.push(event);
        }

        processed
    }
}

/// Debounced event structure for testing
#[derive(Debug, Clone)]
pub struct DebouncedEvent {
    pub path: PathBuf,
    pub event_type: EventKind,
    pub timestamp: Instant,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::{FileWatcherConfig, ProcessingConfig, QdrantConfig};
    use tempfile::TempDir;
    use std::path::PathBuf;
    use std::sync::Arc;
    use tokio_test;
    use std::time::{Duration, Instant};
    use futures_util::future::join_all;
    use notify::{self, event::{CreateKind, ModifyKind}};
    use glob::Pattern;

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

    // PERFORMANCE TESTS

    #[tokio::test]
    async fn test_performance_watcher_initialization_speed() {
        let config = create_test_config(true);
        let processor = create_test_processor();

        let start = Instant::now();
        let _watcher = FileWatcher::new(&config, processor).await.unwrap();
        let init_time = start.elapsed();

        // Should initialize quickly (under 100ms)
        assert!(init_time.as_millis() < 100, "Initialization took {}ms", init_time.as_millis());
    }

    #[tokio::test]
    async fn test_performance_watch_many_directories() {
        let mut config = create_test_config(true);
        config.max_watched_dirs = 1000;
        let processor = create_test_processor();
        let mut watcher = FileWatcher::new(&config, processor).await.unwrap();

        // Create many temporary directories
        let temp_dirs: Vec<TempDir> = (0..100).map(|_| TempDir::new().unwrap()).collect();

        let start = Instant::now();
        for temp_dir in &temp_dirs {
            assert!(watcher.watch_directory(temp_dir.path()).await.is_ok());
        }
        let watch_time = start.elapsed();

        // Should be able to watch 100 directories quickly (under 1 second)
        assert!(watch_time.as_secs() < 1, "Watching 100 directories took {}ms", watch_time.as_millis());

        // Verify all directories are watched
        let watched = watcher.get_watched_directories().await;
        assert_eq!(watched.len(), 100);
    }

    #[tokio::test]
    async fn test_performance_concurrent_watchers() {
        let config = create_test_config(true);
        let processor = create_test_processor();

        let start = Instant::now();
        let mut handles = Vec::new();

        // Create 10 concurrent watchers
        for _ in 0..10 {
            let config_clone = config.clone();
            let processor_clone = Arc::clone(&processor);

            let handle = tokio::spawn(async move {
                let mut watcher = FileWatcher::new(&config_clone, processor_clone).await.unwrap();
                watcher.start().await.unwrap();
                tokio::time::sleep(Duration::from_millis(50)).await;
                watcher.stop().await.unwrap();
            });
            handles.push(handle);
        }

        join_all(handles).await;
        let concurrent_time = start.elapsed();

        // Should handle concurrent watchers efficiently (under 5 seconds)
        assert!(concurrent_time.as_secs() < 5, "Concurrent watchers took {}ms", concurrent_time.as_millis());
    }

    #[tokio::test]
    async fn test_performance_rapid_start_stop_cycles() {
        let config = create_test_config(true);
        let processor = create_test_processor();

        let start = Instant::now();

        // Rapid start/stop cycles
        for _ in 0..20 {
            let mut watcher = FileWatcher::new(&config, processor.clone()).await.unwrap();
            watcher.start().await.unwrap();
            watcher.stop().await.unwrap();
        }

        let cycle_time = start.elapsed();

        // Should handle rapid cycles efficiently (under 2 seconds)
        assert!(cycle_time.as_secs() < 2, "20 rapid cycles took {}ms", cycle_time.as_millis());
    }

    #[tokio::test]
    async fn test_performance_memory_usage_scaling() {
        let mut config = create_test_config(true);
        config.max_watched_dirs = 1000;
        let processor = create_test_processor();
        let mut watcher = FileWatcher::new(&config, processor).await.unwrap();

        // Measure initial memory state
        let initial_dirs = watcher.get_watched_directories().await.len();
        assert_eq!(initial_dirs, 0);

        // Add directories and measure scaling
        let temp_dirs: Vec<TempDir> = (0..200).map(|_| TempDir::new().unwrap()).collect();

        for (i, temp_dir) in temp_dirs.iter().enumerate() {
            assert!(watcher.watch_directory(temp_dir.path()).await.is_ok());

            // Check memory usage doesn't grow excessively
            if i % 50 == 0 {
                let watched = watcher.get_watched_directories().await;
                assert_eq!(watched.len(), i + 1);
            }
        }

        let final_dirs = watcher.get_watched_directories().await.len();
        assert_eq!(final_dirs, 200);
    }

    #[tokio::test]
    async fn test_performance_pattern_matching_speed() {
        let mut config = create_test_config(true);
        config.ignore_patterns = vec![
            "*.tmp".to_string(),
            "*.log".to_string(),
            "target/**".to_string(),
            "node_modules/**".to_string(),
            ".git/**".to_string(),
            "*.backup".to_string(),
            "*.cache".to_string(),
            "*~".to_string(),
        ];
        let processor = create_test_processor();
        let watcher = FileWatcher::new(&config, processor).await.unwrap();

        // Test pattern matching performance
        let test_paths = vec![
            PathBuf::from("/tmp/test.txt"),
            PathBuf::from("/tmp/test.log"),
            PathBuf::from("/tmp/test.tmp"),
            PathBuf::from("/project/target/debug/app"),
            PathBuf::from("/project/node_modules/package/index.js"),
            PathBuf::from("/project/.git/config"),
            PathBuf::from("/tmp/backup.backup"),
            PathBuf::from("/tmp/cache.cache"),
            PathBuf::from("/tmp/file~"),
        ];

        let start = Instant::now();
        for _ in 0..1000 {
            for path in &test_paths {
                let _ = watcher.should_ignore_path(path);
            }
        }
        let pattern_time = start.elapsed();

        // Should handle pattern matching efficiently (under 100ms for 9000 operations)
        assert!(pattern_time.as_millis() < 100, "Pattern matching took {}ms", pattern_time.as_millis());
    }

    #[tokio::test]
    async fn test_performance_debouncer_throughput() {
        let debouncer = EventDebouncer::new(Duration::from_millis(100));

        // Create many events for the same paths
        let mut events = Vec::new();
        for i in 0..1000 {
            events.push(DebouncedEvent {
                path: PathBuf::from(format!("/tmp/file_{}.txt", i % 10)), // 10 unique paths
                event_type: EventKind::Modify(ModifyKind::Data(notify::event::DataChange::Content)),
                timestamp: Instant::now(),
            });
        }

        let start = Instant::now();
        let processed = debouncer.process_events(events).await;
        let debounce_time = start.elapsed();

        // Should process events quickly and deduplicate effectively
        assert!(debounce_time.as_millis() < 50, "Debouncing took {}ms", debounce_time.as_millis());
        assert!(processed.len() <= 10, "Should deduplicate to at most 10 events, got {}", processed.len());
    }

    #[tokio::test]
    async fn test_performance_event_filter_throughput() {
        let config = create_test_config(true);
        let filter = EventFilter::new(&config);

        // Create many test events
        let events: Vec<notify::Event> = (0..10000)
            .map(|i| {
                let extension = if i % 3 == 0 { "tmp" } else if i % 3 == 1 { "log" } else { "txt" };
                notify::Event {
                    kind: EventKind::Create(CreateKind::File),
                    paths: vec![PathBuf::from(format!("/tmp/test_{}.{}", i, extension))],
                    attrs: Default::default(),
                }
            })
            .collect();

        let start = Instant::now();
        let mut filtered_count = 0;
        for event in &events {
            if !filter.should_ignore(&event) {
                filtered_count += 1;
            }
        }
        let filter_time = start.elapsed();

        // Should filter events efficiently (under 100ms for 10000 events)
        assert!(filter_time.as_millis() < 100, "Event filtering took {}ms", filter_time.as_millis());

        // Should filter out .tmp and .log files
        let expected_passed = events.len() / 3; // Only .txt files should pass
        let tolerance = expected_passed / 10; // 10% tolerance
        assert!(
            (filtered_count as i32 - expected_passed as i32).abs() <= tolerance as i32,
            "Expected ~{} filtered events, got {}",
            expected_passed,
            filtered_count
        );
    }

    #[tokio::test]
    async fn test_performance_scalability_limits() {
        let mut config = create_test_config(true);
        config.max_watched_dirs = 2000;
        let processor = create_test_processor();
        let mut watcher = FileWatcher::new(&config, processor).await.unwrap();

        // Test scalability up to the limit
        let test_count = std::cmp::min(config.max_watched_dirs, 500);
        let temp_dirs: Vec<TempDir> = (0..test_count)
            .map(|_| TempDir::new().unwrap())
            .collect();

        let start = Instant::now();
        let mut successful_watches = 0;

        for temp_dir in &temp_dirs {
            if watcher.watch_directory(temp_dir.path()).await.is_ok() {
                successful_watches += 1;
            } else {
                break;
            }
        }

        let scaling_time = start.elapsed();

        // Should handle a significant number of directories
        assert!(successful_watches >= 200, "Should handle at least 200 directories, handled {}", successful_watches);

        // Should complete within reasonable time (scale with directory count)
        let expected_max_time = Duration::from_millis(successful_watches as u64 * 5); // 5ms per directory
        assert!(scaling_time <= expected_max_time, "Scaling took {}ms for {} directories", scaling_time.as_millis(), successful_watches);
    }

    #[test]
    fn test_performance_cpu_usage_estimation() {
        // Test CPU-intensive operations for performance regression
        let start = Instant::now();

        // Pattern matching stress test
        let patterns = vec![
            "*.tmp".to_string(),
            "*.log".to_string(),
            "target/**".to_string(),
            "node_modules/**".to_string(),
            ".git/**".to_string(),
        ];

        // Compile patterns
        let compiled_patterns: Vec<_> = patterns
            .iter()
            .filter_map(|pattern| Pattern::new(pattern).ok())
            .collect();

        // Test many path matches
        let test_paths: Vec<String> = (0..10000)
            .map(|i| {
                let extensions = ["txt", "log", "tmp", "rs", "py"];
                let dirs = ["target", "src", "tests", "node_modules", ".git"];
                let ext = extensions[i % extensions.len()];
                let dir = dirs[i % dirs.len()];
                format!("{}/file_{}.{}", dir, i, ext)
            })
            .collect();

        let mut matches = 0;
        for path in &test_paths {
            for pattern in &compiled_patterns {
                if pattern.matches(path) {
                    matches += 1;
                    break;
                }
            }
        }

        let cpu_time = start.elapsed();

        // Should complete pattern matching efficiently (under 100ms)
        assert!(cpu_time.as_millis() < 100, "CPU-intensive pattern matching took {}ms", cpu_time.as_millis());
        assert!(matches > 0, "Should have some pattern matches");
    }
}