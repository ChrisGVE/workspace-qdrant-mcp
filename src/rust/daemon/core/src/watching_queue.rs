//! File Watching with SQLite Queue Integration
//!
//! This module provides Rust-based file watching that writes directly to the
//! ingestion_queue SQLite table, replacing Python file watching while maintaining
//! compatibility with Python queue processors.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::{Duration, SystemTime};
use chrono::{DateTime, Utc};
use notify::{Event, EventKind, RecursiveMode, Watcher as NotifyWatcher};
use sqlx::SqlitePool;
use tokio::sync::{mpsc, RwLock, Mutex};
use tokio::time::interval;
use tracing::{debug, error, info, warn};
use thiserror::Error;
use glob::Pattern;

use crate::queue_operations::{QueueManager, QueueOperation, QueueError};

/// File watching errors
#[derive(Error, Debug)]
pub enum WatchingQueueError {
    #[error("Notify watcher error: {0}")]
    Notify(#[from] notify::Error),

    #[error("Queue error: {0}")]
    Queue(#[from] QueueError),

    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    #[error("Configuration error: {message}")]
    Config { message: String },

    #[error("Git error: {0}")]
    Git(String),

    #[error("Database error: {0}")]
    Database(#[from] sqlx::Error),
}

pub type WatchingQueueResult<T> = Result<T, WatchingQueueError>;

/// Watch configuration from database
#[derive(Debug, Clone)]
pub struct WatchConfig {
    pub id: String,
    pub path: PathBuf,
    pub collection: String,
    pub patterns: Vec<String>,
    pub ignore_patterns: Vec<String>,
    pub recursive: bool,
    pub debounce_ms: u64,
    pub enabled: bool,
}

/// Compiled patterns for efficient matching
#[derive(Debug)]
struct CompiledPatterns {
    include: Vec<Pattern>,
    exclude: Vec<Pattern>,
}

impl CompiledPatterns {
    fn new(config: &WatchConfig) -> Result<Self, WatchingQueueError> {
        let include = config.patterns
            .iter()
            .map(|p| Pattern::new(p))
            .collect::<Result<Vec<_>, _>>()
            .map_err(|e| WatchingQueueError::Config {
                message: format!("Invalid include pattern: {}", e)
            })?;

        let exclude = config.ignore_patterns
            .iter()
            .map(|p| Pattern::new(p))
            .collect::<Result<Vec<_>, _>>()
            .map_err(|e| WatchingQueueError::Config {
                message: format!("Invalid exclude pattern: {}", e)
            })?;

        Ok(Self { include, exclude })
    }

    fn should_process(&self, path: &Path) -> bool {
        let path_str = path.to_string_lossy();

        // Check exclude patterns first
        for pattern in &self.exclude {
            if pattern.matches(&path_str) {
                return false;
            }
        }

        // If no include patterns, allow all
        if self.include.is_empty() {
            return true;
        }

        // Check include patterns
        for pattern in &self.include {
            if pattern.matches(&path_str) {
                return true;
            }
        }

        false
    }
}

/// File event with metadata
#[derive(Debug, Clone)]
struct FileEvent {
    path: PathBuf,
    event_kind: EventKind,
    timestamp: SystemTime,
}

/// Event debouncer to prevent duplicate processing
#[derive(Debug)]
struct EventDebouncer {
    events: HashMap<PathBuf, FileEvent>,
    debounce_duration: Duration,
}

impl EventDebouncer {
    fn new(debounce_ms: u64) -> Self {
        Self {
            events: HashMap::new(),
            debounce_duration: Duration::from_millis(debounce_ms),
        }
    }

    /// Add event, returns true if should process immediately
    fn add_event(&mut self, event: FileEvent) -> bool {
        let now = SystemTime::now();

        if let Some(existing) = self.events.get(&event.path) {
            if let Ok(elapsed) = now.duration_since(existing.timestamp) {
                if elapsed < self.debounce_duration {
                    self.events.insert(event.path.clone(), event);
                    return false;
                }
            }
        }

        self.events.insert(event.path.clone(), event);
        true
    }

    /// Get events ready to process
    fn get_ready_events(&mut self) -> Vec<FileEvent> {
        let now = SystemTime::now();
        let mut ready = Vec::new();
        let mut to_remove = Vec::new();

        for (path, event) in &self.events {
            if let Ok(elapsed) = now.duration_since(event.timestamp) {
                if elapsed >= self.debounce_duration {
                    ready.push(event.clone());
                    to_remove.push(path.clone());
                }
            }
        }

        for path in to_remove {
            self.events.remove(&path);
        }

        ready
    }
}

/// Main file watcher with queue integration
pub struct FileWatcherQueue {
    config: Arc<RwLock<WatchConfig>>,
    patterns: Arc<RwLock<CompiledPatterns>>,
    queue_manager: Arc<QueueManager>,
    debouncer: Arc<Mutex<EventDebouncer>>,
    watcher: Arc<Mutex<Option<Box<dyn NotifyWatcher + Send + Sync>>>>,
    event_receiver: Arc<Mutex<Option<mpsc::UnboundedReceiver<FileEvent>>>>,
    processor_handle: Arc<Mutex<Option<tokio::task::JoinHandle<()>>>>,

    // Statistics
    events_received: Arc<Mutex<u64>>,
    events_processed: Arc<Mutex<u64>>,
    events_filtered: Arc<Mutex<u64>>,
    queue_errors: Arc<Mutex<u64>>,
}

impl FileWatcherQueue {
    /// Create a new file watcher with queue integration
    pub fn new(
        config: WatchConfig,
        queue_manager: Arc<QueueManager>,
    ) -> WatchingQueueResult<Self> {
        let patterns = CompiledPatterns::new(&config)?;
        let debouncer = EventDebouncer::new(config.debounce_ms);

        Ok(Self {
            config: Arc::new(RwLock::new(config)),
            patterns: Arc::new(RwLock::new(patterns)),
            queue_manager,
            debouncer: Arc::new(Mutex::new(debouncer)),
            watcher: Arc::new(Mutex::new(None)),
            event_receiver: Arc::new(Mutex::new(None)),
            processor_handle: Arc::new(Mutex::new(None)),
            events_received: Arc::new(Mutex::new(0)),
            events_processed: Arc::new(Mutex::new(0)),
            events_filtered: Arc::new(Mutex::new(0)),
            queue_errors: Arc::new(Mutex::new(0)),
        })
    }

    /// Start watching the configured path
    pub async fn start(&self) -> WatchingQueueResult<()> {
        let config = self.config.read().await;

        // Validate path exists
        if !config.path.exists() {
            return Err(WatchingQueueError::Config {
                message: format!("Path does not exist: {}", config.path.display()),
            });
        }

        // Create file system watcher
        let (tx, rx) = mpsc::unbounded_channel();
        let tx_clone = tx.clone();

        let watcher: Box<dyn NotifyWatcher + Send + Sync> = Box::new(
            notify::RecommendedWatcher::new(
                move |result| {
                    if let Ok(event) = result {
                        Self::handle_notify_event(event, &tx_clone);
                    }
                },
                notify::Config::default(),
            )?
        );

        // Add path to watcher
        let recursive_mode = if config.recursive {
            RecursiveMode::Recursive
        } else {
            RecursiveMode::NonRecursive
        };

        {
            let mut watcher_lock = self.watcher.lock().await;
            *watcher_lock = Some(watcher);
            if let Some(ref mut w) = *watcher_lock {
                w.watch(&config.path, recursive_mode)?;
            }
        }

        // Store receiver
        {
            let mut receiver_lock = self.event_receiver.lock().await;
            *receiver_lock = Some(rx);
        }

        // Start event processor
        self.start_event_processor().await?;

        info!("Started file watcher: {} (recursive: {})",
            config.path.display(), config.recursive);

        Ok(())
    }

    /// Stop watching
    pub async fn stop(&self) -> WatchingQueueResult<()> {
        // Stop processor
        {
            let mut handle_lock = self.processor_handle.lock().await;
            if let Some(handle) = handle_lock.take() {
                handle.abort();
            }
        }

        // Clear watcher
        {
            let mut watcher_lock = self.watcher.lock().await;
            *watcher_lock = None;
        }

        // Clear receiver
        {
            let mut receiver_lock = self.event_receiver.lock().await;
            *receiver_lock = None;
        }

        info!("Stopped file watcher");
        Ok(())
    }

    /// Handle notify event
    fn handle_notify_event(event: Event, tx: &mpsc::UnboundedSender<FileEvent>) {
        let timestamp = SystemTime::now();

        for path in event.paths {
            let file_event = FileEvent {
                path,
                event_kind: event.kind,
                timestamp,
            };

            if let Err(e) = tx.send(file_event) {
                error!("Failed to send file event: {}", e);
            }
        }
    }

    /// Start event processing task
    async fn start_event_processor(&self) -> WatchingQueueResult<()> {
        {
            let handle_lock = self.processor_handle.lock().await;
            if handle_lock.is_some() {
                return Ok(());
            }
        }

        let event_receiver = self.event_receiver.clone();
        let debouncer = self.debouncer.clone();
        let patterns = self.patterns.clone();
        let config = self.config.clone();
        let queue_manager = self.queue_manager.clone();
        let events_received = self.events_received.clone();
        let events_processed = self.events_processed.clone();
        let events_filtered = self.events_filtered.clone();
        let queue_errors = self.queue_errors.clone();

        let handle = tokio::spawn(async move {
            Self::event_processing_loop(
                event_receiver,
                debouncer,
                patterns,
                config,
                queue_manager,
                events_received,
                events_processed,
                events_filtered,
                queue_errors,
            ).await;
        });

        {
            let mut handle_lock = self.processor_handle.lock().await;
            *handle_lock = Some(handle);
        }

        Ok(())
    }

    /// Main event processing loop
    async fn event_processing_loop(
        event_receiver: Arc<Mutex<Option<mpsc::UnboundedReceiver<FileEvent>>>>,
        debouncer: Arc<Mutex<EventDebouncer>>,
        patterns: Arc<RwLock<CompiledPatterns>>,
        config: Arc<RwLock<WatchConfig>>,
        queue_manager: Arc<QueueManager>,
        events_received: Arc<Mutex<u64>>,
        events_processed: Arc<Mutex<u64>>,
        events_filtered: Arc<Mutex<u64>>,
        queue_errors: Arc<Mutex<u64>>,
    ) {
        let mut debounce_interval = interval(Duration::from_millis(500));

        loop {
            tokio::select! {
                // Handle incoming events
                event = async {
                    let mut receiver_lock = event_receiver.lock().await;
                    if let Some(ref mut receiver) = *receiver_lock {
                        receiver.recv().await
                    } else {
                        None
                    }
                } => {
                    if let Some(event) = event {
                        Self::process_file_event(
                            event,
                            &debouncer,
                            &patterns,
                            &config,
                            &queue_manager,
                            &events_received,
                            &events_processed,
                            &events_filtered,
                            &queue_errors,
                        ).await;
                    } else {
                        break;
                    }
                },

                // Process debounced events
                _ = debounce_interval.tick() => {
                    Self::process_debounced_events(
                        &debouncer,
                        &patterns,
                        &config,
                        &queue_manager,
                        &events_processed,
                        &queue_errors,
                    ).await;
                },
            }
        }

        info!("Event processing loop stopped");
    }

    /// Process a single file event
    async fn process_file_event(
        event: FileEvent,
        debouncer: &Arc<Mutex<EventDebouncer>>,
        patterns: &Arc<RwLock<CompiledPatterns>>,
        config: &Arc<RwLock<WatchConfig>>,
        queue_manager: &Arc<QueueManager>,
        events_received: &Arc<Mutex<u64>>,
        events_processed: &Arc<Mutex<u64>>,
        events_filtered: &Arc<Mutex<u64>>,
        queue_errors: &Arc<Mutex<u64>>,
    ) {
        // Update stats
        {
            let mut count = events_received.lock().await;
            *count += 1;
        }

        // Check patterns
        {
            let patterns_lock = patterns.read().await;
            if !patterns_lock.should_process(&event.path) {
                let mut count = events_filtered.lock().await;
                *count += 1;
                return;
            }
        }

        // Add to debouncer
        let should_process = {
            let mut debouncer_lock = debouncer.lock().await;
            debouncer_lock.add_event(event.clone())
        };

        if should_process {
            Self::enqueue_file_operation(
                event,
                config,
                queue_manager,
                events_processed,
                queue_errors,
            ).await;
        }
    }

    /// Process debounced events
    async fn process_debounced_events(
        debouncer: &Arc<Mutex<EventDebouncer>>,
        patterns: &Arc<RwLock<CompiledPatterns>>,
        config: &Arc<RwLock<WatchConfig>>,
        queue_manager: &Arc<QueueManager>,
        events_processed: &Arc<Mutex<u64>>,
        queue_errors: &Arc<Mutex<u64>>,
    ) {
        let ready_events = {
            let mut debouncer_lock = debouncer.lock().await;
            debouncer_lock.get_ready_events()
        };

        for event in ready_events {
            // Double-check patterns
            {
                let patterns_lock = patterns.read().await;
                if !patterns_lock.should_process(&event.path) {
                    continue;
                }
            }

            Self::enqueue_file_operation(
                event,
                config,
                queue_manager,
                events_processed,
                queue_errors,
            ).await;
        }
    }

    /// Determine operation type based on event and file state
    fn determine_operation_type(event_kind: EventKind, file_path: &Path) -> QueueOperation {
        match event_kind {
            EventKind::Create(_) => QueueOperation::Ingest,
            EventKind::Remove(_) => QueueOperation::Delete,
            EventKind::Modify(_) => {
                // Check if file still exists
                if file_path.exists() {
                    QueueOperation::Update
                } else {
                    // Race condition: file deleted during debounce
                    QueueOperation::Delete
                }
            },
            _ => QueueOperation::Update,  // Default to update for other events
        }
    }

    /// Calculate priority based on operation type
    fn calculate_priority(operation: QueueOperation) -> i32 {
        match operation {
            QueueOperation::Delete => 8,  // High priority for deletions
            QueueOperation::Update => 5,  // Normal priority for updates
            QueueOperation::Ingest => 5,  // Normal priority for ingestion
        }
    }

    /// Find project root by looking for .git directory
    fn find_project_root(file_path: &Path) -> PathBuf {
        let mut current = file_path.parent().unwrap_or(file_path);

        while current != current.parent().unwrap_or(Path::new("/")) {
            if current.join(".git").exists() {
                return current.to_path_buf();
            }
            current = current.parent().unwrap_or(Path::new("/"));
        }

        // Fallback to file's parent directory
        file_path.parent().unwrap_or(file_path).to_path_buf()
    }

    /// Calculate tenant ID from project root
    async fn calculate_tenant_id(project_root: &Path) -> String {
        // Try to get git remote URL
        if let Ok(output) = tokio::process::Command::new("git")
            .args(&["remote", "get-url", "origin"])
            .current_dir(project_root)
            .output()
            .await
        {
            if output.status.success() {
                if let Ok(remote_url) = String::from_utf8(output.stdout) {
                    let remote_url = remote_url.trim();
                    // Sanitize the URL
                    let sanitized = remote_url
                        .trim_start_matches("https://")
                        .trim_start_matches("http://")
                        .trim_start_matches("git@")
                        .trim_start_matches("ssh://")
                        .replace([':', '/', '.'], "_")
                        .trim_end_matches("_git")
                        .to_lowercase();

                    return sanitized;
                }
            }
        }

        // Fallback: hash of project path
        let path_str = project_root.to_string_lossy();
        let hash = format!("{:x}", md5::compute(path_str.as_bytes()));
        format!("path_{}", &hash[..16])
    }

    /// Get current git branch
    async fn get_current_branch(project_root: &Path) -> String {
        if let Ok(output) = tokio::process::Command::new("git")
            .args(&["rev-parse", "--abbrev-ref", "HEAD"])
            .current_dir(project_root)
            .output()
            .await
        {
            if output.status.success() {
                if let Ok(branch) = String::from_utf8(output.stdout) {
                    return branch.trim().to_string();
                }
            }
        }

        "main".to_string()
    }

    /// Enqueue file operation with retry logic
    async fn enqueue_file_operation(
        event: FileEvent,
        config: &Arc<RwLock<WatchConfig>>,
        queue_manager: &Arc<QueueManager>,
        events_processed: &Arc<Mutex<u64>>,
        queue_errors: &Arc<Mutex<u64>>,
    ) {
        // Skip if not a file
        if !event.path.is_file() && event.event_kind != EventKind::Remove(_) {
            return;
        }

        // Determine operation type
        let operation = Self::determine_operation_type(event.event_kind, &event.path);

        // Calculate priority
        let priority = Self::calculate_priority(operation);

        // Find project root and calculate tenant ID
        let project_root = Self::find_project_root(&event.path);
        let tenant_id = Self::calculate_tenant_id(&project_root).await;
        let branch = Self::get_current_branch(&project_root).await;

        // Get collection from config
        let collection = {
            let config_lock = config.read().await;
            config_lock.collection.clone()
        };

        // Get absolute path
        let file_absolute_path = event.path.to_string_lossy().to_string();

        // Retry logic with exponential backoff
        const MAX_RETRIES: u32 = 3;
        const RETRY_DELAYS_MS: [u64; 3] = [500, 1000, 2000];

        for attempt in 0..MAX_RETRIES {
            match queue_manager.enqueue_file(
                &file_absolute_path,
                &collection,
                &tenant_id,
                &branch,
                operation,
                priority,
                None,
            ).await {
                Ok(_) => {
                    let mut count = events_processed.lock().await;
                    *count += 1;

                    debug!(
                        "Enqueued file: {} (operation={:?}, priority={}, tenant={}, branch={})",
                        file_absolute_path, operation, priority, tenant_id, branch
                    );
                    return;
                },
                Err(QueueError::Database(ref e)) if attempt < MAX_RETRIES - 1 => {
                    // Retry on database errors with backoff
                    let delay = RETRY_DELAYS_MS[attempt as usize];
                    warn!(
                        "Database error enqueueing {}: {}. Retrying in {}ms (attempt {}/{})",
                        file_absolute_path, e, delay, attempt + 1, MAX_RETRIES
                    );
                    tokio::time::sleep(Duration::from_millis(delay)).await;
                },
                Err(e) => {
                    let mut count = queue_errors.lock().await;
                    *count += 1;

                    error!(
                        "Failed to enqueue file {}: {} (attempt {}/{})",
                        file_absolute_path, e, attempt + 1, MAX_RETRIES
                    );
                    return;
                }
            }
        }

        // All retries failed
        let mut count = queue_errors.lock().await;
        *count += 1;
        error!(
            "Failed to enqueue file {} after {} retries",
            file_absolute_path, MAX_RETRIES
        );
    }

    /// Get current statistics
    pub async fn get_stats(&self) -> WatchingQueueStats {
        WatchingQueueStats {
            events_received: *self.events_received.lock().await,
            events_processed: *self.events_processed.lock().await,
            events_filtered: *self.events_filtered.lock().await,
            queue_errors: *self.queue_errors.lock().await,
        }
    }
}

/// Watching statistics
#[derive(Debug, Clone)]
pub struct WatchingQueueStats {
    pub events_received: u64,
    pub events_processed: u64,
    pub events_filtered: u64,
    pub queue_errors: u64,
}

/// Watch manager for multiple watchers
pub struct WatchManager {
    pool: SqlitePool,
    watchers: Arc<RwLock<HashMap<String, Arc<FileWatcherQueue>>>>,
}

impl WatchManager {
    /// Create a new watch manager
    pub fn new(pool: SqlitePool) -> Self {
        Self {
            pool,
            watchers: Arc::new(RwLock::new(HashMap::new())),
        }
    }

    /// Load watch configurations from database and start watchers
    pub async fn start_all_watches(&self) -> WatchingQueueResult<()> {
        let queue_manager = Arc::new(QueueManager::new(self.pool.clone()));

        // Query watch configurations
        let rows = sqlx::query(
            r#"
            SELECT id, path, collection, patterns, ignore_patterns,
                   recursive, debounce_seconds, status
            FROM watch_folders
            WHERE enabled = TRUE
            "#
        )
        .fetch_all(&self.pool)
        .await?;

        for row in rows {
            let id: String = row.get("id");
            let path: String = row.get("path");
            let collection: String = row.get("collection");
            let patterns_json: String = row.get("patterns");
            let ignore_patterns_json: String = row.get("ignore_patterns");
            let recursive: bool = row.get("recursive");
            let debounce_seconds: i64 = row.get("debounce_seconds");

            let patterns: Vec<String> = serde_json::from_str(&patterns_json)
                .unwrap_or_else(|_| vec!["*".to_string()]);
            let ignore_patterns: Vec<String> = serde_json::from_str(&ignore_patterns_json)
                .unwrap_or_else(|_| Vec::new());

            let config = WatchConfig {
                id: id.clone(),
                path: PathBuf::from(path),
                collection,
                patterns,
                ignore_patterns,
                recursive,
                debounce_ms: (debounce_seconds as u64) * 1000,
                enabled: true,
            };

            let watcher = Arc::new(FileWatcherQueue::new(config, queue_manager.clone())?);

            match watcher.start().await {
                Ok(_) => {
                    info!("Started watcher: {}", id);
                    let mut watchers = self.watchers.write().await;
                    watchers.insert(id, watcher);
                },
                Err(e) => {
                    error!("Failed to start watcher {}: {}", id, e);
                }
            }
        }

        Ok(())
    }

    /// Stop all watchers
    pub async fn stop_all_watches(&self) -> WatchingQueueResult<()> {
        let watchers = self.watchers.read().await;

        for (id, watcher) in watchers.iter() {
            if let Err(e) = watcher.stop().await {
                error!("Failed to stop watcher {}: {}", id, e);
            }
        }

        Ok(())
    }

    /// Get statistics for all watchers
    pub async fn get_all_stats(&self) -> HashMap<String, WatchingQueueStats> {
        let watchers = self.watchers.read().await;
        let mut stats = HashMap::new();

        for (id, watcher) in watchers.iter() {
            stats.insert(id.clone(), watcher.get_stats().await);
        }

        stats
    }
}
