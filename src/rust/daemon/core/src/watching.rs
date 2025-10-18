//! File watching system
//!
//! Cross-platform file watching with event debouncing, pattern matching, and priority-based processing.
//! Integrates with the task pipeline system for responsive file processing.

use std::collections::{HashMap, HashSet, VecDeque};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::{Duration, Instant, SystemTime};
use std::num::NonZeroUsize;
use tokio::sync::{mpsc, RwLock, Mutex};
use tokio::time::interval;
use notify::{Watcher as NotifyWatcher, RecursiveMode, Event, EventKind};
use walkdir::WalkDir;
use glob::{Pattern, PatternError};
use serde::{Deserialize, Serialize};
use thiserror::Error;
use lru::LruCache;

use crate::processing::{TaskSubmitter, TaskPriority, TaskSource, TaskPayload};

pub mod platform;
pub use platform::{PlatformWatcherConfig, PlatformWatcherFactory, PlatformWatchingStats};

/// Errors that can occur during file watching
#[derive(Error, Debug)]
pub enum WatchingError {
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
    
    #[error("Notify watcher error: {0}")]
    Notify(#[from] notify::Error),
    
    #[error("Pattern compilation error: {0}")]
    Pattern(#[from] PatternError),
    
    #[error("Configuration error: {message}")]
    Config { message: String },
    
    #[error("Task submission error: {0}")]
    TaskSubmission(String),
}

/// File watching configuration with comprehensive options
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WatcherConfig {
    /// Patterns to include (glob patterns)
    pub include_patterns: Vec<String>,
    
    /// Patterns to exclude (glob patterns) 
    pub exclude_patterns: Vec<String>,
    
    /// Whether to watch directories recursively
    pub recursive: bool,
    
    /// Maximum recursion depth (-1 for unlimited)
    pub max_depth: i32,
    
    /// Debounce time in milliseconds (minimum time between events for the same file)
    pub debounce_ms: u64,
    
    /// Polling interval in milliseconds (for polling-based watching)
    pub polling_interval_ms: u64,
    
    /// Maximum number of events to queue before dropping
    pub max_queue_size: usize,
    
    /// Priority for tasks generated from file watching
    pub task_priority: TaskPriority,
    
    /// Collection name for processed documents
    pub default_collection: String,
    
    /// Whether to process existing files on startup
    pub process_existing: bool,
    
    /// File size limit in bytes (files larger than this are ignored)
    pub max_file_size: Option<u64>,
    
    /// Whether to use polling mode (useful for network drives)
    pub use_polling: bool,

    /// Batch processing settings
    pub batch_processing: BatchConfig,

    /// Maximum number of events to store in debouncer (memory limit)
    pub max_debouncer_capacity: usize,

    /// Maximum total events to store in batcher (memory limit)
    pub max_batcher_capacity: usize,
}

/// Configuration for batch processing of file events
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BatchConfig {
    /// Enable batch processing
    pub enabled: bool,
    
    /// Maximum batch size
    pub max_batch_size: usize,
    
    /// Maximum time to wait for batch to fill (in milliseconds)
    pub max_batch_wait_ms: u64,
    
    /// Whether to group batches by file type
    pub group_by_type: bool,
}

impl Default for WatcherConfig {
    fn default() -> Self {
        Self {
            include_patterns: vec![
                "*.txt".to_string(),
                "*.md".to_string(), 
                "*.pdf".to_string(),
                "*.epub".to_string(),
                "*.docx".to_string(),
                "*.py".to_string(),
                "*.rs".to_string(),
                "*.js".to_string(),
                "*.ts".to_string(),
                "*.json".to_string(),
                "*.yaml".to_string(),
                "*.yml".to_string(),
                "*.toml".to_string(),
            ],
            exclude_patterns: vec![
                "*.tmp".to_string(),
                "*.swp".to_string(),
                "*.bak".to_string(),
                "*~".to_string(),
                ".git/**".to_string(),
                ".svn/**".to_string(),
                "node_modules/**".to_string(),
                "target/**".to_string(),
                "__pycache__/**".to_string(),
                ".pytest_cache/**".to_string(),
                ".DS_Store".to_string(),
                "Thumbs.db".to_string(),
            ],
            recursive: true,
            max_depth: -1,
            debounce_ms: 1000, // 1 second debounce
            polling_interval_ms: 1000,
            max_queue_size: 10000,
            task_priority: TaskPriority::BackgroundWatching,
            default_collection: "documents".to_string(),
            process_existing: false,
            max_file_size: Some(100 * 1024 * 1024), // 100MB limit
            use_polling: false,
            batch_processing: BatchConfig {
                enabled: true,
                max_batch_size: 10,
                max_batch_wait_ms: 5000, // 5 seconds
                group_by_type: true,
            },
            max_debouncer_capacity: 10_000, // 10K events max in debouncer
            max_batcher_capacity: 5_000,    // 5K events max in batcher
        }
    }
}

/// File event with metadata and debouncing information
#[derive(Debug, Clone)]
pub struct FileEvent {
    pub path: PathBuf,
    pub event_kind: EventKind,
    pub timestamp: Instant,
    pub system_time: SystemTime,
    pub size: Option<u64>,
    pub metadata: HashMap<String, String>,
}

/// Compiled patterns for efficient matching
#[derive(Debug)]
struct CompiledPatterns {
    include: Vec<Pattern>,
    exclude: Vec<Pattern>,
}

impl CompiledPatterns {
    fn new(config: &WatcherConfig) -> Result<Self, WatchingError> {
        let include = config.include_patterns
            .iter()
            .map(|p| Pattern::new(p))
            .collect::<Result<Vec<_>, _>>()?;
            
        let exclude = config.exclude_patterns
            .iter()
            .map(|p| Pattern::new(p))
            .collect::<Result<Vec<_>, _>>()?;
            
        Ok(Self { include, exclude })
    }
    
    fn should_process(&self, path: &Path) -> bool {
        let path_str = path.to_string_lossy();
        
        // Check exclude patterns first (more specific)
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

/// Event debouncer to prevent duplicate processing with bounded memory
#[derive(Debug)]
struct EventDebouncer {
    events: LruCache<PathBuf, FileEvent>,
    debounce_duration: Duration,
    evictions: u64,
}

impl EventDebouncer {
    fn new(debounce_ms: u64, capacity: usize) -> Self {
        Self {
            events: LruCache::new(NonZeroUsize::new(capacity).unwrap_or(NonZeroUsize::new(10_000).unwrap())),
            debounce_duration: Duration::from_millis(debounce_ms),
            evictions: 0,
        }
    }
    
    /// Add event to debouncer, returns true if event should be processed immediately
    fn add_event(&mut self, event: FileEvent) -> bool {
        let now = Instant::now();
        let path = event.path.clone();

        if let Some(existing) = self.events.get(&path) {
            // If the existing event is within debounce period, update and don't process
            if now.duration_since(existing.timestamp) < self.debounce_duration {
                // Use push to update and move to front of LRU
                if self.events.push(path, event).is_some() {
                    self.evictions += 1;
                }
                return false;
            }
        }

        // Insert new event, track eviction if cache was full
        if self.events.push(path, event).is_some() {
            self.evictions += 1;
        }
        true
    }
    
    /// Get events that are ready to be processed (past debounce period)
    fn get_ready_events(&mut self) -> Vec<FileEvent> {
        let now = Instant::now();
        let mut ready = Vec::new();
        let mut to_remove = Vec::new();

        // Collect paths and events that are ready
        // Note: iter() doesn't exist on LruCache, so we need to use a different approach
        // We'll peek at entries without removing them first
        for (path, event) in self.events.iter() {
            if now.duration_since(event.timestamp) >= self.debounce_duration {
                ready.push(event.clone());
                to_remove.push(path.clone());
            }
        }

        // Remove ready events from cache
        for path in to_remove {
            self.events.pop(&path);
        }

        ready
    }
    
    /// Clear old events (cleanup)
    fn cleanup(&mut self, max_age: Duration) {
        let now = Instant::now();
        let mut to_remove = Vec::new();

        // Collect paths of old events
        for (path, event) in self.events.iter() {
            if now.duration_since(event.timestamp) >= max_age {
                to_remove.push(path.clone());
            }
        }

        // Remove old events
        for path in to_remove {
            self.events.pop(&path);
        }
    }

    /// Get eviction count
    fn eviction_count(&self) -> u64 {
        self.evictions
    }
}

/// Batch processor for grouping and processing file events efficiently with bounded memory
#[derive(Debug)]
struct EventBatcher {
    batches: HashMap<String, VecDeque<FileEvent>>,
    config: BatchConfig,
    last_flush: Instant,
    max_total_capacity: usize,
    current_total_size: usize,
    evictions: u64,
}

impl EventBatcher {
    fn new(config: BatchConfig, max_total_capacity: usize) -> Self {
        Self {
            batches: HashMap::new(),
            config,
            last_flush: Instant::now(),
            max_total_capacity,
            current_total_size: 0,
            evictions: 0,
        }
    }
    
    fn add_event(&mut self, event: FileEvent) -> Option<Vec<FileEvent>> {
        if !self.config.enabled {
            return Some(vec![event]);
        }

        // Check if we're at capacity - evict oldest event if needed
        if self.current_total_size >= self.max_total_capacity {
            self.evict_oldest_event();
        }

        let key = if self.config.group_by_type {
            event.path.extension()
                .and_then(|ext| ext.to_str())
                .unwrap_or("unknown")
                .to_string()
        } else {
            "default".to_string()
        };

        let batch = self.batches.entry(key).or_insert_with(VecDeque::new);
        batch.push_back(event);
        self.current_total_size += 1;

        // Check if batch is full
        if batch.len() >= self.config.max_batch_size {
            let count = batch.len();
            let events = batch.drain(..).collect();
            self.current_total_size -= count;
            return Some(events);
        }

        // Check if max wait time has elapsed
        let now = Instant::now();
        if now.duration_since(self.last_flush) >= Duration::from_millis(self.config.max_batch_wait_ms) {
            return self.flush_all();
        }

        None
    }
    
    fn flush_all(&mut self) -> Option<Vec<FileEvent>> {
        self.last_flush = Instant::now();

        let mut all_events = Vec::new();
        for batch in self.batches.values_mut() {
            let count = batch.len();
            all_events.extend(batch.drain(..));
            self.current_total_size -= count;
        }

        if all_events.is_empty() {
            None
        } else {
            Some(all_events)
        }
    }

    /// Evict the oldest event from the oldest batch to make room
    fn evict_oldest_event(&mut self) {
        // Find the batch with the oldest event (earliest timestamp)
        let mut oldest_key: Option<String> = None;
        let mut oldest_time = Instant::now();

        for (key, batch) in &self.batches {
            if let Some(event) = batch.front() {
                if event.timestamp < oldest_time {
                    oldest_time = event.timestamp;
                    oldest_key = Some(key.clone());
                }
            }
        }

        // Remove oldest event
        if let Some(key) = oldest_key {
            if let Some(batch) = self.batches.get_mut(&key) {
                if batch.pop_front().is_some() {
                    self.current_total_size -= 1;
                    self.evictions += 1;
                    tracing::debug!("Evicted oldest event from batch '{}' (total evictions: {})", key, self.evictions);

                    // Remove empty batches to free memory
                    if batch.is_empty() {
                        self.batches.remove(&key);
                    }
                }
            }
        }
    }

    /// Get eviction count
    fn eviction_count(&self) -> u64 {
        self.evictions
    }
}

/// Statistics for file watching operations
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WatchingStats {
    pub events_received: u64,
    pub events_processed: u64,
    pub events_debounced: u64,
    pub events_filtered: u64,
    pub tasks_submitted: u64,
    pub errors: u64,
    pub uptime_seconds: u64,
    pub watched_paths: usize,
    pub current_queue_size: usize,
    pub debouncer_evictions: u64,
    pub batcher_evictions: u64,
}

impl Default for WatchingStats {
    fn default() -> Self {
        Self {
            events_received: 0,
            events_processed: 0,
            events_debounced: 0,
            events_filtered: 0,
            tasks_submitted: 0,
            errors: 0,
            uptime_seconds: 0,
            watched_paths: 0,
            current_queue_size: 0,
            debouncer_evictions: 0,
            batcher_evictions: 0,
        }
    }
}

/// Main file watcher implementation with cross-platform support
pub struct FileWatcher {
    /// Configuration for the watcher
    config: Arc<RwLock<WatcherConfig>>,
    
    /// Compiled patterns for efficient matching
    patterns: Arc<RwLock<CompiledPatterns>>,
    
    /// Task submitter for processing pipeline integration
    task_submitter: TaskSubmitter,
    
    /// Event debouncer to prevent duplicate processing
    debouncer: Arc<Mutex<EventDebouncer>>,
    
    /// Event batcher for efficient processing
    batcher: Arc<Mutex<EventBatcher>>,
    
    /// File system watcher
    watcher: Arc<Mutex<Option<Box<dyn NotifyWatcher + Send + Sync>>>>,
    
    /// Channel for receiving file system events
    event_receiver: Arc<Mutex<Option<mpsc::UnboundedReceiver<FileEvent>>>>,
    
    /// Handle to the event processing task
    processor_handle: Arc<Mutex<Option<tokio::task::JoinHandle<()>>>>,
    
    /// Statistics
    stats: Arc<Mutex<WatchingStats>>,
    
    /// Start time for uptime calculation
    start_time: Instant,
    
    /// Currently watched paths
    watched_paths: Arc<RwLock<HashSet<PathBuf>>>,
}

impl FileWatcher {
    /// Create a new file watcher with the given configuration and task submitter
    pub fn new(config: WatcherConfig, task_submitter: TaskSubmitter) -> Result<Self, WatchingError> {
        let patterns = CompiledPatterns::new(&config)?;
        let debouncer = EventDebouncer::new(config.debounce_ms, config.max_debouncer_capacity);
        let batcher = EventBatcher::new(config.batch_processing.clone(), config.max_batcher_capacity);
        
        Ok(Self {
            config: Arc::new(RwLock::new(config)),
            patterns: Arc::new(RwLock::new(patterns)),
            task_submitter,
            debouncer: Arc::new(Mutex::new(debouncer)),
            batcher: Arc::new(Mutex::new(batcher)),
            watcher: Arc::new(Mutex::new(None)),
            event_receiver: Arc::new(Mutex::new(None)),
            processor_handle: Arc::new(Mutex::new(None)),
            stats: Arc::new(Mutex::new(WatchingStats::default())),
            start_time: Instant::now(),
            watched_paths: Arc::new(RwLock::new(HashSet::new())),
        })
    }
    
    /// Start watching the specified path
    pub async fn watch_path(&self, path: &Path) -> Result<(), WatchingError> {
        let config = self.config.read().await;
        
        // Validate path
        if !path.exists() {
            return Err(WatchingError::Config { 
                message: format!("Path does not exist: {}", path.display()) 
            });
        }
        
        // Create file system watcher
        let (tx, rx) = mpsc::unbounded_channel();
        let tx_clone = tx.clone();
        
        let watcher: Box<dyn NotifyWatcher + Send + Sync> = if config.use_polling {
            let config_interval = Duration::from_millis(config.polling_interval_ms);
            Box::new(
                notify::PollWatcher::new(
                    move |result| {
                        if let Ok(event) = result {
                            Self::handle_notify_event(event, &tx_clone);
                        }
                    },
                    notify::Config::default().with_poll_interval(config_interval)
                )?
            )
        } else {
            Box::new(
                notify::RecommendedWatcher::new(
                    move |result| {
                        if let Ok(event) = result {
                            Self::handle_notify_event(event, &tx_clone);
                        }
                    },
                    notify::Config::default()
                )?
            )
        };
        
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
                w.watch(path, recursive_mode)?;
            }
        }
        
        // Store the receiver
        {
            let mut receiver_lock = self.event_receiver.lock().await;
            *receiver_lock = Some(rx);
        }
        
        // Add to watched paths
        {
            let mut watched_paths = self.watched_paths.write().await;
            watched_paths.insert(path.to_path_buf());
        }
        
        // Start the event processor task
        self.start_event_processor().await?;
        
        // Process existing files if configured
        if config.process_existing {
            self.process_existing_files(path).await?;
        }
        
        tracing::info!("Started watching path: {} (recursive: {})", path.display(), config.recursive);
        Ok(())
    }
    
    /// Stop watching all paths
    pub async fn stop_watching(&self) -> Result<(), WatchingError> {
        // Stop the event processor
        {
            let mut handle_lock = self.processor_handle.lock().await;
            if let Some(handle) = handle_lock.take() {
                handle.abort();
            }
        }
        
        // Clear the watcher
        {
            let mut watcher_lock = self.watcher.lock().await;
            *watcher_lock = None;
        }
        
        // Clear the receiver
        {
            let mut receiver_lock = self.event_receiver.lock().await;
            *receiver_lock = None;
        }
        
        // Clear watched paths
        {
            let mut watched_paths = self.watched_paths.write().await;
            watched_paths.clear();
        }
        
        tracing::info!("Stopped file watching");
        Ok(())
    }
    
    /// Update configuration (requires restart to take effect for some settings)
    pub async fn update_config(&self, new_config: WatcherConfig) -> Result<(), WatchingError> {
        let new_patterns = CompiledPatterns::new(&new_config)?;
        
        {
            let mut config_lock = self.config.write().await;
            *config_lock = new_config.clone();
        }
        
        {
            let mut patterns_lock = self.patterns.write().await;
            *patterns_lock = new_patterns;
        }
        
        // Update debouncer and batcher
        {
            let mut debouncer_lock = self.debouncer.lock().await;
            *debouncer_lock = EventDebouncer::new(new_config.debounce_ms, new_config.max_debouncer_capacity);
        }

        {
            let mut batcher_lock = self.batcher.lock().await;
            *batcher_lock = EventBatcher::new(new_config.batch_processing, new_config.max_batcher_capacity);
        }
        
        tracing::info!("Updated file watcher configuration");
        Ok(())
    }
    
    /// Get current statistics
    pub async fn stats(&self) -> WatchingStats {
        let mut stats = self.stats.lock().await.clone();
        stats.uptime_seconds = self.start_time.elapsed().as_secs();

        {
            let watched_paths = self.watched_paths.read().await;
            stats.watched_paths = watched_paths.len();
        }

        // Add debouncer evictions
        {
            let debouncer_lock = self.debouncer.lock().await;
            stats.debouncer_evictions = debouncer_lock.eviction_count();
        }

        // Add batcher evictions
        {
            let batcher_lock = self.batcher.lock().await;
            stats.batcher_evictions = batcher_lock.eviction_count();
        }

        stats
    }
    
    /// Get currently watched paths
    pub async fn watched_paths(&self) -> Vec<PathBuf> {
        let watched_paths = self.watched_paths.read().await;
        watched_paths.iter().cloned().collect()
    }
    
    /// Handle a notify event and convert it to our internal event format
    fn handle_notify_event(event: Event, tx: &mpsc::UnboundedSender<FileEvent>) {
        let now = Instant::now();
        let system_time = SystemTime::now();
        
        for path in event.paths {
            // Get file size if possible
            let size = std::fs::metadata(&path)
                .ok()
                .map(|metadata| metadata.len());
            
            // Create metadata
            let mut metadata = HashMap::new();
            metadata.insert("event_type".to_string(), format!("{:?}", event.kind));
            
            if let Some(file_name) = path.file_name().and_then(|n| n.to_str()) {
                metadata.insert("file_name".to_string(), file_name.to_string());
            }
            
            if let Some(extension) = path.extension().and_then(|e| e.to_str()) {
                metadata.insert("file_extension".to_string(), extension.to_string());
            }
            
            let file_event = FileEvent {
                path,
                event_kind: event.kind,
                timestamp: now,
                system_time,
                size,
                metadata,
            };
            
            if let Err(e) = tx.send(file_event) {
                tracing::error!("Failed to send file event: {}", e);
            }
        }
    }
    
    /// Start the event processing task
    async fn start_event_processor(&self) -> Result<(), WatchingError> {
        // Don't start if already running
        {
            let handle_lock = self.processor_handle.lock().await;
            if handle_lock.is_some() {
                return Ok(());
            }
        }
        
        let event_receiver = self.event_receiver.clone();
        let debouncer = self.debouncer.clone();
        let batcher = self.batcher.clone();
        let patterns = self.patterns.clone();
        let config = self.config.clone();
        let task_submitter = self.task_submitter.clone();
        let stats = self.stats.clone();
        
        let handle = tokio::spawn(async move {
            Self::event_processing_loop(
                event_receiver,
                debouncer, 
                batcher,
                patterns,
                config,
                task_submitter,
                stats
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
        batcher: Arc<Mutex<EventBatcher>>,
        patterns: Arc<RwLock<CompiledPatterns>>,
        config: Arc<RwLock<WatcherConfig>>,
        task_submitter: TaskSubmitter,
        stats: Arc<Mutex<WatchingStats>>,
    ) {
        let mut cleanup_interval = interval(Duration::from_secs(300)); // 5 minute cleanup
        let mut debounce_interval = interval(Duration::from_millis(500)); // Check debounced events
        
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
                            &batcher,
                            &patterns,
                            &config,
                            &task_submitter,
                            &stats
                        ).await;
                    } else {
                        // Channel closed, exit loop
                        break;
                    }
                },
                
                // Process debounced events
                _ = debounce_interval.tick() => {
                    Self::process_debounced_events(
                        &debouncer,
                        &batcher,
                        &patterns,
                        &config,
                        &task_submitter,
                        &stats
                    ).await;
                },
                
                // Cleanup old events
                _ = cleanup_interval.tick() => {
                    Self::cleanup_old_events(&debouncer).await;
                },
            }
        }
        
        tracing::info!("Event processing loop stopped");
    }
    
    /// Process a single file event
    async fn process_file_event(
        event: FileEvent,
        debouncer: &Arc<Mutex<EventDebouncer>>,
        batcher: &Arc<Mutex<EventBatcher>>,
        patterns: &Arc<RwLock<CompiledPatterns>>,
        config: &Arc<RwLock<WatcherConfig>>,
        task_submitter: &TaskSubmitter,
        stats: &Arc<Mutex<WatchingStats>>,
    ) {
        // Update stats
        {
            let mut stats_lock = stats.lock().await;
            stats_lock.events_received += 1;
        }
        
        // Check if we should process this file based on patterns
        {
            let patterns_lock = patterns.read().await;
            if !patterns_lock.should_process(&event.path) {
                let mut stats_lock = stats.lock().await;
                stats_lock.events_filtered += 1;
                return;
            }
        }
        
        // Check file size limit
        {
            let config_lock = config.read().await;
            if let (Some(max_size), Some(file_size)) = (config_lock.max_file_size, event.size) {
                if file_size > max_size {
                    tracing::debug!("Skipping large file: {} ({} bytes)", event.path.display(), file_size);
                    let mut stats_lock = stats.lock().await;
                    stats_lock.events_filtered += 1;
                    return;
                }
            }
        }
        
        // Add to debouncer
        let should_process = {
            let mut debouncer_lock = debouncer.lock().await;
            debouncer_lock.add_event(event.clone())
        };
        
        if should_process {
            Self::handle_ready_event(event, batcher, config, task_submitter, stats).await;
        } else {
            let mut stats_lock = stats.lock().await;
            stats_lock.events_debounced += 1;
        }
    }
    
    /// Process events that are ready after debouncing
    async fn process_debounced_events(
        debouncer: &Arc<Mutex<EventDebouncer>>,
        batcher: &Arc<Mutex<EventBatcher>>,
        patterns: &Arc<RwLock<CompiledPatterns>>,
        config: &Arc<RwLock<WatcherConfig>>,
        task_submitter: &TaskSubmitter,
        stats: &Arc<Mutex<WatchingStats>>,
    ) {
        let ready_events = {
            let mut debouncer_lock = debouncer.lock().await;
            debouncer_lock.get_ready_events()
        };
        
        for event in ready_events {
            // Double-check patterns (they might have changed)
            {
                let patterns_lock = patterns.read().await;
                if !patterns_lock.should_process(&event.path) {
                    continue;
                }
            }
            
            Self::handle_ready_event(event, batcher, config, task_submitter, stats).await;
        }
    }
    
    /// Handle an event that's ready for processing
    async fn handle_ready_event(
        event: FileEvent,
        batcher: &Arc<Mutex<EventBatcher>>,
        config: &Arc<RwLock<WatcherConfig>>,
        task_submitter: &TaskSubmitter,
        stats: &Arc<Mutex<WatchingStats>>,
    ) {
        // Add to batcher
        let ready_batch = {
            let mut batcher_lock = batcher.lock().await;
            batcher_lock.add_event(event)
        };
        
        if let Some(batch) = ready_batch {
            Self::submit_processing_tasks(batch, config, task_submitter, stats).await;
        }
    }
    
    /// Submit processing tasks for a batch of events
    async fn submit_processing_tasks(
        events: Vec<FileEvent>,
        config: &Arc<RwLock<WatcherConfig>>,
        task_submitter: &TaskSubmitter,
        stats: &Arc<Mutex<WatchingStats>>,
    ) {
        let config_lock = config.read().await;
        let task_priority = config_lock.task_priority;
        let default_collection = config_lock.default_collection.clone();
        drop(config_lock);
        
        for event in events {
            // Only process create/modify events, ignore deletes
            match event.event_kind {
                EventKind::Create(_) | EventKind::Modify(_) => {
                    // Check if file still exists (it might have been deleted quickly)
                    if !event.path.exists() {
                        continue;
                    }
                    
                    // Check if it's a file (not directory)
                    if !event.path.is_file() {
                        continue;
                    }
                    
                    let source = match task_priority {
                        TaskPriority::ProjectWatching => TaskSource::ProjectWatcher {
                            project_path: event.path.parent()
                                .unwrap_or_else(|| Path::new("/"))
                                .to_string_lossy()
                                .to_string(),
                        },
                        TaskPriority::BackgroundWatching => TaskSource::BackgroundWatcher {
                            folder_path: event.path.parent()
                                .unwrap_or_else(|| Path::new("/"))
                                .to_string_lossy()
                                .to_string(),
                        },
                        _ => TaskSource::Generic {
                            operation: "file_watching".to_string(),
                        },
                    };
                    
                    let payload = TaskPayload::ProcessDocument {
                        file_path: event.path.clone(),
                        collection: default_collection.clone(),
                    };
                    
                    match task_submitter.submit_task(task_priority, source, payload, None).await {
                        Ok(_) => {
                            let mut stats_lock = stats.lock().await;
                            stats_lock.tasks_submitted += 1;
                            stats_lock.events_processed += 1;
                            tracing::debug!("Submitted processing task for: {}", event.path.display());
                        },
                        Err(e) => {
                            let mut stats_lock = stats.lock().await;
                            stats_lock.errors += 1;
                            tracing::error!("Failed to submit processing task for {}: {}", event.path.display(), e);
                        }
                    }
                },
                _ => {
                    // Ignore other event types (delete, rename, etc.)
                }
            }
        }
    }
    
    /// Process existing files in a directory (for initial scan)
    async fn process_existing_files(&self, root_path: &Path) -> Result<(), WatchingError> {
        let config = self.config.read().await;
        let patterns = self.patterns.read().await;
        
        let max_depth = if config.max_depth < 0 {
            usize::MAX
        } else {
            config.max_depth as usize
        };
        
        let walker = if config.recursive {
            WalkDir::new(root_path).max_depth(max_depth)
        } else {
            WalkDir::new(root_path).max_depth(1)
        };
        
        let mut file_count = 0;
        let start_time = Instant::now();
        
        for entry in walker {
            match entry {
                Ok(entry) => {
                    let path = entry.path();
                    
                    // Skip directories
                    if !path.is_file() {
                        continue;
                    }
                    
                    // Check patterns
                    if !patterns.should_process(path) {
                        continue;
                    }
                    
                    // Check file size
                    if let Some(max_size) = config.max_file_size {
                        if let Ok(metadata) = path.metadata() {
                            if metadata.len() > max_size {
                                continue;
                            }
                        }
                    }
                    
                    // Create a synthetic file event
                    let event = FileEvent {
                        path: path.to_path_buf(),
                        event_kind: EventKind::Create(notify::event::CreateKind::File),
                        timestamp: Instant::now(),
                        system_time: SystemTime::now(),
                        size: path.metadata().ok().map(|m| m.len()),
                        metadata: HashMap::new(),
                    };
                    
                    // Process directly (skip debouncing for initial scan)
                    Self::handle_ready_event(
                        event, 
                        &self.batcher, 
                        &self.config, 
                        &self.task_submitter, 
                        &self.stats
                    ).await;
                    
                    file_count += 1;
                    
                    // Prevent blocking the async runtime for too long
                    if file_count % 100 == 0 {
                        tokio::task::yield_now().await;
                    }
                },
                Err(e) => {
                    tracing::warn!("Error walking directory {}: {}", root_path.display(), e);
                }
            }
        }
        
        // Flush any remaining batched events
        {
            let ready_batch = {
                let mut batcher_lock = self.batcher.lock().await;
                batcher_lock.flush_all()
            };
            
            if let Some(batch) = ready_batch {
                Self::submit_processing_tasks(batch, &self.config, &self.task_submitter, &self.stats).await;
            }
        }
        
        let elapsed = start_time.elapsed();
        tracing::info!(
            "Initial file scan complete: {} files found in {:?}",
            file_count,
            elapsed
        );
        
        Ok(())
    }
    
    /// Clean up old events from debouncer
    async fn cleanup_old_events(debouncer: &Arc<Mutex<EventDebouncer>>) {
        let mut debouncer_lock = debouncer.lock().await;
        debouncer_lock.cleanup(Duration::from_secs(3600)); // 1 hour max age
    }
}


/// Example usage of the FileWatcher integrated with ProcessingEngine
/// 
/// ```rust,no_run
/// use std::path::Path;
/// use workspace_qdrant_core::{ProcessingEngine, watching::{FileWatcher, WatcherConfig}};
/// use workspace_qdrant_core::processing::TaskPriority;
/// 
/// #[tokio::main]
/// async fn main() -> Result<(), Box<dyn std::error::Error>> {
///     // Create processing engine
///     let mut engine = ProcessingEngine::new();
///     engine.start().await?;
///     let task_submitter = engine.task_submitter();
///     
///     // Configure file watcher
///     let config = WatcherConfig {
///         include_patterns: vec!["*.pdf".to_string(), "*.txt".to_string(), "*.md".to_string()],
///         exclude_patterns: vec!["*.tmp".to_string(), ".git/**".to_string()],
///         recursive: true,
///         max_depth: 5,
///         debounce_ms: 2000, // 2 second debounce
///         task_priority: TaskPriority::BackgroundWatching,
///         default_collection: "documents".to_string(),
///         process_existing: true, // Process existing files on startup
///         max_file_size: Some(50 * 1024 * 1024), // 50MB limit
///         batch_processing: workspace_qdrant_core::watching::BatchConfig {
///             enabled: true,
///             max_batch_size: 10,
///             max_batch_wait_ms: 5000,
///             group_by_type: true,
///         },
///         ..Default::default()
///     };
///     
///     // Create and start file watcher
///     let watcher = FileWatcher::new(config, task_submitter)?;
///     watcher.watch_path(Path::new("/path/to/documents")).await?;
///     
///     println!("File watcher started. Monitoring for changes...");
///     
///     // Monitor statistics
///     loop {
///         tokio::time::sleep(std::time::Duration::from_secs(30)).await;
///         let stats = watcher.stats().await;
///         println!(
///             "Stats: {} events received, {} tasks submitted, {} errors",
///             stats.events_received, stats.tasks_submitted, stats.errors
///         );
///     }
/// }
/// ```

// Comprehensive test module (extracted to separate file)
#[cfg(test)]
mod tests;
