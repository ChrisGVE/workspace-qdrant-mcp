//! Main FileWatcher implementation with cross-platform support
//!
//! The static event processing methods are in `watcher_processing.rs`.

use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant, SystemTime};

use notify::{EventKind, RecursiveMode, Watcher as NotifyWatcher};
use tokio::sync::{mpsc, Mutex, RwLock};
use walkdir::WalkDir;

use crate::processing::TaskSubmitter;

use super::compiled_patterns::CompiledPatterns;
use super::config::WatcherConfig;
use super::debouncer::{EventBatcher, EventDebouncer};
use super::events::{FileEvent, PausedEventBuffer};
use super::telemetry::{TelemetryTracker, WatchingStats};
use super::WatchingError;

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

    /// Global pause state (atomic for fast checking in event loop)
    is_paused: Arc<AtomicBool>,

    /// Buffer for events received while paused
    paused_event_buffer: Arc<Mutex<PausedEventBuffer>>,

    /// Telemetry tracking
    telemetry_tracker: Arc<Mutex<TelemetryTracker>>,
}

impl FileWatcher {
    /// Create a new file watcher with the given configuration and task submitter
    pub fn new(
        config: WatcherConfig,
        task_submitter: TaskSubmitter,
    ) -> Result<Self, WatchingError> {
        let patterns = CompiledPatterns::new(&config)?;
        let debouncer = EventDebouncer::new(config.debounce_ms, config.max_debouncer_capacity);
        let batcher =
            EventBatcher::new(config.batch_processing.clone(), config.max_batcher_capacity);

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
            is_paused: Arc::new(AtomicBool::new(false)),
            paused_event_buffer: Arc::new(Mutex::new(PausedEventBuffer::new())),
            telemetry_tracker: Arc::new(Mutex::new(TelemetryTracker::new())),
        })
    }

    /// Start watching the specified path
    pub async fn watch_path(&self, path: &Path) -> Result<(), WatchingError> {
        {
            let mut config = self.config.write().await;
            config.validate_polling_interval();
        }

        let config = self.config.read().await;

        if !path.exists() {
            return Err(WatchingError::Config {
                message: format!("Path does not exist: {}", path.display()),
            });
        }

        let (tx, rx) = mpsc::unbounded_channel();
        let tx_clone = tx.clone();

        let watcher: Box<dyn NotifyWatcher + Send + Sync> = if config.use_polling {
            let config_interval = Duration::from_millis(config.polling_interval_ms);
            Box::new(notify::PollWatcher::new(
                move |result| {
                    if let Ok(event) = result {
                        Self::handle_notify_event(event, &tx_clone);
                    }
                },
                notify::Config::default().with_poll_interval(config_interval),
            )?)
        } else {
            Box::new(notify::RecommendedWatcher::new(
                move |result| {
                    if let Ok(event) = result {
                        Self::handle_notify_event(event, &tx_clone);
                    }
                },
                notify::Config::default(),
            )?)
        };

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

        {
            let mut receiver_lock = self.event_receiver.lock().await;
            *receiver_lock = Some(rx);
        }

        {
            let mut watched_paths = self.watched_paths.write().await;
            watched_paths.insert(path.to_path_buf());
        }

        self.start_event_processor().await?;

        if config.process_existing {
            self.process_existing_files(path).await?;
        }

        tracing::info!(
            "Started watching path: {} (recursive: {})",
            path.display(),
            config.recursive
        );
        Ok(())
    }

    /// Stop watching all paths
    pub async fn stop_watching(&self) -> Result<(), WatchingError> {
        {
            let mut handle_lock = self.processor_handle.lock().await;
            if let Some(handle) = handle_lock.take() {
                handle.abort();
            }
        }
        {
            let mut watcher_lock = self.watcher.lock().await;
            *watcher_lock = None;
        }
        {
            let mut receiver_lock = self.event_receiver.lock().await;
            *receiver_lock = None;
        }
        {
            let mut watched_paths = self.watched_paths.write().await;
            watched_paths.clear();
        }

        tracing::info!("Stopped file watching");
        Ok(())
    }

    /// Update configuration (requires restart to take effect for some settings)
    pub async fn update_config(&self, mut new_config: WatcherConfig) -> Result<(), WatchingError> {
        new_config.validate_polling_interval();
        let new_patterns = CompiledPatterns::new(&new_config)?;

        {
            let mut config_lock = self.config.write().await;
            *config_lock = new_config.clone();
        }
        {
            let mut patterns_lock = self.patterns.write().await;
            *patterns_lock = new_patterns;
        }
        {
            let mut debouncer_lock = self.debouncer.lock().await;
            *debouncer_lock =
                EventDebouncer::new(new_config.debounce_ms, new_config.max_debouncer_capacity);
        }
        {
            let mut batcher_lock = self.batcher.lock().await;
            *batcher_lock =
                EventBatcher::new(new_config.batch_processing, new_config.max_batcher_capacity);
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
        {
            let debouncer_lock = self.debouncer.lock().await;
            stats.debouncer_evictions = debouncer_lock.eviction_count();
        }
        {
            let batcher_lock = self.batcher.lock().await;
            stats.batcher_evictions = batcher_lock.eviction_count();
        }
        {
            let patterns_lock = self.patterns.read().await;
            stats.pattern_cache_size = patterns_lock.cache_len();
        }

        {
            let config = self.config.read().await;
            let telemetry_config = &config.telemetry;

            if telemetry_config.enabled {
                let mut tracker = self.telemetry_tracker.lock().await;
                tracker.record_queue_depth(stats.current_queue_size);

                let elapsed = tracker.last_collection.elapsed();
                if elapsed.as_secs() >= telemetry_config.collection_interval_secs {
                    stats.telemetry =
                        tracker.collect_snapshot(telemetry_config, stats.current_queue_size);
                    stats.telemetry_history = Some(tracker.get_history());
                } else if !tracker.history.is_empty() {
                    stats.telemetry = tracker.history.back().cloned();
                    stats.telemetry_history = Some(tracker.get_history());
                }
            }
        }

        stats.is_paused = self.is_paused.load(Ordering::SeqCst);
        {
            let buffer = self.paused_event_buffer.lock().await;
            stats.buffered_events = buffer.len();
            stats.buffer_evictions = buffer.evictions();
        }

        stats
    }

    /// Check if the watcher is currently paused
    pub fn is_paused(&self) -> bool {
        self.is_paused.load(Ordering::SeqCst)
    }

    /// Pause the watcher - events will be buffered instead of processed
    pub fn pause(&self) {
        let was_paused = self.is_paused.swap(true, Ordering::SeqCst);
        if !was_paused {
            tracing::info!("FileWatcher paused - events will be buffered");
        }
    }

    /// Resume the watcher and return buffered events for processing
    pub async fn resume(&self) -> Vec<FileEvent> {
        let was_paused = self.is_paused.swap(false, Ordering::SeqCst);
        if was_paused {
            let mut buffer = self.paused_event_buffer.lock().await;
            let events = buffer.drain_events();
            tracing::info!(
                "FileWatcher resumed - {} buffered events will be processed",
                events.len()
            );
            events
        } else {
            Vec::new()
        }
    }

    /// Get currently watched paths
    pub async fn watched_paths(&self) -> Vec<PathBuf> {
        let watched_paths = self.watched_paths.read().await;
        watched_paths.iter().cloned().collect()
    }

    /// Start the event processing task
    async fn start_event_processor(&self) -> Result<(), WatchingError> {
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
        let telemetry_tracker = self.telemetry_tracker.clone();
        let is_paused = self.is_paused.clone();
        let paused_event_buffer = self.paused_event_buffer.clone();

        let handle = tokio::spawn(async move {
            Self::event_processing_loop(
                event_receiver,
                debouncer,
                batcher,
                patterns,
                config,
                task_submitter,
                stats,
                telemetry_tracker,
                is_paused,
                paused_event_buffer,
            )
            .await;
        });

        {
            let mut handle_lock = self.processor_handle.lock().await;
            *handle_lock = Some(handle);
        }

        Ok(())
    }

    /// Process existing files in a directory (for initial scan)
    async fn process_existing_files(&self, root_path: &Path) -> Result<(), WatchingError> {
        const YIELD_INTERVAL_MS: u64 = 10;
        const PROGRESS_REPORT_INTERVAL_S: u64 = 5;
        const BATCH_SIZE: usize = 50;

        let config = self.config.read().await;
        let patterns = self.patterns.read().await;
        let walker = build_walker(root_path, &config);

        let mut file_count = 0usize;
        let mut filtered_count = 0usize;
        let start_time = Instant::now();
        let mut last_yield = Instant::now();
        let mut last_progress = Instant::now();
        let mut batch_buffer: Vec<FileEvent> = Vec::with_capacity(BATCH_SIZE);

        for entry in walker {
            match entry {
                Ok(entry) => {
                    let path = entry.path();
                    if !accept_file(path, &patterns, &config, &mut filtered_count) {
                        continue;
                    }

                    batch_buffer.push(make_file_event(path));
                    file_count += 1;

                    if batch_buffer.len() >= BATCH_SIZE {
                        Self::submit_batch_directly(
                            &batch_buffer,
                            &self.batcher,
                            &self.config,
                            &self.task_submitter,
                            &self.stats,
                            &self.telemetry_tracker,
                        )
                        .await;
                        batch_buffer.clear();
                    }

                    let now = Instant::now();
                    if now.duration_since(last_yield) >= Duration::from_millis(YIELD_INTERVAL_MS) {
                        tokio::task::yield_now().await;
                        last_yield = now;
                    }
                    if now.duration_since(last_progress)
                        >= Duration::from_secs(PROGRESS_REPORT_INTERVAL_S)
                    {
                        let rate = file_count as f64 / start_time.elapsed().as_secs_f64();
                        tracing::info!(
                            "Initial scan progress: {} files processed, {} filtered ({:.1} files/sec)",
                            file_count, filtered_count, rate
                        );
                        last_progress = now;
                    }
                }
                Err(e) => tracing::warn!("Error walking directory {}: {}", root_path.display(), e),
            }
        }

        if !batch_buffer.is_empty() {
            Self::submit_batch_directly(
                &batch_buffer,
                &self.batcher,
                &self.config,
                &self.task_submitter,
                &self.stats,
                &self.telemetry_tracker,
            )
            .await;
        }

        // Flush the batcher so no events are stranded
        let ready_batch = {
            let mut batcher_lock = self.batcher.lock().await;
            batcher_lock.flush_all()
        };
        if let Some(batch) = ready_batch {
            Self::submit_processing_tasks(
                batch,
                &self.config,
                &self.task_submitter,
                &self.stats,
                &self.telemetry_tracker,
            )
            .await;
        }

        let elapsed = start_time.elapsed();
        let rate = if elapsed.as_secs() > 0 {
            file_count as f64 / elapsed.as_secs_f64()
        } else {
            0.0
        };
        tracing::info!(
            "Initial file scan complete: {} files processed, {} filtered in {:?} ({:.1} files/sec)",
            file_count,
            filtered_count,
            elapsed,
            rate
        );

        Ok(())
    }
}

/// Build a WalkDir iterator respecting the watcher's depth and recursive settings.
fn build_walker(root_path: &Path, config: &super::config::WatcherConfig) -> WalkDir {
    let max_depth = if config.max_depth < 0 {
        usize::MAX
    } else {
        config.max_depth as usize
    };
    if config.recursive {
        WalkDir::new(root_path).max_depth(max_depth)
    } else {
        WalkDir::new(root_path).max_depth(1)
    }
}

/// Return `true` when a file entry should be enqueued; update `filtered_count` otherwise.
fn accept_file(
    path: &Path,
    patterns: &super::compiled_patterns::CompiledPatterns,
    config: &super::config::WatcherConfig,
    filtered_count: &mut usize,
) -> bool {
    if !path.is_file() {
        return false;
    }
    if !patterns.should_process(path) {
        *filtered_count += 1;
        return false;
    }
    if let Some(max_size) = config.max_file_size {
        if let Ok(metadata) = path.metadata() {
            if metadata.len() > max_size {
                *filtered_count += 1;
                return false;
            }
        }
    }
    true
}

/// Construct a synthetic Create event for an existing file.
fn make_file_event(path: &Path) -> FileEvent {
    FileEvent {
        path: path.to_path_buf(),
        event_kind: EventKind::Create(notify::event::CreateKind::File),
        timestamp: Instant::now(),
        system_time: SystemTime::now(),
        size: path.metadata().ok().map(|m| m.len()),
        metadata: HashMap::new(),
    }
}
