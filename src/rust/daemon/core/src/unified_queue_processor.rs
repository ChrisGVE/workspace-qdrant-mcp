//! Unified Queue Processor Module
//!
//! Implements Task 37.26-29: Background processing loop that dequeues and processes
//! items from the unified_queue with type-specific handlers for content, file,
//! folder, project, library, and other operations.

use chrono::{DateTime, Duration as ChronoDuration, Utc};
use sqlx::SqlitePool;
use std::sync::Arc;
use std::time::{Duration, Instant};
use thiserror::Error;
use tokio::sync::RwLock;
use tokio::task::JoinHandle;
use tokio_util::sync::CancellationToken;
use tracing::{debug, error, info, warn};

use crate::adaptive_resources::ResourceProfile;
use crate::allowed_extensions::AllowedExtensions;
use crate::fairness_scheduler::{FairnessScheduler, FairnessSchedulerConfig};
use crate::queue_health::QueueProcessorHealth;
use crate::lsp::LanguageServerManager;
use crate::queue_operations::QueueManager;
use crate::search_db::SearchDbManager;
use crate::unified_queue_schema::{ItemType, QueueOperation, QueueStatus, UnifiedQueueItem};
use crate::{DocumentProcessor, EmbeddingGenerator, EmbeddingConfig};
use crate::storage::{StorageClient, StorageConfig};
use crate::lexicon::LexiconManager;

/// Unified queue processor errors
#[derive(Error, Debug)]
pub enum UnifiedProcessorError {
    #[error("Queue operation failed: {0}")]
    QueueOperation(String),

    #[error("Processing failed: {0}")]
    ProcessingFailed(String),

    #[error("Storage error: {0}")]
    Storage(String),

    #[error("Embedding error: {0}")]
    Embedding(String),

    #[error("File not found: {0}")]
    FileNotFound(String),

    #[error("Invalid payload: {0}")]
    InvalidPayload(String),

    #[error("Unsupported operation: {0}")]
    UnsupportedOperation(String),

    #[error("Shutdown requested")]
    ShutdownRequested,
}

/// Result type for unified processor operations
pub type UnifiedProcessorResult<T> = Result<T, UnifiedProcessorError>;

/// Warmup state tracker for startup throttling (Task 577)
///
/// Tracks whether the daemon is still in the warmup window after startup.
/// During warmup, the queue processor uses reduced resource limits to avoid
/// CPU spikes.
#[derive(Debug, Clone)]
pub struct WarmupState {
    daemon_start: Instant,
    warmup_window_secs: u64,
}

impl WarmupState {
    /// Create a new warmup state tracker
    pub fn new(warmup_window_secs: u64) -> Self {
        Self {
            daemon_start: Instant::now(),
            warmup_window_secs,
        }
    }

    /// Check if the daemon is still in the warmup window
    pub fn is_in_warmup(&self) -> bool {
        self.daemon_start.elapsed().as_secs() < self.warmup_window_secs
    }

    /// Get the elapsed time since daemon start
    pub fn elapsed_secs(&self) -> u64 {
        self.daemon_start.elapsed().as_secs()
    }
}

/// Processing metrics for unified queue monitoring
#[derive(Debug, Clone, Default)]
pub struct UnifiedProcessingMetrics {
    /// Total items processed by type
    pub items_processed_by_type: std::collections::HashMap<String, u64>,
    /// Total items failed
    pub items_failed: u64,
    /// Current queue depth
    pub queue_depth: i64,
    /// Average processing time (milliseconds)
    pub avg_processing_time_ms: f64,
    /// Items processed per second
    pub items_per_second: f64,
    /// Last metrics update time
    pub last_update: DateTime<Utc>,
    /// Total errors by type
    pub error_counts: std::collections::HashMap<String, u64>,
}

/// Configuration for the unified queue processor
#[derive(Debug, Clone)]
pub struct UnifiedProcessorConfig {
    /// Number of items to process in each batch
    pub batch_size: i32,
    /// Polling interval in milliseconds
    pub poll_interval_ms: u64,
    /// Worker ID for lease acquisition
    pub worker_id: String,
    /// Lease duration in seconds
    pub lease_duration_secs: i64,
    /// Maximum retries before marking as failed
    pub max_retries: i32,
    /// Retry delays (exponential backoff)
    pub retry_delays: Vec<ChronoDuration>,

    // Fairness scheduler settings (asymmetric anti-starvation alternation)
    /// Whether fairness scheduling is enabled (if disabled, falls back to priority DESC always)
    pub fairness_enabled: bool,
    /// Batch size when processing high-priority items (priority DESC direction, default: 10)
    pub high_priority_batch: u64,
    /// Batch size when processing low-priority items (priority ASC / anti-starvation, default: 3)
    pub low_priority_batch: u64,

    // Resource limits (Task 504)
    /// Delay in milliseconds between processing items
    pub inter_item_delay_ms: u64,
    /// Maximum concurrent embedding operations
    pub max_concurrent_embeddings: usize,
    /// Pause processing when memory usage exceeds this percentage
    pub max_memory_percent: u8,

    // Warmup throttling (Task 577)
    /// Duration in seconds of the warmup window with reduced limits
    pub warmup_window_secs: u64,
    /// Max concurrent embeddings during warmup
    pub warmup_max_concurrent_embeddings: usize,
    /// Inter-item delay in ms during warmup
    pub warmup_inter_item_delay_ms: u64,

    // ONNX thread tuning
    /// Number of ONNX intra-op threads per embedding session (default: 2)
    pub onnx_intra_threads: usize,
}

impl Default for UnifiedProcessorConfig {
    fn default() -> Self {
        Self {
            batch_size: 10,
            poll_interval_ms: 500,
            worker_id: format!("unified-worker-{}", uuid::Uuid::new_v4()),
            lease_duration_secs: 300, // 5 minutes
            max_retries: 3,
            retry_delays: vec![
                ChronoDuration::minutes(1),
                ChronoDuration::minutes(5),
                ChronoDuration::minutes(15),
                ChronoDuration::hours(1),
            ],
            // Fairness scheduler defaults (asymmetric anti-starvation)
            fairness_enabled: true,
            high_priority_batch: 10, // Spec: process 10 high-priority items per cycle
            low_priority_batch: 3,   // Spec: process 3 low-priority items per anti-starvation cycle
            // Resource limits defaults (Task 504)
            inter_item_delay_ms: 50,
            max_concurrent_embeddings: 2,
            max_memory_percent: 70,
            // Warmup throttling defaults (Task 577)
            warmup_window_secs: 30,
            warmup_max_concurrent_embeddings: 1,
            warmup_inter_item_delay_ms: 200,
            // ONNX thread tuning
            onnx_intra_threads: 2,
        }
    }
}

/// Unified queue processor manages background processing of unified_queue items
pub struct UnifiedQueueProcessor {
    /// Queue manager for database operations
    queue_manager: QueueManager,

    /// Processor configuration
    config: UnifiedProcessorConfig,

    /// Fairness scheduler for balanced queue processing (Task 34)
    fairness_scheduler: Arc<FairnessScheduler>,

    /// Processing metrics
    metrics: Arc<RwLock<UnifiedProcessingMetrics>>,

    /// Cancellation token for graceful shutdown
    cancellation_token: CancellationToken,

    /// Background task handle
    task_handle: Option<JoinHandle<()>>,

    /// Document processor for file-based operations
    document_processor: Arc<DocumentProcessor>,

    /// Embedding generator for dense/sparse vectors
    embedding_generator: Arc<EmbeddingGenerator>,

    /// Storage client for Qdrant operations
    storage_client: Arc<StorageClient>,

    /// LSP manager for code intelligence enrichment (optional)
    lsp_manager: Option<Arc<RwLock<LanguageServerManager>>>,

    /// Semaphore limiting concurrent embedding operations (Task 504)
    embedding_semaphore: Arc<tokio::sync::Semaphore>,

    /// File type allowlist for ingestion filtering (Task 511)
    allowed_extensions: Arc<AllowedExtensions>,

    /// Warmup state for startup throttling (Task 577)
    warmup_state: Arc<WarmupState>,

    /// Shared health state for gRPC monitoring
    queue_health: Option<Arc<QueueProcessorHealth>>,

    /// Receiver for adaptive resource profile changes (idle/burst mode)
    resource_profile_rx: Option<tokio::sync::watch::Receiver<ResourceProfile>>,

    /// Shared queue depth counter for adaptive resource signaling
    queue_depth_counter: Arc<std::sync::atomic::AtomicUsize>,

    /// Lexicon manager for per-collection BM25 vocabulary persistence (Task 17)
    lexicon_manager: Arc<LexiconManager>,

    /// Search database manager for FTS5 code search index (Task 52)
    search_db: Option<Arc<SearchDbManager>>,

    /// Signal to trigger WatchManager refresh after creating a new watch_folder (Task 12)
    watch_refresh_signal: Option<Arc<tokio::sync::Notify>>,
}

impl UnifiedQueueProcessor {
    /// Create a new unified queue processor
    pub fn new(pool: SqlitePool, config: UnifiedProcessorConfig) -> Self {
        let document_processor = Arc::new(DocumentProcessor::new());
        let embedding_config = EmbeddingConfig {
            num_threads: Some(config.onnx_intra_threads),
            ..EmbeddingConfig::default()
        };
        let embedding_generator = Arc::new(
            EmbeddingGenerator::new(embedding_config.clone())
                .expect("Failed to create embedding generator")
        );
        let storage_config = StorageConfig::default();
        let storage_client = Arc::new(StorageClient::with_config(storage_config));

        // Create lexicon manager for BM25 vocabulary persistence (Task 17)
        let lexicon_manager = Arc::new(LexiconManager::new(pool.clone(), embedding_config.bm25_k1));

        // Create fairness scheduler with config from processor config
        let queue_manager = QueueManager::new(pool);
        let fairness_config = FairnessSchedulerConfig {
            enabled: config.fairness_enabled,
            high_priority_batch: config.high_priority_batch,
            low_priority_batch: config.low_priority_batch,
            worker_id: config.worker_id.clone(),
            lease_duration_secs: config.lease_duration_secs,
        };
        let fairness_scheduler = Arc::new(FairnessScheduler::new(
            queue_manager.clone(),
            fairness_config,
        ));

        // Start with warmup permits, will add more when warmup ends (Task 578)
        let embedding_semaphore = Arc::new(tokio::sync::Semaphore::new(config.warmup_max_concurrent_embeddings));
        let warmup_state = Arc::new(WarmupState::new(config.warmup_window_secs));

        Self {
            queue_manager,
            config,
            fairness_scheduler,
            metrics: Arc::new(RwLock::new(UnifiedProcessingMetrics::default())),
            cancellation_token: CancellationToken::new(),
            task_handle: None,
            document_processor,
            embedding_generator,
            storage_client,
            lsp_manager: None,
            embedding_semaphore,
            allowed_extensions: Arc::new(AllowedExtensions::default()),
            warmup_state,
            queue_health: None,
            resource_profile_rx: None,
            queue_depth_counter: Arc::new(std::sync::atomic::AtomicUsize::new(0)),
            lexicon_manager,
            search_db: None,
            watch_refresh_signal: None,
        }
    }

    /// Create with custom components
    pub fn with_components(
        pool: SqlitePool,
        config: UnifiedProcessorConfig,
        document_processor: Arc<DocumentProcessor>,
        embedding_generator: Arc<EmbeddingGenerator>,
        storage_client: Arc<StorageClient>,
    ) -> Self {
        // Create lexicon manager for BM25 vocabulary persistence (Task 17)
        let lexicon_manager = Arc::new(LexiconManager::new(pool.clone(), EmbeddingConfig::default().bm25_k1));

        // Create fairness scheduler with config from processor config
        let queue_manager = QueueManager::new(pool);
        let fairness_config = FairnessSchedulerConfig {
            enabled: config.fairness_enabled,
            high_priority_batch: config.high_priority_batch,
            low_priority_batch: config.low_priority_batch,
            worker_id: config.worker_id.clone(),
            lease_duration_secs: config.lease_duration_secs,
        };
        let fairness_scheduler = Arc::new(FairnessScheduler::new(
            queue_manager.clone(),
            fairness_config,
        ));

        // Start with warmup permits, will add more when warmup ends (Task 578)
        let embedding_semaphore = Arc::new(tokio::sync::Semaphore::new(config.warmup_max_concurrent_embeddings));
        let warmup_state = Arc::new(WarmupState::new(config.warmup_window_secs));

        Self {
            queue_manager,
            config,
            fairness_scheduler,
            metrics: Arc::new(RwLock::new(UnifiedProcessingMetrics::default())),
            cancellation_token: CancellationToken::new(),
            task_handle: None,
            document_processor,
            embedding_generator,
            storage_client,
            lsp_manager: None,
            embedding_semaphore,
            allowed_extensions: Arc::new(AllowedExtensions::default()),
            warmup_state,
            queue_health: None,
            resource_profile_rx: None,
            queue_depth_counter: Arc::new(std::sync::atomic::AtomicUsize::new(0)),
            lexicon_manager,
            search_db: None,
            watch_refresh_signal: None,
        }
    }

    /// Set the watch refresh signal for triggering WatchManager refresh after new watch_folders (Task 12)
    pub fn with_watch_refresh_signal(mut self, signal: Arc<tokio::sync::Notify>) -> Self {
        self.watch_refresh_signal = Some(signal);
        self
    }

    /// Set the search database manager for FTS5 code search integration (Task 52)
    pub fn with_search_db(mut self, search_db: Arc<SearchDbManager>) -> Self {
        self.search_db = Some(search_db);
        self
    }

    /// Set shared queue processor health state for gRPC monitoring
    pub fn with_queue_health(mut self, health: Arc<QueueProcessorHealth>) -> Self {
        self.queue_health = Some(health);
        self
    }

    /// Set the LSP manager for code intelligence enrichment
    pub fn with_lsp_manager(mut self, lsp_manager: Arc<RwLock<LanguageServerManager>>) -> Self {
        self.lsp_manager = Some(lsp_manager);
        self
    }

    /// Set a custom file type allowlist (Task 511)
    pub fn with_allowed_extensions(mut self, allowed_extensions: Arc<AllowedExtensions>) -> Self {
        self.allowed_extensions = allowed_extensions;
        self
    }

    /// Set the adaptive resource profile receiver for dynamic CPU scaling
    pub fn with_adaptive_resources(mut self, rx: tokio::sync::watch::Receiver<ResourceProfile>) -> Self {
        self.resource_profile_rx = Some(rx);
        self
    }

    /// Get a shared queue depth counter for adaptive resource signaling.
    ///
    /// Returns `Some(Arc<AtomicUsize>)` that tracks the number of pending queue items.
    /// Used by `AdaptiveResourceManager` to detect Active Processing mode.
    pub fn queue_depth(&self) -> Option<Arc<std::sync::atomic::AtomicUsize>> {
        Some(Arc::clone(&self.queue_depth_counter))
    }

    /// Get a reference to the underlying SQLite pool
    pub fn pool(&self) -> &SqlitePool {
        self.queue_manager.pool()
    }

    /// Get a reference to the queue manager
    pub fn queue_manager(&self) -> &QueueManager {
        &self.queue_manager
    }

    /// Recover stale leases at startup (Task 37.19)
    pub async fn recover_stale_leases(&self) -> UnifiedProcessorResult<u64> {
        info!("Recovering stale unified queue leases...");
        let count = self.queue_manager
            .recover_stale_unified_leases()
            .await
            .map_err(|e| UnifiedProcessorError::QueueOperation(e.to_string()))?;

        if count > 0 {
            info!("Recovered {} stale unified queue leases", count);
        }
        Ok(count)
    }

    /// Start the background processing loop
    pub fn start(&mut self) -> UnifiedProcessorResult<()> {
        if self.task_handle.is_some() {
            warn!("Unified queue processor is already running");
            return Ok(());
        }

        info!(
            "Starting unified queue processor (batch_size={}, poll_interval={}ms, worker_id={}, fairness={})",
            self.config.batch_size, self.config.poll_interval_ms, self.config.worker_id, self.config.fairness_enabled
        );

        let queue_manager = self.queue_manager.clone();
        let config = self.config.clone();
        let fairness_scheduler = self.fairness_scheduler.clone();
        let metrics = self.metrics.clone();
        let cancellation_token = self.cancellation_token.clone();
        let document_processor = self.document_processor.clone();
        let embedding_generator = self.embedding_generator.clone();
        let storage_client = self.storage_client.clone();
        let lsp_manager = self.lsp_manager.clone();
        let embedding_semaphore = self.embedding_semaphore.clone();
        let allowed_extensions = self.allowed_extensions.clone();
        let lexicon_manager = self.lexicon_manager.clone();
        let warmup_state = self.warmup_state.clone();
        let queue_health = self.queue_health.clone();
        let resource_profile_rx = self.resource_profile_rx.clone();
        let queue_depth_counter = self.queue_depth_counter.clone();
        let search_db = self.search_db.clone();
        let watch_refresh_signal = self.watch_refresh_signal.clone();

        // Mark as running in health state
        if let Some(ref h) = queue_health {
            h.set_running(true);
        }

        let task_handle = tokio::spawn(async move {
            // One-time cleanup of junk BM25 terms from sparse_vocabulary (Task 22)
            if let Err(e) = lexicon_manager.cleanup_junk_terms().await {
                warn!("Failed to clean junk terms from sparse_vocabulary: {} (non-critical)", e);
            }

            if let Err(e) = Self::processing_loop(
                queue_manager,
                config,
                fairness_scheduler,
                metrics,
                cancellation_token.clone(),
                document_processor,
                embedding_generator,
                storage_client,
                lsp_manager,
                embedding_semaphore,
                allowed_extensions,
                lexicon_manager,
                warmup_state,
                queue_health.clone(),
                resource_profile_rx,
                queue_depth_counter,
                search_db,
                watch_refresh_signal,
            )
            .await
            {
                error!("Unified processing loop failed: {}", e);
            }

            // Mark as stopped in health state
            if let Some(ref h) = queue_health {
                h.set_running(false);
            }
            info!("Unified queue processor stopped");
        });

        self.task_handle = Some(task_handle);
        info!("Unified queue processor started successfully");
        Ok(())
    }

    /// Stop the background processing loop gracefully
    pub async fn stop(&mut self) -> UnifiedProcessorResult<()> {
        info!("Stopping unified queue processor...");

        self.cancellation_token.cancel();

        if let Some(handle) = self.task_handle.take() {
            match tokio::time::timeout(Duration::from_secs(30), handle).await {
                Ok(Ok(())) => {
                    info!("Unified queue processor stopped cleanly");
                }
                Ok(Err(e)) => {
                    error!("Unified queue processor task panicked: {}", e);
                }
                Err(_) => {
                    warn!("Unified queue processor did not stop within timeout");
                }
            }
        }

        Ok(())
    }

    /// Get current processing metrics
    pub async fn get_metrics(&self) -> UnifiedProcessingMetrics {
        self.metrics.read().await.clone()
    }

    /// Check if system memory usage exceeds the configured threshold (Task 504)
    async fn check_memory_pressure(max_memory_percent: u8) -> bool {
        use sysinfo::System;
        let mut sys = System::new();
        sys.refresh_memory();
        let total = sys.total_memory();
        if total == 0 {
            return false;
        }
        let used = sys.used_memory();
        let usage_percent = (used as f64 / total as f64 * 100.0) as u8;
        usage_percent > max_memory_percent
    }

    /// Main processing loop (runs in background task)
    #[allow(clippy::too_many_arguments)]
    async fn processing_loop(
        queue_manager: QueueManager,
        config: UnifiedProcessorConfig,
        fairness_scheduler: Arc<FairnessScheduler>,
        metrics: Arc<RwLock<UnifiedProcessingMetrics>>,
        cancellation_token: CancellationToken,
        document_processor: Arc<DocumentProcessor>,
        embedding_generator: Arc<EmbeddingGenerator>,
        storage_client: Arc<StorageClient>,
        lsp_manager: Option<Arc<RwLock<LanguageServerManager>>>,
        embedding_semaphore: Arc<tokio::sync::Semaphore>,
        allowed_extensions: Arc<AllowedExtensions>,
        lexicon_manager: Arc<LexiconManager>,
        warmup_state: Arc<WarmupState>,
        queue_health: Option<Arc<QueueProcessorHealth>>,
        resource_profile_rx: Option<tokio::sync::watch::Receiver<ResourceProfile>>,
        queue_depth_counter: Arc<std::sync::atomic::AtomicUsize>,
        search_db: Option<Arc<SearchDbManager>>,
        watch_refresh_signal: Option<Arc<tokio::sync::Notify>>,
    ) -> UnifiedProcessorResult<()> {
        let poll_interval = Duration::from_millis(config.poll_interval_ms);
        let mut last_metrics_log = Utc::now();
        let mut resource_profile_rx = resource_profile_rx;
        // Track current adaptive target to detect changes
        let mut adaptive_target_permits: Option<usize> = None;
        let metrics_log_interval = ChronoDuration::minutes(1);
        let mut warmup_logged = false;
        // Metadata uplift tracking
        let mut uplift_config = crate::metadata_uplift::UpliftConfig::default();
        let mut last_uplift_attempt = std::time::Instant::now()
            .checked_sub(Duration::from_secs(uplift_config.min_interval_secs))
            .unwrap_or_else(std::time::Instant::now);

        info!(
            "Unified processing loop started (batch_size={}, worker_id={}, fairness={}, warmup_window={}s)",
            config.batch_size, config.worker_id, config.fairness_enabled,
            config.warmup_window_secs
        );

        loop {
            // Log warmup transition and adjust embedding semaphore (Task 577, Task 578)
            if !warmup_logged && !warmup_state.is_in_warmup() {
                // Add permits to embedding semaphore to reach normal limit (Task 578)
                let permits_to_add = config.max_concurrent_embeddings.saturating_sub(config.warmup_max_concurrent_embeddings);
                if permits_to_add > 0 {
                    embedding_semaphore.add_permits(permits_to_add);
                }

                info!(
                    "Warmup period complete after {}s - switching to normal resource limits (delay: {}ms -> {}ms, max_embeddings: {} -> {})",
                    warmup_state.elapsed_secs(),
                    config.warmup_inter_item_delay_ms,
                    config.inter_item_delay_ms,
                    config.warmup_max_concurrent_embeddings,
                    config.max_concurrent_embeddings
                );
                warmup_logged = true;
            }

            // Check for shutdown signal
            if cancellation_token.is_cancelled() {
                info!("Unified queue shutdown signal received");
                break;
            }

            // Adaptive resource scaling: adjust semaphore permits on profile change
            if let Some(ref mut rx) = resource_profile_rx {
                if rx.has_changed().unwrap_or(false) {
                    let profile = *rx.borrow_and_update();
                    let target = profile.max_concurrent_embeddings;

                    // Only adjust after warmup is complete
                    if warmup_logged {
                        let current_target = adaptive_target_permits.unwrap_or(config.max_concurrent_embeddings);
                        if target > current_target {
                            let permits_to_add = target - current_target;
                            embedding_semaphore.add_permits(permits_to_add);
                            debug!("Adaptive: added {} semaphore permits (target: {})", permits_to_add, target);
                        }
                        // Note: tokio::sync::Semaphore doesn't support shrinking permits.
                        // When scaling down, in-flight operations keep their permits and
                        // new acquires will block once available permits drop below the
                        // new target. We track the target so subsequent transitions are correct.
                        adaptive_target_permits = Some(target);
                    }
                }
            }

            // Record poll for health monitoring
            if let Some(ref h) = queue_health {
                h.record_poll();
            }

            // Check memory pressure before dequeuing (Task 504)
            if Self::check_memory_pressure(config.max_memory_percent).await {
                info!("Memory pressure detected (>{}%), pausing processing for 5s", config.max_memory_percent);
                tokio::time::sleep(Duration::from_secs(5)).await;
                continue;
            }

            // Update queue depth in metrics and adaptive resource counter
            if let Ok(depth) = queue_manager.get_unified_queue_depth(None, None).await {
                let mut m = metrics.write().await;
                m.queue_depth = depth;
                if let Some(ref h) = queue_health {
                    h.set_queue_depth(depth as u64);
                }
                queue_depth_counter.store(depth as usize, std::sync::atomic::Ordering::Relaxed);
            }

            // Dequeue batch of items using fairness scheduler (Task 34)
            // The scheduler handles active project prioritization and starvation prevention
            match fairness_scheduler
                .dequeue_next_batch(config.batch_size)
                .await
            {
                Ok(items) => {
                    if items.is_empty() {
                        // Run metadata uplift when queue is idle (Task 18)
                        let since_last = last_uplift_attempt.elapsed().as_secs();
                        if since_last >= uplift_config.min_interval_secs {
                            debug!("Queue idle — running metadata uplift pass (gen={})", uplift_config.current_generation);
                            let collections = vec!["projects".to_string(), "libraries".to_string()];
                            let stats = crate::metadata_uplift::run_uplift_pass(
                                &storage_client,
                                &lexicon_manager,
                                &collections,
                                &uplift_config,
                            )
                            .await;
                            if stats.scanned > 0 {
                                info!(
                                    "Uplift pass complete: scanned={}, updated={}, skipped={}, errors={}",
                                    stats.scanned, stats.updated, stats.skipped, stats.errors
                                );
                            }
                            if stats.updated == 0 && stats.errors == 0 {
                                // All points uplifted at this generation, advance
                                uplift_config.current_generation += 1;
                            }
                            last_uplift_attempt = std::time::Instant::now();
                        }

                        debug!("Unified queue is empty, waiting {}ms", config.poll_interval_ms);
                        tokio::time::sleep(poll_interval).await;
                        continue;
                    }

                    info!("Dequeued {} unified queue items for processing", items.len());

                    // Check shutdown signal before processing batch
                    if cancellation_token.is_cancelled() {
                        warn!("Shutdown requested, stopping unified batch processing");
                        return Ok(());
                    }

                    // Track tenant_ids with successful processing for activity update
                    let mut processed_tenants = std::collections::HashSet::new();

                    // Process items sequentially
                    for item in items {
                        if cancellation_token.is_cancelled() {
                            warn!("Shutdown requested during item processing");
                            return Ok(());
                        }

                        let start_time = std::time::Instant::now();
                        let item_type_str = format!("{:?}", item.item_type);

                        match Self::process_item(
                            &queue_manager,
                            &item,
                            &config,
                            &document_processor,
                            &embedding_generator,
                            &storage_client,
                            &lsp_manager,
                            &embedding_semaphore,
                            &allowed_extensions,
                            &lexicon_manager,
                            &search_db,
                        )
                        .await
                        {
                            Ok(()) => {
                                let processing_time = start_time.elapsed().as_millis() as u64;

                                // Per-destination state machine (Task 6):
                                // Resolve any destination statuses that weren't explicitly set
                                // by the handler (orchestration-only items, content items, etc.)
                                // before checking finalization. Without this, items whose handlers
                                // don't call update_destination_status() would stay pending forever.
                                let _ = queue_manager
                                    .ensure_destinations_resolved(&item.queue_id)
                                    .await;

                                // check_and_finalize resolves overall status from qdrant_status + search_status.
                                // Delete item only when fully resolved as done.
                                let overall = queue_manager
                                    .check_and_finalize(&item.queue_id)
                                    .await
                                    .unwrap_or(QueueStatus::Done);

                                if overall == QueueStatus::Done {
                                    if let Err(e) = queue_manager
                                        .delete_unified_item(&item.queue_id)
                                        .await
                                    {
                                        error!("Failed to delete item {} from queue: {}", item.queue_id, e);
                                    }
                                } else {
                                    debug!(
                                        "Item {} not fully resolved (status={:?}), keeping in queue",
                                        item.queue_id, overall
                                    );
                                }

                                Self::update_metrics_success(
                                    &metrics,
                                    &item_type_str,
                                    processing_time,
                                ).await;

                                if let Some(ref h) = queue_health {
                                    h.record_success(processing_time);
                                }

                                // Track tenant for implicit activity update
                                processed_tenants.insert(item.tenant_id.clone());

                                info!(
                                    "Successfully processed unified item {} (type={:?}, op={:?}) in {}ms",
                                    item.queue_id, item.item_type, item.op, processing_time
                                );

                                // Task 12: Signal WatchManager to refresh after creating a new watch_folder.
                                // This enables immediate file + git watcher startup for newly registered projects.
                                if item.item_type == ItemType::Tenant && item.op == QueueOperation::Add {
                                    if let Some(ref signal) = watch_refresh_signal {
                                        signal.notify_one();
                                        debug!("Signaled WatchManager refresh after Tenant/Add for {}", item.tenant_id);
                                    }
                                }
                            }
                            Err(e) => {
                                // Classify error into 5 categories for observability
                                let error_category = Self::classify_error(&e);
                                let is_permanent = Self::is_permanent_category(error_category);

                                error!(
                                    error_category = error_category,
                                    permanent = is_permanent,
                                    "Failed to process unified item {} (type={:?}): {}",
                                    item.queue_id, item.item_type, e
                                );

                                // Mark item as failed (with exponential backoff for transient errors)
                                // Prefix error message with category for observability
                                let categorized_msg = format!("[{}] {}", error_category, e);
                                if let Err(mark_err) = queue_manager
                                    .mark_unified_failed(&item.queue_id, &categorized_msg, is_permanent)
                                    .await
                                {
                                    error!("Failed to mark item {} as failed: {}", item.queue_id, mark_err);
                                }

                                Self::update_metrics_failure(&metrics, &e).await;
                                if let Some(ref h) = queue_health {
                                    h.record_failure();
                                }
                            }
                        }

                        // Inter-item delay for resource breathing room (Task 504 / Task 577)
                        // Priority: warmup > adaptive profile > config default
                        let effective_delay_ms = if warmup_state.is_in_warmup() {
                            config.warmup_inter_item_delay_ms
                        } else if let Some(ref rx) = resource_profile_rx {
                            rx.borrow().inter_item_delay_ms
                        } else {
                            config.inter_item_delay_ms
                        };
                        if effective_delay_ms > 0 {
                            tokio::time::sleep(Duration::from_millis(effective_delay_ms)).await;
                        }
                    }

                    // Implicit activity update: refresh last_activity_at for active projects
                    // that had items processed in this batch. Prevents mid-processing deactivation
                    // by the orphan cleanup without requiring explicit heartbeats.
                    if !processed_tenants.is_empty() {
                        let now_str = wqm_common::timestamps::now_utc();
                        for tenant_id in &processed_tenants {
                            if let Err(e) = sqlx::query(
                                "UPDATE watch_folders SET last_activity_at = ?1, updated_at = ?1 \
                                 WHERE tenant_id = ?2 AND collection = 'projects' AND is_active = 1"
                            )
                            .bind(&now_str)
                            .bind(tenant_id)
                            .execute(queue_manager.pool())
                            .await {
                                debug!("Failed to update activity for tenant {}: {}", tenant_id, e);
                            }
                        }
                    }
                }
                Err(e) => {
                    error!("Failed to dequeue unified batch: {}", e);
                    if let Some(ref h) = queue_health {
                        h.record_error();
                    }
                    tokio::time::sleep(poll_interval).await;
                }
            }

            // Log metrics periodically
            let now = Utc::now();
            if now - last_metrics_log >= metrics_log_interval {
                Self::log_metrics(&metrics).await;
                last_metrics_log = now;
            }

            // Brief pause before next batch
            tokio::time::sleep(Duration::from_millis(10)).await;
        }

        Ok(())
    }

    /// Process a single unified queue item based on its type
    #[allow(clippy::too_many_arguments)]
    async fn process_item(
        queue_manager: &QueueManager,
        item: &UnifiedQueueItem,
        _config: &UnifiedProcessorConfig,
        document_processor: &Arc<DocumentProcessor>,
        embedding_generator: &Arc<EmbeddingGenerator>,
        storage_client: &Arc<StorageClient>,
        lsp_manager: &Option<Arc<RwLock<LanguageServerManager>>>,
        embedding_semaphore: &Arc<tokio::sync::Semaphore>,
        allowed_extensions: &Arc<AllowedExtensions>,
        lexicon_manager: &Arc<LexiconManager>,
        search_db: &Option<Arc<SearchDbManager>>,
    ) -> UnifiedProcessorResult<()> {
        debug!(
            "Processing unified item: {} (type={:?}, op={:?}, collection={})",
            item.queue_id, item.item_type, item.op, item.collection
        );

        match item.item_type {
            ItemType::Text => {
                let ctx = crate::context::ProcessingContext::new(
                    queue_manager.pool().clone(),
                    Arc::new(queue_manager.clone()),
                    Arc::clone(storage_client),
                    Arc::clone(embedding_generator),
                    Arc::clone(document_processor),
                    Arc::clone(embedding_semaphore),
                    Arc::clone(lexicon_manager),
                    lsp_manager.clone(),
                    search_db.clone(),
                    Arc::clone(allowed_extensions),
                );
                crate::strategies::processing::text::TextStrategy::process_content_item(&ctx, item).await
            }
            ItemType::File => {
                let ctx = crate::context::ProcessingContext::new(
                    queue_manager.pool().clone(),
                    Arc::new(queue_manager.clone()),
                    Arc::clone(storage_client),
                    Arc::clone(embedding_generator),
                    Arc::clone(document_processor),
                    Arc::clone(embedding_semaphore),
                    Arc::clone(lexicon_manager),
                    lsp_manager.clone(),
                    search_db.clone(),
                    Arc::clone(allowed_extensions),
                );
                crate::strategies::processing::file::FileStrategy::process_file_item(&ctx, item).await
            }
            ItemType::Folder => {
                let ctx = crate::context::ProcessingContext::new(
                    queue_manager.pool().clone(),
                    Arc::new(queue_manager.clone()),
                    Arc::clone(storage_client),
                    Arc::clone(embedding_generator),
                    Arc::clone(document_processor),
                    Arc::clone(embedding_semaphore),
                    Arc::clone(lexicon_manager),
                    lsp_manager.clone(),
                    search_db.clone(),
                    Arc::clone(allowed_extensions),
                );
                crate::strategies::processing::folder::FolderStrategy::process_folder_item(&ctx, item).await
            }
            ItemType::Tenant => {
                let ctx = crate::context::ProcessingContext::new(
                    queue_manager.pool().clone(),
                    Arc::new(queue_manager.clone()),
                    Arc::clone(storage_client),
                    Arc::clone(embedding_generator),
                    Arc::clone(document_processor),
                    Arc::clone(embedding_semaphore),
                    Arc::clone(lexicon_manager),
                    lsp_manager.clone(),
                    search_db.clone(),
                    Arc::clone(allowed_extensions),
                );
                crate::strategies::processing::tenant::TenantStrategy::process_tenant_item(&ctx, item).await
            }
            ItemType::Doc => {
                let ctx = crate::context::ProcessingContext::new(
                    queue_manager.pool().clone(),
                    Arc::new(queue_manager.clone()),
                    Arc::clone(storage_client),
                    Arc::clone(embedding_generator),
                    Arc::clone(document_processor),
                    Arc::clone(embedding_semaphore),
                    Arc::clone(lexicon_manager),
                    lsp_manager.clone(),
                    search_db.clone(),
                    Arc::clone(allowed_extensions),
                );
                crate::strategies::processing::tenant::TenantStrategy::process_doc_item(&ctx, item).await
            }
            ItemType::Url => {
                let ctx = crate::context::ProcessingContext::new(
                    queue_manager.pool().clone(),
                    Arc::new(queue_manager.clone()),
                    Arc::clone(storage_client),
                    Arc::clone(embedding_generator),
                    Arc::clone(document_processor),
                    Arc::clone(embedding_semaphore),
                    Arc::clone(lexicon_manager),
                    lsp_manager.clone(),
                    search_db.clone(),
                    Arc::clone(allowed_extensions),
                );
                crate::strategies::processing::url::UrlStrategy::process_url_item(&ctx, item).await
            }
            ItemType::Website => {
                let ctx = crate::context::ProcessingContext::new(
                    queue_manager.pool().clone(),
                    Arc::new(queue_manager.clone()),
                    Arc::clone(storage_client),
                    Arc::clone(embedding_generator),
                    Arc::clone(document_processor),
                    Arc::clone(embedding_semaphore),
                    Arc::clone(lexicon_manager),
                    lsp_manager.clone(),
                    search_db.clone(),
                    Arc::clone(allowed_extensions),
                );
                crate::strategies::processing::website::WebsiteStrategy::process_website_item(&ctx, item).await
            }
            ItemType::Collection => {
                let ctx = crate::context::ProcessingContext::new(
                    queue_manager.pool().clone(),
                    Arc::new(queue_manager.clone()),
                    Arc::clone(storage_client),
                    Arc::clone(embedding_generator),
                    Arc::clone(document_processor),
                    Arc::clone(embedding_semaphore),
                    Arc::clone(lexicon_manager),
                    lsp_manager.clone(),
                    search_db.clone(),
                    Arc::clone(allowed_extensions),
                );
                crate::strategies::processing::collection::CollectionStrategy::process_collection_item(&ctx, item).await
            }
        }
    }
    /// Update metrics after successful processing
    async fn update_metrics_success(
        metrics: &Arc<RwLock<UnifiedProcessingMetrics>>,
        item_type: &str,
        processing_time_ms: u64,
    ) {
        let mut m = metrics.write().await;

        // Increment counter for this item type
        *m.items_processed_by_type
            .entry(item_type.to_string())
            .or_insert(0) += 1;

        // Update average processing time
        let total_items: u64 = m.items_processed_by_type.values().sum();
        let total_items_f = total_items as f64;
        m.avg_processing_time_ms = (m.avg_processing_time_ms * (total_items_f - 1.0)
            + processing_time_ms as f64)
            / total_items_f;

        // Calculate throughput
        let elapsed_secs = (Utc::now() - m.last_update).num_seconds() as f64;
        if elapsed_secs > 0.0 {
            m.items_per_second = total_items_f / elapsed_secs;
        }
    }

    /// Classify whether an error is permanent (should not be retried).
    ///
    /// Permanent errors: file not found, invalid payload, validation failures.
    /// Transient errors (retryable): storage/Qdrant issues, embedding failures,
    /// queue operation failures (DB locked, etc).
    /// Classify a processing error into one of 5 categories:
    /// - `permanent_data`: invalid payload, unsupported format — no retry
    /// - `permanent_gone`: file deleted, permission denied — no retry
    /// - `transient_infrastructure`: Qdrant down, network error — retry with standard backoff
    /// - `transient_resource`: OOM, embedding failure — retry with longer backoff
    /// - `partial`: partial enrichment — retry enrichment only
    fn classify_error(error: &UnifiedProcessorError) -> &'static str {
        match error {
            // File doesn't exist or was deleted
            UnifiedProcessorError::FileNotFound(_) => "permanent_gone",
            // Malformed payload — retrying won't fix the data
            UnifiedProcessorError::InvalidPayload(_) => "permanent_data",
            // Queue operation errors — check message
            UnifiedProcessorError::QueueOperation(msg) => {
                let lower = msg.to_lowercase();
                if lower.contains("no watch_folder found")
                    || lower.contains("validation")
                    || lower.contains("invalid")
                {
                    "permanent_data"
                } else {
                    "transient_infrastructure"
                }
            }
            // Processing errors — check message for permanent vs transient
            UnifiedProcessorError::ProcessingFailed(msg) => {
                let lower = msg.to_lowercase();
                if lower.contains("permission denied") || lower.contains("access denied") {
                    "permanent_gone"
                } else if lower.contains("invalid format")
                    || lower.contains("malformed")
                    || lower.contains("unsupported")
                {
                    "permanent_data"
                } else {
                    "transient_infrastructure"
                }
            }
            // Qdrant storage errors — transient infrastructure
            UnifiedProcessorError::Storage(_) => "transient_infrastructure",
            // Embedding errors — transient resource (model/memory)
            UnifiedProcessorError::Embedding(_) => "transient_resource",
            // Default: treat as transient infrastructure (retry)
            _ => "transient_infrastructure",
        }
    }

    /// Check if an error category is permanent (should not be retried).
    fn is_permanent_category(category: &str) -> bool {
        category.starts_with("permanent")
    }

    /// Update metrics after processing failure
    async fn update_metrics_failure(
        metrics: &Arc<RwLock<UnifiedProcessingMetrics>>,
        error: &UnifiedProcessorError,
    ) {
        let mut m = metrics.write().await;
        m.items_failed += 1;

        let error_type = match error {
            UnifiedProcessorError::InvalidPayload(_) => "invalid_payload",
            UnifiedProcessorError::ProcessingFailed(_) => "processing_failed",
            UnifiedProcessorError::FileNotFound(_) => "file_not_found",
            UnifiedProcessorError::Storage(_) => "storage_error",
            UnifiedProcessorError::Embedding(_) => "embedding_error",
            _ => "other",
        };

        *m.error_counts.entry(error_type.to_string()).or_insert(0) += 1;
    }

    /// Log current processing metrics
    async fn log_metrics(metrics: &Arc<RwLock<UnifiedProcessingMetrics>>) {
        let m = metrics.read().await;

        let total_processed: u64 = m.items_processed_by_type.values().sum();

        info!(
            "Unified Queue Metrics: processed={}, failed={}, queue_depth={}, avg_time={:.2}ms",
            total_processed,
            m.items_failed,
            m.queue_depth,
            m.avg_processing_time_ms,
        );

        if !m.items_processed_by_type.is_empty() {
            debug!("Items by type: {:?}", m.items_processed_by_type);
        }

        if !m.error_counts.is_empty() {
            debug!("Error breakdown: {:?}", m.error_counts);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::Path;
    use crate::patterns::exclusion::should_exclude_file;
    use crate::unified_queue_schema::QueueStatus;

    #[test]
    fn test_unified_processor_config_default() {
        let config = UnifiedProcessorConfig::default();
        assert_eq!(config.batch_size, 10);
        assert_eq!(config.poll_interval_ms, 500);
        assert_eq!(config.lease_duration_secs, 300);
        assert_eq!(config.max_retries, 3);
        assert!(config.worker_id.starts_with("unified-worker-"));
        // Fairness scheduler settings (asymmetric anti-starvation)
        assert!(config.fairness_enabled);
        assert_eq!(config.high_priority_batch, 10);
        assert_eq!(config.low_priority_batch, 3);
        // Resource limits (Task 504)
        assert_eq!(config.inter_item_delay_ms, 50);
        assert_eq!(config.max_concurrent_embeddings, 2);
        assert_eq!(config.max_memory_percent, 70);
    }

    #[test]
    fn test_unified_processing_metrics_default() {
        let metrics = UnifiedProcessingMetrics::default();
        assert_eq!(metrics.items_failed, 0);
        assert_eq!(metrics.queue_depth, 0);
        assert!(metrics.items_processed_by_type.is_empty());
    }

    #[test]
    fn test_unified_processor_error_display() {
        let err = UnifiedProcessorError::InvalidPayload("missing field".to_string());
        assert_eq!(err.to_string(), "Invalid payload: missing field");

        let err = UnifiedProcessorError::FileNotFound("/path/to/file".to_string());
        assert_eq!(err.to_string(), "File not found: /path/to/file");

        let err = UnifiedProcessorError::Storage("connection refused".to_string());
        assert_eq!(err.to_string(), "Storage error: connection refused");
    }

    /// Test that the exclusion check logic correctly identifies files that should be cleaned up
    /// This tests the core decision logic used by cleanup_excluded_files without needing
    /// Qdrant or SQLite connections.
    #[test]
    fn test_cleanup_exclusion_logic_identifies_hidden_files() {
        let project_root = Path::new("/home/user/project");

        // Simulate file paths as they would be stored in Qdrant (absolute paths)
        let qdrant_paths = vec![
            "/home/user/project/src/main.rs",
            "/home/user/project/.hidden_file",
            "/home/user/project/src/.secret",
            "/home/user/project/.git/config",
            "/home/user/project/src/lib.rs",
            "/home/user/project/node_modules/package/index.js",
            "/home/user/project/.env",
            "/home/user/project/README.md",
            "/home/user/project/src/.cache/data",
            "/home/user/project/.github/workflows/ci.yml",
        ];

        let mut should_delete = Vec::new();
        let mut should_keep = Vec::new();

        for qdrant_file in &qdrant_paths {
            let rel_path = match Path::new(qdrant_file).strip_prefix(project_root) {
                Ok(stripped) => stripped.to_string_lossy().to_string(),
                Err(_) => qdrant_file.to_string(),
            };

            if should_exclude_file(&rel_path) {
                should_delete.push(qdrant_file.to_string());
            } else {
                should_keep.push(qdrant_file.to_string());
            }
        }

        // Hidden files should be marked for deletion
        assert!(
            should_delete.contains(&"/home/user/project/.hidden_file".to_string()),
            "Expected .hidden_file to be excluded"
        );
        assert!(
            should_delete.contains(&"/home/user/project/src/.secret".to_string()),
            "Expected src/.secret to be excluded"
        );
        assert!(
            should_delete.contains(&"/home/user/project/.git/config".to_string()),
            "Expected .git/config to be excluded"
        );
        assert!(
            should_delete.contains(&"/home/user/project/.env".to_string()),
            "Expected .env to be excluded"
        );
        assert!(
            should_delete.contains(&"/home/user/project/src/.cache/data".to_string()),
            "Expected src/.cache/data to be excluded"
        );
        assert!(
            should_delete.contains(&"/home/user/project/node_modules/package/index.js".to_string()),
            "Expected node_modules content to be excluded"
        );

        // Normal files should NOT be deleted
        assert!(
            should_keep.contains(&"/home/user/project/src/main.rs".to_string()),
            "Expected src/main.rs to be kept"
        );
        assert!(
            should_keep.contains(&"/home/user/project/src/lib.rs".to_string()),
            "Expected src/lib.rs to be kept"
        );
        assert!(
            should_keep.contains(&"/home/user/project/README.md".to_string()),
            "Expected README.md to be kept"
        );

        // .github/ should be whitelisted (not excluded)
        assert!(
            should_keep.contains(&"/home/user/project/.github/workflows/ci.yml".to_string()),
            "Expected .github/workflows/ci.yml to be kept (whitelisted)"
        );
    }

    #[test]
    fn test_cleanup_exclusion_logic_with_non_strippable_paths() {
        // Test when Qdrant paths don't share the project root prefix
        let project_root = Path::new("/home/user/project");
        let qdrant_file = "/different/root/src/.hidden";

        let rel_path = match Path::new(qdrant_file).strip_prefix(project_root) {
            Ok(stripped) => stripped.to_string_lossy().to_string(),
            Err(_) => qdrant_file.to_string(),
        };

        // Should still detect hidden component even with full path fallback
        assert!(
            should_exclude_file(&rel_path),
            "Expected .hidden to be excluded even when path can't be stripped"
        );
    }

    #[test]
    fn test_cleanup_exclusion_logic_empty_paths() {
        // Verify no panic with edge cases
        let project_root = Path::new("/home/user/project");
        let qdrant_paths: Vec<String> = vec![];

        let mut count = 0u64;
        for qdrant_file in &qdrant_paths {
            let rel_path = match Path::new(qdrant_file).strip_prefix(project_root) {
                Ok(stripped) => stripped.to_string_lossy().to_string(),
                Err(_) => qdrant_file.clone(),
            };

            if should_exclude_file(&rel_path) {
                count += 1;
            }
        }

        assert_eq!(count, 0, "Empty path list should produce zero deletions");
    }

    /// Test WarmupState tracking (Task 577)
    #[test]
    fn test_warmup_state_tracking() {
        let warmup_state = WarmupState::new(5); // 5 second warmup window

        // Should be in warmup immediately
        assert!(warmup_state.is_in_warmup());
        assert_eq!(warmup_state.elapsed_secs(), 0);

        // Sleep 1 second, still in warmup
        std::thread::sleep(std::time::Duration::from_secs(1));
        assert!(warmup_state.is_in_warmup());

        // Sleep past warmup window
        std::thread::sleep(std::time::Duration::from_secs(5));
        assert!(!warmup_state.is_in_warmup());
        assert!(warmup_state.elapsed_secs() >= 5);
    }

    /// Test embedding semaphore starts with warmup permits (Task 578)
    #[tokio::test]
    async fn test_embedding_semaphore_starts_with_warmup_permits() {
        let config = UnifiedProcessorConfig {
            warmup_max_concurrent_embeddings: 1,
            max_concurrent_embeddings: 2,
            ..Default::default()
        };

        // Create semaphore as done in UnifiedQueueProcessor::new
        let semaphore = Arc::new(tokio::sync::Semaphore::new(config.warmup_max_concurrent_embeddings));

        // Should have exactly 1 permit available (warmup limit)
        assert_eq!(semaphore.available_permits(), 1);

        // Acquire the one warmup permit
        let _permit = semaphore.acquire().await.unwrap();
        assert_eq!(semaphore.available_permits(), 0);

        // Try to acquire another - should fail immediately (would block if we awaited)
        assert!(semaphore.try_acquire().is_err());
    }

    /// Test semaphore transition from warmup to normal limits (Task 578)
    #[tokio::test]
    async fn test_embedding_semaphore_transition_to_normal_limits() {
        let config = UnifiedProcessorConfig {
            warmup_max_concurrent_embeddings: 1,
            max_concurrent_embeddings: 3,
            ..Default::default()
        };

        // Start with warmup permits
        let semaphore = Arc::new(tokio::sync::Semaphore::new(config.warmup_max_concurrent_embeddings));
        assert_eq!(semaphore.available_permits(), 1);

        // Simulate warmup ending: add permits to reach normal limit
        let permits_to_add = config.max_concurrent_embeddings - config.warmup_max_concurrent_embeddings;
        semaphore.add_permits(permits_to_add);

        // Should now have 3 total permits (normal limit)
        assert_eq!(semaphore.available_permits(), 3);

        // Can acquire 3 permits
        let _p1 = semaphore.acquire().await.unwrap();
        let _p2 = semaphore.acquire().await.unwrap();
        let _p3 = semaphore.acquire().await.unwrap();
        assert_eq!(semaphore.available_permits(), 0);

        // Fourth acquire would block
        assert!(semaphore.try_acquire().is_err());
    }

    /// Test warmup config defaults (Task 577)
    #[test]
    fn test_warmup_config_defaults() {
        let config = UnifiedProcessorConfig::default();
        assert_eq!(config.warmup_window_secs, 30);
        assert_eq!(config.warmup_max_concurrent_embeddings, 1);
        assert_eq!(config.warmup_inter_item_delay_ms, 200);
        assert_eq!(config.max_concurrent_embeddings, 2);
        assert_eq!(config.inter_item_delay_ms, 50);
    }

    /// Test that warmup limits are more restrictive than normal limits (Task 578)
    #[test]
    fn test_warmup_limits_are_more_restrictive() {
        let config = UnifiedProcessorConfig::default();
        assert!(
            config.warmup_max_concurrent_embeddings <= config.max_concurrent_embeddings,
            "Warmup max_concurrent_embeddings should be <= normal limit"
        );
        assert!(
            config.warmup_inter_item_delay_ms >= config.inter_item_delay_ms,
            "Warmup inter_item_delay should be >= normal delay (slower processing)"
        );
    }

    // =========================================================================
    // DeleteDocument payload contract tests
    // =========================================================================

    /// Helper to create a minimal UnifiedQueueItem for testing
    fn make_delete_document_item(payload_json: &str) -> UnifiedQueueItem {
        UnifiedQueueItem {
            queue_id: "test-queue-id".to_string(),
            idempotency_key: "test-idempotency".to_string(),
            item_type: ItemType::Doc,
            op: QueueOperation::Delete,
            tenant_id: "test-tenant".to_string(),
            collection: "projects".to_string(),
            priority: wqm_common::constants::priority::LOW,
            status: QueueStatus::InProgress,
            branch: "main".to_string(),
            payload_json: payload_json.to_string(),
            metadata: None,
            created_at: "2026-01-01T00:00:00Z".to_string(),
            updated_at: "2026-01-01T00:00:00Z".to_string(),
            lease_until: None,
            worker_id: None,
            retry_count: 0,
            max_retries: 3,
            error_message: None,
            last_error_at: None,
            file_path: None,
            qdrant_status: None,
            search_status: None,
            decision_json: None,
        }
    }

    #[test]
    fn test_delete_document_payload_uses_document_id_key() {
        // Validates that the payload field is "document_id" (not "doc_id")
        // matching the validation contract in queue_operations.rs
        let item = make_delete_document_item(r#"{"document_id":"doc-abc-123"}"#);
        let payload = item.parse_delete_document_payload()
            .expect("Should parse with document_id key");
        assert_eq!(payload.document_id, "doc-abc-123");
    }

    #[test]
    fn test_delete_document_payload_rejects_wrong_key() {
        // The old code used "doc_id" which would never match the validated payload
        let item = make_delete_document_item(r#"{"doc_id":"doc-abc-123"}"#);
        let payload = item.parse_delete_document_payload();
        // serde will deserialize but document_id will be missing/empty
        // since DeleteDocumentPayload requires "document_id"
        assert!(
            payload.is_err() || payload.unwrap().document_id.is_empty(),
            "Payload with 'doc_id' should fail or produce empty document_id"
        );
    }

    #[test]
    fn test_delete_document_payload_invalid_json() {
        let item = make_delete_document_item("not valid json");
        let result = item.parse_delete_document_payload();
        assert!(result.is_err(), "Invalid JSON should fail deserialization");
    }

    #[test]
    fn test_delete_document_payload_with_point_ids() {
        let item = make_delete_document_item(
            r#"{"document_id":"doc-xyz","point_ids":["p1","p2"]}"#
        );
        let payload = item.parse_delete_document_payload()
            .expect("Should parse with point_ids");
        assert_eq!(payload.document_id, "doc-xyz");
        assert_eq!(payload.point_ids.len(), 2);
    }

    #[test]
    fn test_lsp_enrichment_status_lowercase_in_payload() {
        use crate::lsp::project_manager::{EnrichmentStatus, LspEnrichment};
        use crate::strategies::processing::file::FileStrategy;

        let mut payload = std::collections::HashMap::new();
        let enrichment = LspEnrichment {
            enrichment_status: EnrichmentStatus::Success,
            references: vec![],
            type_info: None,
            resolved_imports: vec![],
            definition: None,
            error_message: None,
        };

        FileStrategy::add_lsp_enrichment_to_payload(&mut payload, &enrichment);
        let status = payload.get("lsp_enrichment_status").unwrap().as_str().unwrap();
        assert_eq!(status, "success", "lsp_enrichment_status must be lowercase");

        let mut payload2 = std::collections::HashMap::new();
        let enrichment2 = LspEnrichment {
            enrichment_status: EnrichmentStatus::Failed,
            references: vec![],
            type_info: None,
            resolved_imports: vec![],
            definition: None,
            error_message: Some("test error".to_string()),
        };

        FileStrategy::add_lsp_enrichment_to_payload(&mut payload2, &enrichment2);
        let status2 = payload2.get("lsp_enrichment_status").unwrap().as_str().unwrap();
        assert_eq!(status2, "failed", "lsp_enrichment_status must be lowercase");
    }


    #[test]
    fn test_file_chunk_tags_construction() {
        use crate::DocumentType;
        use crate::file_classification::is_test_file;

        // Simulate tag construction logic from process_file_item
        let build_tags = |file_type: Option<&str>, doc_type: &DocumentType, file_path: &str| -> Vec<String> {
            let mut tags = Vec::new();
            if let Some(ft) = file_type {
                tags.push(ft.to_lowercase());
            }
            if let Some(lang) = doc_type.language() {
                tags.push(lang.to_string());
            }
            if let Some(ext) = std::path::Path::new(file_path)
                .extension()
                .and_then(|e| e.to_str())
            {
                tags.push(ext.to_lowercase());
            }
            if is_test_file(std::path::Path::new(file_path)) {
                tags.push("test".to_string());
            }
            tags
        };

        // Rust test file
        let tags = build_tags(
            Some("code"),
            &DocumentType::Code("rust".to_string()),
            "/project/src/test_utils.rs",
        );
        assert_eq!(tags, vec!["code", "rust", "rs", "test"]);

        // Python non-test file
        let tags = build_tags(
            Some("code"),
            &DocumentType::Code("python".to_string()),
            "/project/src/main.py",
        );
        assert_eq!(tags, vec!["code", "python", "py"]);

        // Markdown file (no language)
        let tags = build_tags(
            Some("docs"),
            &DocumentType::Markdown,
            "/project/README.md",
        );
        assert_eq!(tags, vec!["docs", "md"]);

        // File with uppercase extension
        let tags = build_tags(
            Some("code"),
            &DocumentType::Code("cpp".to_string()),
            "/project/main.CPP",
        );
        assert_eq!(tags, vec!["code", "cpp", "cpp"]);
    }

    #[test]
    fn test_classify_error_permanent_categories() {
        // FileNotFound → permanent_gone
        let err = UnifiedProcessorError::FileNotFound("/missing.rs".into());
        assert_eq!(UnifiedQueueProcessor::classify_error(&err), "permanent_gone");

        // InvalidPayload → permanent_data
        let err = UnifiedProcessorError::InvalidPayload("bad json".into());
        assert_eq!(UnifiedQueueProcessor::classify_error(&err), "permanent_data");

        // QueueOperation with validation → permanent_data
        let err = UnifiedProcessorError::QueueOperation("no watch_folder found".into());
        assert_eq!(UnifiedQueueProcessor::classify_error(&err), "permanent_data");

        // ProcessingFailed with permission denied → permanent_gone
        let err = UnifiedProcessorError::ProcessingFailed("Permission denied".into());
        assert_eq!(UnifiedQueueProcessor::classify_error(&err), "permanent_gone");

        // ProcessingFailed with unsupported → permanent_data
        let err = UnifiedProcessorError::ProcessingFailed("Unsupported file format".into());
        assert_eq!(UnifiedQueueProcessor::classify_error(&err), "permanent_data");
    }

    #[test]
    fn test_classify_error_transient_categories() {
        // Storage → transient_infrastructure
        let err = UnifiedProcessorError::Storage("connection refused".into());
        assert_eq!(UnifiedQueueProcessor::classify_error(&err), "transient_infrastructure");

        // Embedding → transient_resource
        let err = UnifiedProcessorError::Embedding("out of memory".into());
        assert_eq!(UnifiedQueueProcessor::classify_error(&err), "transient_resource");

        // Generic ProcessingFailed → transient_infrastructure
        let err = UnifiedProcessorError::ProcessingFailed("timeout".into());
        assert_eq!(UnifiedQueueProcessor::classify_error(&err), "transient_infrastructure");

        // Generic QueueOperation → transient_infrastructure
        let err = UnifiedProcessorError::QueueOperation("database locked".into());
        assert_eq!(UnifiedQueueProcessor::classify_error(&err), "transient_infrastructure");
    }

    #[test]
    fn test_is_permanent_category() {
        assert!(UnifiedQueueProcessor::is_permanent_category("permanent_gone"));
        assert!(UnifiedQueueProcessor::is_permanent_category("permanent_data"));
        assert!(!UnifiedQueueProcessor::is_permanent_category("transient_infrastructure"));
        assert!(!UnifiedQueueProcessor::is_permanent_category("transient_resource"));
        assert!(!UnifiedQueueProcessor::is_permanent_category("partial"));
    }
}
