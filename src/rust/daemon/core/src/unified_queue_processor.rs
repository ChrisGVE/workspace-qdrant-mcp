//! Unified Queue Processor Module
//!
//! Implements Task 37.26-29: Background processing loop that dequeues and processes
//! items from the unified_queue with type-specific handlers for content, file,
//! folder, project, library, and other operations.

use chrono::{DateTime, Duration as ChronoDuration, Utc};
use sqlx::SqlitePool;
use std::path::Path;
use std::sync::Arc;
use std::time::{Duration, Instant};
use thiserror::Error;
use tokio::sync::RwLock;
use tokio::task::JoinHandle;
use tokio_util::sync::CancellationToken;
use tracing::{debug, error, info, warn};
use walkdir::WalkDir;

use crate::adaptive_resources::ResourceProfile;
use crate::allowed_extensions::AllowedExtensions;
use crate::fairness_scheduler::{FairnessScheduler, FairnessSchedulerConfig};
use crate::queue_health::QueueProcessorHealth;
use crate::lsp::{
    LanguageServerManager, LspEnrichment, EnrichmentStatus,
};
use crate::queue_operations::QueueManager;
use crate::unified_queue_schema::{
    ItemType, QueueOperation, UnifiedQueueItem,
    ContentPayload, FilePayload, FolderPayload, ProjectPayload, LibraryPayload,
    MemoryPayload, UrlPayload, ScratchpadPayload,
};
use crate::{DocumentProcessor, EmbeddingGenerator, EmbeddingConfig, SparseEmbedding};
use wqm_common::constants::{COLLECTION_PROJECTS, COLLECTION_LIBRARIES};
use crate::storage::{StorageClient, StorageConfig, DocumentPoint};
use crate::patterns::exclusion::{should_exclude_file, should_exclude_directory};
use crate::file_classification::{classify_file_type, is_test_file, get_extension_for_storage};
use crate::tracked_files_schema::{
    self, ProcessingStatus, ChunkType as TrackedChunkType,
};

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
            EmbeddingGenerator::new(embedding_config)
                .expect("Failed to create embedding generator")
        );
        let storage_config = StorageConfig::default();
        let storage_client = Arc::new(StorageClient::with_config(storage_config));

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
        }
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
        let warmup_state = self.warmup_state.clone();
        let queue_health = self.queue_health.clone();
        let resource_profile_rx = self.resource_profile_rx.clone();
        let queue_depth_counter = self.queue_depth_counter.clone();

        // Mark as running in health state
        if let Some(ref h) = queue_health {
            h.set_running(true);
        }

        let task_handle = tokio::spawn(async move {
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
                warmup_state,
                queue_health.clone(),
                resource_profile_rx,
                queue_depth_counter,
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
        warmup_state: Arc<WarmupState>,
        queue_health: Option<Arc<QueueProcessorHealth>>,
        resource_profile_rx: Option<tokio::sync::watch::Receiver<ResourceProfile>>,
        queue_depth_counter: Arc<std::sync::atomic::AtomicUsize>,
    ) -> UnifiedProcessorResult<()> {
        let poll_interval = Duration::from_millis(config.poll_interval_ms);
        let mut last_metrics_log = Utc::now();
        let mut resource_profile_rx = resource_profile_rx;
        // Track current adaptive target to detect changes
        let mut adaptive_target_permits: Option<usize> = None;
        let metrics_log_interval = ChronoDuration::minutes(1);
        let mut warmup_logged = false;

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
                        )
                        .await
                        {
                            Ok(()) => {
                                let processing_time = start_time.elapsed().as_millis() as u64;

                                // Delete item from queue per spec line 813:
                                // "On success: DELETE items from queue"
                                if let Err(e) = queue_manager
                                    .delete_unified_item(&item.queue_id)
                                    .await
                                {
                                    error!("Failed to delete item {} from queue: {}", item.queue_id, e);
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
    ) -> UnifiedProcessorResult<()> {
        debug!(
            "Processing unified item: {} (type={:?}, op={:?}, collection={})",
            item.queue_id, item.item_type, item.op, item.collection
        );

        match item.item_type {
            ItemType::Text => {
                Self::process_content_item(item, embedding_generator, storage_client, embedding_semaphore).await
            }
            ItemType::File => {
                Self::process_file_item(item, queue_manager, document_processor, embedding_generator, storage_client, lsp_manager, embedding_semaphore, allowed_extensions).await
            }
            ItemType::Folder => {
                Self::process_folder_item(item, queue_manager, storage_client, allowed_extensions).await
            }
            ItemType::Tenant => {
                // Tenant consolidates old Project, Library, and DeleteTenant
                match item.op {
                    QueueOperation::Delete => {
                        Self::process_delete_tenant_item(item, queue_manager, storage_client).await
                    }
                    QueueOperation::Rename => {
                        Self::process_tenant_rename_item(item, storage_client).await
                    }
                    _ => {
                        // Add, Scan, Update — route by collection
                        match item.collection.as_str() {
                            "libraries" => Self::process_library_item(item, queue_manager, storage_client, allowed_extensions).await,
                            _ => Self::process_project_item(item, queue_manager, storage_client, allowed_extensions).await,
                        }
                    }
                }
            }
            ItemType::Doc => {
                match item.op {
                    QueueOperation::Delete => {
                        Self::process_delete_document_item(item, storage_client).await
                    }
                    QueueOperation::Uplift => {
                        // Placeholder: no enrichment logic yet
                        info!("Doc uplift placeholder for queue_id={} tenant={}", item.queue_id, item.tenant_id);
                        Ok(())
                    }
                    _ => {
                        warn!("Unsupported operation {:?} for Doc item {}", item.op, item.queue_id);
                        Ok(())
                    }
                }
            }
            ItemType::Url => {
                Self::process_url_item(item, embedding_generator, storage_client, embedding_semaphore).await
            }
            ItemType::Website => {
                // Website processing — placeholder: log and mark done
                info!("Website processing not yet implemented for queue_id={}", item.queue_id);
                Ok(())
            }
            ItemType::Collection => {
                Self::process_collection_item(item, queue_manager, storage_client).await
            }
        }
    }

    /// Process content item (Task 37.27) - direct text ingestion
    ///
    /// When the target collection is "memory", deserializes as MemoryPayload
    /// and carries through all memory-specific fields (label, scope, title,
    /// tags, priority) into the Qdrant point payload.
    async fn process_content_item(
        item: &UnifiedQueueItem,
        embedding_generator: &Arc<EmbeddingGenerator>,
        storage_client: &Arc<StorageClient>,
        embedding_semaphore: &Arc<tokio::sync::Semaphore>,
    ) -> UnifiedProcessorResult<()> {
        info!(
            "Processing content item: {} -> collection: {}",
            item.queue_id, item.collection
        );

        // Ensure collection exists
        if !storage_client
            .collection_exists(&item.collection)
            .await
            .map_err(|e| UnifiedProcessorError::Storage(e.to_string()))?
        {
            info!("Creating collection: {}", item.collection);
            storage_client
                .create_collection(&item.collection, None, None)
                .await
                .map_err(|e| UnifiedProcessorError::Storage(e.to_string()))?;
        }

        // Route to collection-specific or generic content processing
        if item.collection == wqm_common::constants::COLLECTION_MEMORY {
            Self::process_memory_item(item, embedding_generator, storage_client, embedding_semaphore).await
        } else if item.collection == wqm_common::constants::COLLECTION_SCRATCHPAD {
            Self::process_scratchpad_item(item, embedding_generator, storage_client, embedding_semaphore).await
        } else {
            Self::process_generic_content_item(item, embedding_generator, storage_client, embedding_semaphore).await
        }
    }

    /// Process a memory rule item — preserves all memory-specific metadata
    async fn process_memory_item(
        item: &UnifiedQueueItem,
        embedding_generator: &Arc<EmbeddingGenerator>,
        storage_client: &Arc<StorageClient>,
        embedding_semaphore: &Arc<tokio::sync::Semaphore>,
    ) -> UnifiedProcessorResult<()> {
        let payload: MemoryPayload = serde_json::from_str(&item.payload_json)
            .map_err(|e| UnifiedProcessorError::InvalidPayload(format!("Failed to parse MemoryPayload: {}", e)))?;

        let action = payload.action.as_deref().unwrap_or("add");
        let now = wqm_common::timestamps::now_utc();

        // For remove action, delete by label filter and return
        if action == "remove" {
            if let Some(label) = &payload.label {
                info!("Removing memory rule with label: {}", label);
                storage_client
                    .delete_points_by_payload_field(
                        wqm_common::constants::COLLECTION_MEMORY,
                        "label",
                        label,
                    )
                    .await
                    .map_err(|e| UnifiedProcessorError::Storage(e.to_string()))?;
            }
            return Ok(());
        }

        // Generate embedding (semaphore-gated)
        let _permit = embedding_semaphore.acquire().await
            .map_err(|e| UnifiedProcessorError::Embedding(format!("Semaphore closed: {}", e)))?;
        let embedding_result = embedding_generator
            .generate_embedding(&payload.content, "bge-small-en-v1.5")
            .await
            .map_err(|e| UnifiedProcessorError::Embedding(e.to_string()))?;
        drop(_permit);

        // For update action, delete existing point by label first
        if action == "update" {
            if let Some(label) = &payload.label {
                info!("Updating memory rule with label: {} (delete + re-insert)", label);
                let _ = storage_client
                    .delete_points_by_payload_field(
                        wqm_common::constants::COLLECTION_MEMORY,
                        "label",
                        label,
                    )
                    .await;
            }
        }

        let content_doc_id = crate::generate_content_document_id(&item.tenant_id, &payload.content);

        // Build point payload with ALL memory-specific fields
        let mut point_payload = std::collections::HashMap::new();
        point_payload.insert("content".to_string(), serde_json::json!(payload.content));
        point_payload.insert("document_id".to_string(), serde_json::json!(content_doc_id));
        point_payload.insert("tenant_id".to_string(), serde_json::json!(item.tenant_id));
        point_payload.insert("branch".to_string(), serde_json::json!(item.branch));
        point_payload.insert("item_type".to_string(), serde_json::json!("content"));
        point_payload.insert("source_type".to_string(), serde_json::json!(payload.source_type.to_lowercase()));

        // Memory-specific fields
        if let Some(label) = &payload.label {
            point_payload.insert("label".to_string(), serde_json::json!(label));
        }
        if let Some(scope) = &payload.scope {
            point_payload.insert("scope".to_string(), serde_json::json!(scope));
        }
        if let Some(project_id) = &payload.project_id {
            point_payload.insert("project_id".to_string(), serde_json::json!(project_id));
        }
        if let Some(title) = &payload.title {
            point_payload.insert("title".to_string(), serde_json::json!(title));
        }
        if let Some(tags) = &payload.tags {
            // Store as comma-separated string for Qdrant keyword matching
            point_payload.insert("tags".to_string(), serde_json::json!(tags.join(",")));
        }
        if let Some(priority) = payload.priority {
            point_payload.insert("priority".to_string(), serde_json::json!(priority));
        }

        // Timestamps
        if action == "add" {
            point_payload.insert("created_at".to_string(), serde_json::json!(&now));
        }
        point_payload.insert("updated_at".to_string(), serde_json::json!(&now));

        let point = DocumentPoint {
            id: crate::generate_point_id(&item.tenant_id, &item.branch, &content_doc_id, 0),
            dense_vector: embedding_result.dense.vector,
            sparse_vector: Self::sparse_embedding_to_map(&embedding_result.sparse),
            payload: point_payload,
        };

        storage_client
            .insert_points_batch(&item.collection, vec![point], Some(1))
            .await
            .map_err(|e| UnifiedProcessorError::Storage(e.to_string()))?;

        info!(
            "Successfully processed memory item {} (action={}, label={:?}) -> {}",
            item.queue_id, action, payload.label, item.collection
        );

        Ok(())
    }

    /// Process a scratchpad item — persistent LLM scratch space
    async fn process_scratchpad_item(
        item: &UnifiedQueueItem,
        embedding_generator: &Arc<EmbeddingGenerator>,
        storage_client: &Arc<StorageClient>,
        embedding_semaphore: &Arc<tokio::sync::Semaphore>,
    ) -> UnifiedProcessorResult<()> {
        let payload: ScratchpadPayload = serde_json::from_str(&item.payload_json)
            .map_err(|e| UnifiedProcessorError::InvalidPayload(format!("Failed to parse ScratchpadPayload: {}", e)))?;

        let now = wqm_common::timestamps::now_utc();

        // Generate document ID from content hash (for idempotent updates)
        let content_doc_id = crate::generate_content_document_id(&item.tenant_id, &payload.content);

        // Generate embedding (semaphore-gated)
        let _permit = embedding_semaphore.acquire().await
            .map_err(|e| UnifiedProcessorError::Embedding(format!("Semaphore closed: {}", e)))?;
        let embedding_result = embedding_generator
            .generate_embedding(&payload.content, "all-MiniLM-L6-v2")
            .await
            .map_err(|e| UnifiedProcessorError::Embedding(e.to_string()))?;
        drop(_permit);

        // Build Qdrant payload
        let mut point_payload = std::collections::HashMap::new();
        point_payload.insert("content".to_string(), serde_json::json!(payload.content));
        point_payload.insert("document_id".to_string(), serde_json::json!(content_doc_id));
        point_payload.insert("tenant_id".to_string(), serde_json::json!(item.tenant_id));
        point_payload.insert("source_type".to_string(), serde_json::json!("scratchpad"));
        point_payload.insert("item_type".to_string(), serde_json::json!("content"));
        point_payload.insert("branch".to_string(), serde_json::json!(item.branch));
        point_payload.insert("created_at".to_string(), serde_json::json!(&now));
        point_payload.insert("updated_at".to_string(), serde_json::json!(&now));

        if let Some(ref title) = payload.title {
            point_payload.insert("title".to_string(), serde_json::json!(title));
        }
        if !payload.tags.is_empty() {
            point_payload.insert("tags".to_string(), serde_json::json!(payload.tags));
        }

        let point = DocumentPoint {
            id: crate::generate_point_id(&item.tenant_id, &item.branch, &content_doc_id, 0),
            dense_vector: embedding_result.dense.vector,
            sparse_vector: Self::sparse_embedding_to_map(&embedding_result.sparse),
            payload: point_payload,
        };

        storage_client
            .insert_points_batch(&item.collection, vec![point], Some(1))
            .await
            .map_err(|e| UnifiedProcessorError::Storage(e.to_string()))?;

        info!(
            "Successfully processed scratchpad item {} (tenant={}, title={:?}) -> {}",
            item.queue_id, item.tenant_id, payload.title, item.collection
        );

        Ok(())
    }

    /// Process a generic content item (non-memory)
    async fn process_generic_content_item(
        item: &UnifiedQueueItem,
        embedding_generator: &Arc<EmbeddingGenerator>,
        storage_client: &Arc<StorageClient>,
        embedding_semaphore: &Arc<tokio::sync::Semaphore>,
    ) -> UnifiedProcessorResult<()> {
        let payload: ContentPayload = serde_json::from_str(&item.payload_json)
            .map_err(|e| UnifiedProcessorError::InvalidPayload(format!("Failed to parse ContentPayload: {}", e)))?;

        // Generate embedding (semaphore-gated, Task 504)
        let _permit = embedding_semaphore.acquire().await
            .map_err(|e| UnifiedProcessorError::Embedding(format!("Semaphore closed: {}", e)))?;
        let embedding_result = embedding_generator
            .generate_embedding(&payload.content, "bge-small-en-v1.5")
            .await
            .map_err(|e| UnifiedProcessorError::Embedding(e.to_string()))?;
        drop(_permit);

        let content_doc_id = crate::generate_content_document_id(&item.tenant_id, &payload.content);

        // Build payload with metadata
        let mut point_payload = std::collections::HashMap::new();
        point_payload.insert("content".to_string(), serde_json::json!(payload.content));
        point_payload.insert("document_id".to_string(), serde_json::json!(content_doc_id));
        point_payload.insert("tenant_id".to_string(), serde_json::json!(item.tenant_id));
        point_payload.insert("branch".to_string(), serde_json::json!(item.branch));
        point_payload.insert("item_type".to_string(), serde_json::json!("content"));
        point_payload.insert("source_type".to_string(), serde_json::json!(payload.source_type.to_lowercase()));

        if let Some(main_tag) = &payload.main_tag {
            point_payload.insert("main_tag".to_string(), serde_json::json!(main_tag));
        }
        if let Some(full_tag) = &payload.full_tag {
            point_payload.insert("full_tag".to_string(), serde_json::json!(full_tag));
        }

        let point = DocumentPoint {
            id: crate::generate_point_id(&item.tenant_id, &item.branch, &content_doc_id, 0),
            dense_vector: embedding_result.dense.vector,
            sparse_vector: Self::sparse_embedding_to_map(&embedding_result.sparse),
            payload: point_payload,
        };

        storage_client
            .insert_points_batch(&item.collection, vec![point], Some(1))
            .await
            .map_err(|e| UnifiedProcessorError::Storage(e.to_string()))?;

        info!(
            "Successfully processed content item {} -> {}",
            item.queue_id, item.collection
        );

        Ok(())
    }

    /// Process file item (Task 37.28 + Task 506) - file-based ingestion with tracked_files
    #[allow(clippy::too_many_arguments)]
    async fn process_file_item(
        item: &UnifiedQueueItem,
        queue_manager: &QueueManager,
        document_processor: &Arc<DocumentProcessor>,
        embedding_generator: &Arc<EmbeddingGenerator>,
        storage_client: &Arc<StorageClient>,
        lsp_manager: &Option<Arc<RwLock<LanguageServerManager>>>,
        embedding_semaphore: &Arc<tokio::sync::Semaphore>,
        allowed_extensions: &Arc<AllowedExtensions>,
    ) -> UnifiedProcessorResult<()> {
        info!(
            "Processing file item: {} -> collection: {} (op={:?})",
            item.queue_id, item.collection, item.op
        );

        // Parse the file payload
        let payload: FilePayload = serde_json::from_str(&item.payload_json)
            .map_err(|e| UnifiedProcessorError::InvalidPayload(format!("Failed to parse FilePayload: {}", e)))?;

        // File type allowlist check (Task 511) - skip for delete operations
        if item.op != QueueOperation::Delete {
            if !allowed_extensions.is_allowed(&payload.file_path, &item.collection) {
                debug!(
                    "File type not in allowlist, skipping: {} (collection={})",
                    payload.file_path, item.collection
                );
                return Ok(());
            }
        }

        let file_path = Path::new(&payload.file_path);
        let pool = queue_manager.pool();

        // Look up watch_folder for tracked_files context
        let watch_info = tracked_files_schema::lookup_watch_folder(
            pool, &item.tenant_id, &item.collection,
        )
        .await
        .map_err(|e| UnifiedProcessorError::QueueOperation(format!("Failed to lookup watch_folder: {}", e)))?;

        // CRITICAL: watch_folders lookup MUST succeed before ingestion.
        // This ensures tracked_files can be updated after Qdrant write, maintaining
        // tracked_files as authoritative inventory and preventing orphaned Qdrant data.
        //
        // For library-routed files from project folders (Task 568), the item's collection
        // is "libraries" but the watch_folder has collection="projects". Fall back to
        // looking up by "projects" when the primary lookup fails.
        let (watch_folder_id, base_path) = match watch_info {
            Some((wid, bp)) => (wid, bp),
            None if item.collection == COLLECTION_LIBRARIES => {
                // Try fallback: file may originate from a project watch folder
                let fallback = tracked_files_schema::lookup_watch_folder(
                    pool, &item.tenant_id, COLLECTION_PROJECTS,
                ).await
                .map_err(|e| UnifiedProcessorError::QueueOperation(format!("Fallback watch_folder lookup failed: {}", e)))?;

                match fallback {
                    Some((wid, bp)) => {
                        debug!(
                            "Library-routed file resolved via project watch_folder: tenant={}, watch_id={}",
                            item.tenant_id, wid
                        );
                        (wid, bp)
                    }
                    None => {
                        error!(
                            "watch_folders validation failed: tenant_id={}, collection={} (also tried 'projects') -- refusing ingestion",
                            item.tenant_id, item.collection
                        );
                        return Err(UnifiedProcessorError::QueueOperation(format!(
                            "No watch_folder found for tenant_id={}, collection={} or projects. Cannot ingest without tracked_files context.",
                            item.tenant_id, item.collection
                        )));
                    }
                }
            }
            None => {
                error!(
                    "watch_folders validation failed: tenant_id={}, collection={} -- refusing ingestion to prevent orphaned data",
                    item.tenant_id, item.collection
                );
                return Err(UnifiedProcessorError::QueueOperation(format!(
                    "No watch_folder found for tenant_id={}, collection={}. Cannot ingest without tracked_files context.",
                    item.tenant_id, item.collection
                )));
            }
        };

        let relative_path = tracked_files_schema::compute_relative_path(&payload.file_path, &base_path)
            .unwrap_or_else(|| payload.file_path.clone());

        // Ensure collection exists
        if !storage_client
            .collection_exists(&item.collection)
            .await
            .map_err(|e| UnifiedProcessorError::Storage(e.to_string()))?
        {
            info!("Creating collection: {}", item.collection);
            storage_client
                .create_collection(&item.collection, None, None)
                .await
                .map_err(|e| UnifiedProcessorError::Storage(e.to_string()))?;
        }

        // === DELETE OPERATION ===
        if item.op == QueueOperation::Delete {
            return Self::process_file_delete(
                item, pool, storage_client, &watch_folder_id, &relative_path, &payload.file_path,
            ).await;
        }

        // For ingest/update: check if file exists on disk
        if !file_path.exists() {
            // File gone -- if tracked, clean up Qdrant points + SQLite records; otherwise just report not found
            if let Ok(Some(existing)) = tracked_files_schema::lookup_tracked_file(
                pool, &watch_folder_id, &relative_path, Some(item.branch.as_str()),
            ).await {
                debug!("File no longer exists, cleaning up tracked record and Qdrant points: {}", relative_path);

                // Get point IDs from qdrant_chunks before deletion
                let point_ids = tracked_files_schema::get_chunk_point_ids(pool, existing.file_id)
                    .await
                    .unwrap_or_default();

                // Delete Qdrant points first (irreversible), scoped to tenant
                if !point_ids.is_empty() {
                    if let Err(e) = storage_client
                        .delete_points_by_filter(&item.collection, &payload.file_path, &item.tenant_id)
                        .await
                    {
                        // Qdrant deletion failed but may already be gone - log and continue cleanup
                        warn!(
                            "Qdrant point deletion failed for missing file {}: {} (points may already be gone)",
                            relative_path, e
                        );
                    }
                }

                // Clean up SQLite records in a transaction (CASCADE handles qdrant_chunks)
                let tx_result: Result<(), UnifiedProcessorError> = async {
                    let mut tx = pool.begin().await
                        .map_err(|e| UnifiedProcessorError::QueueOperation(format!("Failed to begin transaction: {}", e)))?;
                    tracked_files_schema::delete_tracked_file_tx(&mut tx, existing.file_id)
                        .await
                        .map_err(|e| UnifiedProcessorError::QueueOperation(format!("tracked_files delete failed: {}", e)))?;
                    tx.commit().await
                        .map_err(|e| UnifiedProcessorError::QueueOperation(format!("Transaction commit failed: {}", e)))?;
                    Ok(())
                }.await;

                if let Err(e) = tx_result {
                    warn!(
                        "SQLite transaction failed during file-not-found cleanup for {}: {}. Marked for reconciliation on next startup.",
                        relative_path, e
                    );
                    let _ = tracked_files_schema::mark_needs_reconcile(
                        pool, existing.file_id,
                        &format!("file_not_found_cleanup_tx_failed: {}", e),
                    ).await;
                } else {
                    info!("Cleaned up {} Qdrant points and tracked record for missing file: {}", point_ids.len(), relative_path);
                }
            }
            return Err(UnifiedProcessorError::FileNotFound(payload.file_path.clone()));
        }

        // === UPDATE OPERATION: hash comparison to skip unchanged files ===
        if item.op == QueueOperation::Update {
            let new_hash = tracked_files_schema::compute_file_hash(file_path)
                .map_err(|e| UnifiedProcessorError::ProcessingFailed(format!("Failed to hash file: {}", e)))?;

            if let Ok(Some(existing)) = tracked_files_schema::lookup_tracked_file(
                pool, &watch_folder_id, &relative_path, Some(item.branch.as_str()),
            ).await {
                if existing.file_hash == new_hash {
                    info!("File unchanged (hash match), skipping update: {}", relative_path);
                    return Ok(());
                }
                // File changed -- delete old Qdrant points via tracked qdrant_chunks
                let old_point_ids = tracked_files_schema::get_chunk_point_ids(pool, existing.file_id)
                    .await
                    .unwrap_or_default();
                if !old_point_ids.is_empty() {
                    // Delete old points by filter (file_path + tenant_id) as we lack batch-by-id API
                    storage_client
                        .delete_points_by_filter(&item.collection, &payload.file_path, &item.tenant_id)
                        .await
                        .map_err(|e| UnifiedProcessorError::Storage(e.to_string()))?;
                }
                // Old chunk records will be cleaned up atomically in the transaction below
            } else {
                // Not tracked yet -- defensive cleanup: delete by filter as fallback for update
                storage_client
                    .delete_points_by_filter(&item.collection, &payload.file_path, &item.tenant_id)
                    .await
                    .map_err(|e| UnifiedProcessorError::Storage(e.to_string()))?;
            }
        }

        // === INGEST / UPDATE: process file content ===
        let document_content = document_processor
            .process_file_content(file_path, &item.collection)
            .await
            .map_err(|e| UnifiedProcessorError::ProcessingFailed(e.to_string()))?;

        info!(
            "Extracted {} chunks from {}",
            document_content.chunks.len(),
            payload.file_path
        );

        // Check if LSP enrichment is available for this project
        let (is_project_active, lsp_mgr_guard) = if let Some(lsp_mgr) = lsp_manager {
            let mgr = lsp_mgr.read().await;
            let is_active = mgr.has_active_servers(&item.tenant_id).await;
            if is_active {
                debug!(
                    "LSP enrichment available for project {} on file {}",
                    item.tenant_id, payload.file_path
                );
            }
            (is_active, Some(lsp_mgr.clone()))
        } else {
            (false, None)
        };

        // Determine LSP/treesitter status for tracked_files
        let mut lsp_status = ProcessingStatus::None;
        let mut treesitter_status = ProcessingStatus::None;

        // Generate stable document_id for this file (deterministic from tenant + path)
        let file_document_id = crate::generate_document_id(&item.tenant_id, &payload.file_path);

        // Process each chunk and build points + chunk metadata
        let mut points = Vec::new();
        let mut chunk_records: Vec<(String, i32, String, Option<TrackedChunkType>, Option<String>, Option<i32>, Option<i32>)> = Vec::new();
        let embedding_start = std::time::Instant::now();

        for (chunk_idx, chunk) in document_content.chunks.iter().enumerate() {
            // Semaphore-gated embedding generation (Task 504)
            let _permit = embedding_semaphore.acquire().await
                .map_err(|e| UnifiedProcessorError::Embedding(format!("Semaphore closed: {}", e)))?;
            let embedding_result = embedding_generator
                .generate_embedding(&chunk.content, "bge-small-en-v1.5")
                .await
                .map_err(|e| UnifiedProcessorError::Embedding(e.to_string()))?;
            drop(_permit);

            let mut point_payload = std::collections::HashMap::new();
            point_payload.insert("content".to_string(), serde_json::json!(chunk.content));
            point_payload.insert("chunk_index".to_string(), serde_json::json!(chunk.chunk_index));
            point_payload.insert("file_path".to_string(), serde_json::json!(payload.file_path));
            point_payload.insert("document_id".to_string(), serde_json::json!(file_document_id));
            point_payload.insert("tenant_id".to_string(), serde_json::json!(item.tenant_id));
            point_payload.insert("branch".to_string(), serde_json::json!(item.branch));
            point_payload.insert(
                "document_type".to_string(),
                serde_json::json!(document_content.document_type.as_str()),
            );
            if let Some(lang) = document_content.document_type.language() {
                point_payload.insert("language".to_string(), serde_json::json!(lang));
            }
            // Add file extension as metadata (e.g., "rs", "py", "md") — lowercase for consistency
            if let Some(ext) = std::path::Path::new(&payload.file_path)
                .extension()
                .and_then(|e| e.to_str())
            {
                point_payload.insert("file_extension".to_string(), serde_json::json!(ext.to_lowercase()));
            }
            point_payload.insert("item_type".to_string(), serde_json::json!("file"));

            if let Some(file_type) = &payload.file_type {
                point_payload.insert("file_type".to_string(), serde_json::json!(file_type.to_lowercase()));
            }

            // Build tags array from static metadata for filtering/aggregation
            {
                let mut tags = Vec::new();
                if let Some(ft) = &payload.file_type {
                    tags.push(ft.to_lowercase());
                }
                if let Some(lang) = document_content.document_type.language() {
                    tags.push(lang.to_string());
                }
                if let Some(ext) = std::path::Path::new(&payload.file_path)
                    .extension()
                    .and_then(|e| e.to_str())
                {
                    tags.push(ext.to_lowercase());
                }
                if is_test_file(file_path) {
                    tags.push("test".to_string());
                }
                if !tags.is_empty() {
                    point_payload.insert("tags".to_string(), serde_json::json!(tags));
                }
            }

            // Extract chunk metadata for tracked record
            let symbol_name = chunk.metadata.get("symbol_name").cloned();
            let start_line = chunk.metadata.get("start_line").and_then(|s| s.parse::<i32>().ok());
            let end_line = chunk.metadata.get("end_line").and_then(|s| s.parse::<i32>().ok());
            let chunk_type_str = chunk.metadata.get("chunk_type");
            let chunk_type = chunk_type_str.and_then(|s| TrackedChunkType::from_str(s));

            // Detect tree-sitter status from chunk metadata
            if chunk.metadata.contains_key("chunk_type") {
                treesitter_status = ProcessingStatus::Done;
            }

            for (key, value) in &chunk.metadata {
                point_payload.insert(format!("chunk_{}", key), serde_json::json!(value));
            }

            // LSP enrichment (if available)
            if let Some(lsp_mgr) = &lsp_mgr_guard {
                let mgr = lsp_mgr.read().await;

                let sym_name = chunk.metadata.get("symbol_name")
                    .map(|s| s.as_str())
                    .unwrap_or("unknown");

                let sl = chunk.metadata.get("start_line")
                    .and_then(|s| s.parse::<u32>().ok())
                    .unwrap_or(chunk_idx as u32 * 20);

                let el = chunk.metadata.get("end_line")
                    .and_then(|s| s.parse::<u32>().ok())
                    .unwrap_or(sl + 20);

                let enrichment = mgr.enrich_chunk(
                    &item.tenant_id,
                    file_path,
                    sym_name,
                    sl,
                    el,
                    is_project_active,
                ).await;

                Self::add_lsp_enrichment_to_payload(&mut point_payload, &enrichment);
                lsp_status = ProcessingStatus::Done;
            }

            let point_id = crate::generate_point_id(&item.tenant_id, &item.branch, &payload.file_path, chunk_idx);
            let content_hash = tracked_files_schema::compute_content_hash(&chunk.content);

            let point = DocumentPoint {
                id: point_id.clone(),
                dense_vector: embedding_result.dense.vector,
                sparse_vector: Self::sparse_embedding_to_map(&embedding_result.sparse),
                payload: point_payload,
            };

            points.push(point);
            chunk_records.push((
                point_id,
                chunk_idx as i32,
                content_hash,
                chunk_type,
                symbol_name,
                start_line,
                end_line,
            ));
        }

        info!("Embedding generation completed: {} chunks in {}ms", chunk_records.len(), embedding_start.elapsed().as_millis());

        // Upsert points to Qdrant
        // Task 555: If insert fails after old points were deleted (update path),
        // clean up stale SQLite chunk records before propagating the error.
        let qdrant_insert_failed = if !points.is_empty() {
            info!("Inserting {} points into {}", points.len(), item.collection);
            let upsert_start = std::time::Instant::now();
            match storage_client
                .insert_points_batch(&item.collection, points, Some(100))
                .await
            {
                Ok(_stats) => {
                    info!("Qdrant upsert completed: {} points in {}ms", chunk_records.len(), upsert_start.elapsed().as_millis());
                    None
                }
                Err(e) => Some(e.to_string()),
            }
        } else {
            None
        };

        // If Qdrant insert failed, clean up stale SQLite state before propagating
        if let Some(ref qdrant_err) = qdrant_insert_failed {
            // Old Qdrant points were deleted but new ones failed to insert.
            // Clean up stale qdrant_chunks so SQLite doesn't reference non-existent points.
            if let Ok(Some(existing)) = tracked_files_schema::lookup_tracked_file(
                pool, &watch_folder_id, &relative_path, Some(item.branch.as_str()),
            ).await {
                let cleanup_result: Result<(), String> = async {
                    let mut tx = pool.begin().await
                        .map_err(|e| format!("begin tx: {}", e))?;
                    tracked_files_schema::delete_qdrant_chunks_tx(&mut tx, existing.file_id)
                        .await
                        .map_err(|e| format!("delete chunks: {}", e))?;
                    tx.commit().await
                        .map_err(|e| format!("commit: {}", e))?;
                    Ok(())
                }.await;

                match cleanup_result {
                    Ok(()) => {
                        warn!(
                            "Qdrant insert failed for {}; cleaned up stale SQLite chunks. Error: {}",
                            relative_path, qdrant_err
                        );
                    }
                    Err(cleanup_err) => {
                        warn!(
                            "Qdrant insert failed AND chunk cleanup failed for {}: insert={}, cleanup={}",
                            relative_path, qdrant_err, cleanup_err
                        );
                        let _ = tracked_files_schema::mark_needs_reconcile(
                            pool, existing.file_id,
                            &format!("qdrant_insert_failed_cleanup_failed: {}", cleanup_err),
                        ).await;
                    }
                }
            }
            return Err(UnifiedProcessorError::Storage(qdrant_err.clone()));
        }

        // After Qdrant success: record in tracked_files + qdrant_chunks atomically (Task 519)
        let file_hash = tracked_files_schema::compute_file_hash(file_path)
            .unwrap_or_else(|_| "unknown".to_string());
        let file_mtime = tracked_files_schema::get_file_mtime(file_path)
            .unwrap_or_else(|_| wqm_common::timestamps::now_utc());
        let language = chunk_records.first()
            .and_then(|_| document_content.metadata.get("language"))
            .cloned();
        let chunking_method = if treesitter_status == ProcessingStatus::Done {
            Some("tree_sitter")
        } else {
            Some("text")
        };
        let extension = get_extension_for_storage(file_path);
        let is_test = is_test_file(file_path);

        // Check if file is already tracked (read outside transaction)
        let existing = tracked_files_schema::lookup_tracked_file(
            pool, &watch_folder_id, &relative_path, Some(item.branch.as_str()),
        )
        .await
        .map_err(|e| UnifiedProcessorError::QueueOperation(format!("tracked_files lookup failed: {}", e)))?;

        // Begin SQLite transaction for atomic tracked_files + qdrant_chunks writes
        let tx_result: Result<(), UnifiedProcessorError> = async {
            let mut tx = pool.begin().await
                .map_err(|e| UnifiedProcessorError::QueueOperation(format!("Failed to begin transaction: {}", e)))?;

            let file_id = match &existing {
                Some(existing_file) => {
                    // Update existing record
                    tracked_files_schema::update_tracked_file_tx(
                        &mut tx,
                        existing_file.file_id,
                        &file_mtime,
                        &file_hash,
                        chunk_records.len() as i32,
                        chunking_method,
                        lsp_status,
                        treesitter_status,
                    )
                    .await
                    .map_err(|e| UnifiedProcessorError::QueueOperation(format!("tracked_files update failed: {}", e)))?;
                    // Delete old chunks before inserting new
                    tracked_files_schema::delete_qdrant_chunks_tx(&mut tx, existing_file.file_id)
                        .await
                        .map_err(|e| UnifiedProcessorError::QueueOperation(format!("qdrant_chunks delete failed: {}", e)))?;
                    existing_file.file_id
                }
                None => {
                    // Insert new record
                    tracked_files_schema::insert_tracked_file_tx(
                        &mut tx,
                        &watch_folder_id,
                        &relative_path,
                        Some(item.branch.as_str()),
                        payload.file_type.as_deref(),
                        language.as_deref(),
                        &file_mtime,
                        &file_hash,
                        chunk_records.len() as i32,
                        chunking_method,
                        lsp_status,
                        treesitter_status,
                        Some(&item.collection),
                        extension.as_deref(),
                        is_test,
                    )
                    .await
                    .map_err(|e| UnifiedProcessorError::QueueOperation(format!("tracked_files insert failed: {}", e)))?
                }
            };

            // Insert qdrant_chunks
            if !chunk_records.is_empty() {
                tracked_files_schema::insert_qdrant_chunks_tx(&mut tx, file_id, &chunk_records)
                    .await
                    .map_err(|e| UnifiedProcessorError::QueueOperation(format!("qdrant_chunks insert failed: {}", e)))?;
            }

            tx.commit().await
                .map_err(|e| UnifiedProcessorError::QueueOperation(format!("Transaction commit failed: {}", e)))?;

            debug!(
                "Recorded {} chunks in tracked_files for file_id={} ({})",
                chunk_records.len(), file_id, relative_path
            );
            Ok(())
        }.await;

        // Handle transaction failure: Qdrant has points but SQLite state is inconsistent.
        // Propagate the error so the queue item enters retry instead of being deleted.
        if let Err(ref e) = tx_result {
            warn!(
                "SQLite transaction failed after Qdrant upsert for {}: {}. Queue item will be retried.",
                relative_path, e
            );
            // Mark for reconciliation as a safety net (startup recovery can fix if retries exhaust)
            if let Some(existing_file) = &existing {
                let _ = tracked_files_schema::mark_needs_reconcile(
                    pool, existing_file.file_id,
                    &format!("ingest_tx_failed: {}", e),
                ).await;
            }
        }
        tx_result?;

        info!(
            "Successfully processed file item {} ({})",
            item.queue_id, payload.file_path
        );

        Ok(())
    }

    /// Process file delete operation with tracked_files awareness (Task 506 + Task 519)
    async fn process_file_delete(
        item: &UnifiedQueueItem,
        pool: &SqlitePool,
        storage_client: &Arc<StorageClient>,
        watch_folder_id: &str,
        relative_path: &str,
        abs_file_path: &str,
    ) -> UnifiedProcessorResult<()> {
        let delete_start = std::time::Instant::now();

        // Try tracked_files lookup first for precise deletion
        // watch_folder_id is guaranteed non-empty by caller validation (Task 556)
        if let Ok(Some(existing)) = tracked_files_schema::lookup_tracked_file(
            pool, watch_folder_id, relative_path, Some(item.branch.as_str()),
        ).await {
            // Get point_ids from qdrant_chunks for targeted deletion
            let point_ids = tracked_files_schema::get_chunk_point_ids(pool, existing.file_id)
                .await
                .unwrap_or_default();

            if !point_ids.is_empty() {
                // Delete from Qdrant first (irreversible), scoped to tenant
                storage_client
                    .delete_points_by_filter(&item.collection, abs_file_path, &item.tenant_id)
                    .await
                    .map_err(|e| UnifiedProcessorError::Storage(e.to_string()))?;
            }

            // Clean up SQLite records in a transaction (CASCADE handles qdrant_chunks)
            let tx_result: Result<(), UnifiedProcessorError> = async {
                let mut tx = pool.begin().await
                    .map_err(|e| UnifiedProcessorError::QueueOperation(format!("Failed to begin transaction: {}", e)))?;
                tracked_files_schema::delete_tracked_file_tx(&mut tx, existing.file_id)
                    .await
                    .map_err(|e| UnifiedProcessorError::QueueOperation(format!("tracked_files delete failed: {}", e)))?;
                tx.commit().await
                    .map_err(|e| UnifiedProcessorError::QueueOperation(format!("Transaction commit failed: {}", e)))?;
                Ok(())
            }.await;

            if let Err(e) = tx_result {
                warn!(
                    "SQLite transaction failed after Qdrant delete for {}: {}. Marked for reconciliation on next startup.",
                    relative_path, e
                );
                let _ = tracked_files_schema::mark_needs_reconcile(
                    pool, existing.file_id,
                    &format!("delete_tx_failed: {}", e),
                ).await;
            } else {
                info!("Deleted tracked file and {} Qdrant points for: {} in {}ms", point_ids.len(), relative_path, delete_start.elapsed().as_millis());
            }
            return Ok(());
        }

        // Fallback: file not in tracked_files, attempt Qdrant filter delete (tenant-scoped)
        warn!("File not in tracked_files, falling back to Qdrant filter delete: {}", abs_file_path);
        storage_client
            .delete_points_by_filter(&item.collection, abs_file_path, &item.tenant_id)
            .await
            .map_err(|e| UnifiedProcessorError::Storage(e.to_string()))?;

        info!("Deleted points for file (fallback) in {}ms: {}", delete_start.elapsed().as_millis(), abs_file_path);
        Ok(())
    }

    /// Convert a SparseEmbedding to the HashMap format expected by DocumentPoint
    fn sparse_embedding_to_map(sparse: &SparseEmbedding) -> Option<std::collections::HashMap<u32, f32>> {
        if sparse.indices.is_empty() {
            return None;
        }
        let map: std::collections::HashMap<u32, f32> = sparse.indices.iter()
            .zip(sparse.values.iter())
            .map(|(&idx, &val)| (idx, val))
            .collect();
        Some(map)
    }

    /// Add LSP enrichment data to a point payload
    fn add_lsp_enrichment_to_payload(
        payload: &mut std::collections::HashMap<String, serde_json::Value>,
        enrichment: &LspEnrichment,
    ) {
        // Add enrichment status (lowercase for consistent metadata filtering)
        payload.insert(
            "lsp_enrichment_status".to_string(),
            serde_json::json!(enrichment.enrichment_status.as_str()),
        );

        // Skip adding empty data for non-success status
        if enrichment.enrichment_status == EnrichmentStatus::Skipped
            || enrichment.enrichment_status == EnrichmentStatus::Failed
        {
            if let Some(error) = &enrichment.error_message {
                payload.insert("lsp_enrichment_error".to_string(), serde_json::json!(error));
            }
            return;
        }

        // Add references (limited to avoid huge payloads)
        if !enrichment.references.is_empty() {
            let refs: Vec<_> = enrichment.references.iter().take(20).map(|r| {
                serde_json::json!({
                    "file": r.file,
                    "line": r.line,
                    "column": r.column
                })
            }).collect();
            payload.insert("lsp_references".to_string(), serde_json::json!(refs));
            payload.insert(
                "lsp_references_count".to_string(),
                serde_json::json!(enrichment.references.len()),
            );
        }

        // Add type info
        if let Some(type_info) = &enrichment.type_info {
            payload.insert("lsp_type_signature".to_string(), serde_json::json!(type_info.type_signature));
            payload.insert("lsp_type_kind".to_string(), serde_json::json!(type_info.kind));
            if let Some(doc) = &type_info.documentation {
                // Truncate long docs
                let truncated = if doc.len() > 500 {
                    format!("{}...", &doc[..500])
                } else {
                    doc.clone()
                };
                payload.insert("lsp_type_documentation".to_string(), serde_json::json!(truncated));
            }
        }

        // Add resolved imports
        if !enrichment.resolved_imports.is_empty() {
            let imports: Vec<_> = enrichment.resolved_imports.iter().map(|imp| {
                serde_json::json!({
                    "name": imp.import_name,
                    "target_file": imp.target_file,
                    "is_stdlib": imp.is_stdlib,
                    "resolved": imp.resolved
                })
            }).collect();
            payload.insert("lsp_imports".to_string(), serde_json::json!(imports));
        }

        // Add definition location
        if let Some(def) = &enrichment.definition {
            payload.insert("lsp_definition".to_string(), serde_json::json!({
                "file": def.file,
                "line": def.line,
                "column": def.column
            }));
        }
    }

    /// Process folder item - scanning, deletion, and update operations
    async fn process_folder_item(
        item: &UnifiedQueueItem,
        queue_manager: &QueueManager,
        storage_client: &Arc<StorageClient>,
        allowed_extensions: &Arc<AllowedExtensions>,
    ) -> UnifiedProcessorResult<()> {
        info!(
            "Processing folder item: {} (op={:?}, collection={})",
            item.queue_id, item.op, item.collection
        );

        let payload: FolderPayload = serde_json::from_str(&item.payload_json)
            .map_err(|e| UnifiedProcessorError::InvalidPayload(format!("Failed to parse FolderPayload: {}", e)))?;

        match item.op {
            QueueOperation::Scan => {
                Self::scan_library_directory(item, &payload.folder_path, queue_manager, storage_client, allowed_extensions).await
            }
            QueueOperation::Delete => {
                Self::process_folder_delete(item, &payload, queue_manager).await
            }
            QueueOperation::Update | QueueOperation::Add => {
                // Folder update/add is equivalent to a rescan
                info!(
                    "Folder {:?} operation treated as rescan for: {}",
                    item.op, payload.folder_path
                );
                Self::scan_library_directory(item, &payload.folder_path, queue_manager, storage_client, allowed_extensions).await
            }
            QueueOperation::Rename => {
                // Folder rename: not yet implemented
                info!("Folder rename not yet implemented for queue_id={}", item.queue_id);
                Ok(())
            }
            _ => {
                // Uplift, Reset not valid for folders
                warn!("Unsupported operation {:?} for folder item {}", item.op, item.queue_id);
                Ok(())
            }
        }
    }

    /// Delete all tracked files under a folder path
    ///
    /// Looks up tracked files whose relative path falls under the folder,
    /// then enqueues individual (File, Delete) items for each one.
    async fn process_folder_delete(
        item: &UnifiedQueueItem,
        payload: &FolderPayload,
        queue_manager: &QueueManager,
    ) -> UnifiedProcessorResult<()> {
        let start = std::time::Instant::now();
        let pool = queue_manager.pool();

        // Look up watch_folder to resolve paths
        let watch_info = tracked_files_schema::lookup_watch_folder(
            pool, &item.tenant_id, &item.collection,
        )
        .await
        .map_err(|e| UnifiedProcessorError::QueueOperation(format!("Failed to lookup watch_folder: {}", e)))?;

        let (watch_folder_id, base_path) = match watch_info {
            Some((wid, bp)) => (wid, bp),
            None => {
                warn!(
                    "No watch_folder for tenant_id={}, collection={} -- nothing to delete",
                    item.tenant_id, item.collection
                );
                return Ok(());
            }
        };

        // Compute relative folder path from absolute folder_path and base_path
        let relative_folder = tracked_files_schema::compute_relative_path(&payload.folder_path, &base_path)
            .unwrap_or_else(|| payload.folder_path.clone());

        // Get all tracked files under this folder prefix
        let tracked_files = tracked_files_schema::get_tracked_files_by_prefix(
            pool, &watch_folder_id, &relative_folder,
        )
        .await
        .map_err(|e| UnifiedProcessorError::QueueOperation(format!("Failed to query tracked files: {}", e)))?;

        if tracked_files.is_empty() {
            info!(
                "No tracked files found under folder '{}' (relative='{}') -- nothing to delete",
                payload.folder_path, relative_folder
            );
            return Ok(());
        }

        info!(
            "Folder delete: found {} tracked files under '{}', enqueueing file deletions",
            tracked_files.len(), relative_folder
        );

        let mut files_queued = 0u64;
        let mut errors = 0u64;

        for (file_id, rel_path, _branch) in &tracked_files {
            // Reconstruct absolute path for the file payload
            let abs_path = std::path::Path::new(&base_path).join(rel_path);
            let abs_path_str = abs_path.to_string_lossy().to_string();

            let file_payload = FilePayload {
                file_path: abs_path_str.clone(),
                file_type: None,
                file_hash: None,
                size_bytes: None,
                old_path: None,
            };

            let payload_json = serde_json::to_string(&file_payload)
                .map_err(|e| UnifiedProcessorError::ProcessingFailed(format!("Failed to serialize FilePayload: {}", e)))?;

            match queue_manager.enqueue_unified(
                ItemType::File,
                QueueOperation::Delete,
                &item.tenant_id,
                &item.collection,
                &payload_json,
                0,
                Some(&item.branch),
                None,
            ).await {
                Ok((_queue_id, is_new)) => {
                    if is_new {
                        files_queued += 1;
                        debug!("Queued file for deletion: {} (file_id={})", abs_path_str, file_id);
                    } else {
                        debug!("File deletion already in queue (deduplicated): {}", abs_path_str);
                    }
                }
                Err(e) => {
                    warn!("Failed to queue file deletion for {}: {}", abs_path_str, e);
                    errors += 1;
                }
            }
        }

        let elapsed = start.elapsed();
        info!(
            "Folder delete complete: {} files queued for deletion, {} errors in {:?} (folder={})",
            files_queued, errors, elapsed, payload.folder_path
        );

        if errors > 0 {
            warn!(
                "Folder delete had {} errors out of {} files for folder: {}",
                errors, tracked_files.len(), payload.folder_path
            );
        }

        Ok(())
    }

    /// Process project item (Task 37.29) - create/manage project collections
    async fn process_project_item(
        item: &UnifiedQueueItem,
        queue_manager: &QueueManager,
        storage_client: &Arc<StorageClient>,
        allowed_extensions: &Arc<AllowedExtensions>,
    ) -> UnifiedProcessorResult<()> {
        info!(
            "Processing project item: {} (op={:?})",
            item.queue_id, item.op
        );

        let payload: ProjectPayload = serde_json::from_str(&item.payload_json)
            .map_err(|e| UnifiedProcessorError::InvalidPayload(format!("Failed to parse ProjectPayload: {}", e)))?;

        match item.op {
            QueueOperation::Add => {
                // 1. Ensure the collection exists
                if !storage_client
                    .collection_exists(&item.collection)
                    .await
                    .map_err(|e| UnifiedProcessorError::Storage(e.to_string()))?
                {
                    info!("Creating project collection: {}", item.collection);
                    storage_client
                        .create_collection(&item.collection, None, None)
                        .await
                        .map_err(|e| UnifiedProcessorError::Storage(e.to_string()))?;
                }

                // 2. Create watch_folder entry (idempotent — skip if already exists)
                let now = wqm_common::timestamps::now_utc();
                let watch_id = uuid::Uuid::new_v4().to_string();
                let is_active: i32 = if payload.is_active.unwrap_or(false) { 1 } else { 0 };
                let insert_result = sqlx::query(
                    r#"INSERT OR IGNORE INTO watch_folders (
                        watch_id, path, collection, tenant_id, is_active,
                        git_remote_url, last_activity_at, follow_symlinks, enabled,
                        cleanup_on_disable, created_at, updated_at
                    ) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, 0, 1, 0, ?7, ?7)"#,
                )
                .bind(&watch_id)
                .bind(&payload.project_root)
                .bind(&item.collection)
                .bind(&item.tenant_id)
                .bind(is_active)
                .bind(&payload.git_remote)
                .bind(&now)
                .execute(queue_manager.pool())
                .await;

                match insert_result {
                    Ok(result) => {
                        if result.rows_affected() > 0 {
                            info!(
                                "Created watch_folder for tenant={} path={} (active={})",
                                item.tenant_id, payload.project_root, is_active
                            );
                        } else {
                            info!(
                                "Watch folder already exists for tenant={} (idempotent)",
                                item.tenant_id
                            );
                        }
                    }
                    Err(e) => {
                        return Err(UnifiedProcessorError::ProcessingFailed(
                            format!("Failed to create watch_folder: {}", e)
                        ));
                    }
                }

                // 3. Enqueue (Tenant, Scan) to trigger directory scanning
                let scan_payload_json = serde_json::to_string(&payload)
                    .map_err(|e| UnifiedProcessorError::InvalidPayload(
                        format!("Failed to serialize scan payload: {}", e)
                    ))?;

                match queue_manager.enqueue_unified(
                    ItemType::Tenant,
                    QueueOperation::Scan,
                    &item.tenant_id,
                    &item.collection,
                    &scan_payload_json,
                    0,
                    None,
                    None,
                ).await {
                    Ok((queue_id, is_new)) => {
                        if is_new {
                            info!(
                                "Enqueued project scan for tenant={} queue_id={}",
                                item.tenant_id, queue_id
                            );
                        }
                    }
                    Err(e) => {
                        warn!(
                            "Failed to enqueue project scan for tenant={}: {} (non-critical)",
                            item.tenant_id, e
                        );
                    }
                }
            }
            QueueOperation::Scan => {
                // Scan project directory and queue file ingestion items
                Self::scan_project_directory(item, &payload, queue_manager, storage_client, allowed_extensions).await?;

                // Update last_scan timestamp for this project's watch_folder
                let update_result = sqlx::query(
                    "UPDATE watch_folders SET last_scan = strftime('%Y-%m-%dT%H:%M:%fZ', 'now'), updated_at = strftime('%Y-%m-%dT%H:%M:%fZ', 'now') WHERE tenant_id = ?1 AND collection = ?2"
                )
                    .bind(&item.tenant_id)
                    .bind(COLLECTION_PROJECTS)
                    .execute(queue_manager.pool())
                    .await;

                match update_result {
                    Ok(result) => {
                        if result.rows_affected() > 0 {
                            info!("Updated last_scan for project tenant_id={}", item.tenant_id);
                        } else {
                            debug!("No watch_folder found for tenant_id={} (may not be watched)", item.tenant_id);
                        }
                    }
                    Err(e) => {
                        warn!("Failed to update last_scan for tenant_id={}: {} (non-critical)", item.tenant_id, e);
                    }
                }
            }
            QueueOperation::Delete => {
                // Delete project data (tenant-scoped, not the whole collection)
                if storage_client
                    .collection_exists(&item.collection)
                    .await
                    .map_err(|e| UnifiedProcessorError::Storage(e.to_string()))?
                {
                    info!("Deleting project data for tenant={} from collection={}", item.tenant_id, item.collection);
                    storage_client
                        .delete_points_by_tenant(&item.collection, &item.tenant_id)
                        .await
                        .map_err(|e| UnifiedProcessorError::Storage(e.to_string()))?;
                }
            }
            QueueOperation::Uplift => {
                // Query tracked_files for all files of this tenant → enqueue (Doc, Uplift) for each
                let files: Vec<(String, String)> = sqlx::query_as(
                    "SELECT file_id, file_path FROM tracked_files WHERE tenant_id = ?1"
                )
                    .bind(&item.tenant_id)
                    .fetch_all(queue_manager.pool())
                    .await
                    .map_err(|e| UnifiedProcessorError::ProcessingFailed(
                        format!("Failed to query tracked_files for uplift: {}", e)
                    ))?;

                let mut enqueued = 0u32;
                for (file_id, file_path) in &files {
                    let doc_payload = serde_json::json!({
                        "file_id": file_id,
                        "file_path": file_path,
                    }).to_string();

                    if let Ok((_, true)) = queue_manager.enqueue_unified(
                        ItemType::Doc,
                        QueueOperation::Uplift,
                        &item.tenant_id,
                        &item.collection,
                        &doc_payload,
                        0,
                        None,
                        None,
                    ).await {
                        enqueued += 1;
                    }
                }
                info!(
                    "Tenant uplift: enqueued {}/{} doc uplift items for tenant={}",
                    enqueued, files.len(), item.tenant_id
                );
            }
            _ => {
                warn!(
                    "Unsupported operation {:?} for project item {}",
                    item.op, item.queue_id
                );
            }
        }

        info!(
            "Successfully processed project item {} (project_root={})",
            item.queue_id, payload.project_root
        );

        Ok(())
    }

    /// Scan a project directory and queue file ingestion items
    ///
    /// Walks the project directory recursively, filters files using exclusion rules
    /// and the file type allowlist (Task 511), and queues (File, Add) items for
    /// each eligible file.
    async fn scan_project_directory(
        item: &UnifiedQueueItem,
        payload: &ProjectPayload,
        queue_manager: &QueueManager,
        storage_client: &Arc<StorageClient>,
        allowed_extensions: &Arc<AllowedExtensions>,
    ) -> UnifiedProcessorResult<()> {
        let project_root = Path::new(&payload.project_root);

        if !project_root.exists() {
            return Err(UnifiedProcessorError::FileNotFound(format!(
                "Project root does not exist: {}",
                payload.project_root
            )));
        }

        if !project_root.is_dir() {
            return Err(UnifiedProcessorError::InvalidPayload(format!(
                "Project root is not a directory: {}",
                payload.project_root
            )));
        }

        // Ensure collection exists before scanning
        if !storage_client
            .collection_exists(&item.collection)
            .await
            .map_err(|e| UnifiedProcessorError::Storage(e.to_string()))?
        {
            info!("Creating project collection for scan: {}", item.collection);
            storage_client
                .create_collection(&item.collection, None, None)
                .await
                .map_err(|e| UnifiedProcessorError::Storage(e.to_string()))?;
        }

        info!(
            "Scanning project directory: {} (tenant_id={})",
            payload.project_root, item.tenant_id
        );

        let mut files_queued = 0u64;
        let mut files_excluded = 0u64;
        let mut errors = 0u64;
        let start_time = std::time::Instant::now();

        // Walk directory recursively, skipping excluded directories entirely
        for entry in WalkDir::new(project_root)
            .follow_links(false)  // Don't follow symlinks to avoid cycles
            .into_iter()
            .filter_entry(|e| {
                if e.file_type().is_dir() && e.depth() > 0 {
                    let dir_name = e.file_name().to_string_lossy();
                    !should_exclude_directory(&dir_name)
                } else {
                    true
                }
            })
            .filter_map(|e| e.ok())  // Skip entries with errors
        {
            let path = entry.path();

            // Skip directories - we only process files
            if !path.is_file() {
                continue;
            }

            // Get relative path for pattern matching
            let rel_path = path
                .strip_prefix(project_root)
                .unwrap_or(path)
                .to_string_lossy();

            // Check exclusion rules using the relative path
            if should_exclude_file(&rel_path) {
                files_excluded += 1;
                continue;
            }

            // Also check absolute path for completeness
            let abs_path = path.to_string_lossy();
            if should_exclude_file(&abs_path) {
                files_excluded += 1;
                continue;
            }

            // Check file type allowlist (Task 511)
            if !allowed_extensions.is_allowed(&abs_path, &item.collection) {
                files_excluded += 1;
                continue;
            }

            // Get file metadata for the payload
            let metadata = match path.metadata() {
                Ok(m) => m,
                Err(e) => {
                    warn!("Failed to get metadata for {}: {}", abs_path, e);
                    errors += 1;
                    continue;
                }
            };

            // Skip files that are too large (100MB limit)
            const MAX_FILE_SIZE: u64 = 100 * 1024 * 1024;
            if metadata.len() > MAX_FILE_SIZE {
                debug!("Skipping large file: {} ({} bytes)", abs_path, metadata.len());
                files_excluded += 1;
                continue;
            }

            // Classify file type
            let file_type = classify_file_type(path);

            // Create file payload
            let file_payload = FilePayload {
                file_path: abs_path.to_string(),
                file_type: Some(file_type.as_str().to_string()),
                file_hash: None,  // Hash will be computed during processing
                size_bytes: Some(metadata.len()),
                old_path: None,
            };

            let payload_json = serde_json::to_string(&file_payload)
                .map_err(|e| UnifiedProcessorError::ProcessingFailed(format!("Failed to serialize FilePayload: {}", e)))?;

            // Queue the file for ingestion
            // Priority is computed at dequeue time via CASE/JOIN, not stored
            match queue_manager.enqueue_unified(
                ItemType::File,
                QueueOperation::Add,
                &item.tenant_id,
                &item.collection,
                &payload_json,
                0,  // Priority is dynamic (computed at dequeue time)
                Some(&item.branch),
                None,
            ).await {
                Ok((queue_id, is_new)) => {
                    if is_new {
                        files_queued += 1;
                        debug!("Queued file for ingestion: {} (queue_id={})", abs_path, queue_id);
                    } else {
                        debug!("File already in queue (deduplicated): {}", abs_path);
                    }
                }
                Err(e) => {
                    warn!("Failed to queue file {}: {}", abs_path, e);
                    errors += 1;
                }
            }

            // Yield periodically to avoid blocking the async runtime
            if files_queued % 100 == 0 && files_queued > 0 {
                tokio::task::yield_now().await;
            }
        }

        // After scanning, clean up any excluded files that were previously indexed
        let files_cleaned = Self::cleanup_excluded_files(
            item,
            project_root,
            queue_manager,
            storage_client,
            allowed_extensions,
        ).await?;

        let elapsed = start_time.elapsed();
        info!(
            "Project scan complete: {} files queued, {} excluded, {} queued for deletion, {} errors in {:?} (project={})",
            files_queued, files_excluded, files_cleaned, errors, elapsed, payload.project_root
        );

        Ok(())
    }

    /// Scan a library directory and enqueue files for ingestion (Task 523)
    ///
    /// Similar to scan_project_directory but for library folders:
    /// - Uses tenant_id as library name
    /// - Targets the 'libraries' collection
    /// - No branch tracking (libraries are not Git repos)
    async fn scan_library_directory(
        item: &UnifiedQueueItem,
        folder_path: &str,
        queue_manager: &QueueManager,
        storage_client: &Arc<StorageClient>,
        allowed_extensions: &Arc<AllowedExtensions>,
    ) -> UnifiedProcessorResult<()> {
        let library_root = Path::new(folder_path);

        if !library_root.exists() {
            return Err(UnifiedProcessorError::FileNotFound(format!(
                "Library path does not exist: {}", folder_path
            )));
        }

        if !library_root.is_dir() {
            return Err(UnifiedProcessorError::InvalidPayload(format!(
                "Library path is not a directory: {}", folder_path
            )));
        }

        // Ensure the libraries collection exists
        if !storage_client
            .collection_exists(&item.collection)
            .await
            .map_err(|e| UnifiedProcessorError::Storage(e.to_string()))?
        {
            info!("Creating libraries collection for scan: {}", item.collection);
            storage_client
                .create_collection(&item.collection, None, None)
                .await
                .map_err(|e| UnifiedProcessorError::Storage(e.to_string()))?;
        }

        info!(
            "Scanning library directory: {} (tenant_id={})",
            folder_path, item.tenant_id
        );

        let mut files_queued = 0u64;
        let mut files_excluded = 0u64;
        let mut errors = 0u64;
        let start_time = std::time::Instant::now();

        for entry in WalkDir::new(library_root)
            .follow_links(false)
            .into_iter()
            .filter_entry(|e| {
                if e.file_type().is_dir() && e.depth() > 0 {
                    let dir_name = e.file_name().to_string_lossy();
                    !should_exclude_directory(&dir_name)
                } else {
                    true
                }
            })
            .filter_map(|e| e.ok())
        {
            let path = entry.path();

            if !path.is_file() {
                continue;
            }

            let rel_path = path
                .strip_prefix(library_root)
                .unwrap_or(path)
                .to_string_lossy();

            if should_exclude_file(&rel_path) {
                files_excluded += 1;
                continue;
            }

            let abs_path = path.to_string_lossy();
            if should_exclude_file(&abs_path) {
                files_excluded += 1;
                continue;
            }

            if !allowed_extensions.is_allowed(&abs_path, &item.collection) {
                files_excluded += 1;
                continue;
            }

            let metadata = match path.metadata() {
                Ok(m) => m,
                Err(e) => {
                    warn!("Failed to get metadata for {}: {}", abs_path, e);
                    errors += 1;
                    continue;
                }
            };

            const MAX_FILE_SIZE: u64 = 100 * 1024 * 1024;
            if metadata.len() > MAX_FILE_SIZE {
                debug!("Skipping large file: {} ({} bytes)", abs_path, metadata.len());
                files_excluded += 1;
                continue;
            }

            let file_type = classify_file_type(path);

            let file_payload = FilePayload {
                file_path: abs_path.to_string(),
                file_type: Some(file_type.as_str().to_string()),
                file_hash: None,
                size_bytes: Some(metadata.len()),
                old_path: None,
            };

            let payload_json = serde_json::to_string(&file_payload)
                .map_err(|e| UnifiedProcessorError::ProcessingFailed(format!("Failed to serialize FilePayload: {}", e)))?;

            match queue_manager.enqueue_unified(
                ItemType::File,
                QueueOperation::Add,
                &item.tenant_id,
                &item.collection,
                &payload_json,
                0,
                Some(""),  // No branch for libraries
                None,
            ).await {
                Ok((queue_id, is_new)) => {
                    if is_new {
                        files_queued += 1;
                        debug!("Queued library file for ingestion: {} (queue_id={})", abs_path, queue_id);
                    } else {
                        debug!("Library file already in queue (deduplicated): {}", abs_path);
                    }
                }
                Err(e) => {
                    warn!("Failed to queue library file {}: {}", abs_path, e);
                    errors += 1;
                }
            }

            if files_queued % 100 == 0 && files_queued > 0 {
                tokio::task::yield_now().await;
            }
        }

        // Update last_scan timestamp for this library's watch_folder
        let update_result = sqlx::query(
            "UPDATE watch_folders SET last_scan = strftime('%Y-%m-%dT%H:%M:%fZ', 'now'), updated_at = strftime('%Y-%m-%dT%H:%M:%fZ', 'now') WHERE tenant_id = ?1 AND collection = ?2"
        )
            .bind(&item.tenant_id)
            .bind(COLLECTION_LIBRARIES)
            .execute(queue_manager.pool())
            .await;

        if let Err(e) = update_result {
            warn!("Failed to update last_scan for library {}: {}", item.tenant_id, e);
        }

        let elapsed = start_time.elapsed();
        info!(
            "Library scan complete: {} files queued, {} excluded, {} errors in {:?} (library={})",
            files_queued, files_excluded, errors, elapsed, folder_path
        );

        Ok(())
    }

    /// Clean up excluded files after a scan completes (Task 506)
    ///
    /// Queries tracked_files (fast SQLite) instead of scrolling Qdrant.
    /// Checks each tracked file against current exclusion rules and the
    /// file type allowlist (Task 511), queuing deletion for files that
    /// now match exclusion patterns or are no longer in the allowlist.
    async fn cleanup_excluded_files(
        item: &UnifiedQueueItem,
        project_root: &Path,
        queue_manager: &QueueManager,
        storage_client: &Arc<StorageClient>,
        allowed_extensions: &Arc<AllowedExtensions>,
    ) -> UnifiedProcessorResult<u64> {
        let pool = queue_manager.pool();

        // Look up watch_folder for this project
        let watch_info = tracked_files_schema::lookup_watch_folder(
            pool, &item.tenant_id, &item.collection,
        )
        .await
        .map_err(|e| UnifiedProcessorError::QueueOperation(format!("Failed to lookup watch_folder: {}", e)))?;

        let (watch_folder_id, base_path) = match &watch_info {
            Some((wid, bp)) => (wid.as_str(), bp.as_str()),
            None => {
                // No watch_folder -- fall back to Qdrant scroll for backward compatibility
                debug!("No watch_folder for tenant_id={}, falling back to Qdrant scroll for cleanup", item.tenant_id);
                return Self::cleanup_excluded_files_qdrant_fallback(
                    item, project_root, queue_manager, storage_client, allowed_extensions,
                ).await;
            }
        };

        // Query tracked_files for all files in this project (fast SQLite query)
        let tracked_files = tracked_files_schema::get_tracked_file_paths(pool, watch_folder_id)
            .await
            .map_err(|e| {
                error!("Failed to query tracked_files for exclusion cleanup: {}", e);
                UnifiedProcessorError::QueueOperation(e.to_string())
            })?;

        if tracked_files.is_empty() {
            debug!(
                "No tracked files for watch_folder_id='{}', skipping exclusion cleanup",
                watch_folder_id
            );
            return Ok(0);
        }

        info!(
            "Checking {} tracked files against exclusion rules (watch_folder_id={})",
            tracked_files.len(), watch_folder_id
        );

        let mut files_cleaned = 0u64;

        for (_file_id, rel_path, _branch) in &tracked_files {
            // Check if this file should now be excluded (pattern or allowlist)
            let should_clean = should_exclude_file(rel_path) || {
                let abs_path = Path::new(base_path).join(rel_path);
                !allowed_extensions.is_allowed(&abs_path.to_string_lossy(), &item.collection)
            };

            if !should_clean {
                continue;
            }

            // Reconstruct absolute path for the queue payload
            let abs_path = Path::new(base_path).join(rel_path);
            let abs_path_str = abs_path.to_string_lossy().to_string();

            let file_payload = FilePayload {
                file_path: abs_path_str.clone(),
                file_type: None,
                file_hash: None,
                size_bytes: None,
                old_path: None,
            };

            let payload_json = serde_json::to_string(&file_payload).map_err(|e| {
                UnifiedProcessorError::ProcessingFailed(format!(
                    "Failed to serialize FilePayload for deletion: {}",
                    e
                ))
            })?;

            match queue_manager
                .enqueue_unified(
                    ItemType::File,
                    QueueOperation::Delete,
                    &item.tenant_id,
                    &item.collection,
                    &payload_json,
                    0,
                    Some(&item.branch),
                    None,
                )
                .await
            {
                Ok((_queue_id, is_new)) => {
                    if is_new {
                        files_cleaned += 1;
                        debug!(
                            "Queued excluded file for deletion: {} (rel={})",
                            abs_path_str, rel_path
                        );
                    } else {
                        debug!(
                            "Excluded file deletion already in queue (deduplicated): {}",
                            abs_path_str
                        );
                    }
                }
                Err(e) => {
                    warn!(
                        "Failed to queue excluded file for deletion {}: {}",
                        abs_path_str, e
                    );
                }
            }
        }

        if files_cleaned > 0 {
            info!(
                "Queued {} excluded files for deletion (watch_folder_id={})",
                files_cleaned, watch_folder_id
            );
        }

        Ok(files_cleaned)
    }

    /// Fallback cleanup using Qdrant scroll (for when tracked_files is not available)
    async fn cleanup_excluded_files_qdrant_fallback(
        item: &UnifiedQueueItem,
        project_root: &Path,
        queue_manager: &QueueManager,
        storage_client: &Arc<StorageClient>,
        allowed_extensions: &Arc<AllowedExtensions>,
    ) -> UnifiedProcessorResult<u64> {
        if !storage_client
            .collection_exists(&item.collection)
            .await
            .map_err(|e| UnifiedProcessorError::Storage(e.to_string()))?
        {
            return Ok(0);
        }

        let qdrant_file_paths = match storage_client
            .scroll_file_paths_by_tenant(&item.collection, &item.tenant_id)
            .await
        {
            Ok(paths) => paths,
            Err(e) => {
                error!("Failed to scroll Qdrant for exclusion cleanup: {}", e);
                return Ok(0);
            }
        };

        if qdrant_file_paths.is_empty() {
            return Ok(0);
        }

        let mut files_cleaned = 0u64;

        for qdrant_file in &qdrant_file_paths {
            let rel_path = match Path::new(qdrant_file).strip_prefix(project_root) {
                Ok(stripped) => stripped.to_string_lossy().to_string(),
                Err(_) => qdrant_file.clone(),
            };

            // Check exclusion patterns and allowlist (Task 511)
            let should_clean = should_exclude_file(&rel_path)
                || !allowed_extensions.is_allowed(qdrant_file, &item.collection);

            if !should_clean {
                continue;
            }

            let file_payload = FilePayload {
                file_path: qdrant_file.clone(),
                file_type: None,
                file_hash: None,
                size_bytes: None,
                old_path: None,
            };

            let payload_json = serde_json::to_string(&file_payload).map_err(|e| {
                UnifiedProcessorError::ProcessingFailed(format!(
                    "Failed to serialize FilePayload for deletion: {}",
                    e
                ))
            })?;

            match queue_manager
                .enqueue_unified(
                    ItemType::File,
                    QueueOperation::Delete,
                    &item.tenant_id,
                    &item.collection,
                    &payload_json,
                    0,
                    Some(&item.branch),
                    None,
                )
                .await
            {
                Ok((_queue_id, is_new)) => {
                    if is_new {
                        files_cleaned += 1;
                    }
                }
                Err(e) => {
                    warn!("Failed to queue excluded file for deletion {}: {}", qdrant_file, e);
                }
            }
        }

        if files_cleaned > 0 {
            info!(
                "Queued {} excluded files for deletion via Qdrant fallback (tenant_id={})",
                files_cleaned, item.tenant_id
            );
        }

        Ok(files_cleaned)
    }

    /// Process library item - create/manage library collections
    async fn process_library_item(
        item: &UnifiedQueueItem,
        queue_manager: &QueueManager,
        storage_client: &Arc<StorageClient>,
        allowed_extensions: &Arc<AllowedExtensions>,
    ) -> UnifiedProcessorResult<()> {
        info!(
            "Processing library item: {} (op={:?})",
            item.queue_id, item.op
        );

        let payload: LibraryPayload = serde_json::from_str(&item.payload_json)
            .map_err(|e| UnifiedProcessorError::InvalidPayload(format!("Failed to parse LibraryPayload: {}", e)))?;

        match item.op {
            QueueOperation::Add => {
                // Create collection for the library
                if !storage_client
                    .collection_exists(&item.collection)
                    .await
                    .map_err(|e| UnifiedProcessorError::Storage(e.to_string()))?
                {
                    info!("Creating library collection: {}", item.collection);
                    storage_client
                        .create_collection(&item.collection, None, None)
                        .await
                        .map_err(|e| UnifiedProcessorError::Storage(e.to_string()))?;
                }
            }
            QueueOperation::Scan => {
                // Scan library directory - look up path from watch_folders
                let pool = queue_manager.pool();
                let folder_path: Option<String> = sqlx::query_scalar(
                    "SELECT path FROM watch_folders WHERE tenant_id = ?1 AND collection = ?2"
                )
                    .bind(&item.tenant_id)
                    .bind(COLLECTION_LIBRARIES)
                    .fetch_optional(pool)
                    .await
                    .map_err(|e| UnifiedProcessorError::QueueOperation(format!("Failed to lookup library path: {}", e)))?;

                match folder_path {
                    Some(path) => {
                        Self::scan_library_directory(item, &path, queue_manager, storage_client, allowed_extensions).await?;
                    }
                    None => {
                        warn!("Library '{}' not found in watch_folders", item.tenant_id);
                    }
                }
            }
            QueueOperation::Delete => {
                // Delete library data from collection
                if storage_client
                    .collection_exists(&item.collection)
                    .await
                    .map_err(|e| UnifiedProcessorError::Storage(e.to_string()))?
                {
                    info!("Deleting library data for tenant={} from collection={}", item.tenant_id, item.collection);
                    storage_client
                        .delete_points_by_tenant(&item.collection, &item.tenant_id)
                        .await
                        .map_err(|e| UnifiedProcessorError::Storage(e.to_string()))?;
                }
            }
            _ => {
                warn!(
                    "Unsupported operation {:?} for library item {}",
                    item.op, item.queue_id
                );
            }
        }

        info!(
            "Successfully processed library item {} (library={})",
            item.queue_id, payload.library_name
        );

        Ok(())
    }

    /// Process collection item — Uplift or Reset operations
    ///
    /// - Uplift: cascade to all tenants in the collection → (Tenant, Uplift) each
    /// - Reset: delete all Qdrant points + SQLite data for all tenants, preserve schema
    async fn process_collection_item(
        item: &UnifiedQueueItem,
        queue_manager: &QueueManager,
        storage_client: &Arc<StorageClient>,
    ) -> UnifiedProcessorResult<()> {
        info!(
            "Processing collection item: {} (op={:?}, collection={})",
            item.queue_id, item.op, item.collection
        );

        match item.op {
            QueueOperation::Uplift => {
                // Query all tenants in this collection → enqueue (Tenant, Uplift) for each
                let tenants: Vec<(String,)> = sqlx::query_as(
                    "SELECT DISTINCT tenant_id FROM watch_folders WHERE collection = ?1"
                )
                    .bind(&item.collection)
                    .fetch_all(queue_manager.pool())
                    .await
                    .map_err(|e| UnifiedProcessorError::ProcessingFailed(
                        format!("Failed to query tenants for collection uplift: {}", e)
                    ))?;

                let mut enqueued = 0u32;
                for (tenant_id,) in &tenants {
                    let payload = serde_json::json!({
                        "project_root": "",
                    }).to_string();

                    if let Ok((_, true)) = queue_manager.enqueue_unified(
                        ItemType::Tenant,
                        QueueOperation::Uplift,
                        tenant_id,
                        &item.collection,
                        &payload,
                        0,
                        None,
                        None,
                    ).await {
                        enqueued += 1;
                    }
                }
                info!(
                    "Collection uplift: enqueued {}/{} tenant uplift items for collection={}",
                    enqueued, tenants.len(), item.collection
                );
            }
            QueueOperation::Reset => {
                // 1. Delete all Qdrant points in the collection
                if storage_client
                    .collection_exists(&item.collection)
                    .await
                    .map_err(|e| UnifiedProcessorError::Storage(e.to_string()))?
                {
                    // Get all tenant_ids in this collection
                    let tenants: Vec<(String,)> = sqlx::query_as(
                        "SELECT DISTINCT tenant_id FROM watch_folders WHERE collection = ?1"
                    )
                        .bind(&item.collection)
                        .fetch_all(queue_manager.pool())
                        .await
                        .map_err(|e| UnifiedProcessorError::ProcessingFailed(
                            format!("Failed to query tenants for reset: {}", e)
                        ))?;

                    for (tenant_id,) in &tenants {
                        storage_client
                            .delete_points_by_tenant(&item.collection, tenant_id)
                            .await
                            .map_err(|e| UnifiedProcessorError::Storage(e.to_string()))?;
                    }
                    info!("Reset: deleted Qdrant points for {} tenants in collection={}", tenants.len(), item.collection);
                }

                // 2. Delete SQLite data in a transaction (preserve watch_folders)
                let pool = queue_manager.pool();
                let mut tx = pool.begin().await
                    .map_err(|e| UnifiedProcessorError::ProcessingFailed(
                        format!("Failed to begin reset transaction: {}", e)
                    ))?;

                // Get all tenant_ids from watch_folders for this collection
                let tenant_ids: Vec<(String,)> = sqlx::query_as(
                    "SELECT DISTINCT tenant_id FROM watch_folders WHERE collection = ?1"
                )
                    .bind(&item.collection)
                    .fetch_all(&mut *tx)
                    .await
                    .map_err(|e| UnifiedProcessorError::ProcessingFailed(
                        format!("Failed to query tenants: {}", e)
                    ))?;

                for (tenant_id,) in &tenant_ids {
                    // Delete qdrant_chunks for this tenant
                    let _ = sqlx::query(
                        r#"DELETE FROM qdrant_chunks WHERE file_id IN (
                            SELECT file_id FROM tracked_files WHERE tenant_id = ?1
                        )"#
                    )
                        .bind(tenant_id)
                        .execute(&mut *tx)
                        .await;

                    // Delete tracked_files for this tenant
                    let _ = sqlx::query("DELETE FROM tracked_files WHERE tenant_id = ?1")
                        .bind(tenant_id)
                        .execute(&mut *tx)
                        .await;
                }

                tx.commit().await
                    .map_err(|e| UnifiedProcessorError::ProcessingFailed(
                        format!("Failed to commit reset transaction: {}", e)
                    ))?;

                info!(
                    "Reset: cleared SQLite data for {} tenants in collection={} (watch_folders preserved)",
                    tenant_ids.len(), item.collection
                );
            }
            _ => {
                warn!(
                    "Unsupported operation {:?} for collection item {}",
                    item.op, item.queue_id
                );
            }
        }

        Ok(())
    }

    /// Process delete tenant item - delete all data for a tenant
    ///
    /// Full deletion cascade:
    /// 1. Delete Qdrant points (tenant-scoped)
    /// 2. Delete SQLite: qdrant_chunks, tracked_files, watch_folders
    async fn process_delete_tenant_item(
        item: &UnifiedQueueItem,
        queue_manager: &QueueManager,
        storage_client: &Arc<StorageClient>,
    ) -> UnifiedProcessorResult<()> {
        info!(
            "Processing delete tenant item: {} (tenant={}, collection={})",
            item.queue_id, item.tenant_id, item.collection
        );

        // 1. Delete Qdrant points (tenant-scoped)
        if storage_client
            .collection_exists(&item.collection)
            .await
            .map_err(|e| UnifiedProcessorError::Storage(e.to_string()))?
        {
            storage_client
                .delete_points_by_tenant(&item.collection, &item.tenant_id)
                .await
                .map_err(|e| UnifiedProcessorError::Storage(e.to_string()))?;
            info!("Deleted Qdrant points for tenant={} collection={}", item.tenant_id, item.collection);
        }

        // 2. Delete SQLite state in a transaction (child tables first)
        let pool = queue_manager.pool();
        let mut tx = pool.begin().await
            .map_err(|e| UnifiedProcessorError::ProcessingFailed(
                format!("Failed to begin delete transaction: {}", e)
            ))?;

        // 2a. Delete qdrant_chunks (child of tracked_files)
        match sqlx::query(
            r#"DELETE FROM qdrant_chunks WHERE file_id IN (
                SELECT file_id FROM tracked_files WHERE tenant_id = ?1
            )"#
        )
            .bind(&item.tenant_id)
            .execute(&mut *tx)
            .await
        {
            Ok(result) => {
                if result.rows_affected() > 0 {
                    info!("Deleted {} qdrant_chunks rows for tenant={}", result.rows_affected(), item.tenant_id);
                }
            }
            Err(e) => {
                // Table may not exist yet, non-fatal
                debug!("qdrant_chunks delete for tenant={}: {}", item.tenant_id, e);
            }
        }

        // 2b. Delete tracked_files
        match sqlx::query("DELETE FROM tracked_files WHERE tenant_id = ?1")
            .bind(&item.tenant_id)
            .execute(&mut *tx)
            .await
        {
            Ok(result) => {
                if result.rows_affected() > 0 {
                    info!("Deleted {} tracked_files for tenant={}", result.rows_affected(), item.tenant_id);
                }
            }
            Err(e) => {
                debug!("tracked_files delete for tenant={}: {}", item.tenant_id, e);
            }
        }

        // 2c. Delete watch_folders
        match sqlx::query("DELETE FROM watch_folders WHERE tenant_id = ?1")
            .bind(&item.tenant_id)
            .execute(&mut *tx)
            .await
        {
            Ok(result) => {
                if result.rows_affected() > 0 {
                    info!("Deleted {} watch_folders for tenant={}", result.rows_affected(), item.tenant_id);
                }
            }
            Err(e) => {
                debug!("watch_folders delete for tenant={}: {}", item.tenant_id, e);
            }
        }

        tx.commit().await
            .map_err(|e| UnifiedProcessorError::ProcessingFailed(
                format!("Failed to commit delete transaction: {}", e)
            ))?;

        info!(
            "Successfully processed delete tenant item {} (tenant={})",
            item.queue_id, item.tenant_id
        );

        Ok(())
    }

    /// Process delete document item - delete specific document by document_id
    async fn process_delete_document_item(
        item: &UnifiedQueueItem,
        storage_client: &Arc<StorageClient>,
    ) -> UnifiedProcessorResult<()> {
        info!(
            "Processing delete document item: {}",
            item.queue_id
        );

        // Use typed payload deserialization (matches validation in queue_operations.rs)
        let payload = item.parse_delete_document_payload()
            .map_err(|e| UnifiedProcessorError::InvalidPayload(format!("Failed to parse DeleteDocumentPayload: {}", e)))?;

        if payload.document_id.trim().is_empty() {
            return Err(UnifiedProcessorError::InvalidPayload(
                "document_id must not be empty".to_string(),
            ));
        }

        if storage_client
            .collection_exists(&item.collection)
            .await
            .map_err(|e| UnifiedProcessorError::Storage(e.to_string()))?
        {
            storage_client
                .delete_points_by_document_id(&item.collection, &payload.document_id)
                .await
                .map_err(|e| UnifiedProcessorError::Storage(e.to_string()))?;
        }

        info!(
            "Successfully deleted document {} from {} (tenant={})",
            payload.document_id, item.collection, item.tenant_id
        );

        Ok(())
    }

    /// Process tenant rename item - update tenant_id on all matching Qdrant points.
    ///
    /// Uses ProjectPayload with old_tenant_id field.
    async fn process_tenant_rename_item(
        item: &UnifiedQueueItem,
        storage_client: &Arc<StorageClient>,
    ) -> UnifiedProcessorResult<()> {
        let payload: ProjectPayload = serde_json::from_str(&item.payload_json)
            .map_err(|e| UnifiedProcessorError::InvalidPayload(format!("Failed to parse ProjectPayload for rename: {}", e)))?;

        let old_tenant = payload.old_tenant_id.as_deref()
            .ok_or_else(|| UnifiedProcessorError::InvalidPayload("Missing old_tenant_id in tenant rename payload".to_string()))?;
        let new_tenant = &item.tenant_id;

        // Extract reason from metadata if available
        let reason = item.metadata.as_deref()
            .and_then(|m| serde_json::from_str::<serde_json::Value>(m).ok())
            .and_then(|v| v.get("reason").and_then(|r| r.as_str().map(String::from)))
            .unwrap_or_else(|| "unknown".to_string());

        info!(
            "Processing tenant rename: {} -> {} in collection '{}' (reason: {})",
            old_tenant, new_tenant, item.collection, reason
        );

        use qdrant_client::qdrant::{Condition, Filter};
        let filter = Filter::must([
            Condition::matches("tenant_id", old_tenant.to_string()),
        ]);

        let mut new_payload = std::collections::HashMap::new();
        new_payload.insert("tenant_id".to_string(), serde_json::Value::String(new_tenant.to_string()));

        storage_client
            .set_payload_by_filter(&item.collection, filter, new_payload)
            .await
            .map_err(|e| UnifiedProcessorError::Storage(e.to_string()))?;

        info!(
            "Successfully processed tenant rename {} -> {} in '{}'",
            old_tenant, new_tenant, item.collection
        );

        Ok(())
    }

    /// Process URL fetch and ingestion item
    ///
    /// Fetches content from a URL, extracts text (using html2text for HTML),
    /// generates embeddings, and stores in Qdrant. Supports both single-page
    /// fetch and crawl mode.
    async fn process_url_item(
        item: &UnifiedQueueItem,
        embedding_generator: &Arc<EmbeddingGenerator>,
        storage_client: &Arc<StorageClient>,
        embedding_semaphore: &Arc<tokio::sync::Semaphore>,
    ) -> UnifiedProcessorResult<()> {
        let payload: UrlPayload = serde_json::from_str(&item.payload_json)
            .map_err(|e| UnifiedProcessorError::InvalidPayload(
                format!("Failed to parse UrlPayload: {}", e)
            ))?;

        info!("Processing URL item: {} (url={})", item.queue_id, payload.url);

        // Fetch URL content
        let response = reqwest::get(&payload.url)
            .await
            .map_err(|e| UnifiedProcessorError::ProcessingFailed(
                format!("Failed to fetch URL {}: {}", payload.url, e)
            ))?;

        let status = response.status();
        if !status.is_success() {
            return Err(UnifiedProcessorError::ProcessingFailed(
                format!("HTTP {} for URL {}", status, payload.url)
            ));
        }

        let content_type = response
            .headers()
            .get("content-type")
            .and_then(|v| v.to_str().ok())
            .unwrap_or("text/html")
            .to_string();

        let body = response.text().await.map_err(|e| {
            UnifiedProcessorError::ProcessingFailed(format!("Failed to read response body: {}", e))
        })?;

        let is_html = content_type.contains("text/html");

        // Extract title from HTML before text extraction
        let title = payload.title.unwrap_or_else(|| {
            if is_html {
                // Simple title extraction: find <title>...</title>
                let lower = body.to_lowercase();
                if let Some(start) = lower.find("<title>") {
                    let title_start = start + 7;
                    if let Some(end) = lower[title_start..].find("</title>") {
                        return body[title_start..title_start + end].trim().to_string();
                    }
                }
            }
            payload.url.clone()
        });

        // Extract text based on content type
        let extracted_text = if is_html {
            html2text::from_read(body.as_bytes(), 80)
        } else {
            body
        };

        if extracted_text.trim().is_empty() {
            warn!("URL {} yielded empty content after extraction", payload.url);
            return Ok(());
        }

        // Generate document ID from URL (stable across re-fetches)
        let document_id = {
            use sha2::{Sha256, Digest};
            let mut hasher = Sha256::new();
            hasher.update(payload.url.as_bytes());
            format!("{:x}", hasher.finalize())[..32].to_string()
        };

        // Generate embedding (semaphore-gated)
        let _permit = embedding_semaphore.acquire().await
            .map_err(|e| UnifiedProcessorError::Embedding(format!("Semaphore closed: {}", e)))?;

        let embedding_result = embedding_generator
            .generate_embedding(&extracted_text, "all-MiniLM-L6-v2")
            .await
            .map_err(|e| UnifiedProcessorError::Embedding(e.to_string()))?;

        drop(_permit);

        // Build Qdrant payload
        let mut point_payload = std::collections::HashMap::new();
        point_payload.insert("content".to_string(), serde_json::json!(extracted_text));
        point_payload.insert("document_id".to_string(), serde_json::json!(document_id));
        point_payload.insert("tenant_id".to_string(), serde_json::json!(item.tenant_id));
        point_payload.insert("source_url".to_string(), serde_json::json!(payload.url));
        point_payload.insert("title".to_string(), serde_json::json!(title));
        point_payload.insert("source_type".to_string(), serde_json::json!("web"));
        point_payload.insert("item_type".to_string(), serde_json::json!("url"));
        point_payload.insert("branch".to_string(), serde_json::json!(item.branch));

        if let Some(ref lib_name) = payload.library_name {
            point_payload.insert("library_name".to_string(), serde_json::json!(lib_name));
        }

        let point = DocumentPoint {
            id: crate::generate_point_id(&item.tenant_id, &item.branch, &payload.url, 0),
            dense_vector: embedding_result.dense.vector,
            sparse_vector: Self::sparse_embedding_to_map(&embedding_result.sparse),
            payload: point_payload,
        };

        storage_client
            .insert_points_batch(&item.collection, vec![point], Some(1))
            .await
            .map_err(|e| UnifiedProcessorError::Storage(e.to_string()))?;

        info!(
            "Successfully processed URL item {} (url={}, content_length={})",
            item.queue_id, payload.url, extracted_text.len()
        );

        Ok(())
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

        let mut payload = std::collections::HashMap::new();
        let enrichment = LspEnrichment {
            enrichment_status: EnrichmentStatus::Success,
            references: vec![],
            type_info: None,
            resolved_imports: vec![],
            definition: None,
            error_message: None,
        };

        UnifiedQueueProcessor::add_lsp_enrichment_to_payload(&mut payload, &enrichment);
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

        UnifiedQueueProcessor::add_lsp_enrichment_to_payload(&mut payload2, &enrichment2);
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
