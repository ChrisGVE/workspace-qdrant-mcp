//! Unified Queue Processor Module
//!
//! Implements Task 37.26-29: Background processing loop that dequeues and processes
//! items from the unified_queue with type-specific handlers for content, file,
//! folder, project, library, and other operations.

use chrono::{DateTime, Duration as ChronoDuration, Utc};
use sqlx::SqlitePool;
use std::path::Path;
use std::sync::Arc;
use std::time::Duration;
use thiserror::Error;
use tokio::sync::RwLock;
use tokio::task::JoinHandle;
use tokio_util::sync::CancellationToken;
use tracing::{debug, error, info, warn};
use walkdir::WalkDir;

use crate::allowed_extensions::AllowedExtensions;
use crate::fairness_scheduler::{FairnessScheduler, FairnessSchedulerConfig};
use crate::lsp::{
    LanguageServerManager, LspEnrichment, EnrichmentStatus,
};
use crate::queue_operations::QueueManager;
use crate::unified_queue_schema::{
    ItemType, QueueOperation, UnifiedQueueItem,
    ContentPayload, FilePayload, FolderPayload, ProjectPayload, LibraryPayload,
};
use crate::{DocumentProcessor, EmbeddingGenerator, EmbeddingConfig, SparseEmbedding};
use crate::storage::{StorageClient, StorageConfig, DocumentPoint};
use crate::patterns::exclusion::should_exclude_file;
use crate::file_classification::classify_file_type;
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

    // Fairness scheduler settings (Task 21 - Anti-starvation alternation)
    /// Whether fairness scheduling is enabled (if disabled, falls back to priority DESC always)
    pub fairness_enabled: bool,
    /// Number of items between priority direction flips (default: 10)
    /// Every N items, alternates between priority DESC and ASC to prevent starvation
    pub items_per_flip: u64,

    // Resource limits (Task 504)
    /// Delay in milliseconds between processing items
    pub inter_item_delay_ms: u64,
    /// Maximum concurrent embedding operations
    pub max_concurrent_embeddings: usize,
    /// Pause processing when memory usage exceeds this percentage
    pub max_memory_percent: u8,
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
            // Fairness scheduler defaults (Task 21 - Anti-starvation)
            fairness_enabled: true,
            items_per_flip: 10, // Spec: every 10 items, flip priority direction
            // Resource limits defaults (Task 504)
            inter_item_delay_ms: 50,
            max_concurrent_embeddings: 2,
            max_memory_percent: 70,
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
}

impl UnifiedQueueProcessor {
    /// Create a new unified queue processor
    pub fn new(pool: SqlitePool, config: UnifiedProcessorConfig) -> Self {
        let document_processor = Arc::new(DocumentProcessor::new());
        let embedding_config = EmbeddingConfig::default();
        let embedding_generator = Arc::new(
            EmbeddingGenerator::new(embedding_config)
                .expect("Failed to create embedding generator")
        );
        let storage_config = StorageConfig::default();
        let storage_client = Arc::new(StorageClient::with_config(storage_config));

        // Create fairness scheduler with config from processor config (Task 21)
        let queue_manager = QueueManager::new(pool);
        let fairness_config = FairnessSchedulerConfig {
            enabled: config.fairness_enabled,
            items_per_flip: config.items_per_flip,
            worker_id: config.worker_id.clone(),
            lease_duration_secs: config.lease_duration_secs,
        };
        let fairness_scheduler = Arc::new(FairnessScheduler::new(
            queue_manager.clone(),
            fairness_config,
        ));

        let embedding_semaphore = Arc::new(tokio::sync::Semaphore::new(config.max_concurrent_embeddings));

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
        // Create fairness scheduler with config from processor config (Task 21)
        let queue_manager = QueueManager::new(pool);
        let fairness_config = FairnessSchedulerConfig {
            enabled: config.fairness_enabled,
            items_per_flip: config.items_per_flip,
            worker_id: config.worker_id.clone(),
            lease_duration_secs: config.lease_duration_secs,
        };
        let fairness_scheduler = Arc::new(FairnessScheduler::new(
            queue_manager.clone(),
            fairness_config,
        ));

        let embedding_semaphore = Arc::new(tokio::sync::Semaphore::new(config.max_concurrent_embeddings));

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
        }
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
            )
            .await
            {
                error!("Unified processing loop failed: {}", e);
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
    ) -> UnifiedProcessorResult<()> {
        let poll_interval = Duration::from_millis(config.poll_interval_ms);
        let inter_item_delay = Duration::from_millis(config.inter_item_delay_ms);
        let mut last_metrics_log = Utc::now();
        let metrics_log_interval = ChronoDuration::minutes(1);

        info!(
            "Unified processing loop started (batch_size={}, worker_id={}, fairness={}, inter_item_delay={}ms, max_embeddings={}, max_memory={}%)",
            config.batch_size, config.worker_id, config.fairness_enabled,
            config.inter_item_delay_ms, config.max_concurrent_embeddings, config.max_memory_percent
        );

        loop {
            // Check for shutdown signal
            if cancellation_token.is_cancelled() {
                info!("Unified queue shutdown signal received");
                break;
            }

            // Check memory pressure before dequeuing (Task 504)
            if Self::check_memory_pressure(config.max_memory_percent).await {
                info!("Memory pressure detected (>{}%), pausing processing for 5s", config.max_memory_percent);
                tokio::time::sleep(Duration::from_secs(5)).await;
                continue;
            }

            // Update queue depth in metrics
            if let Ok(depth) = queue_manager.get_unified_queue_depth(None, None).await {
                let mut m = metrics.write().await;
                m.queue_depth = depth;
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

                                info!(
                                    "Successfully processed unified item {} (type={:?}, op={:?}) in {}ms",
                                    item.queue_id, item.item_type, item.op, processing_time
                                );
                            }
                            Err(e) => {
                                error!(
                                    "Failed to process unified item {} (type={:?}): {}",
                                    item.queue_id, item.item_type, e
                                );

                                // Classify error: permanent errors skip retry
                                let is_permanent = Self::is_permanent_error(&e);

                                // Mark item as failed (with exponential backoff for transient errors)
                                if let Err(mark_err) = queue_manager
                                    .mark_unified_failed(&item.queue_id, &e.to_string(), is_permanent)
                                    .await
                                {
                                    error!("Failed to mark item {} as failed: {}", item.queue_id, mark_err);
                                }

                                Self::update_metrics_failure(&metrics, &e).await;
                            }
                        }

                        // Inter-item delay for resource breathing room (Task 504)
                        if inter_item_delay > Duration::ZERO {
                            tokio::time::sleep(inter_item_delay).await;
                        }
                    }
                }
                Err(e) => {
                    error!("Failed to dequeue unified batch: {}", e);
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
            ItemType::Content => {
                Self::process_content_item(item, embedding_generator, storage_client, embedding_semaphore).await
            }
            ItemType::File => {
                Self::process_file_item(item, queue_manager, document_processor, embedding_generator, storage_client, lsp_manager, embedding_semaphore, allowed_extensions).await
            }
            ItemType::Folder => {
                Self::process_folder_item(item, queue_manager, storage_client, allowed_extensions).await
            }
            ItemType::Project => {
                Self::process_project_item(item, queue_manager, storage_client, allowed_extensions).await
            }
            ItemType::Library => {
                Self::process_library_item(item, queue_manager, storage_client, allowed_extensions).await
            }
            ItemType::DeleteTenant => {
                Self::process_delete_tenant_item(item, storage_client).await
            }
            ItemType::DeleteDocument => {
                Self::process_delete_document_item(item, storage_client).await
            }
            ItemType::Rename => {
                Self::process_rename_item(item, storage_client).await
            }
        }
    }

    /// Process content item (Task 37.27) - direct text ingestion
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

        // Parse the content payload
        let payload: ContentPayload = serde_json::from_str(&item.payload_json)
            .map_err(|e| UnifiedProcessorError::InvalidPayload(format!("Failed to parse ContentPayload: {}", e)))?;

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

        // Generate embedding for the content (semaphore-gated, Task 504)
        let _permit = embedding_semaphore.acquire().await
            .map_err(|e| UnifiedProcessorError::Embedding(format!("Semaphore closed: {}", e)))?;
        let embedding_result = embedding_generator
            .generate_embedding(&payload.content, "bge-small-en-v1.5")
            .await
            .map_err(|e| UnifiedProcessorError::Embedding(e.to_string()))?;
        drop(_permit);

        // Generate stable document_id for content (deterministic from tenant + content)
        let content_doc_id = crate::generate_content_document_id(&item.tenant_id, &payload.content);

        // Build payload with metadata
        let mut point_payload = std::collections::HashMap::new();
        point_payload.insert("content".to_string(), serde_json::json!(payload.content));
        point_payload.insert("document_id".to_string(), serde_json::json!(content_doc_id));
        point_payload.insert("tenant_id".to_string(), serde_json::json!(item.tenant_id));
        point_payload.insert("branch".to_string(), serde_json::json!(item.branch));
        point_payload.insert("item_type".to_string(), serde_json::json!("content"));
        point_payload.insert("source_type".to_string(), serde_json::json!(payload.source_type));

        // Add tag metadata from payload
        if let Some(main_tag) = &payload.main_tag {
            point_payload.insert("main_tag".to_string(), serde_json::json!(main_tag));
        }
        if let Some(full_tag) = &payload.full_tag {
            point_payload.insert("full_tag".to_string(), serde_json::json!(full_tag));
        }

        // Create document point with stable point ID and sparse vector for hybrid search
        let point = DocumentPoint {
            id: crate::generate_point_id(&content_doc_id, 0),
            dense_vector: embedding_result.dense.vector,
            sparse_vector: Self::sparse_embedding_to_map(&embedding_result.sparse),
            payload: point_payload,
        };

        // Insert point
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
        let (watch_folder_id, base_path) = match watch_info {
            Some((wid, bp)) => (wid, bp),
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
                        "SQLite transaction failed during file-not-found cleanup for {}: {}. Will be reconciled on next startup.",
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
                serde_json::json!(format!("{:?}", document_content.document_type)),
            );
            point_payload.insert("item_type".to_string(), serde_json::json!("file"));

            if let Some(file_type) = &payload.file_type {
                point_payload.insert("file_type".to_string(), serde_json::json!(file_type));
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

            let point_id = crate::generate_point_id(&file_document_id, chunk_idx);
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
            .unwrap_or_else(|_| chrono::Utc::now().to_rfc3339());
        let language = chunk_records.first()
            .and_then(|_| document_content.metadata.get("language"))
            .cloned();
        let chunking_method = if treesitter_status == ProcessingStatus::Done {
            Some("tree_sitter")
        } else {
            Some("text")
        };

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

        // Handle transaction failure: Qdrant has points but SQLite state is inconsistent
        if let Err(e) = tx_result {
            warn!(
                "SQLite transaction failed after Qdrant upsert for {}: {}. File will be reconciled on next startup.",
                relative_path, e
            );
            // If the file was already tracked, mark it for reconciliation
            if let Some(existing_file) = &existing {
                let _ = tracked_files_schema::mark_needs_reconcile(
                    pool, existing_file.file_id,
                    &format!("ingest_tx_failed: {}", e),
                ).await;
            }
            // Don't propagate error - Qdrant has the data, startup recovery will fix SQLite
        }

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
                    "SQLite transaction failed after Qdrant delete for {}: {}. Will be reconciled on next startup.",
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
        // Add enrichment status
        payload.insert(
            "lsp_enrichment_status".to_string(),
            serde_json::json!(format!("{:?}", enrichment.enrichment_status)),
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
                Self::process_folder_delete(item, &payload, queue_manager, storage_client).await
            }
            QueueOperation::Update | QueueOperation::Ingest => {
                // Folder update/ingest is equivalent to a rescan
                info!(
                    "Folder {:?} operation treated as rescan for: {}",
                    item.op, payload.folder_path
                );
                Self::scan_library_directory(item, &payload.folder_path, queue_manager, storage_client, allowed_extensions).await
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
        storage_client: &Arc<StorageClient>,
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
            QueueOperation::Ingest => {
                // Create collection for the project
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
                } else {
                    info!("Project collection {} already exists", item.collection);
                }
            }
            QueueOperation::Scan => {
                // Scan project directory and queue file ingestion items
                Self::scan_project_directory(item, &payload, queue_manager, storage_client, allowed_extensions).await?;

                // Update last_scan timestamp for this project's watch_folder
                let update_result = sqlx::query(
                    "UPDATE watch_folders SET last_scan = strftime('%Y-%m-%dT%H:%M:%fZ', 'now'), updated_at = strftime('%Y-%m-%dT%H:%M:%fZ', 'now') WHERE tenant_id = ?1 AND collection = 'projects'"
                )
                    .bind(&item.tenant_id)
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
    /// and the file type allowlist (Task 511), and queues (File, Ingest) items for
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

        // Walk directory recursively
        for entry in WalkDir::new(project_root)
            .follow_links(false)  // Don't follow symlinks to avoid cycles
            .into_iter()
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
            };

            let payload_json = serde_json::to_string(&file_payload)
                .map_err(|e| UnifiedProcessorError::ProcessingFailed(format!("Failed to serialize FilePayload: {}", e)))?;

            // Queue the file for ingestion
            // Priority is computed at dequeue time via CASE/JOIN, not stored
            match queue_manager.enqueue_unified(
                ItemType::File,
                QueueOperation::Ingest,
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
            };

            let payload_json = serde_json::to_string(&file_payload)
                .map_err(|e| UnifiedProcessorError::ProcessingFailed(format!("Failed to serialize FilePayload: {}", e)))?;

            match queue_manager.enqueue_unified(
                ItemType::File,
                QueueOperation::Ingest,
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
            "UPDATE watch_folders SET last_scan = strftime('%Y-%m-%dT%H:%M:%fZ', 'now'), updated_at = strftime('%Y-%m-%dT%H:%M:%fZ', 'now') WHERE tenant_id = ?1 AND collection = 'libraries'"
        )
            .bind(&item.tenant_id)
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
            QueueOperation::Ingest => {
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
                    "SELECT path FROM watch_folders WHERE tenant_id = ?1 AND collection = 'libraries'"
                )
                    .bind(&item.tenant_id)
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

    /// Process delete tenant item - delete all data for a tenant
    async fn process_delete_tenant_item(
        item: &UnifiedQueueItem,
        storage_client: &Arc<StorageClient>,
    ) -> UnifiedProcessorResult<()> {
        info!(
            "Processing delete tenant item: {} (tenant={})",
            item.queue_id, item.tenant_id
        );

        // Delete all points with matching tenant_id from the collection (tenant-scoped)
        if storage_client
            .collection_exists(&item.collection)
            .await
            .map_err(|e| UnifiedProcessorError::Storage(e.to_string()))?
        {
            storage_client
                .delete_points_by_tenant(&item.collection, &item.tenant_id)
                .await
                .map_err(|e| UnifiedProcessorError::Storage(e.to_string()))?;
        }

        info!(
            "Successfully processed delete tenant item {} (tenant={})",
            item.queue_id, item.tenant_id
        );

        Ok(())
    }

    /// Process delete document item - delete specific document
    async fn process_delete_document_item(
        item: &UnifiedQueueItem,
        storage_client: &Arc<StorageClient>,
    ) -> UnifiedProcessorResult<()> {
        info!(
            "Processing delete document item: {}",
            item.queue_id
        );

        // Parse payload to get document identifier
        let payload: serde_json::Value = serde_json::from_str(&item.payload_json)
            .map_err(|e| UnifiedProcessorError::InvalidPayload(format!("Failed to parse payload: {}", e)))?;

        let doc_id = payload
            .get("doc_id")
            .and_then(|v| v.as_str())
            .ok_or_else(|| UnifiedProcessorError::InvalidPayload("Missing doc_id in payload".to_string()))?;

        if storage_client
            .collection_exists(&item.collection)
            .await
            .map_err(|e| UnifiedProcessorError::Storage(e.to_string()))?
        {
            storage_client
                .delete_points_by_filter(&item.collection, doc_id, &item.tenant_id)
                .await
                .map_err(|e| UnifiedProcessorError::Storage(e.to_string()))?;
        }

        info!(
            "Successfully deleted document {} from {} (tenant={})",
            doc_id, item.collection, item.tenant_id
        );

        Ok(())
    }

    /// Process rename item - update file paths in metadata
    async fn process_rename_item(
        item: &UnifiedQueueItem,
        storage_client: &Arc<StorageClient>,
    ) -> UnifiedProcessorResult<()> {
        info!(
            "Processing rename item: {}",
            item.queue_id
        );

        // Parse payload to get old and new paths
        let payload: serde_json::Value = serde_json::from_str(&item.payload_json)
            .map_err(|e| UnifiedProcessorError::InvalidPayload(format!("Failed to parse payload: {}", e)))?;

        let old_path = payload
            .get("old_path")
            .and_then(|v| v.as_str())
            .ok_or_else(|| UnifiedProcessorError::InvalidPayload("Missing old_path in payload".to_string()))?;

        let _new_path = payload
            .get("new_path")
            .and_then(|v| v.as_str())
            .ok_or_else(|| UnifiedProcessorError::InvalidPayload("Missing new_path in payload".to_string()))?;

        // For now, rename is handled as delete old + re-ingest new
        // The file watcher will detect the new file and enqueue it
        if storage_client
            .collection_exists(&item.collection)
            .await
            .map_err(|e| UnifiedProcessorError::Storage(e.to_string()))?
        {
            storage_client
                .delete_points_by_filter(&item.collection, old_path, &item.tenant_id)
                .await
                .map_err(|e| UnifiedProcessorError::Storage(e.to_string()))?;
        }

        info!(
            "Successfully processed rename item {} (deleted old path: {})",
            item.queue_id, old_path
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
    fn is_permanent_error(error: &UnifiedProcessorError) -> bool {
        match error {
            // File doesn't exist - retrying won't help
            UnifiedProcessorError::FileNotFound(_) => true,
            // Malformed payload - retrying won't fix the data
            UnifiedProcessorError::InvalidPayload(_) => true,
            // Check error message for permanent patterns
            UnifiedProcessorError::QueueOperation(msg) => {
                let lower = msg.to_lowercase();
                lower.contains("no watch_folder found")
                    || lower.contains("validation")
                    || lower.contains("invalid")
            }
            UnifiedProcessorError::ProcessingFailed(msg) => {
                let lower = msg.to_lowercase();
                lower.contains("permission denied")
                    || lower.contains("invalid format")
                    || lower.contains("malformed")
                    || lower.contains("unsupported")
            }
            // Storage and embedding errors are transient (Qdrant may be temporarily down)
            UnifiedProcessorError::Storage(_) => false,
            UnifiedProcessorError::Embedding(_) => false,
            // Default: treat as transient (retry)
            _ => false,
        }
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

    #[test]
    fn test_unified_processor_config_default() {
        let config = UnifiedProcessorConfig::default();
        assert_eq!(config.batch_size, 10);
        assert_eq!(config.poll_interval_ms, 500);
        assert_eq!(config.lease_duration_secs, 300);
        assert_eq!(config.max_retries, 3);
        assert!(config.worker_id.starts_with("unified-worker-"));
        // Fairness scheduler settings (Task 21 - Anti-starvation)
        assert!(config.fairness_enabled);
        assert_eq!(config.items_per_flip, 10); // Spec: every 10 items
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
}
