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

use crate::fairness_scheduler::{FairnessScheduler, FairnessSchedulerConfig};
use crate::lsp::{
    LanguageServerManager, LspEnrichment, EnrichmentStatus,
};
use crate::queue_operations::QueueManager;
use crate::unified_queue_schema::{
    ItemType, QueueOperation, UnifiedQueueItem,
    ContentPayload, FilePayload, ProjectPayload, LibraryPayload,
};
use crate::{DocumentProcessor, EmbeddingGenerator, EmbeddingConfig};
use crate::storage::{StorageClient, StorageConfig, DocumentPoint};
use crate::patterns::exclusion::should_exclude_file;
use crate::file_classification::classify_file_type;

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
        }
    }

    /// Set the LSP manager for code intelligence enrichment
    pub fn with_lsp_manager(mut self, lsp_manager: Arc<RwLock<LanguageServerManager>>) -> Self {
        self.lsp_manager = Some(lsp_manager);
        self
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
    ) -> UnifiedProcessorResult<()> {
        let poll_interval = Duration::from_millis(config.poll_interval_ms);
        let mut last_metrics_log = Utc::now();
        let metrics_log_interval = ChronoDuration::minutes(1);

        info!(
            "Unified processing loop started (batch_size={}, worker_id={}, fairness={})",
            config.batch_size, config.worker_id, config.fairness_enabled
        );

        loop {
            // Check for shutdown signal
            if cancellation_token.is_cancelled() {
                info!("Unified queue shutdown signal received");
                break;
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
                                    "Successfully processed unified item {} (type={:?}, op={:?})",
                                    item.queue_id, item.item_type, item.op
                                );
                            }
                            Err(e) => {
                                error!(
                                    "Failed to process unified item {} (type={:?}): {}",
                                    item.queue_id, item.item_type, e
                                );

                                // Mark item as failed
                                if let Err(mark_err) = queue_manager
                                    .mark_unified_failed(&item.queue_id, &e.to_string())
                                    .await
                                {
                                    error!("Failed to mark item {} as failed: {}", item.queue_id, mark_err);
                                }

                                Self::update_metrics_failure(&metrics, &e).await;
                            }
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
    ) -> UnifiedProcessorResult<()> {
        debug!(
            "Processing unified item: {} (type={:?}, op={:?}, collection={})",
            item.queue_id, item.item_type, item.op, item.collection
        );

        match item.item_type {
            ItemType::Content => {
                Self::process_content_item(item, embedding_generator, storage_client).await
            }
            ItemType::File => {
                Self::process_file_item(item, document_processor, embedding_generator, storage_client, lsp_manager).await
            }
            ItemType::Folder => {
                // Folder operations typically trigger file enqueues
                Self::process_folder_item(item).await
            }
            ItemType::Project => {
                Self::process_project_item(item, queue_manager, storage_client).await
            }
            ItemType::Library => {
                Self::process_library_item(item, storage_client).await
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
    ) -> UnifiedProcessorResult<()> {
        info!(
            "Processing content item: {} → collection: {}",
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

        // Generate embedding for the content
        let embedding_result = embedding_generator
            .generate_embedding(&payload.content, "bge-small-en-v1.5")
            .await
            .map_err(|e| UnifiedProcessorError::Embedding(e.to_string()))?;

        // Build payload with metadata
        let mut point_payload = std::collections::HashMap::new();
        point_payload.insert("content".to_string(), serde_json::json!(payload.content));
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

        // Create document point with generated ID
        let point = DocumentPoint {
            id: uuid::Uuid::new_v4().to_string(),
            dense_vector: embedding_result.dense.vector,
            sparse_vector: None,
            payload: point_payload,
        };

        // Insert point
        storage_client
            .insert_points_batch(&item.collection, vec![point], Some(1))
            .await
            .map_err(|e| UnifiedProcessorError::Storage(e.to_string()))?;

        info!(
            "Successfully processed content item {} → {}",
            item.queue_id, item.collection
        );

        Ok(())
    }

    /// Process file item (Task 37.28) - file-based ingestion
    async fn process_file_item(
        item: &UnifiedQueueItem,
        document_processor: &Arc<DocumentProcessor>,
        embedding_generator: &Arc<EmbeddingGenerator>,
        storage_client: &Arc<StorageClient>,
        lsp_manager: &Option<Arc<RwLock<LanguageServerManager>>>,
    ) -> UnifiedProcessorResult<()> {
        info!(
            "Processing file item: {} → collection: {}",
            item.queue_id, item.collection
        );

        // Parse the file payload
        let payload: FilePayload = serde_json::from_str(&item.payload_json)
            .map_err(|e| UnifiedProcessorError::InvalidPayload(format!("Failed to parse FilePayload: {}", e)))?;

        let file_path = Path::new(&payload.file_path);

        // Check if file exists
        if !file_path.exists() {
            return Err(UnifiedProcessorError::FileNotFound(payload.file_path.clone()));
        }

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

        // For delete operations, just delete existing points
        if item.op == QueueOperation::Delete {
            storage_client
                .delete_points_by_filter(&item.collection, &payload.file_path)
                .await
                .map_err(|e| UnifiedProcessorError::Storage(e.to_string()))?;

            info!("Deleted points for file: {}", payload.file_path);
            return Ok(());
        }

        // For update operations, delete existing first
        if item.op == QueueOperation::Update {
            storage_client
                .delete_points_by_filter(&item.collection, &payload.file_path)
                .await
                .map_err(|e| UnifiedProcessorError::Storage(e.to_string()))?;
        }

        // Process file content
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
            // We need to drop the read lock before processing chunks
            // to avoid holding it for too long
            (is_active, Some(lsp_mgr.clone()))
        } else {
            (false, None)
        };

        // Process each chunk
        let mut points = Vec::new();
        for (chunk_idx, chunk) in document_content.chunks.iter().enumerate() {
            let embedding_result = embedding_generator
                .generate_embedding(&chunk.content, "bge-small-en-v1.5")
                .await
                .map_err(|e| UnifiedProcessorError::Embedding(e.to_string()))?;

            let mut point_payload = std::collections::HashMap::new();
            point_payload.insert("content".to_string(), serde_json::json!(chunk.content));
            point_payload.insert("chunk_index".to_string(), serde_json::json!(chunk.chunk_index));
            point_payload.insert("file_path".to_string(), serde_json::json!(payload.file_path));
            point_payload.insert("tenant_id".to_string(), serde_json::json!(item.tenant_id));
            point_payload.insert("branch".to_string(), serde_json::json!(item.branch));
            point_payload.insert(
                "document_type".to_string(),
                serde_json::json!(format!("{:?}", document_content.document_type)),
            );
            point_payload.insert("item_type".to_string(), serde_json::json!("file"));

            // Add file metadata
            if let Some(file_type) = &payload.file_type {
                point_payload.insert("file_type".to_string(), serde_json::json!(file_type));
            }

            // Add chunk metadata
            for (key, value) in &chunk.metadata {
                point_payload.insert(format!("chunk_{}", key), serde_json::json!(value));
            }

            // LSP enrichment (if available)
            if let Some(lsp_mgr) = &lsp_mgr_guard {
                let mgr = lsp_mgr.read().await;

                // Extract symbol name from metadata if available
                let symbol_name = chunk.metadata.get("symbol_name")
                    .map(|s| s.as_str())
                    .unwrap_or("unknown");

                // Get start line from metadata or estimate from chunk index
                let start_line = chunk.metadata.get("start_line")
                    .and_then(|s| s.parse::<u32>().ok())
                    .unwrap_or(chunk_idx as u32 * 20); // Rough estimate

                let end_line = chunk.metadata.get("end_line")
                    .and_then(|s| s.parse::<u32>().ok())
                    .unwrap_or(start_line + 20);

                let enrichment = mgr.enrich_chunk(
                    &item.tenant_id,
                    file_path,
                    symbol_name,
                    start_line,
                    end_line,
                    is_project_active,
                ).await;

                // Add enrichment data to payload
                Self::add_lsp_enrichment_to_payload(&mut point_payload, &enrichment);
            }

            let point = DocumentPoint {
                id: uuid::Uuid::new_v4().to_string(),
                dense_vector: embedding_result.dense.vector,
                sparse_vector: None,
                payload: point_payload,
            };

            points.push(point);
        }

        // Insert points in batch
        if !points.is_empty() {
            info!("Inserting {} points into {}", points.len(), item.collection);
            storage_client
                .insert_points_batch(&item.collection, points, Some(100))
                .await
                .map_err(|e| UnifiedProcessorError::Storage(e.to_string()))?;
        }

        info!(
            "Successfully processed file item {} ({})",
            item.queue_id, payload.file_path
        );

        Ok(())
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

    /// Process folder item - typically triggers scanning
    async fn process_folder_item(item: &UnifiedQueueItem) -> UnifiedProcessorResult<()> {
        info!(
            "Processing folder item: {} (op={:?})",
            item.queue_id, item.op
        );

        // Folder operations are typically handled by the file watcher
        // This is a placeholder for folder-level metadata operations
        warn!(
            "Folder item processing is a no-op (handled by watcher): {}",
            item.queue_id
        );

        Ok(())
    }

    /// Process project item (Task 37.29) - create/manage project collections
    async fn process_project_item(
        item: &UnifiedQueueItem,
        queue_manager: &QueueManager,
        storage_client: &Arc<StorageClient>,
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
                Self::scan_project_directory(item, &payload, queue_manager, storage_client).await?;
            }
            QueueOperation::Delete => {
                // Delete project collection
                if storage_client
                    .collection_exists(&item.collection)
                    .await
                    .map_err(|e| UnifiedProcessorError::Storage(e.to_string()))?
                {
                    info!("Deleting project collection: {}", item.collection);
                    storage_client
                        .delete_collection(&item.collection)
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
    /// Walks the project directory recursively, filters files using exclusion rules,
    /// and queues (File, Ingest) items for each eligible file.
    async fn scan_project_directory(
        item: &UnifiedQueueItem,
        payload: &ProjectPayload,
        queue_manager: &QueueManager,
        storage_client: &Arc<StorageClient>,
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
            // Priority 5 is default/normal - not urgent but not low
            match queue_manager.enqueue_unified(
                ItemType::File,
                QueueOperation::Ingest,
                &item.tenant_id,
                &item.collection,
                &payload_json,
                5,  // Normal priority for bulk scan items
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
        ).await?;

        let elapsed = start_time.elapsed();
        info!(
            "Project scan complete: {} files queued, {} excluded, {} queued for deletion, {} errors in {:?} (project={})",
            files_queued, files_excluded, files_cleaned, errors, elapsed, payload.project_root
        );

        Ok(())
    }

    /// Clean up excluded files from Qdrant after a scan completes
    ///
    /// Scrolls through all indexed file paths for the tenant, checks each against
    /// current exclusion rules, and queues deletion for any files that now match
    /// exclusion patterns. This handles retroactive cleanup when exclusion rules
    /// are updated (e.g., hidden files that were indexed before the exclusion fix).
    async fn cleanup_excluded_files(
        item: &UnifiedQueueItem,
        project_root: &Path,
        queue_manager: &QueueManager,
        storage_client: &Arc<StorageClient>,
    ) -> UnifiedProcessorResult<u64> {
        // Check if collection exists before attempting scroll
        if !storage_client
            .collection_exists(&item.collection)
            .await
            .map_err(|e| UnifiedProcessorError::Storage(e.to_string()))?
        {
            debug!(
                "Collection '{}' does not exist, skipping exclusion cleanup",
                item.collection
            );
            return Ok(0);
        }

        let qdrant_file_paths = storage_client
            .scroll_file_paths_by_tenant(&item.collection, &item.tenant_id)
            .await
            .map_err(|e| {
                // Log but don't propagate - cleanup failure shouldn't fail the scan
                error!("Failed to scroll Qdrant for exclusion cleanup: {}", e);
                UnifiedProcessorError::Storage(e.to_string())
            });

        let qdrant_file_paths = match qdrant_file_paths {
            Ok(paths) => paths,
            Err(_) => return Ok(0), // Already logged above
        };

        if qdrant_file_paths.is_empty() {
            debug!(
                "No indexed files found for tenant_id='{}', skipping exclusion cleanup",
                item.tenant_id
            );
            return Ok(0);
        }

        info!(
            "Checking {} indexed files against exclusion rules (tenant_id={})",
            qdrant_file_paths.len(),
            item.tenant_id
        );

        let mut files_cleaned = 0u64;

        for qdrant_file in &qdrant_file_paths {
            // Get relative path for exclusion check
            let rel_path = match Path::new(qdrant_file).strip_prefix(project_root) {
                Ok(stripped) => stripped.to_string_lossy().to_string(),
                Err(_) => qdrant_file.clone(),
            };

            // Check if this file should now be excluded
            if !should_exclude_file(&rel_path) {
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
                    7, // Higher priority than scan items (5)
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
                            qdrant_file, rel_path
                        );
                    } else {
                        debug!(
                            "Excluded file deletion already in queue (deduplicated): {}",
                            qdrant_file
                        );
                    }
                }
                Err(e) => {
                    warn!(
                        "Failed to queue excluded file for deletion {}: {}",
                        qdrant_file, e
                    );
                }
            }
        }

        if files_cleaned > 0 {
            info!(
                "Queued {} excluded files for deletion (tenant_id={})",
                files_cleaned, item.tenant_id
            );
        }

        Ok(files_cleaned)
    }

    /// Process library item - create/manage library collections
    async fn process_library_item(
        item: &UnifiedQueueItem,
        storage_client: &Arc<StorageClient>,
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
            QueueOperation::Delete => {
                // Delete library collection
                if storage_client
                    .collection_exists(&item.collection)
                    .await
                    .map_err(|e| UnifiedProcessorError::Storage(e.to_string()))?
                {
                    info!("Deleting library collection: {}", item.collection);
                    storage_client
                        .delete_collection(&item.collection)
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

        // Delete all points with matching tenant_id from the collection
        // This requires a filter-based deletion
        if storage_client
            .collection_exists(&item.collection)
            .await
            .map_err(|e| UnifiedProcessorError::Storage(e.to_string()))?
        {
            // For now, we delete the entire collection if it's a tenant-specific collection
            // In future, implement filter-based deletion by tenant_id
            warn!(
                "Tenant deletion for {} - deleting collection {}",
                item.tenant_id, item.collection
            );
            storage_client
                .delete_collection(&item.collection)
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
                .delete_points_by_filter(&item.collection, doc_id)
                .await
                .map_err(|e| UnifiedProcessorError::Storage(e.to_string()))?;
        }

        info!(
            "Successfully deleted document {} from {}",
            doc_id, item.collection
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
                .delete_points_by_filter(&item.collection, old_path)
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
