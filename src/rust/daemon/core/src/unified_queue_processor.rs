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

use crate::queue_operations::QueueManager;
use crate::unified_queue_schema::{
    ItemType, QueueOperation, UnifiedQueueItem,
    ContentPayload, FilePayload, ProjectPayload, LibraryPayload,
};
use crate::{DocumentProcessor, EmbeddingGenerator, EmbeddingConfig};
use crate::storage::{StorageClient, StorageConfig, DocumentPoint};

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
        }
    }
}

/// Unified queue processor manages background processing of unified_queue items
pub struct UnifiedQueueProcessor {
    /// Queue manager for database operations
    queue_manager: QueueManager,

    /// Processor configuration
    config: UnifiedProcessorConfig,

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

        Self {
            queue_manager: QueueManager::new(pool),
            config,
            metrics: Arc::new(RwLock::new(UnifiedProcessingMetrics::default())),
            cancellation_token: CancellationToken::new(),
            task_handle: None,
            document_processor,
            embedding_generator,
            storage_client,
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
        Self {
            queue_manager: QueueManager::new(pool),
            config,
            metrics: Arc::new(RwLock::new(UnifiedProcessingMetrics::default())),
            cancellation_token: CancellationToken::new(),
            task_handle: None,
            document_processor,
            embedding_generator,
            storage_client,
        }
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
            "Starting unified queue processor (batch_size={}, poll_interval={}ms, worker_id={})",
            self.config.batch_size, self.config.poll_interval_ms, self.config.worker_id
        );

        let queue_manager = self.queue_manager.clone();
        let config = self.config.clone();
        let metrics = self.metrics.clone();
        let cancellation_token = self.cancellation_token.clone();
        let document_processor = self.document_processor.clone();
        let embedding_generator = self.embedding_generator.clone();
        let storage_client = self.storage_client.clone();

        let task_handle = tokio::spawn(async move {
            if let Err(e) = Self::processing_loop(
                queue_manager,
                config,
                metrics,
                cancellation_token.clone(),
                document_processor,
                embedding_generator,
                storage_client,
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
        metrics: Arc<RwLock<UnifiedProcessingMetrics>>,
        cancellation_token: CancellationToken,
        document_processor: Arc<DocumentProcessor>,
        embedding_generator: Arc<EmbeddingGenerator>,
        storage_client: Arc<StorageClient>,
    ) -> UnifiedProcessorResult<()> {
        let poll_interval = Duration::from_millis(config.poll_interval_ms);
        let mut last_metrics_log = Utc::now();
        let metrics_log_interval = ChronoDuration::minutes(1);

        // Garbage collection tracking (Task 36)
        let mut last_gc_run = Utc::now();
        let gc_interval = ChronoDuration::hours(1); // Run every hour
        let max_inactive_hours = 24; // Remove projects inactive for 24+ hours

        info!(
            "Unified processing loop started (batch_size={}, worker_id={})",
            config.batch_size, config.worker_id
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

            // Dequeue batch of items
            match queue_manager
                .dequeue_unified(
                    config.batch_size,
                    &config.worker_id,
                    Some(config.lease_duration_secs),
                    None, // all tenants
                    None, // all item types
                )
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
                        )
                        .await
                        {
                            Ok(()) => {
                                let processing_time = start_time.elapsed().as_millis() as u64;

                                // Mark item as done
                                if let Err(e) = queue_manager
                                    .mark_unified_done(&item.queue_id)
                                    .await
                                {
                                    error!("Failed to mark item {} as done: {}", item.queue_id, e);
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

            // Garbage collect stale active projects periodically (Task 36)
            if now - last_gc_run >= gc_interval {
                debug!("Running active projects garbage collection...");
                match queue_manager.garbage_collect_stale_projects(Some(max_inactive_hours)).await {
                    Ok(removed) => {
                        if removed > 0 {
                            info!("Garbage collected {} stale active projects (inactive > {} hours)",
                                removed, max_inactive_hours);
                        } else {
                            debug!("No stale active projects to garbage collect");
                        }
                    }
                    Err(e) => {
                        warn!("Active projects garbage collection failed: {}", e);
                    }
                }
                last_gc_run = now;
            }

            // Brief pause before next batch
            tokio::time::sleep(Duration::from_millis(10)).await;
        }

        Ok(())
    }

    /// Process a single unified queue item based on its type
    #[allow(clippy::too_many_arguments)]
    async fn process_item(
        _queue_manager: &QueueManager,
        item: &UnifiedQueueItem,
        _config: &UnifiedProcessorConfig,
        document_processor: &Arc<DocumentProcessor>,
        embedding_generator: &Arc<EmbeddingGenerator>,
        storage_client: &Arc<StorageClient>,
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
                Self::process_file_item(item, document_processor, embedding_generator, storage_client).await
            }
            ItemType::Folder => {
                // Folder operations typically trigger file enqueues
                Self::process_folder_item(item).await
            }
            ItemType::Project => {
                Self::process_project_item(item, storage_client).await
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

        // Process each chunk
        let mut points = Vec::new();
        for chunk in document_content.chunks {
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
            for (key, value) in chunk.metadata {
                point_payload.insert(format!("chunk_{}", key), serde_json::json!(value));
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
}
