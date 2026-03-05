//! Unified Queue Processor Module
//!
//! Implements Task 37.26-29: Background processing loop that dequeues and processes
//! items from the unified_queue with type-specific handlers for content, file,
//! folder, project, library, and other operations.

pub mod config;
pub mod error;
mod metrics;
mod processing_loop;
#[cfg(test)]
mod tests;

pub use config::{UnifiedProcessingMetrics, UnifiedProcessorConfig, WarmupState};
pub use error::{UnifiedProcessorError, UnifiedProcessorResult};

use sqlx::SqlitePool;
use std::sync::Arc;
use std::time::Duration;
use tokio::sync::RwLock;
use tokio::task::JoinHandle;
use tokio_util::sync::CancellationToken;
use tracing::{error, info, warn};

use crate::adaptive_resources::ResourceProfile;
use crate::allowed_extensions::AllowedExtensions;
use crate::fairness_scheduler::{FairnessScheduler, FairnessSchedulerConfig};
use crate::lexicon::LexiconManager;
use crate::lsp::LanguageServerManager;
use crate::queue_health::QueueProcessorHealth;
use crate::queue_operations::QueueManager;
use crate::search_db::SearchDbManager;
use crate::storage::{StorageClient, StorageConfig};
use crate::tree_sitter::GrammarManager;
use crate::{DocumentProcessor, EmbeddingConfig, EmbeddingGenerator};

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

    /// Graph store for code relationship extraction and storage (graph-rag)
    graph_store: Option<crate::graph::SharedGraphStore<crate::graph::SqliteGraphStore>>,

    /// Signal to trigger WatchManager refresh after creating a new watch_folder (Task 12)
    watch_refresh_signal: Option<Arc<tokio::sync::Notify>>,

    /// Grammar manager for dynamic tree-sitter grammar loading
    grammar_manager: Option<Arc<RwLock<GrammarManager>>>,
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
                .expect("Failed to create embedding generator"),
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
        let embedding_semaphore = Arc::new(tokio::sync::Semaphore::new(
            config.warmup_max_concurrent_embeddings,
        ));
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
            graph_store: None,
            watch_refresh_signal: None,
            grammar_manager: None,
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
        let lexicon_manager = Arc::new(LexiconManager::new(
            pool.clone(),
            EmbeddingConfig::default().bm25_k1,
        ));

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
        let embedding_semaphore = Arc::new(tokio::sync::Semaphore::new(
            config.warmup_max_concurrent_embeddings,
        ));
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
            graph_store: None,
            watch_refresh_signal: None,
            grammar_manager: None,
        }
    }

    /// Set the grammar manager for dynamic tree-sitter grammar loading
    pub fn with_grammar_manager(mut self, manager: Arc<RwLock<GrammarManager>>) -> Self {
        self.grammar_manager = Some(manager);
        self
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

    /// Set the graph store for code relationship extraction (graph-rag)
    pub fn with_graph_store(
        mut self,
        store: crate::graph::SharedGraphStore<crate::graph::SqliteGraphStore>,
    ) -> Self {
        self.graph_store = Some(store);
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
    pub fn with_adaptive_resources(
        mut self,
        rx: tokio::sync::watch::Receiver<ResourceProfile>,
    ) -> Self {
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
        let count = self
            .queue_manager
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
        let graph_store = self.graph_store.clone();
        let watch_refresh_signal = self.watch_refresh_signal.clone();
        let grammar_manager = self.grammar_manager.clone();

        // Mark as running in health state
        if let Some(ref h) = queue_health {
            h.set_running(true);
        }

        let task_handle = tokio::spawn(async move {
            // One-time cleanup of junk BM25 terms from sparse_vocabulary (Task 22)
            if let Err(e) = lexicon_manager.cleanup_junk_terms().await {
                warn!(
                    "Failed to clean junk terms from sparse_vocabulary: {} (non-critical)",
                    e
                );
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
                graph_store,
                watch_refresh_signal,
                grammar_manager,
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
}
