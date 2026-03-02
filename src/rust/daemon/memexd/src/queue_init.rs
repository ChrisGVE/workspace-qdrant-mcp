//! Phase 5: Embedding, storage, queue processor initialization, and post-start tasks.
//!
//! Creates the document processor, embedding generator, storage client, and
//! the unified queue processor with all its optional attachments (LSP, search DB,
//! graph store, etc.). Also handles IPC server setup.

use std::sync::Arc;

use sqlx::SqlitePool;
use tokio::sync::{Notify, RwLock};
use tracing::{debug, error, info, warn};

use workspace_qdrant_core::{
    config::Config,
    config::DaemonConfig,
    ipc::IpcServer,
    ProcessorConfig,
    UnifiedQueueProcessor, UnifiedProcessorConfig, QueueProcessorHealth,
    DocumentProcessor, EmbeddingGenerator, EmbeddingConfig,
    StorageClient, StorageConfig,
    LanguageServerManager,
    AllowedExtensions,
    SearchDbManager,
    create_grammar_manager,
    HierarchyBuilder, HierarchyRebuildConfig,
    adaptive_resources::{AdaptiveResourceManager, AdaptiveResourceConfig, AdaptiveResourceState},
};

use crate::database::ConcreteGraphStore;

/// All components produced by queue initialization for use by gRPC and watchers.
pub struct QueueComponents {
    pub unified_queue_processor: UnifiedQueueProcessor,
    pub allowed_extensions: Arc<AllowedExtensions>,
    pub hierarchy_builder: Arc<HierarchyBuilder>,
    pub adaptive_shutdown_token: tokio_util::sync::CancellationToken,
    pub adaptive_state: Arc<AdaptiveResourceState>,
    pub queue_health: Arc<QueueProcessorHealth>,
    /// Pool clone for WatchManager (taken before pool is moved into queue processor).
    pub watch_pool: SqlitePool,
    /// Mirror storage for rules backfill.
    pub mirror_storage: Arc<StorageClient>,
}

/// Initialize the IPC server with spill-to-disk and rollback storage.
pub async fn setup_ipc_server(
    config: &Config,
    queue_pool: &SqlitePool,
) -> Result<IpcServer, Box<dyn std::error::Error>> {
    info!("Initializing IPC server");
    let max_concurrent = config.max_concurrent_tasks.unwrap_or(8);
    let (mut ipc_server, _ipc_client) = IpcServer::new(max_concurrent);

    let ipc_spill_qm = Arc::new(
        workspace_qdrant_core::queue_operations::QueueManager::new(queue_pool.clone()),
    );
    ipc_server.set_spill_queue(ipc_spill_qm);

    let ipc_rollback_storage = Arc::new(StorageClient::with_config(StorageConfig::daemon_mode()));
    ipc_server.set_rollback_storage(ipc_rollback_storage);
    info!("IPC server configured with spill-to-disk and rollback storage");

    ipc_server.start().await.map_err(|e| {
        error!("Failed to start IPC server: {}", e);
        e
    })?;
    info!("IPC server started successfully");
    Ok(ipc_server)
}

/// Create the embedding generator from daemon configuration.
fn create_embedding_generator(
    daemon_config: &DaemonConfig,
    config: &Config,
) -> Result<Arc<EmbeddingGenerator>, Box<dyn std::error::Error>> {
    let embedding_config = EmbeddingConfig {
        max_cache_size: daemon_config.embedding.cache_max_entries,
        model_cache_dir: daemon_config.embedding.model_cache_dir.clone(),
        num_threads: Some(config.resource_limits.onnx_intra_threads),
        ..EmbeddingConfig::default()
    };

    if let Some(ref cache_dir) = embedding_config.model_cache_dir {
        info!("Using model cache directory: {}", cache_dir.display());
    }

    let generator = EmbeddingGenerator::new(embedding_config)
        .map_err(|e| format!("Failed to create embedding generator: {}", e))?;
    Ok(Arc::new(generator))
}

/// Initialize multi-tenant Qdrant collections (idempotent).
async fn init_qdrant_collections(storage_client: &StorageClient) {
    info!("Initializing Qdrant collections...");
    match storage_client.initialize_multi_tenant_collections(None).await {
        Ok(result) => {
            info!(
                "Qdrant collections initialized: projects={}, libraries={}, rules={}",
                result.projects_created, result.libraries_created, result.rules_created
            );
            if result.is_complete() {
                info!("All multi-tenant collections ready with dense+sparse vector support");
            }
        }
        Err(e) => {
            warn!("Failed to initialize Qdrant collections (will retry on use): {}", e);
        }
    }
}

/// Build the `UnifiedProcessorConfig` from daemon and core configs.
fn build_unified_config(
    config: &Config,
    daemon_config: &DaemonConfig,
    processor_config: &ProcessorConfig,
) -> UnifiedProcessorConfig {
    UnifiedProcessorConfig {
        batch_size: processor_config.batch_size,
        poll_interval_ms: processor_config.poll_interval_ms,
        worker_id: format!("memexd-{}", std::process::id()),
        lease_duration_secs: 300,
        max_retries: 3,
        inter_item_delay_ms: config.resource_limits.inter_item_delay_ms,
        max_concurrent_embeddings: config.resource_limits.max_concurrent_embeddings,
        max_memory_percent: config.resource_limits.max_memory_percent,
        warmup_window_secs: daemon_config.startup.warmup_window_secs,
        warmup_max_concurrent_embeddings: daemon_config.startup.warmup_max_concurrent_embeddings,
        warmup_inter_item_delay_ms: daemon_config.startup.warmup_inter_item_delay_ms,
        onnx_intra_threads: config.resource_limits.onnx_intra_threads,
        ..UnifiedProcessorConfig::default()
    }
}

/// Attach optional components (LSP, allowlist, search DB, graph) to the queue processor.
fn attach_optional_components(
    uqp: UnifiedQueueProcessor,
    lsp_manager: &Option<Arc<RwLock<LanguageServerManager>>>,
    allowed_extensions: &Arc<AllowedExtensions>,
    search_db: &Arc<SearchDbManager>,
    graph_store: &Option<ConcreteGraphStore>,
    watch_refresh_signal: &Arc<Notify>,
) -> UnifiedQueueProcessor {
    let mut uqp = uqp;
    if let Some(ref lsp) = lsp_manager {
        uqp = uqp.with_lsp_manager(Arc::clone(lsp));
        info!("LSP manager attached to unified queue processor for code enrichment");
    }
    uqp = uqp.with_allowed_extensions(Arc::clone(allowed_extensions));
    uqp = uqp.with_search_db(Arc::clone(search_db));
    if let Some(ref gs) = graph_store {
        uqp = uqp.with_graph_store(gs.clone());
    }
    uqp = uqp.with_watch_refresh_signal(Arc::clone(watch_refresh_signal));
    uqp
}

/// Initialize all queue-related components and return them for gRPC and watcher wiring.
#[allow(clippy::too_many_arguments)]
pub async fn initialize(
    config: &Config,
    daemon_config: &DaemonConfig,
    queue_pool: SqlitePool,
    search_db: &Arc<SearchDbManager>,
    graph_store: &Option<ConcreteGraphStore>,
    lsp_manager: &Option<Arc<RwLock<LanguageServerManager>>>,
    watch_refresh_signal: &Arc<Notify>,
) -> Result<QueueComponents, Box<dyn std::error::Error>> {
    info!("Initializing queue processor...");

    let processor_config = ProcessorConfig {
        batch_size: config.queue_batch_size.unwrap_or(10) as i32,
        poll_interval_ms: config.queue_poll_interval_ms.unwrap_or(500),
        worker_count: config.queue_worker_count.unwrap_or(4),
        backpressure_threshold: config.queue_backpressure_threshold.unwrap_or(1000),
        ..ProcessorConfig::default()
    };

    let embedding_generator = create_embedding_generator(daemon_config, config)?;
    let storage_client = Arc::new(StorageClient::with_config(StorageConfig::daemon_mode()));
    init_qdrant_collections(&storage_client).await;

    let allowed_extensions = Arc::new(AllowedExtensions::default());
    info!("File type allowlist initialized (90+ project extensions, library extensions active)");

    let unified_config = build_unified_config(config, daemon_config, &processor_config);
    let watch_pool = queue_pool.clone();
    let hierarchy_builder = Arc::new(HierarchyBuilder::new(
        queue_pool.clone(),
        Arc::clone(&embedding_generator),
        HierarchyRebuildConfig::default(),
    ));
    let mirror_storage = Arc::clone(&storage_client);

    let uqp = UnifiedQueueProcessor::with_components(
        queue_pool,
        unified_config.clone(),
        Arc::new(DocumentProcessor::new()),
        embedding_generator,
        storage_client,
    );

    // Create grammar manager for dynamic tree-sitter grammar loading
    let grammar_manager = Arc::new(RwLock::new(
        create_grammar_manager(daemon_config.grammars.clone())
    ));
    info!(
        "Grammar manager created (auto_download={}, cache_dir={:?})",
        daemon_config.grammars.auto_download,
        daemon_config.grammars.expanded_cache_dir()
    );

    let uqp = uqp.with_grammar_manager(Arc::clone(&grammar_manager));

    let mut uqp = attach_optional_components(
        uqp, lsp_manager, &allowed_extensions, search_db, graph_store, watch_refresh_signal,
    );

    // Adaptive resources + health monitoring
    let adaptive_shutdown_token = tokio_util::sync::CancellationToken::new();
    let adaptive_config = AdaptiveResourceConfig::from_resource_limits(&config.resource_limits);
    let adaptive_manager =
        AdaptiveResourceManager::start(adaptive_config, adaptive_shutdown_token.clone());
    let adaptive_state = adaptive_manager.state();
    uqp = uqp.with_adaptive_resources(adaptive_manager.subscribe());

    let queue_health = Arc::new(QueueProcessorHealth::new());
    uqp = uqp.with_queue_health(Arc::clone(&queue_health));

    info!(
        "Unified queue processor configured (batch_size={}, poll_interval={}ms, worker_id={})",
        unified_config.batch_size, unified_config.poll_interval_ms, unified_config.worker_id
    );

    Ok(QueueComponents {
        unified_queue_processor: uqp,
        allowed_extensions,
        hierarchy_builder,
        adaptive_shutdown_token,
        adaptive_state,
        queue_health,
        watch_pool,
        mirror_storage,
    })
}

/// Recover stale leases, apply warmup delay, and start the queue processor.
pub async fn start_processor(
    uqp: &mut UnifiedQueueProcessor,
    daemon_config: &DaemonConfig,
) -> Result<(), Box<dyn std::error::Error>> {
    if let Err(e) = uqp.recover_stale_leases().await {
        warn!("Failed to recover stale unified queue leases: {}", e);
    }

    let warmup_delay_secs = daemon_config.startup.warmup_delay_secs;
    if warmup_delay_secs > 0 {
        info!(
            "Applying startup warmup delay of {}s before queue processing begins...",
            warmup_delay_secs
        );
        tokio::time::sleep(tokio::time::Duration::from_secs(warmup_delay_secs)).await;
    }

    uqp.start()
        .map_err(|e| format!("Failed to start unified queue processor: {}", e))?;
    Ok(())
}

/// Spawn post-start background recovery tasks.
pub fn spawn_recovery_tasks(
    uqp: &UnifiedQueueProcessor,
    allowed_extensions: &Arc<AllowedExtensions>,
    mirror_storage: &Arc<StorageClient>,
    daemon_config: &DaemonConfig,
) {
    spawn_base_point_migration(uqp);
    spawn_startup_recovery(uqp, allowed_extensions, daemon_config);
    spawn_rules_mirror_backfill(uqp, mirror_storage);
    spawn_component_backfill(uqp);
}

fn spawn_base_point_migration(uqp: &UnifiedQueueProcessor) {
    let pool = uqp.pool().clone();
    let qm = uqp.queue_manager().clone();
    tokio::spawn(async move {
        match workspace_qdrant_core::startup::check_base_point_migration(&pool, &qm).await {
            Ok(true) => info!("base_point migration triggered"),
            Ok(false) => {}
            Err(e) => warn!("base_point migration check failed (non-fatal): {}", e),
        }
    });
}

fn spawn_startup_recovery(
    uqp: &UnifiedQueueProcessor,
    allowed_extensions: &Arc<AllowedExtensions>,
    daemon_config: &DaemonConfig,
) {
    let pool = uqp.pool().clone();
    let qm = uqp.queue_manager().clone();
    let ext = Arc::clone(allowed_extensions);
    let startup_config = daemon_config.startup.clone();
    tokio::spawn(async move {
        match workspace_qdrant_core::startup::run_startup_recovery(
            &pool, &qm, &ext, &startup_config,
        )
        .await
        {
            Ok(stats) => {
                let total = stats.total_queued();
                if total > 0 {
                    info!(
                        "Startup recovery complete: {} folders processed, {} items queued",
                        stats.folders_processed, total
                    );
                } else {
                    info!("Startup recovery complete: no changes detected");
                }
            }
            Err(e) => warn!("Startup recovery failed (non-fatal): {}", e),
        }
    });
}

fn spawn_rules_mirror_backfill(
    uqp: &UnifiedQueueProcessor,
    mirror_storage: &Arc<StorageClient>,
) {
    let pool = uqp.pool().clone();
    let storage = Arc::clone(mirror_storage);
    tokio::spawn(async move {
        match workspace_qdrant_core::startup::backfill_rules_mirror(&pool, &storage).await {
            Ok(stats) => {
                if stats.inserted > 0 {
                    info!(
                        "Rules mirror backfill: {} inserted, {} already existed",
                        stats.inserted, stats.already_exists
                    );
                } else {
                    debug!(
                        "Rules mirror backfill: no new entries ({} existed)",
                        stats.already_exists
                    );
                }
            }
            Err(e) => warn!("Rules mirror backfill failed (non-fatal): {}", e),
        }
    });
}

fn spawn_component_backfill(uqp: &UnifiedQueueProcessor) {
    let pool = uqp.pool().clone();
    tokio::spawn(async move {
        match workspace_qdrant_core::component_detection::backfill_components(&pool, 100).await {
            Ok(stats) => {
                if stats.files_updated > 0 {
                    info!(
                        "Component backfill: {} files updated, {} unmatched across {} folders",
                        stats.files_updated, stats.files_unmatched, stats.folders_processed
                    );
                } else {
                    debug!("Component backfill: no NULL components to backfill");
                }
            }
            Err(e) => warn!("Component backfill failed (non-fatal): {}", e),
        }
    });
}
