//! Phase 6: gRPC server creation and dependency wiring.
//!
//! Constructs the `GrpcServer` with all its optional dependencies (database pool,
//! pause flag, search DB, graph store, LSP manager, etc.) and spawns it.

use std::sync::Arc;

use sqlx::SqlitePool;
use tokio::sync::{Notify, RwLock};
use tokio::task::JoinHandle;
use tracing::{error, info};

use workspace_qdrant_core::{
    adaptive_resources::AdaptiveResourceState, HierarchyBuilder, LanguageServerManager,
    ProjectLspConfig, QueueProcessorHealth, SearchDbManager,
};
use workspace_qdrant_grpc::{GrpcServer, ServerConfig as GrpcServerConfig};

use crate::database::ConcreteGraphStore;
use crate::DaemonArgs;

/// Initialize the LSP lifecycle manager (Task 1.1).
pub async fn init_lsp_manager(
    daemon_config: &workspace_qdrant_core::config::DaemonConfig,
) -> Option<Arc<RwLock<LanguageServerManager>>> {
    let lsp_config = ProjectLspConfig::from(daemon_config.lsp.clone());
    info!(
        "Initializing LSP lifecycle manager (max_servers={}, health_interval={}s)...",
        lsp_config.max_servers_per_project, lsp_config.health_check_interval_secs
    );
    match LanguageServerManager::new(lsp_config).await {
        Ok(mut manager) => {
            if let Err(e) = manager.initialize().await {
                tracing::warn!("Failed to initialize LSP manager: {}", e);
            }
            let manager = Arc::new(RwLock::new(manager));
            info!("LSP lifecycle manager initialized");
            Some(manager)
        }
        Err(e) => {
            tracing::warn!(
                "Failed to create LSP manager, continuing without LSP: {}",
                e
            );
            None
        }
    }
}

/// Spawn the gRPC server with all dependencies wired.
#[allow(clippy::too_many_arguments)]
pub fn spawn_grpc_server(
    args: &DaemonArgs,
    grpc_db_pool: SqlitePool,
    pause_flag: Arc<std::sync::atomic::AtomicBool>,
    watch_refresh_signal: Arc<Notify>,
    queue_health: Arc<QueueProcessorHealth>,
    adaptive_state: Arc<AdaptiveResourceState>,
    search_db: Arc<SearchDbManager>,
    graph_store: Option<ConcreteGraphStore>,
    lsp_manager: Option<Arc<RwLock<LanguageServerManager>>>,
    hierarchy_builder: Arc<HierarchyBuilder>,
    lexicon_pool: SqlitePool,
) -> Result<JoinHandle<()>, Box<dyn std::error::Error>> {
    let grpc_port = args.grpc_port;
    let grpc_addr = format!("127.0.0.1:{}", grpc_port)
        .parse()
        .map_err(|e| format!("Invalid gRPC address: {}", e))?;
    let grpc_config = GrpcServerConfig::new(grpc_addr);

    let grpc_lexicon_manager = Arc::new(workspace_qdrant_core::LexiconManager::new(
        lexicon_pool,
        workspace_qdrant_core::EmbeddingConfig::default().bm25_k1,
    ));

    info!("Starting gRPC server on port {}", grpc_port);
    let handle = tokio::spawn(async move {
        let mut grpc_server = GrpcServer::new(grpc_config)
            .with_database_pool(grpc_db_pool)
            .with_pause_flag(pause_flag)
            .with_watch_refresh_signal(watch_refresh_signal)
            .with_queue_health(queue_health)
            .with_adaptive_state(adaptive_state)
            .with_search_db(search_db)
            .with_hierarchy_builder(hierarchy_builder)
            .with_lexicon_manager(grpc_lexicon_manager);

        if let Some(lsp_manager) = lsp_manager {
            grpc_server = grpc_server.with_lsp_manager(lsp_manager);
        }

        if let Some(gs) = graph_store {
            grpc_server = grpc_server.with_graph_store(gs);
        }

        if let Err(e) = grpc_server.start().await {
            error!("gRPC server error: {}", e);
        }
    });
    info!(
        "gRPC server started on 127.0.0.1:{} with ProjectService enabled",
        grpc_port
    );

    Ok(handle)
}
