//! Fluent builder for configuring `GrpcServer` with optional service dependencies.
//!
//! Each `with_*` method injects an optional dependency that the server factory
//! uses when wiring up gRPC services during `start()`.

use std::sync::atomic::AtomicBool;
use std::sync::Arc;

use sqlx::SqlitePool;
use tokio::sync::{Notify, RwLock};
use workspace_qdrant_core::adaptive_resources::AdaptiveResourceState;
use workspace_qdrant_core::LanguageServerManager;
use workspace_qdrant_core::SearchDbManager;

use crate::GrpcServer;

impl GrpcServer {
    pub fn with_shutdown_signal(mut self, receiver: tokio::sync::oneshot::Receiver<()>) -> Self {
        self.shutdown_signal = Some(receiver);
        self
    }

    /// Set the database pool for ProjectService.
    ///
    /// If provided, ProjectService will be registered with the gRPC server,
    /// enabling project registration, heartbeat, and priority management.
    pub fn with_database_pool(mut self, pool: SqlitePool) -> Self {
        self.db_pool = Some(pool);
        self
    }

    /// Enable LSP lifecycle management in ProjectService.
    ///
    /// When enabled, ProjectService will automatically start/stop LSP servers
    /// for projects when they are registered/deprioritized.
    pub fn with_lsp_enabled(mut self, enable: bool) -> Self {
        self.enable_lsp = enable;
        self
    }

    /// Set an external LSP manager for lifecycle control.
    ///
    /// When provided, this manager is used instead of creating one internally.
    /// This allows the daemon to manage the LSP lifecycle across components
    /// (e.g., sharing with UnifiedQueueProcessor).
    pub fn with_lsp_manager(mut self, manager: Arc<RwLock<LanguageServerManager>>) -> Self {
        self.lsp_manager = Some(manager);
        self.enable_lsp = true; // Automatically enable LSP when manager is provided
        self
    }

    /// Set a shared pause flag for watcher pause/resume propagation.
    /// This flag is shared with SystemServiceImpl so that gRPC pause/resume
    /// endpoints toggle it in addition to updating the database.
    pub fn with_pause_flag(mut self, flag: Arc<AtomicBool>) -> Self {
        self.pause_flag = Some(flag);
        self
    }

    /// Set the watch refresh signal for triggering WatchManager refresh
    /// when gRPC calls modify watch_folders state.
    pub fn with_watch_refresh_signal(mut self, signal: Arc<Notify>) -> Self {
        self.watch_refresh_signal = Some(signal);
        self
    }

    /// Set shared queue processor health state for health monitoring.
    /// This is the same `Arc` passed to `UnifiedQueueProcessor`.
    pub fn with_queue_health(
        mut self,
        health: Arc<workspace_qdrant_core::QueueProcessorHealth>,
    ) -> Self {
        self.queue_health = Some(health);
        self
    }

    /// Set adaptive resource state for idle/burst mode reporting in system status.
    pub fn with_adaptive_state(mut self, state: Arc<AdaptiveResourceState>) -> Self {
        self.adaptive_state = Some(state);
        self
    }

    /// Set the search database manager for TextSearchService.
    ///
    /// If provided, TextSearchService will be registered with the gRPC server,
    /// enabling FTS5-based code search via gRPC.
    pub fn with_search_db(mut self, search_db: Arc<SearchDbManager>) -> Self {
        self.search_db = Some(search_db);
        self
    }

    /// Set the graph store for GraphService.
    ///
    /// If provided, GraphService will be registered with the gRPC server,
    /// enabling code relationship queries via gRPC.
    pub fn with_graph_store(
        mut self,
        graph_store: workspace_qdrant_core::graph::SharedGraphStore<
            workspace_qdrant_core::graph::SqliteGraphStore,
        >,
    ) -> Self {
        self.graph_store = Some(graph_store);
        self
    }

    /// Set the hierarchy builder for tag hierarchy rebuild via RebuildIndex RPC.
    pub fn with_hierarchy_builder(
        mut self,
        builder: Arc<workspace_qdrant_core::HierarchyBuilder>,
    ) -> Self {
        self.hierarchy_builder = Some(builder);
        self
    }

    /// Set the lexicon manager for vocabulary rebuild via RebuildIndex RPC.
    pub fn with_lexicon_manager(
        mut self,
        lexicon: Arc<workspace_qdrant_core::LexiconManager>,
    ) -> Self {
        self.lexicon_manager = Some(lexicon);
        self
    }

    /// Set the storage client for Qdrant operations (rules rebuild).
    pub fn with_storage_client(
        mut self,
        client: Arc<workspace_qdrant_core::StorageClient>,
    ) -> Self {
        self.storage_client = Some(client);
        self
    }
}
