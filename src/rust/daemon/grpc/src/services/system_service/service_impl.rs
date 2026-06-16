//! SystemServiceImpl struct definition, constructor, and builder methods.
//!
//! Helper methods (health probes, queue metrics, folder scans, lifecycle) live
//! in the sibling `helpers` module.

use std::collections::HashMap;
use std::sync::atomic::AtomicBool;
use std::sync::Arc;
use std::time::SystemTime;

use tokio::sync::{Mutex, Notify, RwLock};
use workspace_qdrant_core::adaptive_resources::AdaptiveResourceState;
use workspace_qdrant_core::config::EmbeddingSettings;
use workspace_qdrant_core::embedding::provider::{DenseProvider, SharedProbeCache};
use workspace_qdrant_core::{EwmaState, QueueProcessorHealth};

use super::types::ServerStatusStore;

/// SystemService implementation
///
/// Provides health monitoring, status reporting, and lifecycle management.
/// Can be connected to actual queue processor health state for real metrics.
pub struct SystemServiceImpl {
    pub(super) start_time: SystemTime,
    /// Optional queue processor health state
    pub(super) queue_health: Option<Arc<QueueProcessorHealth>>,
    /// Optional shared dual-rate EWMA state (#133). Same `Arc` as the processor's
    /// — verdict reads here observe the samples the processor feeds in.
    pub(super) ewma_state: Option<Arc<EwmaState>>,
    /// Optional database pool for refresh signal operations
    pub(super) db_pool: Option<sqlx::SqlitePool>,
    /// Server status store for tracking component status
    pub(super) status_store: ServerStatusStore,
    /// Shared pause flag for propagation to file watchers.
    /// When the gRPC endpoint pauses/resumes, this flag is toggled atomically
    /// so that any FileWatcher sharing this flag reacts immediately.
    pub(super) pause_flag: Arc<AtomicBool>,
    /// Signal to trigger immediate WatchManager refresh
    pub(super) watch_refresh_signal: Option<Arc<Notify>>,
    /// Adaptive resource state for idle/burst mode reporting
    pub(super) adaptive_state: Option<Arc<AdaptiveResourceState>>,
    /// Hierarchy builder for tag hierarchy rebuild via RebuildIndex RPC
    pub(super) hierarchy_builder: Option<Arc<workspace_qdrant_core::HierarchyBuilder>>,
    /// Search database manager for FTS5 rebuild
    pub(super) search_db: Option<Arc<workspace_qdrant_core::SearchDbManager>>,
    /// Lexicon manager for vocabulary rebuild
    pub(super) lexicon_manager: Option<Arc<workspace_qdrant_core::LexiconManager>>,
    /// Storage client for Qdrant operations (rules rebuild)
    pub(super) storage_client: Option<Arc<workspace_qdrant_core::StorageClient>>,
    /// Active dense embedding provider for the embedding_provider health
    /// component and `GetEmbeddingProviderStatus` RPC.
    pub(super) dense_provider: Option<Arc<dyn DenseProvider>>,
    /// Embedding settings — authoritative `output_dim`, model id, base url.
    pub(super) embedding_settings: Option<Arc<EmbeddingSettings>>,
    /// Shared probe cache written by `ProviderHealthMonitor` (background) and
    /// read/written by on-demand `GetEmbeddingProviderStatus` RPC.
    pub(super) embedding_probe_cache: Arc<Mutex<SharedProbeCache>>,
}

impl std::fmt::Debug for SystemServiceImpl {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SystemServiceImpl")
            .field("start_time", &self.start_time)
            .field("queue_health", &self.queue_health.is_some())
            .field("ewma_state", &self.ewma_state.is_some())
            .field("db_pool", &self.db_pool.is_some())
            .field("hierarchy_builder", &self.hierarchy_builder.is_some())
            .field("search_db", &self.search_db.is_some())
            .field("lexicon_manager", &self.lexicon_manager.is_some())
            .field("storage_client", &self.storage_client.is_some())
            .finish()
    }
}

impl SystemServiceImpl {
    /// Create a new SystemService
    pub fn new() -> Self {
        Self {
            start_time: SystemTime::now(),
            queue_health: None,
            ewma_state: None,
            db_pool: None,
            status_store: Arc::new(RwLock::new(HashMap::new())),
            pause_flag: Arc::new(AtomicBool::new(false)),
            watch_refresh_signal: None,
            adaptive_state: None,
            hierarchy_builder: None,
            search_db: None,
            lexicon_manager: None,
            storage_client: None,
            dense_provider: None,
            embedding_settings: None,
            embedding_probe_cache: SharedProbeCache::new(),
        }
    }

    /// Set queue processor health state for monitoring
    pub fn with_queue_health(mut self, queue_health: Arc<QueueProcessorHealth>) -> Self {
        self.queue_health = Some(queue_health);
        self
    }

    /// Set the shared dual-rate EWMA state (#133). This is the same `Arc` handed
    /// to the `UnifiedQueueProcessor`, so verdict reads here see live samples.
    pub fn with_ewma_state(mut self, ewma_state: Arc<EwmaState>) -> Self {
        self.ewma_state = Some(ewma_state);
        self
    }

    /// Set the database pool for refresh signal operations
    pub fn with_database_pool(mut self, pool: sqlx::SqlitePool) -> Self {
        self.db_pool = Some(pool);
        self
    }

    /// Set a shared pause flag for propagation to file watchers.
    /// The returned `Arc<AtomicBool>` should be passed to the FileWatcher so both
    /// the gRPC endpoint and the watcher share the same atomic flag.
    pub fn with_pause_flag(mut self, flag: Arc<AtomicBool>) -> Self {
        self.pause_flag = flag;
        self
    }

    /// Set the watch refresh signal for triggering WatchManager refresh
    pub fn with_watch_refresh_signal(mut self, signal: Arc<Notify>) -> Self {
        self.watch_refresh_signal = Some(signal);
        self
    }

    /// Set the adaptive resource state for idle/burst mode reporting
    pub fn with_adaptive_state(mut self, state: Arc<AdaptiveResourceState>) -> Self {
        self.adaptive_state = Some(state);
        self
    }

    /// Set the hierarchy builder for tag hierarchy rebuild
    pub fn with_hierarchy_builder(
        mut self,
        builder: Arc<workspace_qdrant_core::HierarchyBuilder>,
    ) -> Self {
        self.hierarchy_builder = Some(builder);
        self
    }

    /// Set the search database manager for FTS5 rebuild
    pub fn with_search_db(
        mut self,
        search_db: Arc<workspace_qdrant_core::SearchDbManager>,
    ) -> Self {
        self.search_db = Some(search_db);
        self
    }

    /// Set the lexicon manager for vocabulary rebuild
    pub fn with_lexicon_manager(
        mut self,
        lexicon: Arc<workspace_qdrant_core::LexiconManager>,
    ) -> Self {
        self.lexicon_manager = Some(lexicon);
        self
    }

    /// Set the storage client for Qdrant operations (rules rebuild)
    pub fn with_storage_client(
        mut self,
        client: Arc<workspace_qdrant_core::StorageClient>,
    ) -> Self {
        self.storage_client = Some(client);
        self
    }

    /// Set the active dense embedding provider for the
    /// `embedding_provider` health component and `GetEmbeddingProviderStatus`.
    pub fn with_dense_provider(mut self, provider: Arc<dyn DenseProvider>) -> Self {
        self.dense_provider = Some(provider);
        self
    }

    /// Set the embedding settings (model, output_dim, base_url, probe TTL).
    pub fn with_embedding_settings(mut self, settings: Arc<EmbeddingSettings>) -> Self {
        self.embedding_settings = Some(settings);
        self
    }

    /// Set a shared probe cache (same instance passed to ProviderHealthMonitor).
    pub fn with_probe_cache(mut self, cache: Arc<Mutex<SharedProbeCache>>) -> Self {
        self.embedding_probe_cache = cache;
        self
    }

    /// Get a clone of the pause flag for sharing with file watchers
    pub fn pause_flag(&self) -> Arc<AtomicBool> {
        Arc::clone(&self.pause_flag)
    }
}

impl Default for SystemServiceImpl {
    fn default() -> Self {
        Self::new()
    }
}
