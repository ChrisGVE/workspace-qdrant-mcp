//! Processing context bundling all dependencies needed by processing strategies.
//!
//! Replaces the 11+ argument function signatures in `unified_queue_processor.rs`
//! with a single struct that carries all shared state.

use sqlx::SqlitePool;
use std::sync::Arc;
use tokio::sync::{RwLock, Semaphore};

use std::collections::{HashMap, HashSet};

use crate::allowed_extensions::AllowedExtensions;
use crate::component_detection::ComponentMap;
use crate::config::{IngestionLimitsConfig, UrlIngestionConfig};
use crate::document_processor::DocumentProcessor;
use crate::embedding::EmbeddingGenerator;
use crate::git::BranchCache;
use crate::graph::GraphStore;
use crate::keyword_extraction::cooccurrence_graph::CentralityCache;
use crate::lexicon::LexiconManager;
use crate::lsp::LanguageServerManager;
use crate::patterns::GitattributesOverrides;
use crate::queue_operations::QueueManager;
use crate::search_db::SearchDbManager;
use crate::storage::StorageClient;
use crate::tagging::Tier2Tagger;
use crate::tree_sitter::GrammarManager;

/// Per-tenant mutex registry for serializing branch-array mutations.
///
/// When content-hash deduplication detects that identical file content already
/// exists under a different branch, the Qdrant point `branches` payload and
/// the SQLite `tracked_files.branches` JSON array are updated via
/// read-modify-write. This struct provides a per-tenant async mutex to
/// prevent concurrent mutations from racing on the same array.
pub struct TenantBranchLocks {
    locks: std::sync::Mutex<HashMap<String, Arc<tokio::sync::Mutex<()>>>>,
}

impl TenantBranchLocks {
    /// Create an empty lock registry.
    pub fn new() -> Self {
        Self {
            locks: std::sync::Mutex::new(HashMap::new()),
        }
    }

    /// Get (or create) the async mutex for `tenant_id`.
    ///
    /// The returned `Arc<Mutex<()>>` is shared across all callers for the same
    /// tenant, so acquiring it serializes branch-array mutations for that tenant.
    pub fn get(&self, tenant_id: &str) -> Arc<tokio::sync::Mutex<()>> {
        let mut map = self.locks.lock().expect("TenantBranchLocks poisoned");
        Arc::clone(
            map.entry(tenant_id.to_string())
                .or_insert_with(|| Arc::new(tokio::sync::Mutex::new(()))),
        )
    }
}

impl Default for TenantBranchLocks {
    fn default() -> Self {
        Self::new()
    }
}

/// Tracks which (watch_folder_id, branch) pairs have had discovery checked.
///
/// This prevents re-running the expensive filesystem scan on every file
/// event for a branch that's already been discovered or confirmed as known.
pub struct DiscoveryTracker {
    checked: std::sync::Mutex<HashSet<(String, String)>>,
}

impl DiscoveryTracker {
    pub fn new() -> Self {
        Self {
            checked: std::sync::Mutex::new(HashSet::new()),
        }
    }

    /// Process-global shared tracker.
    ///
    /// Discovery dedup is inherently per-daemon-run, so all processing
    /// contexts must share one instance. `ProcessingContext::new` is called
    /// per queue item; a fresh per-item tracker meant `is_checked` always
    /// returned false, so branch discovery re-ran a full-tree filesystem
    /// scan for EVERY item — stalling the processor for minutes on large
    /// tenants (119K-file library). Sharing one tracker makes discovery run
    /// once per (watch_folder_id, branch) as intended.
    pub fn global() -> Arc<DiscoveryTracker> {
        static GLOBAL: std::sync::OnceLock<Arc<DiscoveryTracker>> = std::sync::OnceLock::new();
        GLOBAL
            .get_or_init(|| Arc::new(DiscoveryTracker::new()))
            .clone()
    }

    /// Returns true if this (watch_folder_id, branch) has already been checked.
    pub fn is_checked(&self, watch_folder_id: &str, branch: &str) -> bool {
        let set = self.checked.lock().expect("DiscoveryTracker poisoned");
        set.contains(&(watch_folder_id.to_string(), branch.to_string()))
    }

    /// Mark a (watch_folder_id, branch) as checked.
    pub fn mark_checked(&self, watch_folder_id: &str, branch: &str) {
        let mut set = self.checked.lock().expect("DiscoveryTracker poisoned");
        set.insert((watch_folder_id.to_string(), branch.to_string()));
    }
}

impl Default for DiscoveryTracker {
    fn default() -> Self {
        Self::new()
    }
}

/// Bundled processing dependencies for queue item strategies.
///
/// Instead of passing 11+ arguments through every function call in the
/// queue processing chain, strategies receive a single `ProcessingContext`
/// reference that provides access to all shared components.
pub struct ProcessingContext {
    /// SQLite connection pool for state operations.
    pub pool: SqlitePool,

    /// Queue manager for enqueue/dequeue/status operations.
    pub queue_manager: Arc<QueueManager>,

    /// Qdrant storage client for vector operations.
    pub storage_client: Arc<StorageClient>,

    /// Dense + sparse embedding generator.
    pub embedding_generator: Arc<EmbeddingGenerator>,

    /// Document content extraction and chunking.
    pub document_processor: Arc<DocumentProcessor>,

    /// Semaphore limiting concurrent embedding operations.
    pub embedding_semaphore: Arc<Semaphore>,

    /// Per-collection BM25 vocabulary persistence for IDF-weighted sparse vectors.
    pub lexicon_manager: Arc<LexiconManager>,

    /// LSP manager for code intelligence enrichment (optional — not all deployments have LSP).
    pub lsp_manager: Option<Arc<RwLock<LanguageServerManager>>>,

    /// FTS5 search database manager (optional — can be disabled).
    pub search_db: Option<Arc<SearchDbManager>>,

    /// File type allowlist for ingestion filtering.
    pub allowed_extensions: Arc<AllowedExtensions>,

    /// TTL-based cache for symbol co-occurrence centrality scores.
    pub cooccurrence_cache: Arc<tokio::sync::Mutex<CentralityCache>>,

    /// Graph store for code relationship storage (optional — initialized when graph.db available).
    pub graph_store: Option<Arc<dyn GraphStore>>,

    /// Cache of detected project components, keyed by watch_folder_id.
    /// Lazily populated on first file processed per watch folder.
    pub component_cache: Arc<RwLock<HashMap<String, ComponentMap>>>,

    /// Grammar manager for dynamic tree-sitter grammar loading (optional).
    /// Provides on-demand grammar download and caching for semantic code chunking.
    pub grammar_manager: Option<Arc<RwLock<GrammarManager>>>,

    /// Per-extension ingestion size limits (Task 14).
    pub ingestion_limits: Arc<IngestionLimitsConfig>,

    /// Languages with in-flight background grammar downloads.
    /// Prevents duplicate download spawns for the same language.
    pub pending_grammar_downloads: Arc<tokio::sync::Mutex<HashSet<String>>>,

    /// Per-project `.gitattributes` overrides cache, keyed by project root path.
    /// Lazily populated on first file processed per project.
    pub gitattributes_cache: Arc<RwLock<HashMap<String, GitattributesOverrides>>>,

    /// URL ingestion limits and SSRF policy (T5).
    pub url_ingestion: Arc<UrlIngestionConfig>,

    /// Tier 2 taxonomy-based tagger (optional — initialized at daemon startup
    /// after embedding generator and taxonomy YAML are available).
    pub tier2_tagger: Option<Arc<Tier2Tagger>>,

    /// TTL-based cache for resolving the current git branch from `.git/HEAD`.
    /// Shared across all file items so rapid successive items from the same
    /// project avoid repeated filesystem reads.
    pub branch_cache: Arc<BranchCache>,

    /// Per-tenant mutex registry for serializing branch-array mutations
    /// during content-hash deduplication. Prevents read-modify-write races
    /// when adding a branch to an existing Qdrant point's `branches` payload.
    pub branch_locks: Arc<TenantBranchLocks>,

    /// Tracks (watch_folder_id, branch) pairs that have already been checked
    /// for discovery. Prevents re-running discovery on every file event for
    /// a branch that's already been processed.
    pub discovery_tracker: Arc<DiscoveryTracker>,

    /// Optional dedicated local FastEmbed generator for keyword extraction.
    /// When present, keyword/tag embedding uses this instead of the main
    /// `embedding_generator`, freeing the main provider for chunk embeddings.
    pub keyword_embedding_generator: Option<Arc<EmbeddingGenerator>>,
}

impl ProcessingContext {
    /// Create a new processing context from individual components.
    pub fn new(
        pool: SqlitePool,
        queue_manager: Arc<QueueManager>,
        storage_client: Arc<StorageClient>,
        embedding_generator: Arc<EmbeddingGenerator>,
        document_processor: Arc<DocumentProcessor>,
        embedding_semaphore: Arc<Semaphore>,
        lexicon_manager: Arc<LexiconManager>,
        lsp_manager: Option<Arc<RwLock<LanguageServerManager>>>,
        search_db: Option<Arc<SearchDbManager>>,
        allowed_extensions: Arc<AllowedExtensions>,
    ) -> Self {
        Self {
            pool,
            queue_manager,
            storage_client,
            embedding_generator,
            document_processor,
            embedding_semaphore,
            lexicon_manager,
            lsp_manager,
            search_db,
            allowed_extensions,
            cooccurrence_cache: Arc::new(tokio::sync::Mutex::new(CentralityCache::default())),
            graph_store: None,
            component_cache: Arc::new(RwLock::new(HashMap::new())),
            grammar_manager: None,
            ingestion_limits: Arc::new(IngestionLimitsConfig::default()),
            pending_grammar_downloads: Arc::new(tokio::sync::Mutex::new(HashSet::new())),
            gitattributes_cache: Arc::new(RwLock::new(HashMap::new())),
            url_ingestion: Arc::new(UrlIngestionConfig::default()),
            tier2_tagger: None,
            branch_cache: Arc::new(BranchCache::new()),
            branch_locks: Arc::new(TenantBranchLocks::new()),
            discovery_tracker: DiscoveryTracker::global(),
            keyword_embedding_generator: None,
        }
    }
}

impl ProcessingContext {
    /// Attach a graph store to the processing context.
    pub fn with_graph_store(mut self, store: Arc<dyn GraphStore>) -> Self {
        self.graph_store = Some(store);
        self
    }

    /// Attach a grammar manager for dynamic tree-sitter grammar loading.
    pub fn with_grammar_manager(mut self, manager: Arc<RwLock<GrammarManager>>) -> Self {
        self.grammar_manager = Some(manager);
        self
    }

    /// Override per-extension ingestion size limits (Task 14).
    pub fn with_ingestion_limits(mut self, limits: Arc<IngestionLimitsConfig>) -> Self {
        self.ingestion_limits = limits;
        self
    }

    /// Override URL ingestion config (T5).
    pub fn with_url_ingestion(mut self, cfg: Arc<UrlIngestionConfig>) -> Self {
        self.url_ingestion = cfg;
        self
    }

    /// Attach a Tier 2 taxonomy tagger.
    pub fn with_tier2_tagger(mut self, tagger: Arc<Tier2Tagger>) -> Self {
        self.tier2_tagger = Some(tagger);
        self
    }

    /// Attach a dedicated keyword embedding generator.
    pub fn with_keyword_embedding_generator(mut self, gen: Arc<EmbeddingGenerator>) -> Self {
        self.keyword_embedding_generator = Some(gen);
        self
    }

    /// Returns the keyword-specific generator if available, else the main one.
    pub fn keyword_generator(&self) -> &Arc<EmbeddingGenerator> {
        self.keyword_embedding_generator
            .as_ref()
            .unwrap_or(&self.embedding_generator)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // ProcessingContext is a data struct — its test surface is covered by
    // the strategies and pipeline handlers that consume it.
    // This test verifies the struct is constructible with the expected field names.
    #[test]
    fn test_context_field_names_compile() {
        // Compile-time check: all field names are valid identifiers.
        fn _assert_fields(ctx: &ProcessingContext) {
            let _ = &ctx.pool;
            let _ = &ctx.queue_manager;
            let _ = &ctx.storage_client;
            let _ = &ctx.embedding_generator;
            let _ = &ctx.document_processor;
            let _ = &ctx.embedding_semaphore;
            let _ = &ctx.lexicon_manager;
            let _ = &ctx.lsp_manager;
            let _ = &ctx.search_db;
            let _ = &ctx.allowed_extensions;
            let _ = &ctx.cooccurrence_cache;
            let _ = &ctx.graph_store;
            let _ = &ctx.component_cache;
            let _ = &ctx.grammar_manager;
            let _ = &ctx.ingestion_limits;
            let _ = &ctx.pending_grammar_downloads;
            let _ = &ctx.gitattributes_cache;
            let _ = &ctx.url_ingestion;
            let _ = &ctx.tier2_tagger;
            let _ = &ctx.branch_cache;
            let _ = &ctx.branch_locks;
            let _ = &ctx.discovery_tracker;
            let _ = &ctx.keyword_embedding_generator;
        }
    }

    #[test]
    fn discovery_tracker_global_is_shared_and_persists() {
        // The global tracker must be one shared instance so discovery dedup
        // survives across per-item ProcessingContext construction.
        let a = DiscoveryTracker::global();
        let b = DiscoveryTracker::global();
        assert!(
            Arc::ptr_eq(&a, &b),
            "global() must return the same instance"
        );

        a.mark_checked("watch-xyz", "main");
        assert!(
            b.is_checked("watch-xyz", "main"),
            "a mark via one handle must be visible through another"
        );
        // Distinct (watch, branch) pairs remain independent.
        assert!(!b.is_checked("watch-xyz", "dev"));
    }

    #[test]
    fn test_tenant_branch_locks_returns_same_lock_for_same_tenant() {
        let locks = TenantBranchLocks::new();
        let lock1 = locks.get("tenant_a");
        let lock2 = locks.get("tenant_a");
        // Same Arc pointer
        assert!(Arc::ptr_eq(&lock1, &lock2));
    }

    #[test]
    fn test_tenant_branch_locks_returns_different_lock_for_different_tenant() {
        let locks = TenantBranchLocks::new();
        let lock_a = locks.get("tenant_a");
        let lock_b = locks.get("tenant_b");
        // Different Arc pointers
        assert!(!Arc::ptr_eq(&lock_a, &lock_b));
    }

    #[test]
    fn test_tenant_branch_locks_default() {
        let locks = TenantBranchLocks::default();
        let lock = locks.get("test");
        // Verify we can get a lock from default-constructed instance
        assert!(Arc::strong_count(&lock) >= 1);
    }
}
