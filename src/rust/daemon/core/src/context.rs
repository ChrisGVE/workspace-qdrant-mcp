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
use crate::config::IngestionLimitsConfig;
use crate::document_processor::DocumentProcessor;
use crate::embedding::EmbeddingGenerator;
use crate::graph::{SharedGraphStore, SqliteGraphStore};
use crate::keyword_extraction::cooccurrence_graph::CentralityCache;
use crate::lexicon::LexiconManager;
use crate::lsp::LanguageServerManager;
use crate::patterns::GitattributesOverrides;
use crate::queue_operations::QueueManager;
use crate::search_db::SearchDbManager;
use crate::storage::StorageClient;
use crate::tree_sitter::GrammarManager;

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
    pub graph_store: Option<SharedGraphStore<SqliteGraphStore>>,

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
        }
    }
}

impl ProcessingContext {
    /// Attach a graph store to the processing context.
    pub fn with_graph_store(mut self, store: SharedGraphStore<SqliteGraphStore>) -> Self {
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
        }
    }
}
