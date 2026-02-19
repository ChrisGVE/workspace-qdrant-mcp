//! Processing context bundling all dependencies needed by processing strategies.
//!
//! Replaces the 11+ argument function signatures in `unified_queue_processor.rs`
//! with a single struct that carries all shared state.

use std::sync::Arc;
use sqlx::SqlitePool;
use tokio::sync::{RwLock, Semaphore};

use crate::allowed_extensions::AllowedExtensions;
use crate::embedding::EmbeddingGenerator;
use crate::document_processor::DocumentProcessor;
use crate::lexicon::LexiconManager;
use crate::lsp::LanguageServerManager;
use crate::queue_operations::QueueManager;
use crate::search_db::SearchDbManager;
use crate::storage::StorageClient;

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
        }
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
        }
    }
}
