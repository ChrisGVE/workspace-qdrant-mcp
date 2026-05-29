use std::sync::atomic::AtomicBool;
use std::sync::Arc;

use sqlx::SqlitePool;

use workspace_qdrant_core::config::EmbeddingSettings;
use workspace_qdrant_core::embedding::provider::DenseProvider;
use workspace_qdrant_core::storage::StorageClient;

// Re-export the canonical names from wqm_common so callers don't import two
// different aliases for the same constant.
pub use wqm_common::constants::CANONICAL_COLLECTIONS;

/// Shared dependencies wired into `AdminWriteServiceImpl` to support
/// `TriggerReembed`. Optional on the impl: when any field is missing the
/// RPC returns `Status::failed_precondition`.
#[derive(Clone)]
pub struct ReembedContext {
    pub settings: Arc<EmbeddingSettings>,
    pub provider: Arc<dyn DenseProvider>,
    pub storage_client: Arc<StorageClient>,
    pub pool: SqlitePool,
    pub pause_flag: Arc<AtomicBool>,
}
