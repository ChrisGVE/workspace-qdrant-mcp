use std::sync::atomic::AtomicBool;
use std::sync::Arc;

use sqlx::SqlitePool;

use workspace_qdrant_core::config::EmbeddingSettings;
use workspace_qdrant_core::embedding::provider::DenseProvider;
use workspace_qdrant_core::storage::StorageClient;

/// Names of the four canonical Qdrant collections affected by a reembed.
pub const CANONICAL_COLLECTIONS: &[&str] = &["projects", "libraries", "rules", "scratchpad"];

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
