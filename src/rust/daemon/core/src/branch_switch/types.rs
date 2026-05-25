//! Types for the branch switch protocol.

use std::sync::Arc;

use crate::context::TenantBranchLocks;
use crate::search_db::SearchDbManager;
use crate::storage::StorageClient;

/// Dependencies for branch-array mutations during branch switch.
///
/// Groups the Qdrant client, search database, and per-tenant locks
/// needed to add a branch to existing points without re-embedding.
pub struct BranchUpdateContext {
    pub storage_client: Arc<StorageClient>,
    pub search_db: Option<Arc<SearchDbManager>>,
    pub branch_locks: Arc<TenantBranchLocks>,
}

/// Result of a branch switch operation
#[derive(Debug, Clone, Default)]
pub struct BranchSwitchStats {
    /// Files where new branch was added to branches[] (no re-ingestion)
    pub branch_added: u64,
    /// Files enqueued for re-ingestion (content changed)
    pub enqueued_changed: u64,
    /// Files enqueued for addition (new on target branch)
    pub enqueued_added: u64,
    /// Files enqueued for deletion (removed on target branch)
    pub enqueued_deleted: u64,
    /// Errors during processing
    pub errors: u64,
}
