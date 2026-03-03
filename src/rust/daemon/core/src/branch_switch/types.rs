//! Types for the branch switch protocol.

/// Result of a branch switch operation
#[derive(Debug, Clone, Default)]
pub struct BranchSwitchStats {
    /// Files batch-updated (branch metadata only, no re-ingestion)
    pub batch_updated: u64,
    /// Files enqueued for re-ingestion (content changed)
    pub enqueued_changed: u64,
    /// Files enqueued for addition (new on target branch)
    pub enqueued_added: u64,
    /// Files enqueued for deletion (removed on target branch)
    pub enqueued_deleted: u64,
    /// Errors during processing
    pub errors: u64,
}
