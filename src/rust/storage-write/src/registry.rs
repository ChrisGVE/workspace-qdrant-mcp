//! Per-branch write-handle registry.
//!
//! Location: `wqm-storage-write/src/registry.rs`. Logical context: the write
//! crate enforces single-writer-per-branch. This registry hands out one shared
//! async serialization lock per branch id so concurrent write requests for the
//! same branch queue instead of racing, while different branches proceed in
//! parallel. It is the concurrent-map foundation later write features build on.
//!
//! Binds the dashmap 6.x API (`DashMap::entry().or_insert_with`) — the F1
//! 5->6 major add (AC-F1.3): the live lock carried only a transitive dashmap
//! 5.5.3; this is the first direct 6.x consumer.
//!
//! Neighbors: [`crate::qdrant::QdrantWriteClient`] (the mutating Qdrant surface
//! these locks serialize around).

use std::sync::Arc;

use dashmap::DashMap;
use tokio::sync::Mutex;

/// Concurrent map of branch id -> per-branch write lock.
#[derive(Default)]
pub struct WriteHandleRegistry {
    locks: DashMap<String, Arc<Mutex<()>>>,
}

impl WriteHandleRegistry {
    /// Create an empty registry.
    pub fn new() -> Self {
        Self {
            locks: DashMap::new(),
        }
    }

    /// Return the shared write lock for `branch_id`, creating it on first use.
    /// Callers `.lock().await` the returned mutex to serialize writes to that
    /// branch; distinct branches get distinct locks and never contend.
    pub fn lock_for(&self, branch_id: &str) -> Arc<Mutex<()>> {
        self.locks
            .entry(branch_id.to_string())
            .or_insert_with(|| Arc::new(Mutex::new(())))
            .clone()
    }

    /// Number of branches with a registered lock.
    pub fn len(&self) -> usize {
        self.locks.len()
    }

    /// Whether any branch lock is registered.
    pub fn is_empty(&self) -> bool {
        self.locks.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn same_branch_shares_one_lock_distinct_branches_differ() {
        let reg = WriteHandleRegistry::new();
        let a1 = reg.lock_for("branch-a");
        let a2 = reg.lock_for("branch-a");
        let b1 = reg.lock_for("branch-b");
        assert!(Arc::ptr_eq(&a1, &a2), "same branch must share one lock");
        assert!(!Arc::ptr_eq(&a1, &b1), "distinct branches must not share");
        assert_eq!(reg.len(), 2);
    }
}
