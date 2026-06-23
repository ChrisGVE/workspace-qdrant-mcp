//! `ProjectBinding` — the resolved (tenant, branch, store path) triple.
//!
//! Location: `wqm-storage/src/types/binding.rs`. Logical context: what the
//! registry resolves a working directory to before any store operation — the
//! owning tenant, the active branch, and the per-project `store.db` path. The
//! facade opens that path and scopes every query/write to this binding.

use std::path::PathBuf;

use wqm_common::domain::{BranchId, TenantId};

/// A resolved project binding: which tenant + branch a request targets and where
/// that tenant's per-project `store.db` lives on disk.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ProjectBinding {
    /// Owning tenant — the partition key for the store and every Qdrant filter.
    pub tenant_id: TenantId,
    /// Active branch within the tenant.
    pub branch_id: BranchId,
    /// Filesystem path to this tenant's per-project SQLite store.
    pub db_path: PathBuf,
}

impl ProjectBinding {
    /// Construct a binding from its parts.
    pub fn new(tenant_id: TenantId, branch_id: BranchId, db_path: impl Into<PathBuf>) -> Self {
        Self {
            tenant_id,
            branch_id,
            db_path: db_path.into(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn binding_holds_its_parts() {
        let b = ProjectBinding::new(
            TenantId::new("acme"),
            BranchId::new("deadbeef"),
            "/tmp/acme/store.db",
        );
        assert_eq!(b.tenant_id.as_str(), "acme");
        assert_eq!(b.branch_id.as_str(), "deadbeef");
        assert_eq!(b.db_path, PathBuf::from("/tmp/acme/store.db"));
    }
}
