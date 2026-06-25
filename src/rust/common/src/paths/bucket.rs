//! Store-bucket folder layout (AC-F16.3).
//!
//! Per-tenant `store.db` files live under three sibling top-level buckets in
//! the wqm data directory, keyed by tenant class (arch §3 canonical layout):
//!
//! ```text
//! <data_dir>/
//!   projects/<tenant_id>/store.db   -- registered projects
//!   libraries/<tenant_id>/store.db  -- reference libraries (branchless)
//!   global/<tenant_id>/store.db     -- global bucket (orphan re-home target)
//! ```
//!
//! All three are structurally identical `store.db` files; the bucket prefix
//! alone records tenant class. This module is the single source of truth for
//! the name-to-path mapping (FP-2 — no per-call-site bucket-string literals).
//!
//! Consumed by:
//!   - F13 / registration: daemon sets `projects.db_path` via
//!     `store_bucket_path(data_dir, StoreBucket::Projects, tenant_id)`.
//!   - AC-F16.1 library open: `store_bucket_path(data_dir, StoreBucket::Libraries, tenant_id)`.
//!   - AC-F16.5 orphan re-home: `store_bucket_path(data_dir, StoreBucket::Global, tenant_id)`.

use std::path::{Path, PathBuf};

// ---------------------------------------------------------------------------
// StoreBucket enum
// ---------------------------------------------------------------------------

/// The three top-level storage buckets under the wqm data directory.
///
/// Per-tenant `store.db` files live under three sibling buckets keyed by
/// tenant class (arch §3 canonical layout, AC-F16.3). This enum is the
/// single name-to-path mapping (FP-2 — one location, no per-call-site
/// bucket-string literals).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StoreBucket {
    /// Registered project stores (`projects/<tenant_id>/store.db`).
    Projects,
    /// Reference library stores (`libraries/<tenant_id>/store.db`).
    Libraries,
    /// Global bucket — orphan re-home target (`global/<tenant_id>/store.db`).
    Global,
}

impl StoreBucket {
    /// Directory name of this bucket as it appears under the data directory.
    pub fn dir_name(self) -> &'static str {
        match self {
            StoreBucket::Projects => "projects",
            StoreBucket::Libraries => "libraries",
            StoreBucket::Global => "global",
        }
    }
}

// ---------------------------------------------------------------------------
// Path helpers
// ---------------------------------------------------------------------------

/// Canonical absolute path to a tenant's `store.db` inside `data_dir`.
///
/// Produces `<data_dir>/<bucket>/<tenant_id>/store.db` — the single path
/// construction rule shared by all bucket types (AC-F16.3, FP-2: one
/// bucket-path function, not per-call-site joins).
///
/// The parent directory (`<data_dir>/<bucket>/<tenant_id>/`) is NOT created
/// here; call [`ensure_store_dir`] before opening the store for writing.
pub fn store_bucket_path(data_dir: &Path, bucket: StoreBucket, tenant_id: &str) -> PathBuf {
    data_dir
        .join(bucket.dir_name())
        .join(tenant_id)
        .join("store.db")
}

/// Ensure the parent directory of a `store.db` path exists, creating it and
/// all intermediate directories if necessary.
///
/// Call this before opening a store for writing. The path argument is the
/// full `store.db` file path (as returned by [`store_bucket_path`]); the
/// function creates the parent directory (`<data_dir>/<bucket>/<tenant_id>/`).
///
/// Returns `Ok(())` when the directory exists (regardless of whether it was
/// just created or already present) or an `Err` when creation fails.
pub fn ensure_store_dir(store_db_path: &Path) -> std::io::Result<()> {
    if let Some(parent) = store_db_path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Tests (AC-F16.3)
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    /// T-F16.3-projects: projects bucket produces <data_dir>/projects/<id>/store.db.
    #[test]
    fn t_f16_3_projects_bucket_path() {
        let data = Path::new("/wqm/data");
        let path = store_bucket_path(data, StoreBucket::Projects, "tenant-abc");
        assert_eq!(
            path,
            PathBuf::from("/wqm/data/projects/tenant-abc/store.db"),
            "projects bucket path must be <data_dir>/projects/<tenant_id>/store.db"
        );
    }

    /// T-F16.3-libraries: libraries bucket produces <data_dir>/libraries/<id>/store.db.
    #[test]
    fn t_f16_3_libraries_bucket_path() {
        let data = Path::new("/wqm/data");
        let path = store_bucket_path(data, StoreBucket::Libraries, "lib-xyz");
        assert_eq!(
            path,
            PathBuf::from("/wqm/data/libraries/lib-xyz/store.db"),
            "libraries bucket path must be <data_dir>/libraries/<tenant_id>/store.db"
        );
    }

    /// T-F16.3-global: global bucket produces <data_dir>/global/<id>/store.db.
    #[test]
    fn t_f16_3_global_bucket_path() {
        let data = Path::new("/wqm/data");
        let path = store_bucket_path(data, StoreBucket::Global, "global-sentinel");
        assert_eq!(
            path,
            PathBuf::from("/wqm/data/global/global-sentinel/store.db"),
            "global bucket path must be <data_dir>/global/<tenant_id>/store.db"
        );
    }

    /// T-F16.3-ensure-dir: ensure_store_dir creates the parent directory for a
    /// store.db path and is idempotent (calling twice does not error).
    #[test]
    fn t_f16_3_ensure_store_dir_creates_parent() {
        let tmp = tempfile::tempdir().expect("tempdir");
        let store_path = store_bucket_path(tmp.path(), StoreBucket::Projects, "t1");
        // Parent must not exist yet.
        assert!(!store_path.parent().unwrap().exists());
        // First call creates it.
        ensure_store_dir(&store_path).expect("ensure_store_dir must succeed");
        assert!(
            store_path.parent().unwrap().is_dir(),
            "parent dir must exist after ensure_store_dir"
        );
        // Second call is idempotent.
        ensure_store_dir(&store_path).expect("ensure_store_dir must be idempotent");
    }

    /// T-F16.3-all-buckets-distinct: all three buckets for the same tenant_id
    /// produce distinct paths (no bucket collision).
    #[test]
    fn t_f16_3_bucket_paths_are_distinct() {
        let data = Path::new("/wqm/data");
        let tid = "tenant-999";
        let p = store_bucket_path(data, StoreBucket::Projects, tid);
        let l = store_bucket_path(data, StoreBucket::Libraries, tid);
        let g = store_bucket_path(data, StoreBucket::Global, tid);
        assert_ne!(p, l, "projects and libraries buckets must differ");
        assert_ne!(p, g, "projects and global buckets must differ");
        assert_ne!(l, g, "libraries and global buckets must differ");
    }
}
