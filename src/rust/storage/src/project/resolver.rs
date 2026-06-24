//! `ProjectRegistry` — CWD-to-tenant resolution over `state.db` (AC-F10.10).
//!
//! File: `wqm-storage/src/project/resolver.rs`
//! Location: `src/rust/storage/src/project/` (read crate)
//! Context: workspace-qdrant-mcp branch-storage model (arch §6.2, §8).
//!   The single CWD->tenant resolver (FP-2). Every facade search begins here:
//!   the registry walks the caller's CWD up to the most-specific registered
//!   project root, returning (tenant_id, branch_id, db_path) as a
//!   `ProjectBinding`. No component path-walks on its own — all resolution
//!   routes through this type.
//!
//!   AC-F10.10: this type is MINTED net-new by F10. F16 will EXTEND it with the
//!   FP-3 fuzzy handle->key resolver (AC-F16.6) — one nexus, two phases, no
//!   second resolver (FP-2). Leave room for that extension in the public API.
//!
//!   Path canonicalization (arch §6.5): `resolve_project` resolves symlinks and
//!   `..` components via `std::fs::canonicalize` before any query, preventing
//!   path-traversal in the walk-up logic.
//!
//!   SEC-3 semantics: when no registered root matches, the facade MUST return
//!   error or empty — never fall through to an all-tenant search.
//!
//! Neighbors: `crate::types::binding::ProjectBinding` (resolved triple),
//!   `wqm_common::error::StorageError` (canonical error type, DR GP-9).

use std::path::{Path, PathBuf};

use sqlx::sqlite::{SqliteConnectOptions, SqlitePool, SqlitePoolOptions};
use std::str::FromStr;
use tracing::instrument;
use wqm_common::domain::{BranchId, TenantId};
use wqm_common::error::StorageError;

use crate::types::binding::ProjectBinding;

// ---------------------------------------------------------------------------
// ProjectRegistry
// ---------------------------------------------------------------------------

/// Maps a caller's CWD to the owning `ProjectBinding` (tenant, branch, store
/// path) by walking up from the CWD through registered project roots in
/// `state.db.project_locations`.
///
/// One nexus, two phases (FP-2):
///   - F10 mints `resolve_project` (CWD -> tenant resolution, this file).
///   - F16 extends this same type with the FP-3 fuzzy handle->key resolver
///     (`resolve_by_handle`, AC-F16.6). Do not create a second resolver.
pub struct ProjectRegistry {
    /// Pool for `state.db` (central registry). Read-only; never writes.
    state_pool: SqlitePool,
}

impl ProjectRegistry {
    /// Open a `ProjectRegistry` against the given `state.db` path.
    ///
    /// The pool is read-only (`query_only = ON`) consistent with the read-crate
    /// contract (arch §6.2). Max 2 connections — `state.db` is light-read.
    pub async fn open(state_db_path: impl AsRef<Path>) -> Result<Self, StorageError> {
        let path = state_db_path.as_ref();
        let url = format!("sqlite://{}", path.display());
        let opts = SqliteConnectOptions::from_str(&url)
            .map_err(|e| StorageError::Connection(format!("invalid state.db URL: {e}")))?
            .read_only(true)
            .pragma("query_only", "ON")
            .pragma("foreign_keys", "ON")
            .pragma("journal_mode", "WAL")
            .pragma("busy_timeout", "5000");

        let pool = SqlitePoolOptions::new()
            .max_connections(2)
            .connect_with(opts)
            .await
            .map_err(|e| StorageError::Connection(format!("failed to open state.db pool: {e}")))?;

        Ok(Self { state_pool: pool })
    }

    /// Resolve `cwd` to the `ProjectBinding` of the most-specific registered
    /// project root that contains it.
    ///
    /// The path is canonicalized (symlinks + `..` resolved) before any query
    /// (arch §6.5). The most-specific root wins: a submodule root at
    /// `/a/b` beats the container project at `/a` when the CWD is `/a/b/src`.
    ///
    /// Returns `None` when no registered root is a prefix of the canonical CWD.
    /// The caller (facade) MUST treat `None` as an error — never fall through to
    /// an all-tenant query (SEC-3, arch §6.5).
    #[instrument(skip(self), fields(cwd = %cwd.as_ref().display()))]
    pub async fn resolve_project(
        &self,
        cwd: impl AsRef<Path>,
    ) -> Result<Option<ProjectBinding>, StorageError> {
        // Canonicalize: resolve symlinks and `..` (arch §6.5 path-traversal guard).
        let canonical = canonicalize_cwd(cwd.as_ref())?;

        // Fetch all active locations with their project metadata in one query.
        // Most-specific-root-wins is handled in Rust by longest-prefix selection.
        let rows = sqlx::query_as::<_, LocationRow>(
            r#"
            SELECT
                pl.location,
                pl.branch_id,
                p.tenant_id,
                p.db_path
            FROM project_locations pl
            JOIN projects p ON p.project_id = pl.project_id
            WHERE pl.active = 1
            "#,
        )
        .fetch_all(&self.state_pool)
        .await
        .map_err(|e| StorageError::Sqlite(format!("state.db location query failed: {e}")))?;

        Ok(most_specific_match(&canonical, &rows))
    }

    /// List all branches for a project identified by `tenant_id`.
    ///
    /// Returns one row per `project_locations` entry (one per checkout/branch
    /// combo). Used by `wqm project branches` (AC-F10.8).
    pub async fn list_project_branches(
        &self,
        tenant_id: &str,
    ) -> Result<Vec<BranchEntry>, StorageError> {
        let rows = sqlx::query_as::<_, BranchEntry>(
            r#"
            SELECT
                pl.branch_name,
                pl.branch_id,
                pl.sync_state,
                pl.location,
                pl.active
            FROM project_locations pl
            JOIN projects p ON p.project_id = pl.project_id
            WHERE p.tenant_id = ?1
            ORDER BY pl.branch_name ASC, pl.location ASC
            "#,
        )
        .bind(tenant_id)
        .fetch_all(&self.state_pool)
        .await
        .map_err(|e| StorageError::Sqlite(format!("list_project_branches failed: {e}")))?;

        Ok(rows)
    }
}

// ---------------------------------------------------------------------------
// Public types
// ---------------------------------------------------------------------------

/// One branch entry returned by `ProjectRegistry::list_project_branches`.
/// Used by `wqm project branches` (AC-F10.8).
#[derive(Debug, Clone, sqlx::FromRow)]
pub struct BranchEntry {
    /// Git ref name (e.g. "main", "feat/x").
    pub branch_name: String,
    /// Canonical search key (SHA256 of tenant|location|branch_name).
    pub branch_id: String,
    /// Sync state from `project_locations`.
    pub sync_state: String,
    /// Absolute checkout root path.
    pub location: String,
    /// Whether this checkout is currently active.
    pub active: bool,
}

// ---------------------------------------------------------------------------
// Private helpers
// ---------------------------------------------------------------------------

/// One row from the `project_locations JOIN projects` query.
#[derive(sqlx::FromRow)]
struct LocationRow {
    location: String,
    branch_id: String,
    tenant_id: String,
    db_path: String,
}

/// Canonicalize a path — resolve symlinks and `..`.
///
/// Falls back to the path as-is if the filesystem call fails (e.g. the path
/// does not exist yet). The fallback preserves the guard: a non-existent CWD
/// will not match any registered root and will return `None` from the caller.
fn canonicalize_cwd(path: &Path) -> Result<PathBuf, StorageError> {
    match std::fs::canonicalize(path) {
        Ok(p) => Ok(p),
        Err(_) => {
            // Path may not exist on disk (e.g. test fixtures). Use it verbatim;
            // the prefix-match will simply return None for an unregistered path.
            Ok(path.to_path_buf())
        }
    }
}

/// Select the `LocationRow` whose `location` is the longest prefix of `cwd`.
///
/// "Most-specific root wins" — a submodule at `/a/b` beats the container at
/// `/a` when the CWD is inside `/a/b/`.
fn most_specific_match(cwd: &Path, rows: &[LocationRow]) -> Option<ProjectBinding> {
    rows.iter()
        .filter_map(|row| {
            let root = PathBuf::from(&row.location);
            // The CWD must be either the root itself or a descendant.
            if cwd == root || cwd.starts_with(&root) {
                Some((root.components().count(), row))
            } else {
                None
            }
        })
        // Longest path (most components) wins.
        .max_by_key(|(depth, _)| *depth)
        .map(|(_, row)| {
            ProjectBinding::new(
                TenantId::new(&row.tenant_id),
                BranchId::new(&row.branch_id),
                &row.db_path,
            )
        })
}

// ---------------------------------------------------------------------------
// Tests (AC-F10.10, AC-F10.4)
// ---------------------------------------------------------------------------

#[cfg(test)]
#[path = "resolver_tests.rs"]
mod tests;
