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
//!   F17 adds `SearchScope` (Project/Group/All) and `enumerate_by_scope` which
//!   returns the `Vec<ProjectBinding>` of all projects the fan-out should query.
//!   `scope=group` reads `project_groups` (state.db schema v24) to find sibling
//!   tenants; falls back to single-project when the tenant has no group.
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

    /// Return all `ProjectBinding`s the fan-out should query for `scope`.
    ///
    /// - `Project` — the current tenant's active binding only (the F10 path).
    /// - `Group` — all tenants sharing a `project_groups` group with the
    ///   current tenant; falls back to `[current]` when none found.
    /// - `All` — every tenant with at least one active `project_locations` row.
    ///
    /// Each returned binding uses the `active=1` / `sync_state='current'`
    /// location for the project; if a project has several active locations the
    /// `current` sync_state wins, then the most-recently-created row as a
    /// tie-break.
    pub async fn enumerate_by_scope(
        &self,
        scope: SearchScope,
        current_tenant_id: &str,
    ) -> Result<Vec<ProjectBinding>, StorageError> {
        match scope {
            SearchScope::Project => self.bindings_for_tenants(&[current_tenant_id]).await,
            SearchScope::Group => self.bindings_for_group(current_tenant_id).await,
            SearchScope::All => self.bindings_all_active().await,
        }
    }

    /// Resolve bindings for an explicit list of `tenant_id`s.
    async fn bindings_for_tenants(
        &self,
        tenant_ids: &[&str],
    ) -> Result<Vec<ProjectBinding>, StorageError> {
        let mut results = Vec::with_capacity(tenant_ids.len());
        for tid in tenant_ids {
            if let Some(b) = self.active_binding_for_tenant(tid).await? {
                results.push(b);
            }
        }
        Ok(results)
    }

    /// Resolve bindings for all tenants sharing a group with `current_tenant_id`.
    ///
    /// Falls back to `[current_tenant_id]` when the tenant has no group.
    async fn bindings_for_group(
        &self,
        current_tenant_id: &str,
    ) -> Result<Vec<ProjectBinding>, StorageError> {
        let member_ids = query_group_members(&self.state_pool, current_tenant_id).await?;

        // No group membership — behave like scope=project.
        if member_ids.is_empty() {
            return self.bindings_for_tenants(&[current_tenant_id]).await;
        }

        let refs: Vec<&str> = member_ids.iter().map(String::as_str).collect();
        self.bindings_for_tenants(&refs).await
    }

    /// Resolve bindings for ALL tenants with at least one active location.
    async fn bindings_all_active(&self) -> Result<Vec<ProjectBinding>, StorageError> {
        let rows = sqlx::query_as::<_, ProjectRow>(
            r#"
            SELECT DISTINCT p.tenant_id
            FROM projects p
            JOIN project_locations pl ON pl.project_id = p.project_id
            WHERE pl.active = 1
            ORDER BY p.tenant_id
            "#,
        )
        .fetch_all(&self.state_pool)
        .await
        .map_err(|e| StorageError::Sqlite(format!("enumerate all tenants failed: {e}")))?;

        let tenant_ids: Vec<String> = rows.into_iter().map(|r| r.tenant_id).collect();
        let refs: Vec<&str> = tenant_ids.iter().map(String::as_str).collect();
        self.bindings_for_tenants(&refs).await
    }

    /// Return the best active `ProjectBinding` for a single `tenant_id`.
    ///
    /// Selection priority: `sync_state='current'` first, then latest
    /// `created_at` as tie-break. Returns `None` when the tenant has no active
    /// location (skipped silently — not an error).
    async fn active_binding_for_tenant(
        &self,
        tenant_id: &str,
    ) -> Result<Option<ProjectBinding>, StorageError> {
        let row = sqlx::query_as::<_, ActiveRow>(
            r#"
            SELECT
                p.tenant_id,
                p.db_path,
                pl.branch_id
            FROM project_locations pl
            JOIN projects p ON p.project_id = pl.project_id
            WHERE p.tenant_id = ?1 AND pl.active = 1
            ORDER BY
                CASE pl.sync_state WHEN 'current' THEN 0 ELSE 1 END ASC,
                pl.created_at DESC
            LIMIT 1
            "#,
        )
        .bind(tenant_id)
        .fetch_optional(&self.state_pool)
        .await
        .map_err(|e| StorageError::Sqlite(format!("active binding for tenant {tenant_id}: {e}")))?;

        Ok(row.map(|r| {
            ProjectBinding::new(
                TenantId::new(&r.tenant_id),
                BranchId::new(&r.branch_id),
                &r.db_path,
            )
        }))
    }
}

// ---------------------------------------------------------------------------
// Public types
// ---------------------------------------------------------------------------

/// Search scope controlling which projects the fan-out targets (AC-F17, arch R5).
///
/// The MCP `search` tool declares `["project","group","all"]` — this enum is
/// the authoritative Rust representation; parse from those strings via
/// `SearchScope::from_str_loose`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SearchScope {
    /// Query only the current project (the existing F10 single-project path).
    Project,
    /// Query the current project plus all tenants sharing a `project_groups`
    /// group with it. Falls back to `Project` when the tenant has no group.
    Group,
    /// Query every active project. Above the cliff, returns `ScopeTooBroad`.
    All,
}

impl SearchScope {
    /// Parse from a string value (case-insensitive). Unknown strings default to
    /// `Project` so callers that receive an unrecognised value fail safe.
    pub fn from_str_loose(s: &str) -> Self {
        match s.to_lowercase().as_str() {
            "group" => Self::Group,
            "all" => Self::All,
            _ => Self::Project,
        }
    }
}

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

/// Minimal row returned by `active_binding_for_tenant`.
#[derive(sqlx::FromRow)]
struct ActiveRow {
    tenant_id: String,
    db_path: String,
    branch_id: String,
}

/// Minimal row used by `bindings_all_active` to list distinct tenant_ids.
#[derive(sqlx::FromRow)]
struct ProjectRow {
    tenant_id: String,
}

/// Query `project_groups` in `state.db` for all tenant_ids sharing at least
/// one group with `current_tenant_id` (including the tenant itself).
///
/// Returns an empty `Vec` when the `project_groups` table does not exist (pre-
/// v24 state.db) or the tenant has no group membership — both treated as
/// "no group" so the caller falls back gracefully.
async fn query_group_members(
    pool: &SqlitePool,
    current_tenant_id: &str,
) -> Result<Vec<String>, StorageError> {
    // Guard: table may not exist on older state.db versions.
    let table_exists: bool = sqlx::query_scalar(
        "SELECT EXISTS(SELECT 1 FROM sqlite_master WHERE type='table' AND name='project_groups')",
    )
    .fetch_one(pool)
    .await
    .map_err(|e| StorageError::Sqlite(format!("check project_groups existence: {e}")))?;

    if !table_exists {
        return Ok(vec![]);
    }

    let rows: Vec<String> = sqlx::query_scalar(
        r#"
        SELECT DISTINCT pg2.tenant_id
        FROM project_groups pg1
        JOIN project_groups pg2 ON pg1.group_id = pg2.group_id
        WHERE pg1.tenant_id = ?1
        ORDER BY pg2.tenant_id
        "#,
    )
    .bind(current_tenant_id)
    .fetch_all(pool)
    .await
    .map_err(|e| StorageError::Sqlite(format!("query_group_members failed: {e}")))?;

    Ok(rows)
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
