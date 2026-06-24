//! AC-F15.2: re-sync `store.db.branches` from authoritative `state.db.project_locations`.
//!
//! File: `wqm-storage-write/src/reconcile/branches_sync.rs`
//! Location: `src/rust/storage-write/src/reconcile/`
//! Context: On daemon startup, reconcile re-syncs `store.db.branches` from the
//!   authoritative `state.db.project_locations` (arch R1 mirror-consistency
//!   invariant). `state.db` wins on disagreement: any branch present in
//!   `project_locations` but absent from `store.db.branches` is inserted;
//!   branches present in `store.db.branches` but absent from `project_locations`
//!   are NOT deleted here (the full branch-delete path runs in case-3; this
//!   function is purely additive).
//!
//!   The `state.db` pool is passed in separately from the `store.db` pool so the
//!   caller controls the seam. Offline tests supply an in-memory `state.db`
//!   fixture; the live daemon supplies its real `state.db` pool.
//!
//! ## Additive-only contract
//!
//! This function ONLY inserts missing rows. It does NOT delete rows that appear
//! in `store.db.branches` but not in `project_locations` -- that is case-3's job
//! (via `branch_delete`). The delete path requires git confirmation; this path
//! does not.
//!
//! Neighbors: `state.db` schema (project_locations table -- arch §5.1),
//!   `store.db` schema (branches table -- `schema/files.rs`).

use sqlx::{Row, SqlitePool};
use wqm_common::error::StorageError;

/// One row from `state.db.project_locations` for a given project (tenant).
#[derive(Debug, Clone)]
pub struct ProjectLocation {
    /// The canonical branch_id (UUID, stable across renames).
    pub branch_id: String,
    /// The human-readable branch name.
    pub branch_name: String,
    /// The filesystem location (repo root path).
    pub location: String,
    /// Whether this branch is currently active.
    pub active: bool,
}

/// Fetch `project_locations` rows for `tenant_id` from `state.db`.
///
/// The `project_locations` table is in `state.db`, which is the authoritative
/// source per arch R1. Rows are fetched for the specific tenant (project) that
/// owns the `store.db` we are syncing.
pub async fn fetch_project_locations(
    state_pool: &SqlitePool,
    tenant_id: &str,
) -> Result<Vec<ProjectLocation>, StorageError> {
    let rows = sqlx::query(
        "SELECT pl.branch_id, pl.branch_name, pl.location, pl.active \
         FROM project_locations pl \
         JOIN projects p ON p.project_id = pl.project_id \
         WHERE p.tenant_id = ? \
         ORDER BY pl.branch_id",
    )
    .bind(tenant_id)
    .fetch_all(state_pool)
    .await
    .map_err(|e| StorageError::Sqlite(format!("branches_sync fetch_locations: {e}")))?;

    Ok(rows
        .into_iter()
        .map(|r| ProjectLocation {
            branch_id: r.get("branch_id"),
            branch_name: r.get("branch_name"),
            location: r.get("location"),
            active: r.get::<i64, _>("active") != 0,
        })
        .collect())
}

/// Re-sync `store.db.branches` from `state.db.project_locations` (additive only).
///
/// For each `ProjectLocation` row in `state.db`:
///   - If the `branch_id` already exists in `store.db.branches` -> skip (ON CONFLICT
///     DO NOTHING keeps existing row; existing data wins to avoid losing `sync_state`).
///   - If absent -> INSERT a new row with `sync_state = 'pending'`.
///
/// Returns the count of rows inserted.
pub async fn sync_branches_from_state(
    store_pool: &SqlitePool,
    locations: &[ProjectLocation],
) -> Result<u64, StorageError> {
    let now = "2026-01-01"; // placeholder; real daemon injects current timestamp
    let mut inserted = 0u64;

    for loc in locations {
        let active_int: i64 = if loc.active { 1 } else { 0 };
        let rows_affected = sqlx::query(
            "INSERT INTO branches(branch_id, branch_name, location, active, sync_state, \
             created_at, updated_at) \
             VALUES (?, ?, ?, ?, 'pending', ?, ?) \
             ON CONFLICT(branch_id) DO NOTHING",
        )
        .bind(&loc.branch_id)
        .bind(&loc.branch_name)
        .bind(&loc.location)
        .bind(active_int)
        .bind(now)
        .bind(now)
        .execute(store_pool)
        .await
        .map_err(|e| StorageError::Sqlite(format!("branches_sync insert: {e}")))?
        .rows_affected();

        inserted += rows_affected;
    }

    Ok(inserted)
}

/// Full branches sync: fetch from state.db and apply to store.db.
///
/// Convenience wrapper that combines `fetch_project_locations` and
/// `sync_branches_from_state`. Returns the count of newly inserted branch rows.
pub async fn run_branches_sync(
    store_pool: &SqlitePool,
    state_pool: &SqlitePool,
    tenant_id: &str,
) -> Result<u64, StorageError> {
    let locations = fetch_project_locations(state_pool, tenant_id).await?;
    sync_branches_from_state(store_pool, &locations).await
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
#[path = "branches_sync_tests.rs"]
mod tests;
