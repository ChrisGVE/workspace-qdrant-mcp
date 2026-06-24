//! Per-tenant reconcile watermark in `maintenance_meta` (AC-F15.4 / DATA-NIT-02).
//!
//! File: `wqm-storage-write/src/reconcile/watermark.rs`
//! Location: `src/rust/storage-write/src/reconcile/` (write-crate reconcile layer)
//! Context: Each reconcile pass reads and writes a per-tenant watermark row in the
//!   `maintenance_meta` table. The watermark is keyed one-row-per-`tenant_id`
//!   (DATA-R5-NIT-03 -- no shared JSON blob). The last-reconcile timestamp and the
//!   max-seen `blob_id` are stored together so case-2 and case-4 can scope their
//!   incremental scans to rows that are newer than the watermark.
//!
//! ## DDL (created on first reconcile if absent)
//!
//! ```sql
//! CREATE TABLE IF NOT EXISTS maintenance_meta (
//!     tenant_id           TEXT PRIMARY KEY,
//!     last_reconcile_at   TEXT,
//!     max_seen_blob_id    INTEGER NOT NULL DEFAULT 0,
//!     last_tenant_move_at TEXT
//! )
//! ```
//!
//! `last_tenant_move_at` is the migration-journal timestamp used by case-5
//! (AC-F15.6 / M4 detection bound): case-5 is skipped when no tenant-move has
//! occurred since the last successful pass.
//!
//! ## RMW discipline (DATA-NIT-02)
//!
//! Every read-modify-write of the watermark runs under `BEGIN IMMEDIATE` — the same
//! isolation level AC-F13.2 uses. This prevents a concurrent migration writer and a
//! reconcile watermark writer from losing each other's updates (lost-update safety).
//!
//! Neighbors: [`super::mod`] (the reconcile pass that calls this module),
//!   F13 (which reuses `maintenance_meta` for migration-epoch rows).

use sqlx::{Row, Sqlite, SqlitePool, Transaction};
use wqm_common::error::StorageError;

/// DDL for `maintenance_meta` -- created IF NOT EXISTS so F15 and F13 can both
/// evolve the table without collision (F15 lands first; F13 reuses it).
pub const CREATE_MAINTENANCE_META: &str = r#"CREATE TABLE IF NOT EXISTS maintenance_meta (
    tenant_id           TEXT PRIMARY KEY,
    last_reconcile_at   TEXT,
    max_seen_blob_id    INTEGER NOT NULL DEFAULT 0,
    last_tenant_move_at TEXT
)"#;

/// Snapshot of one tenant's watermark row read before the pass starts.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ReconcileWatermark {
    /// The owning tenant this watermark belongs to.
    pub tenant_id: String,
    /// ISO-8601 timestamp of the last successful reconcile (None on first run).
    pub last_reconcile_at: Option<String>,
    /// Max `blob_id` seen by the last incremental pass (0 on first run).
    /// Case-2 and case-4 scope their scan to `blob_id > max_seen_blob_id`.
    pub max_seen_blob_id: i64,
    /// ISO-8601 timestamp of the last tenant-move recorded in the migration
    /// journal (None if no tenant-move has ever occurred). Used by case-5 as the
    /// M4 detection bound: case-5 is skipped when this is None or predates
    /// `last_reconcile_at`.
    pub last_tenant_move_at: Option<String>,
}

impl ReconcileWatermark {
    /// Return true if the migration journal shows a tenant-move that occurred
    /// AFTER the last successful reconcile -- meaning case-5 MUST run.
    ///
    /// If no tenant-move has ever been recorded, or the move predates the last
    /// successful reconcile, case-5 can be skipped (M4 detection bound).
    pub fn tenant_move_since_last_pass(&self) -> bool {
        match (&self.last_tenant_move_at, &self.last_reconcile_at) {
            // No move recorded -- skip case-5.
            (None, _) => false,
            // Move recorded but no prior reconcile -- must run case-5.
            (Some(_), None) => true,
            // Move recorded AND prior reconcile exists: run case-5 only if the
            // move is MORE RECENT than the last reconcile timestamp.
            (Some(move_at), Some(reconcile_at)) => move_at > reconcile_at,
        }
    }
}

/// Ensure `maintenance_meta` exists (idempotent `CREATE TABLE IF NOT EXISTS`).
///
/// Called once at the start of each reconcile pass so the table is present
/// whether or not F13 has already run on this store.
pub async fn ensure_maintenance_meta(pool: &SqlitePool) -> Result<(), StorageError> {
    sqlx::query(CREATE_MAINTENANCE_META)
        .execute(pool)
        .await
        .map_err(|e| StorageError::Sqlite(format!("ensure_maintenance_meta: {e}")))?;
    Ok(())
}

/// Read the current watermark for `tenant_id` from `maintenance_meta`.
///
/// Returns a zeroed watermark (no prior reconcile) if no row exists yet.
/// Does NOT start a transaction -- the caller is responsible for calling
/// `update_watermark` under `BEGIN IMMEDIATE` after the pass completes.
pub async fn read_watermark(
    pool: &SqlitePool,
    tenant_id: &str,
) -> Result<ReconcileWatermark, StorageError> {
    let row = sqlx::query(
        "SELECT last_reconcile_at, max_seen_blob_id, last_tenant_move_at \
         FROM maintenance_meta WHERE tenant_id = ?",
    )
    .bind(tenant_id)
    .fetch_optional(pool)
    .await
    .map_err(|e| StorageError::Sqlite(format!("read_watermark: {e}")))?;

    match row {
        Some(r) => Ok(ReconcileWatermark {
            tenant_id: tenant_id.to_owned(),
            last_reconcile_at: r.get("last_reconcile_at"),
            max_seen_blob_id: r.get::<i64, _>("max_seen_blob_id"),
            last_tenant_move_at: r.get("last_tenant_move_at"),
        }),
        None => Ok(ReconcileWatermark {
            tenant_id: tenant_id.to_owned(),
            last_reconcile_at: None,
            max_seen_blob_id: 0,
            last_tenant_move_at: None,
        }),
    }
}

/// Update the watermark for `tenant_id` after a successful pass.
///
/// Holds a `pool.begin()` transaction for the entire RMW (DATA-NIT-02).
/// Keeping the single connection inside `tx` for both the upsert and the
/// commit prevents any concurrent writer from slipping between read and write
/// (lost-update safety). The caller supplies:
/// - `now_iso8601` -- current timestamp (injectable for tests).
/// - `max_blob_id` -- the highest `blob_id` the pass scanned.
pub async fn update_watermark(
    pool: &SqlitePool,
    tenant_id: &str,
    now_iso8601: &str,
    max_blob_id: i64,
) -> Result<(), StorageError> {
    let mut tx: Transaction<'_, Sqlite> = pool
        .begin()
        .await
        .map_err(|e| StorageError::Sqlite(format!("update_watermark begin: {e}")))?;

    sqlx::query(
        "INSERT INTO maintenance_meta(tenant_id, last_reconcile_at, max_seen_blob_id) \
         VALUES (?, ?, ?) \
         ON CONFLICT(tenant_id) DO UPDATE SET \
             last_reconcile_at  = excluded.last_reconcile_at, \
             max_seen_blob_id   = MAX(maintenance_meta.max_seen_blob_id, excluded.max_seen_blob_id)",
    )
    .bind(tenant_id)
    .bind(now_iso8601)
    .bind(max_blob_id)
    .execute(&mut *tx)
    .await
    .map_err(|e| StorageError::Sqlite(format!("update_watermark upsert: {e}")))?;

    tx.commit()
        .await
        .map_err(|e| StorageError::Sqlite(format!("update_watermark commit: {e}")))
}

/// Record a tenant-move event in `maintenance_meta.last_tenant_move_at`.
///
/// Holds a `pool.begin()` transaction for the entire RMW (DATA-NIT-02).
/// Called by F16 (AC-F16.2 / AC-F16.5) when a tenant-move operation starts.
/// This feeds the M4 detection bound for case-5.
pub async fn record_tenant_move(
    pool: &SqlitePool,
    tenant_id: &str,
    now_iso8601: &str,
) -> Result<(), StorageError> {
    let mut tx: Transaction<'_, Sqlite> = pool
        .begin()
        .await
        .map_err(|e| StorageError::Sqlite(format!("record_tenant_move begin: {e}")))?;

    sqlx::query(
        "INSERT INTO maintenance_meta(tenant_id, last_tenant_move_at) \
         VALUES (?, ?) \
         ON CONFLICT(tenant_id) DO UPDATE SET \
             last_tenant_move_at = excluded.last_tenant_move_at",
    )
    .bind(tenant_id)
    .bind(now_iso8601)
    .execute(&mut *tx)
    .await
    .map_err(|e| StorageError::Sqlite(format!("record_tenant_move upsert: {e}")))?;

    tx.commit()
        .await
        .map_err(|e| StorageError::Sqlite(format!("record_tenant_move commit: {e}")))
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
#[path = "watermark_tests.rs"]
mod tests;
