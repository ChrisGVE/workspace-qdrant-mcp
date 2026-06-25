//! Branchless library store helpers (arch §3, §5.4, AC-F16.1).
//!
//! File: `wqm-storage-write/src/library.rs`
//! Location: `src/rust/storage-write/src/` (write-crate)
//! Context: Libraries (external reference docs, textbooks, web pages) are indexed
//!   into their own per-tenant `store.db` files in the `libraries/<tenant_id>/`
//!   bucket. The store.db structure is IDENTICAL to the project store (same 9-table
//!   DDL, same cross-tenant trigger, same `store_meta` row), but libraries have NO
//!   git branches. Content isolation between tenants is guaranteed by the tenant_id
//!   occupying the FIRST slot of `content_key_v4`, so two library tenants (or a
//!   library tenant and a project tenant) with byte-identical content produce
//!   DIFFERENT `content_key` and `point_id` values — a different Qdrant point identity
//!   per tenant (arch §5.4).
//!
//! ## Sentinel branch convention (AC-F16.1)
//!
//! The schema requires every `blob_refs`, `files`, and `concrete` row to carry a
//! valid `branch_id` foreign key into `branches`. A library has no git branches, so
//! every library store.db contains EXACTLY ONE sentinel branch row:
//!
//!   - `branch_id`   = [`LIBRARY_SENTINEL_BRANCH_ID`] (fixed string constant)
//!   - `branch_name` = [`LIBRARY_SENTINEL_BRANCH_NAME`] (fixed string constant)
//!   - `location`    = `""` (no checkout path for a library)
//!   - `sync_state`  = `"current"` (always ready; no git lifecycle applies)
//!
//! Using a fixed constant (not a hash of the tenant) is safe because `branch_id` is
//! PRIMARY KEY within a single store.db file, and each library tenant has its OWN
//! isolated store.db — there is no cross-tenant foreign key. The sentinel constant is
//! declared here once (FP-2 / DR GP-9) so no call site re-invents it.
//!
//! ## Open protocol
//!
//! [`open_library_store`] combines `open_store_write` + DDL application + `store_meta`
//! insert + sentinel branch insert into ONE call. After this call the store is fully
//! initialized and ready for library blob ingestion via the standard ladder
//! (`crate::blob::dedup::ingest_file`) with the sentinel branch_id.
//!
//! Neighbors: [`crate::connection`] (open_store_write), [`crate::schema`] (DDL),
//!   [`wqm_common::hashing`] (content_key_v4 / point_id — cross-tenant isolation proof).

use std::path::Path;

use sqlx::SqlitePool;
use wqm_common::error::StorageError;
use wqm_common::timestamps::now_utc;

use crate::connection::open_store_write;
use crate::schema::ddl_statements;

// ---------------------------------------------------------------------------
// Sentinel branch constants (single source of truth — FP-2 / DR GP-9)
// ---------------------------------------------------------------------------

/// The fixed `branch_id` value used for the single sentinel branch row in every
/// library store.db (AC-F16.1 branchless-tenant convention).
///
/// Fixed rather than derived from the tenant because each library store.db is
/// tenant-isolated already (separate file); the constant is unambiguous and
/// impossible to accidentally reuse as a real git branch_id (which is a 64-char
/// SHA-256 hex string).
pub const LIBRARY_SENTINEL_BRANCH_ID: &str = "_library_sentinel";

/// The `branch_name` value for the sentinel branch row.
pub const LIBRARY_SENTINEL_BRANCH_NAME: &str = "_library";

// ---------------------------------------------------------------------------
// Public surface
// ---------------------------------------------------------------------------

/// Open (or create) a branchless library `store.db`, applying the full schema,
/// inserting the `store_meta` tenant row, and inserting the single sentinel
/// branch row (AC-F16.1).
///
/// After this call the store is ready for blob ingestion: callers pass
/// [`LIBRARY_SENTINEL_BRANCH_ID`] wherever a `branch_id` is required.
///
/// `path` — absolute path to the `store.db` file (created if absent via
/// `open_store_write`'s `create_if_missing = true`).
///
/// `tenant_id` — the library tenant's stable UUID string. Stored in `store_meta`
/// and required to match `blobs.tenant_id` on every insert (enforced by the
/// `blobs_bi` cross-tenant trigger — AC-F3.4).
pub async fn open_library_store(
    path: impl AsRef<Path>,
    tenant_id: &str,
) -> Result<SqlitePool, StorageError> {
    let pool = open_store_write(path).await?;

    // Apply the 9-table DDL (idempotent on a fresh file; errors if called
    // twice on an already-initialized store — caller is responsible for
    // calling this only on store creation, not on every open).
    for stmt in ddl_statements() {
        sqlx::query(stmt)
            .execute(&pool)
            .await
            .map_err(|e| StorageError::Sqlite(format!("library store DDL: {e}")))?;
    }

    // Bind the store to its tenant (store_meta single-row constraint).
    sqlx::query("INSERT INTO store_meta(tenant_id) VALUES (?)")
        .bind(tenant_id)
        .execute(&pool)
        .await
        .map_err(|e| StorageError::Sqlite(format!("library store store_meta insert: {e}")))?;

    // Insert the single sentinel branch row.
    let now = now_utc();
    sqlx::query(
        "INSERT INTO branches(branch_id, branch_name, location, active, sync_state, \
         created_at, updated_at) VALUES (?, ?, '', 1, 'current', ?, ?)",
    )
    .bind(LIBRARY_SENTINEL_BRANCH_ID)
    .bind(LIBRARY_SENTINEL_BRANCH_NAME)
    .bind(&now)
    .bind(&now)
    .execute(&pool)
    .await
    .map_err(|e| StorageError::Sqlite(format!("library store sentinel branch insert: {e}")))?;

    Ok(pool)
}

// ---------------------------------------------------------------------------
// Tests (AC-F16.1) — extracted to sibling file for codesize compliance
// ---------------------------------------------------------------------------

#[cfg(test)]
#[path = "library_tests.rs"]
mod tests;
