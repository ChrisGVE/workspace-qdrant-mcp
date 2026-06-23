//! Connection-open protocol for per-project `store.db` (arch §5.2, AC-F3.3).
//!
//! File: `wqm-storage-write/src/connection.rs`
//! Location: `src/rust/storage-write/src/` (write-crate)
//! Context: Every connection to a per-project `store.db` — whether from the daemon
//!   write path or from the read-only MCP/CLI path — MUST run the same base protocol:
//!     PRAGMA foreign_keys = ON;   (SQLite default is OFF; RESTRICT/CASCADE are silently
//!                                   inactive without this; not persisted across closes)
//!     PRAGMA journal_mode = WAL;  (already set at DB creation; idempotent to re-assert)
//!
//!   Daemon write connections additionally set:
//!     PRAGMA wal_autocheckpoint = 1000;  (keep WAL from growing unbounded)
//!     PRAGMA busy_timeout = 5000;        (5 s retry window for WAL lock contention)
//!
//!   DR GP-9: the protocol is enforced in ONE place (this module), not duplicated per
//!   call site. The two public entry points are:
//!     - `open_store` — base protocol only (suitable for read-only connections).
//!     - `open_store_write` — base protocol + write-only PRAGMAs (daemon write path).
//!
//! Neighbors: [`crate::schema`] (DDL applied after `open_store_write` on first open),
//!   [`crate::registry`] (write-handle registry that serializes per-branch mutations).

use sqlx::sqlite::{SqliteConnectOptions, SqlitePool, SqlitePoolOptions};
use std::path::Path;
use std::str::FromStr;
use tracing::instrument;
use wqm_common::error::StorageError;

/// Open a `store.db` connection pool with the base connection-open protocol applied
/// to every connection (AC-F3.3):
///   - `PRAGMA foreign_keys = ON`
///   - `PRAGMA journal_mode = WAL`
///
/// Suitable for read-only consumers (MCP server, CLI). Does NOT set
/// `wal_autocheckpoint` or `busy_timeout` (write-path-only PRAGMAs).
///
/// `path` must point to the `store.db` file (which must already exist or be created
/// by the caller before this function is invoked for read-only use).
#[instrument(skip_all, fields(path = %path.as_ref().display()))]
pub async fn open_store(path: impl AsRef<Path>) -> Result<SqlitePool, StorageError> {
    let url = sqlite_url(path.as_ref());
    let opts = base_options(&url)?;
    build_pool(opts, 1).await
}

/// Open a `store.db` connection pool with the full write connection-open protocol
/// applied to every connection (AC-F3.3):
///   - `PRAGMA foreign_keys = ON`
///   - `PRAGMA journal_mode = WAL`
///   - `PRAGMA wal_autocheckpoint = 1000`
///   - `PRAGMA busy_timeout = 5000`
///
/// Intended for the daemon write path only. The single-writer invariant (GP-9) is
/// enforced by the `WriteHandleRegistry` in `crate::registry`.
#[instrument(skip_all, fields(path = %path.as_ref().display()))]
pub async fn open_store_write(path: impl AsRef<Path>) -> Result<SqlitePool, StorageError> {
    let url = sqlite_url(path.as_ref());
    let opts = base_options(&url)?
        .pragma("wal_autocheckpoint", "1000")
        .pragma("busy_timeout", "5000");
    // Single writer: pool size 1 for the write path.
    build_pool(opts, 1).await
}

// ---------------------------------------------------------------------------
// Private helpers
// ---------------------------------------------------------------------------

fn sqlite_url(path: &Path) -> String {
    format!("sqlite://{}", path.display())
}

/// Build `SqliteConnectOptions` with the base protocol (foreign_keys + WAL).
fn base_options(url: &str) -> Result<SqliteConnectOptions, StorageError> {
    let opts = SqliteConnectOptions::from_str(url)
        .map_err(|e| StorageError::Connection(format!("invalid store.db URL: {e}")))?
        .create_if_missing(true)
        .pragma("foreign_keys", "ON")
        .pragma("journal_mode", "WAL");
    Ok(opts)
}

async fn build_pool(
    opts: SqliteConnectOptions,
    max_connections: u32,
) -> Result<SqlitePool, StorageError> {
    SqlitePoolOptions::new()
        .max_connections(max_connections)
        .connect_with(opts)
        .await
        .map_err(|e| StorageError::Connection(format!("failed to open store.db pool: {e}")))
}

// ---------------------------------------------------------------------------
// Tests (AC-F3.3)
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::NamedTempFile;

    /// Helper: apply the store.db DDL on a freshly opened write pool.
    async fn apply_schema(pool: &SqlitePool) -> Result<(), sqlx::Error> {
        for stmt in crate::schema::ddl_statements() {
            sqlx::query(stmt).execute(pool).await?;
        }
        Ok(())
    }

    // AC-F3.3 — base protocol: foreign_keys = ON is active on every connection.
    #[tokio::test]
    async fn base_protocol_foreign_keys_on() {
        let tmp = NamedTempFile::new().expect("tempfile");
        let pool = open_store(tmp.path()).await.expect("open_store");
        apply_schema(&pool).await.expect("schema");

        // Insert a branch row first.
        sqlx::query(
            "INSERT INTO branches(branch_id, branch_name, location, created_at, updated_at) \
             VALUES ('b1','main','/repo','2024-01-01','2024-01-01')",
        )
        .execute(&pool)
        .await
        .expect("branch insert");

        // Insert a file referencing that branch.
        sqlx::query(
            "INSERT INTO files(branch_id, relative_path, is_test, collection, created_at, updated_at) \
             VALUES ('b1','src/lib.rs',0,'projects','2024-01-01','2024-01-01')",
        )
        .execute(&pool)
        .await
        .expect("file insert");

        // Now delete the branch — ON DELETE CASCADE should remove the file too.
        sqlx::query("DELETE FROM branches WHERE branch_id = 'b1'")
            .execute(&pool)
            .await
            .expect("branch delete");

        let count: i64 = sqlx::query_scalar("SELECT COUNT(*) FROM files WHERE branch_id = 'b1'")
            .fetch_one(&pool)
            .await
            .expect("count");

        assert_eq!(count, 0, "CASCADE delete requires foreign_keys = ON");
    }

    // AC-F3.3 — write protocol: wal_autocheckpoint and busy_timeout are set.
    #[tokio::test]
    async fn write_protocol_pragmas_are_set() {
        let tmp = NamedTempFile::new().expect("tempfile");
        let pool = open_store_write(tmp.path())
            .await
            .expect("open_store_write");
        apply_schema(&pool).await.expect("schema");

        let fk: i64 = sqlx::query_scalar("PRAGMA foreign_keys")
            .fetch_one(&pool)
            .await
            .expect("pragma fk");
        assert_eq!(fk, 1, "foreign_keys must be ON");

        let wcp: i64 = sqlx::query_scalar("PRAGMA wal_autocheckpoint")
            .fetch_one(&pool)
            .await
            .expect("pragma wcp");
        assert_eq!(wcp, 1000, "wal_autocheckpoint must be 1000");

        let bt: i64 = sqlx::query_scalar("PRAGMA busy_timeout")
            .fetch_one(&pool)
            .await
            .expect("pragma bt");
        assert_eq!(bt, 5000, "busy_timeout must be 5000");
    }

    // AC-F3.3 — WAL mode is active.
    #[tokio::test]
    async fn journal_mode_is_wal() {
        let tmp = NamedTempFile::new().expect("tempfile");
        let pool = open_store_write(tmp.path())
            .await
            .expect("open_store_write");

        let mode: String = sqlx::query_scalar("PRAGMA journal_mode")
            .fetch_one(&pool)
            .await
            .expect("pragma journal_mode");
        assert_eq!(mode, "wal", "journal_mode must be WAL");
    }

    // AC-F3.3 — RESTRICT FK actually fires on blob_refs DELETE RESTRICT.
    // Proves foreign_keys=ON is active end-to-end (not just PRAGMA value).
    #[tokio::test]
    async fn restrict_fk_fires_on_blob_delete() {
        let tmp = NamedTempFile::new().expect("tempfile");
        let pool = open_store_write(tmp.path())
            .await
            .expect("open_store_write");
        apply_schema(&pool).await.expect("schema");

        // Populate store_meta so the blobs_bi trigger does not fire.
        sqlx::query("INSERT INTO store_meta(tenant_id) VALUES ('tenant-a')")
            .execute(&pool)
            .await
            .expect("store_meta");

        // Insert branch.
        sqlx::query(
            "INSERT INTO branches(branch_id, branch_name, location, created_at, updated_at) \
             VALUES ('b1','main','/repo','2024-01-01','2024-01-01')",
        )
        .execute(&pool)
        .await
        .expect("branch");

        // Insert file.
        sqlx::query(
            "INSERT INTO files(branch_id, relative_path, is_test, collection, created_at, updated_at) \
             VALUES ('b1','a.rs',0,'projects','2024-01-01','2024-01-01')",
        )
        .execute(&pool)
        .await
        .expect("file");

        let file_id: i64 =
            sqlx::query_scalar("SELECT file_id FROM files WHERE relative_path='a.rs'")
                .fetch_one(&pool)
                .await
                .expect("file_id");

        // Insert a blob.
        sqlx::query(
            "INSERT INTO blobs(content_key, chunk_content_hash, point_id, tenant_id, \
             raw_text, dense_vec, sparse_vec, created_at) \
             VALUES ('ck1','hash1','pt1','tenant-a','hello',X'',X'','2024-01-01')",
        )
        .execute(&pool)
        .await
        .expect("blob");

        let blob_id: i64 = sqlx::query_scalar("SELECT blob_id FROM blobs WHERE content_key='ck1'")
            .fetch_one(&pool)
            .await
            .expect("blob_id");

        // Insert blob_ref.
        sqlx::query(
            "INSERT INTO blob_refs(branch_id, file_id, chunk_index, blob_id) VALUES ('b1',?,0,?)",
        )
        .bind(file_id)
        .bind(blob_id)
        .execute(&pool)
        .await
        .expect("blob_ref");

        // Attempt to delete the blob while a blob_ref still points at it.
        // ON DELETE RESTRICT means this must fail when foreign_keys = ON.
        let result = sqlx::query("DELETE FROM blobs WHERE blob_id = ?")
            .bind(blob_id)
            .execute(&pool)
            .await;

        assert!(
            result.is_err(),
            "DELETE RESTRICT must prevent blob deletion while blob_refs exist"
        );
    }
}
