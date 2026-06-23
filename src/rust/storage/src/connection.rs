//! Read-only connection-open protocol for per-project `store.db` (AC-F14.2).
//!
//! File: `wqm-storage/src/connection.rs`
//! Location: `src/rust/storage/` (read-crate)
//! Context: Every non-daemon consumer (MCP server, CLI) that needs a direct
//!   SQLite connection to a per-project `store.db` MUST use this opener. It
//!   enforces the two-layer read-only guarantee from GP-9 (arch §2 B1):
//!     1. `SQLITE_OPEN_READONLY` — opens the file without write permission.
//!     2. `PRAGMA query_only = ON` — a connection property that causes any write
//!        attempt (INSERT / UPDATE / DELETE / CREATE / DROP) to return an error
//!        immediately, regardless of the schema or file-system permissions.
//!
//!   The second layer (`query_only`) is defence-in-depth: even if an attacker or
//!   bug somehow obtained a writable fd, the PRAGMA would still reject writes.
//!   WAL readers proceed without blocking the daemon writer (WAL reader/writer
//!   concurrency is the reason WAL mode is required here; see also
//!   `wqm-storage-write/src/connection.rs` for the write-path opener).
//!
//!   DESIGN NOTE — deliberate PRAGMA duplication across the read/write boundary:
//!   Guard-3 forbids `wqm-storage` from depending on `wqm-storage-write`. The
//!   `open_store` / `open_store_write` functions live in the write crate; this
//!   module provides an independent opener for the read crate. Some PRAGMA overlap
//!   (`foreign_keys`, `journal_mode`, `busy_timeout`) is intentional and necessary
//!   across this strict crate boundary. DR GP-9 (one error type) is honoured —
//!   both crates surface `wqm_common::StorageError`.
//!
//! Neighbors: `wqm-storage-write/src/connection.rs` (structural template, write
//!   crate), `wqm_common::StorageError` (canonical error type, DR GP-9).

use sqlx::sqlite::{SqliteConnectOptions, SqlitePool, SqlitePoolOptions};
use std::path::Path;
use std::str::FromStr;
use tracing::instrument;
use wqm_common::error::StorageError;

/// Open a `store.db` connection pool in strict read-only mode (AC-F14.2).
///
/// The pool enforces the two-layer read-only guarantee (GP-9 B1):
///   - `.read_only(true)` sets `SQLITE_OPEN_READONLY` in the SQLite VFS.
///   - `PRAGMA query_only = ON` rejects any write statement at the SQL level.
///
/// Additional PRAGMAs carried over from the write-crate base protocol:
///   - `PRAGMA foreign_keys = ON`  (referential integrity; not persisted across opens)
///   - `PRAGMA journal_mode = WAL` (idempotent; enables WAL reader concurrency)
///   - `PRAGMA busy_timeout = 5000` (5 s retry window for WAL lock contention)
///
/// `create_if_missing` is NOT set — a missing DB is a caller error, not an
/// invitation to create a writable file.
///
/// Pool `max_connections` defaults to 4, suitable for concurrent readers.
/// WAL mode allows multiple simultaneous readers without blocking.
///
/// # Errors
///
/// Returns `StorageError::Connection` if the path is invalid or the pool cannot
/// be established (e.g. file does not exist).
#[instrument(skip_all, fields(path = %path.as_ref().display()))]
pub async fn open_store_readonly(path: impl AsRef<Path>) -> Result<SqlitePool, StorageError> {
    let url = sqlite_url(path.as_ref());
    let opts = readonly_options(&url)?;
    build_readonly_pool(opts, 4).await
}

// ---------------------------------------------------------------------------
// Private helpers
// ---------------------------------------------------------------------------

fn sqlite_url(path: &Path) -> String {
    format!("sqlite://{}", path.display())
}

/// Build `SqliteConnectOptions` with the read-only protocol.
fn readonly_options(url: &str) -> Result<SqliteConnectOptions, StorageError> {
    let opts = SqliteConnectOptions::from_str(url)
        .map_err(|e| StorageError::Connection(format!("invalid store.db URL: {e}")))?
        // Layer 1: SQLITE_OPEN_READONLY — no write permission at the VFS level.
        .read_only(true)
        // Layer 2: query_only = ON — rejects writes at the SQL statement level.
        .pragma("query_only", "ON")
        // Base connection protocol (mirrors write-crate base_options).
        .pragma("foreign_keys", "ON")
        .pragma("journal_mode", "WAL")
        .pragma("busy_timeout", "5000");
    Ok(opts)
}

async fn build_readonly_pool(
    opts: SqliteConnectOptions,
    max_connections: u32,
) -> Result<SqlitePool, StorageError> {
    SqlitePoolOptions::new()
        .max_connections(max_connections)
        .connect_with(opts)
        .await
        .map_err(|e| {
            StorageError::Connection(format!("failed to open read-only store.db pool: {e}"))
        })
}

// ---------------------------------------------------------------------------
// Tests (AC-F14.2)
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use sqlx::sqlite::{SqliteConnectOptions, SqlitePool, SqlitePoolOptions};
    use std::str::FromStr;
    use tempfile::NamedTempFile;

    /// Create a minimal writable DB (without depending on wqm-storage-write).
    /// We use raw sqlx with write options to bootstrap the test DB.
    async fn create_writable_pool(path: &std::path::Path) -> SqlitePool {
        let url = format!("sqlite://{}", path.display());
        let opts = SqliteConnectOptions::from_str(&url)
            .expect("url")
            .create_if_missing(true)
            .pragma("foreign_keys", "ON")
            .pragma("journal_mode", "WAL");
        SqlitePoolOptions::new()
            .max_connections(1)
            .connect_with(opts)
            .await
            .expect("writable pool")
    }

    // AC-F14.2 — write attempt on a read-only connection returns an error.
    #[tokio::test]
    async fn t_f14_readonly_rejects_writes() {
        let tmp = NamedTempFile::new().expect("tempfile");

        // Bootstrap: create DB and a table via a writable pool, then close it.
        {
            let write_pool = create_writable_pool(tmp.path()).await;
            sqlx::query("CREATE TABLE test_tbl (id INTEGER PRIMARY KEY, val TEXT)")
                .execute(&write_pool)
                .await
                .expect("create table");
            write_pool.close().await;
        }

        // Open the same DB read-only.
        let ro_pool = open_store_readonly(tmp.path())
            .await
            .expect("open_store_readonly");

        // INSERT must be rejected (query_only = ON).
        let insert_result = sqlx::query("INSERT INTO test_tbl(val) VALUES ('x')")
            .execute(&ro_pool)
            .await;
        assert!(
            insert_result.is_err(),
            "INSERT on read-only connection must fail (query_only=ON)"
        );

        // CREATE TABLE must also be rejected.
        let create_result = sqlx::query("CREATE TABLE another_tbl (id INTEGER)")
            .execute(&ro_pool)
            .await;
        assert!(
            create_result.is_err(),
            "CREATE TABLE on read-only connection must fail (query_only=ON)"
        );
    }

    // AC-F14.2 — SELECT proceeds on a read-only connection.
    #[tokio::test]
    async fn t_f14_readonly_allows_select() {
        let tmp = NamedTempFile::new().expect("tempfile");

        // Bootstrap: create DB with a row.
        {
            let write_pool = create_writable_pool(tmp.path()).await;
            sqlx::query("CREATE TABLE kv (k TEXT, v TEXT)")
                .execute(&write_pool)
                .await
                .expect("create");
            sqlx::query("INSERT INTO kv(k, v) VALUES ('hello', 'world')")
                .execute(&write_pool)
                .await
                .expect("insert");
            write_pool.close().await;
        }

        let ro_pool = open_store_readonly(tmp.path())
            .await
            .expect("open_store_readonly");

        let val: Option<String> = sqlx::query_scalar("SELECT v FROM kv WHERE k = 'hello'")
            .fetch_optional(&ro_pool)
            .await
            .expect("select");
        assert_eq!(
            val.as_deref(),
            Some("world"),
            "SELECT must work on read-only pool"
        );
    }

    // AC-F14.2 — UPDATE is also rejected (belt-and-suspenders; covers non-INSERT writes).
    #[tokio::test]
    async fn t_f14_readonly_rejects_update() {
        let tmp = NamedTempFile::new().expect("tempfile");

        {
            let write_pool = create_writable_pool(tmp.path()).await;
            sqlx::query("CREATE TABLE kv (k TEXT, v TEXT)")
                .execute(&write_pool)
                .await
                .expect("create");
            sqlx::query("INSERT INTO kv(k, v) VALUES ('a', 'b')")
                .execute(&write_pool)
                .await
                .expect("insert");
            write_pool.close().await;
        }

        let ro_pool = open_store_readonly(tmp.path())
            .await
            .expect("open_store_readonly");

        let update_result = sqlx::query("UPDATE kv SET v = 'changed' WHERE k = 'a'")
            .execute(&ro_pool)
            .await;
        assert!(
            update_result.is_err(),
            "UPDATE on read-only connection must fail (query_only=ON)"
        );
    }
}
