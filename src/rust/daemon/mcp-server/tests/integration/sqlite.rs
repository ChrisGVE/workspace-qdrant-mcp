// Integration tests: SQLite StateManager — busy_timeout under concurrent writes.
//
// it_sqlite_busy_timeout_does_not_panic — open a read-only StateManager against a
//     WAL-mode DB while another thread holds a write lock; verify the
//     busy_timeout (5 s) absorbs the contention and the manager opens without panic.
// it_sqlite_read_only_rejects_write     — confirms the read-only flag is enforced.
// it_sqlite_degraded_on_missing_file    — StateManager::open_at on a nonexistent
//     path is degraded (no panic, no crash).
// it_sqlite_list_rules_empty_no_panic   — list_rules on a connected but empty
//     rules_mirror table returns [] without panic.

use mcp_server::sqlite::{DegradedReason, QueryStatus, StateManager};
use rusqlite::Connection;
use tempfile::TempDir;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Create a WAL-mode SQLite database in `dir` with the minimal schema.
fn make_wal_db(dir: &TempDir) -> std::path::PathBuf {
    let path = dir.path().join("state.db");
    let setup = Connection::open(&path).unwrap();
    setup
        .execute_batch(
            "PRAGMA journal_mode=WAL; PRAGMA synchronous=NORMAL;
             CREATE TABLE IF NOT EXISTS unified_queue (
                 queue_id   TEXT PRIMARY KEY,
                 item_type  TEXT NOT NULL,
                 op         TEXT NOT NULL,
                 tenant_id  TEXT NOT NULL,
                 collection TEXT NOT NULL,
                 status     TEXT NOT NULL DEFAULT 'pending',
                 lease_until TEXT,
                 created_at TEXT NOT NULL DEFAULT (datetime('now')),
                 updated_at TEXT NOT NULL DEFAULT (datetime('now'))
             );
             CREATE TABLE IF NOT EXISTS rules_mirror (
                 rule_id    TEXT PRIMARY KEY,
                 rule_text  TEXT NOT NULL,
                 scope      TEXT,
                 tenant_id  TEXT,
                 created_at TEXT NOT NULL,
                 updated_at TEXT NOT NULL
             );",
        )
        .unwrap();
    drop(setup);
    path
}

// ---------------------------------------------------------------------------
// it_sqlite_degraded_on_missing_file
// ---------------------------------------------------------------------------

#[test]
fn it_sqlite_degraded_on_missing_file() {
    let mgr = StateManager::open_at("/nonexistent/path/to/state.db");
    assert!(
        !mgr.is_connected(),
        "StateManager must be degraded when file does not exist"
    );
    assert!(mgr.connection().is_none());
    let result = mgr.init_result();
    assert_eq!(result.status, QueryStatus::Degraded);
    assert_eq!(result.reason, Some(DegradedReason::DatabaseNotFound));
}

// ---------------------------------------------------------------------------
// it_sqlite_read_only_rejects_write
// ---------------------------------------------------------------------------

#[test]
fn it_sqlite_read_only_rejects_write() {
    let dir = TempDir::new().unwrap();
    let path = make_wal_db(&dir);

    let mgr = StateManager::open_at(&path);
    assert!(mgr.is_connected(), "StateManager must open successfully");

    let conn = mgr.connection().unwrap();
    let result = conn.execute(
        "INSERT INTO unified_queue \
         VALUES ('test_id','text','add','tenant','coll','pending',NULL,datetime('now'),datetime('now'))",
        [],
    );
    assert!(
        result.is_err(),
        "write on read-only connection must return Err"
    );
}

// ---------------------------------------------------------------------------
// it_sqlite_list_rules_empty_no_panic
// ---------------------------------------------------------------------------

#[test]
fn it_sqlite_list_rules_empty_no_panic() {
    let dir = TempDir::new().unwrap();
    let path = make_wal_db(&dir);

    let mgr = StateManager::open_at(&path);
    assert!(mgr.is_connected());

    // Empty rules_mirror table — must return [] without panic.
    let rows = mcp_server::sqlite::rules_mirror::list_rules(mgr.connection(), None, None, 50);
    assert!(rows.is_empty(), "empty rules_mirror must return empty Vec");
}

// ---------------------------------------------------------------------------
// it_sqlite_busy_timeout_absorbs_write_contention
// ---------------------------------------------------------------------------

/// Opens a WAL-mode DB, spawns a writer thread that holds a long BEGIN
/// EXCLUSIVE transaction (500 ms), and simultaneously opens a StateManager
/// (read-only, busy_timeout=5s).  The manager must open without panicking and
/// its connection must become available when the writer releases the lock.
///
/// This validates that `conn.busy_timeout(5000 ms)` (set inside StateManager)
/// allows readers to wait rather than immediately returning SQLITE_BUSY.
#[test]
fn it_sqlite_busy_timeout_absorbs_write_contention() {
    let dir = TempDir::new().unwrap();
    let path = make_wal_db(&dir);

    // Writer thread: opens a write-mode connection, starts an EXCLUSIVE
    // transaction, holds it for 300 ms, then commits and closes.
    let writer_path = path.clone();
    let writer = std::thread::spawn(move || {
        let wconn = Connection::open(&writer_path).unwrap();
        wconn.execute_batch("BEGIN EXCLUSIVE;").unwrap();
        std::thread::sleep(std::time::Duration::from_millis(300));
        wconn.execute_batch("COMMIT;").unwrap();
    });

    // Give the writer a moment to acquire the lock before the reader tries.
    std::thread::sleep(std::time::Duration::from_millis(30));

    // Open the StateManager while the writer holds EXCLUSIVE.
    // busy_timeout=5000 ms means it will wait up to 5 s; 300 ms hold << 5 s.
    let mgr = StateManager::open_at(&path);

    // Join writer before asserting (ensures it released the lock).
    writer.join().expect("writer thread must not panic");

    // The manager must be connected — the busy_timeout covered the contention.
    assert!(
        mgr.is_connected(),
        "StateManager must be connected after write contention resolved"
    );
    assert!(mgr.connection().is_some());
}

// ---------------------------------------------------------------------------
// it_sqlite_open_at_existing_wal_db_succeeds
// ---------------------------------------------------------------------------

#[test]
fn it_sqlite_open_at_existing_wal_db_succeeds() {
    let dir = TempDir::new().unwrap();
    let path = make_wal_db(&dir);

    let mgr = StateManager::open_at(&path);
    assert!(mgr.is_connected());
    assert!(mgr.connection().is_some());

    let result = mgr.init_result();
    assert_eq!(result.status, QueryStatus::Ok);
    assert!(result.data);
}
