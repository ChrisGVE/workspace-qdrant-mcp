//! Queue statistics queries — direct SQLite reads from `unified_queue`.
//!
//! SQL is verbatim from `queue-operations.ts:174-209`.  Only read operations
//! live here; enqueue (write) goes via daemon gRPC and is not implemented in
//! this module.

use rusqlite::Connection;
use std::collections::HashMap;

// ---------------------------------------------------------------------------
// Public return types
// ---------------------------------------------------------------------------

/// Queue statistics grouped by status, item type, and collection.
///
/// Mirrors the `QueueStats` type in `types/state.ts`.
#[derive(Debug, Clone, PartialEq)]
pub struct QueueStats {
    pub total_pending: i64,
    pub total_in_progress: i64,
    pub total_done: i64,
    pub total_failed: i64,
    /// Pending count per `item_type`.
    pub by_item_type: HashMap<String, i64>,
    /// Pending count per `collection`.
    pub by_collection: Vec<CollectionCount>,
    /// Items in `in_progress` with `lease_until < datetime('now')`.
    pub stale_items_count: i64,
}

#[derive(Debug, Clone, PartialEq)]
pub struct CollectionCount {
    pub collection: String,
    pub count: i64,
}

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

/// "no such table" → degrade, anything else → propagate.
fn is_no_such_table(e: &rusqlite::Error) -> bool {
    e.to_string().contains("no such table")
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/// Return status counts from `unified_queue` grouped by `status`.
///
/// SQL: `SELECT status, COUNT(*) as count FROM unified_queue GROUP BY status`
pub fn get_status_counts(
    conn: Option<&Connection>,
) -> Result<HashMap<String, i64>, rusqlite::Error> {
    let Some(conn) = conn else {
        return Ok(HashMap::new());
    };
    let mut stmt =
        conn.prepare("SELECT status, COUNT(*) as count FROM unified_queue GROUP BY status")?;
    let rows = stmt.query_map([], |row| {
        let status: String = row.get(0)?;
        let count: i64 = row.get(1)?;
        Ok((status, count))
    })?;
    let mut map = HashMap::new();
    for row in rows {
        let (status, count) = row?;
        map.insert(status, count);
    }
    Ok(map)
}

/// Return pending-item counts grouped by `item_type`.
///
/// SQL: `SELECT item_type, COUNT(*) as count FROM unified_queue
///       WHERE status = 'pending' GROUP BY item_type`
pub fn get_pending_by_type(
    conn: Option<&Connection>,
) -> Result<HashMap<String, i64>, rusqlite::Error> {
    let Some(conn) = conn else {
        return Ok(HashMap::new());
    };
    let mut stmt = conn.prepare(
        "SELECT item_type, COUNT(*) as count FROM unified_queue \
         WHERE status = 'pending' GROUP BY item_type",
    )?;
    let rows = stmt.query_map([], |row| {
        let item_type: String = row.get(0)?;
        let count: i64 = row.get(1)?;
        Ok((item_type, count))
    })?;
    let mut map = HashMap::new();
    for row in rows {
        let (item_type, count) = row?;
        map.insert(item_type, count);
    }
    Ok(map)
}

/// Return pending-item counts grouped by `collection`.
///
/// SQL: `SELECT collection, COUNT(*) as count FROM unified_queue
///       WHERE status = 'pending' GROUP BY collection`
pub fn get_pending_by_collection(
    conn: Option<&Connection>,
) -> Result<Vec<CollectionCount>, rusqlite::Error> {
    let Some(conn) = conn else {
        return Ok(Vec::new());
    };
    let mut stmt = conn.prepare(
        "SELECT collection, COUNT(*) as count FROM unified_queue \
         WHERE status = 'pending' GROUP BY collection",
    )?;
    let rows = stmt.query_map([], |row| {
        let collection: String = row.get(0)?;
        let count: i64 = row.get(1)?;
        Ok(CollectionCount { collection, count })
    })?;
    rows.collect::<Result<Vec<_>, _>>()
}

/// Count items that are `in_progress` but whose `lease_until` has expired.
///
/// SQL: `SELECT COUNT(*) as count FROM unified_queue
///       WHERE status = 'in_progress' AND lease_until < datetime('now')`
pub fn get_stale_lease_count(conn: Option<&Connection>) -> Result<i64, rusqlite::Error> {
    let Some(conn) = conn else {
        return Ok(0);
    };
    conn.query_row(
        "SELECT COUNT(*) as count FROM unified_queue \
         WHERE status = 'in_progress' AND lease_until < datetime('now')",
        [],
        |row| row.get(0),
    )
}

/// Aggregate all four queries into a single `QueueStats` value.
///
/// Returns a degraded (all-zero) result wrapped in `Ok` when the database is
/// absent (`conn = None`) or when the `unified_queue` table does not exist yet
/// (daemon not initialised).  Any other SQLite error is propagated.
pub fn get_queue_stats(conn: Option<&Connection>) -> Result<QueueStats, rusqlite::Error> {
    let status_counts = match get_status_counts(conn) {
        Ok(m) => m,
        Err(e) if is_no_such_table(&e) => HashMap::new(),
        Err(e) => return Err(e),
    };
    let by_item_type = match get_pending_by_type(conn) {
        Ok(m) => m,
        Err(e) if is_no_such_table(&e) => HashMap::new(),
        Err(e) => return Err(e),
    };
    let by_collection = match get_pending_by_collection(conn) {
        Ok(v) => v,
        Err(e) if is_no_such_table(&e) => Vec::new(),
        Err(e) => return Err(e),
    };
    let stale_items_count = match get_stale_lease_count(conn) {
        Ok(n) => n,
        Err(e) if is_no_such_table(&e) => 0,
        Err(e) => return Err(e),
    };

    Ok(QueueStats {
        total_pending: *status_counts.get("pending").unwrap_or(&0),
        total_in_progress: *status_counts.get("in_progress").unwrap_or(&0),
        total_done: *status_counts.get("done").unwrap_or(&0),
        total_failed: *status_counts.get("failed").unwrap_or(&0),
        by_item_type,
        by_collection,
        stale_items_count,
    })
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use rusqlite::{Connection, OpenFlags};
    use tempfile::TempDir;

    fn make_db(dir: &TempDir) -> Connection {
        let path = dir.path().join("state.db");
        let setup = Connection::open(&path).unwrap();
        setup
            .execute_batch(
                "PRAGMA journal_mode=WAL; PRAGMA synchronous=NORMAL;
                 CREATE TABLE unified_queue (
                     queue_id    TEXT PRIMARY KEY,
                     item_type   TEXT NOT NULL,
                     op          TEXT NOT NULL,
                     tenant_id   TEXT NOT NULL,
                     collection  TEXT NOT NULL,
                     status      TEXT NOT NULL DEFAULT 'pending',
                     lease_until TEXT,
                     created_at  TEXT NOT NULL DEFAULT (datetime('now')),
                     updated_at  TEXT NOT NULL DEFAULT (datetime('now'))
                 );",
            )
            .unwrap();
        drop(setup);
        Connection::open_with_flags(&path, OpenFlags::SQLITE_OPEN_READ_ONLY).unwrap()
    }

    fn insert_item(
        setup_conn: &Connection,
        id: &str,
        item_type: &str,
        collection: &str,
        status: &str,
        lease_until: Option<&str>,
    ) {
        setup_conn
            .execute(
                "INSERT INTO unified_queue
                 (queue_id, item_type, op, tenant_id, collection, status, lease_until)
                 VALUES (?1, ?2, 'add', 'tenant1', ?3, ?4, ?5)",
                rusqlite::params![id, item_type, collection, status, lease_until],
            )
            .unwrap();
    }

    #[test]
    fn none_connection_returns_zeros() {
        let stats = get_queue_stats(None).unwrap();
        assert_eq!(stats.total_pending, 0);
        assert_eq!(stats.total_in_progress, 0);
        assert_eq!(stats.total_done, 0);
        assert_eq!(stats.total_failed, 0);
        assert!(stats.by_item_type.is_empty());
        assert!(stats.by_collection.is_empty());
        assert_eq!(stats.stale_items_count, 0);
    }

    #[test]
    fn empty_table_returns_zeros() {
        let dir = TempDir::new().unwrap();
        let conn = make_db(&dir);
        let stats = get_queue_stats(Some(&conn)).unwrap();
        assert_eq!(stats.total_pending, 0);
        assert_eq!(stats.stale_items_count, 0);
    }

    #[test]
    fn status_counts_aggregated_correctly() {
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("state.db");
        let setup = Connection::open(&path).unwrap();
        setup
            .execute_batch(
                "PRAGMA journal_mode=WAL;
                 CREATE TABLE unified_queue (
                     queue_id TEXT PRIMARY KEY, item_type TEXT NOT NULL, op TEXT NOT NULL,
                     tenant_id TEXT NOT NULL, collection TEXT NOT NULL,
                     status TEXT NOT NULL DEFAULT 'pending', lease_until TEXT,
                     created_at TEXT NOT NULL DEFAULT (datetime('now')),
                     updated_at TEXT NOT NULL DEFAULT (datetime('now'))
                 );",
            )
            .unwrap();
        insert_item(&setup, "q1", "document", "projects", "pending", None);
        insert_item(&setup, "q2", "document", "projects", "pending", None);
        insert_item(&setup, "q3", "rule", "rules", "in_progress", None);
        insert_item(&setup, "q4", "document", "libraries", "done", None);
        insert_item(&setup, "q5", "document", "projects", "failed", None);
        drop(setup);

        let conn = Connection::open_with_flags(&path, OpenFlags::SQLITE_OPEN_READ_ONLY).unwrap();
        let stats = get_queue_stats(Some(&conn)).unwrap();

        assert_eq!(stats.total_pending, 2);
        assert_eq!(stats.total_in_progress, 1);
        assert_eq!(stats.total_done, 1);
        assert_eq!(stats.total_failed, 1);
        assert_eq!(stats.by_item_type.get("document"), Some(&2));
        assert_eq!(stats.by_item_type.get("rule"), None); // not pending
    }

    #[test]
    fn stale_lease_count() {
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("state.db");
        let setup = Connection::open(&path).unwrap();
        setup
            .execute_batch(
                "PRAGMA journal_mode=WAL;
                 CREATE TABLE unified_queue (
                     queue_id TEXT PRIMARY KEY, item_type TEXT NOT NULL, op TEXT NOT NULL,
                     tenant_id TEXT NOT NULL, collection TEXT NOT NULL,
                     status TEXT NOT NULL DEFAULT 'pending', lease_until TEXT,
                     created_at TEXT NOT NULL DEFAULT (datetime('now')),
                     updated_at TEXT NOT NULL DEFAULT (datetime('now'))
                 );",
            )
            .unwrap();
        // Stale: in_progress with expired lease
        insert_item(
            &setup,
            "s1",
            "document",
            "projects",
            "in_progress",
            Some("2000-01-01T00:00:00Z"),
        );
        // Active: in_progress with future lease
        insert_item(
            &setup,
            "s2",
            "document",
            "projects",
            "in_progress",
            Some("2099-01-01T00:00:00Z"),
        );
        // Not in_progress
        insert_item(
            &setup,
            "s3",
            "document",
            "projects",
            "pending",
            Some("2000-01-01T00:00:00Z"),
        );
        drop(setup);

        let conn = Connection::open_with_flags(&path, OpenFlags::SQLITE_OPEN_READ_ONLY).unwrap();
        let count = get_stale_lease_count(Some(&conn)).unwrap();
        assert_eq!(count, 1);
    }

    #[test]
    fn by_collection_groups_correctly() {
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("state.db");
        let setup = Connection::open(&path).unwrap();
        setup
            .execute_batch(
                "PRAGMA journal_mode=WAL;
                 CREATE TABLE unified_queue (
                     queue_id TEXT PRIMARY KEY, item_type TEXT NOT NULL, op TEXT NOT NULL,
                     tenant_id TEXT NOT NULL, collection TEXT NOT NULL,
                     status TEXT NOT NULL DEFAULT 'pending', lease_until TEXT,
                     created_at TEXT NOT NULL DEFAULT (datetime('now')),
                     updated_at TEXT NOT NULL DEFAULT (datetime('now'))
                 );",
            )
            .unwrap();
        insert_item(&setup, "c1", "document", "projects", "pending", None);
        insert_item(&setup, "c2", "document", "projects", "pending", None);
        insert_item(&setup, "c3", "rule", "rules", "pending", None);
        insert_item(&setup, "c4", "document", "projects", "done", None); // not pending
        drop(setup);

        let conn = Connection::open_with_flags(&path, OpenFlags::SQLITE_OPEN_READ_ONLY).unwrap();
        let counts = get_pending_by_collection(Some(&conn)).unwrap();
        let proj = counts.iter().find(|c| c.collection == "projects").unwrap();
        let rules = counts.iter().find(|c| c.collection == "rules").unwrap();
        assert_eq!(proj.count, 2);
        assert_eq!(rules.count, 1);
    }

    #[test]
    fn missing_table_degrades_gracefully() {
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("empty.db");
        // Create DB but no tables
        let setup = Connection::open(&path).unwrap();
        setup.execute_batch("PRAGMA journal_mode=WAL;").unwrap();
        drop(setup);

        let conn = Connection::open_with_flags(&path, OpenFlags::SQLITE_OPEN_READ_ONLY).unwrap();
        // get_queue_stats should NOT error; it should degrade to zeros.
        let stats = get_queue_stats(Some(&conn)).unwrap();
        assert_eq!(stats.total_pending, 0);
        assert_eq!(stats.stale_items_count, 0);
    }
}
