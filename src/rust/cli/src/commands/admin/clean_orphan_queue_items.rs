//! Clean-orphan-queue-items subcommand handler
//!
//! Removes rows from `unified_queue` whose `tenant_id` does not match any
//! row in `watch_folders`. These are typically left over after a watch
//! folder was disabled/deleted but the worker had already enqueued items.
//!
//! Dry-run by default: prints counts and breakdown by tenant/status. With
//! `--apply`, deletes the matching rows in a single transaction.

use std::collections::BTreeMap;

use anyhow::{Context, Result};
use rusqlite::{Connection, OptionalExtension};

use crate::output;

/// Group key for the breakdown summary.
#[derive(Debug, Clone, Eq, PartialEq, Ord, PartialOrd)]
struct Group {
    tenant_id: String,
    status: String,
}

/// Open state.db in read-write mode.
///
/// Mirrors `rebalance_idf::open_state_db_rw` (admin commands that mutate
/// SQLite directly outside the daemon write actor — same pattern as
/// `recover-state` and IDF rebalancing).
fn open_state_db_rw() -> Result<Connection> {
    let db_path = crate::config::get_database_path().map_err(|e| anyhow::anyhow!("{}", e))?;

    if !db_path.exists() {
        anyhow::bail!(
            "Database not found at {}. Is the daemon running? Start it with: wqm service start",
            db_path.display()
        );
    }

    let conn = Connection::open(&db_path).context("Failed to open state database read-write")?;
    conn.execute_batch("PRAGMA busy_timeout=10000; PRAGMA journal_mode=WAL;")
        .context("Failed to configure SQLite connection")?;
    Ok(conn)
}

/// Detect and optionally delete `unified_queue` rows whose tenant has no
/// matching `watch_folders` entry (orphan queue items).
pub fn execute(apply: bool, limit: Option<u64>) -> Result<()> {
    output::section("Orphan Queue Item Cleanup");

    if !apply {
        output::info("Dry run — no rows will be deleted");
    }
    if let Some(n) = limit {
        output::kv("Limit", n.to_string());
    }
    output::separator();

    let conn = open_state_db_rw()?;

    let groups = scan_orphan_groups(&conn)?;
    let total: u64 = groups.values().sum();

    if total == 0 {
        output::success("No orphan queue items found.");
        return Ok(());
    }

    print_breakdown(&groups, total);
    output::separator();

    if !apply {
        let msg = match limit {
            Some(n) if n < total => format!(
                "DRY RUN — would delete up to {} of {} orphan row(s). \
                 Re-run with --apply to commit.",
                n, total
            ),
            _ => format!(
                "DRY RUN — would delete {} orphan row(s). \
                 Re-run with --apply to commit.",
                total
            ),
        };
        output::info(msg);
        return Ok(());
    }

    let deleted = delete_orphan_rows(&conn, limit)?;
    output::success(format!("Deleted {} orphan queue row(s).", deleted));

    Ok(())
}

/// Group orphan rows by (tenant_id, status).
fn scan_orphan_groups(conn: &Connection) -> Result<BTreeMap<Group, u64>> {
    let mut stmt = conn
        .prepare(
            "SELECT tenant_id, status, COUNT(*) \
             FROM unified_queue \
             WHERE tenant_id NOT IN (SELECT tenant_id FROM watch_folders) \
             GROUP BY tenant_id, status \
             ORDER BY tenant_id, status",
        )
        .context("Failed to prepare orphan scan query")?;

    let rows = stmt
        .query_map([], |row| {
            Ok((
                row.get::<_, String>(0)?,
                row.get::<_, String>(1)?,
                row.get::<_, i64>(2)? as u64,
            ))
        })
        .context("Failed to query orphan groups")?;

    let mut groups: BTreeMap<Group, u64> = BTreeMap::new();
    for row in rows {
        let (tenant_id, status, count) = row.context("Failed to read orphan group row")?;
        groups.insert(Group { tenant_id, status }, count);
    }
    Ok(groups)
}

/// Print the (tenant_id, status) breakdown.
fn print_breakdown(groups: &BTreeMap<Group, u64>, total: u64) {
    output::warning(format!(
        "Found {} orphan queue row(s) spanning {} (tenant, status) group(s):",
        total,
        groups.len()
    ));
    for (group, count) in groups {
        output::kv(
            format!("  {} [{}]", group.tenant_id, group.status),
            count.to_string(),
        );
    }
}

/// Delete orphan rows, optionally limited to N rows.
///
/// Runs inside a single transaction. SQLite does not allow `DELETE ... LIMIT`
/// without compile-time `SQLITE_ENABLE_UPDATE_DELETE_LIMIT`, so when a limit
/// is specified we use a subquery on the rowid index instead.
fn delete_orphan_rows(conn: &Connection, limit: Option<u64>) -> Result<u64> {
    let tx = conn
        .unchecked_transaction()
        .context("Failed to begin transaction")?;

    let deleted = match limit {
        Some(n) => tx
            .execute(
                "DELETE FROM unified_queue \
                 WHERE rowid IN ( \
                   SELECT rowid FROM unified_queue \
                   WHERE tenant_id NOT IN (SELECT tenant_id FROM watch_folders) \
                   LIMIT ?1 \
                 )",
                rusqlite::params![n as i64],
            )
            .context("Failed to delete orphan rows (limited)")?,
        None => tx
            .execute(
                "DELETE FROM unified_queue \
                 WHERE tenant_id NOT IN (SELECT tenant_id FROM watch_folders)",
                [],
            )
            .context("Failed to delete orphan rows")?,
    };

    tx.commit().context("Failed to commit deletion")?;
    Ok(deleted as u64)
}

/// Count remaining orphan rows — helper for tests / verification.
#[allow(dead_code)]
fn count_orphans(conn: &Connection) -> Result<u64> {
    let count: Option<i64> = conn
        .query_row(
            "SELECT COUNT(*) FROM unified_queue \
             WHERE tenant_id NOT IN (SELECT tenant_id FROM watch_folders)",
            [],
            |row| row.get(0),
        )
        .optional()
        .context("Failed to count orphan rows")?;
    Ok(count.unwrap_or(0) as u64)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Minimal schema for testing orphan detection.
    fn setup_in_memory_db() -> Connection {
        let conn = Connection::open_in_memory().unwrap();
        conn.execute_batch(
            "CREATE TABLE watch_folders (
                watch_id TEXT PRIMARY KEY,
                path TEXT NOT NULL,
                collection TEXT NOT NULL,
                tenant_id TEXT NOT NULL
            );
            CREATE TABLE unified_queue (
                queue_id TEXT PRIMARY KEY,
                idempotency_key TEXT UNIQUE NOT NULL,
                item_type TEXT NOT NULL,
                op TEXT NOT NULL,
                tenant_id TEXT NOT NULL,
                collection TEXT NOT NULL,
                status TEXT NOT NULL DEFAULT 'pending',
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                retry_count INTEGER NOT NULL DEFAULT 0,
                payload_json TEXT NOT NULL DEFAULT '{}'
            );",
        )
        .unwrap();
        conn
    }

    fn insert_watch(conn: &Connection, tenant: &str) {
        conn.execute(
            "INSERT INTO watch_folders (watch_id, path, collection, tenant_id) \
             VALUES (?1, ?2, 'projects', ?3)",
            rusqlite::params![format!("w-{tenant}"), format!("/tmp/{tenant}"), tenant],
        )
        .unwrap();
    }

    fn insert_queue_row(conn: &Connection, queue_id: &str, tenant: &str, status: &str) {
        conn.execute(
            "INSERT INTO unified_queue \
             (queue_id, idempotency_key, item_type, op, tenant_id, collection, \
              status, created_at, updated_at) \
             VALUES (?1, ?2, 'file', 'add', ?3, 'projects', ?4, \
                     '2026-01-01T00:00:00Z', '2026-01-01T00:00:00Z')",
            rusqlite::params![queue_id, queue_id, tenant, status],
        )
        .unwrap();
    }

    #[test]
    fn no_orphans_when_all_tenants_have_watch_folders() {
        let conn = setup_in_memory_db();
        insert_watch(&conn, "live");
        insert_queue_row(&conn, "q1", "live", "pending");
        insert_queue_row(&conn, "q2", "live", "failed");

        let groups = scan_orphan_groups(&conn).unwrap();
        assert!(groups.is_empty());
        assert_eq!(count_orphans(&conn).unwrap(), 0);
    }

    #[test]
    fn orphans_detected_when_tenant_missing_from_watch_folders() {
        let conn = setup_in_memory_db();
        insert_watch(&conn, "live");
        insert_queue_row(&conn, "q1", "live", "pending");
        insert_queue_row(&conn, "q2", "orphan-1", "pending");
        insert_queue_row(&conn, "q3", "orphan-1", "failed");
        insert_queue_row(&conn, "q4", "orphan-2", "pending");

        let groups = scan_orphan_groups(&conn).unwrap();
        assert_eq!(groups.len(), 3);
        let total: u64 = groups.values().sum();
        assert_eq!(total, 3);
        assert_eq!(count_orphans(&conn).unwrap(), 3);
    }

    #[test]
    fn delete_removes_only_orphan_rows() {
        let conn = setup_in_memory_db();
        insert_watch(&conn, "live");
        insert_queue_row(&conn, "q-live", "live", "pending");
        insert_queue_row(&conn, "q-orphan-1", "orphan", "pending");
        insert_queue_row(&conn, "q-orphan-2", "orphan", "failed");

        let deleted = delete_orphan_rows(&conn, None).unwrap();
        assert_eq!(deleted, 2);

        let remaining: i64 = conn
            .query_row("SELECT COUNT(*) FROM unified_queue", [], |r| r.get(0))
            .unwrap();
        assert_eq!(remaining, 1);

        let still_there: i64 = conn
            .query_row(
                "SELECT COUNT(*) FROM unified_queue WHERE tenant_id = 'live'",
                [],
                |r| r.get(0),
            )
            .unwrap();
        assert_eq!(still_there, 1);
        assert_eq!(count_orphans(&conn).unwrap(), 0);
    }

    #[test]
    fn delete_respects_limit() {
        let conn = setup_in_memory_db();
        insert_watch(&conn, "live");
        insert_queue_row(&conn, "q-live", "live", "pending");
        for i in 0..5 {
            insert_queue_row(&conn, &format!("q-orphan-{i}"), "orphan", "pending");
        }

        let deleted = delete_orphan_rows(&conn, Some(2)).unwrap();
        assert_eq!(deleted, 2);
        assert_eq!(count_orphans(&conn).unwrap(), 3);

        let live_intact: i64 = conn
            .query_row(
                "SELECT COUNT(*) FROM unified_queue WHERE tenant_id = 'live'",
                [],
                |r| r.get(0),
            )
            .unwrap();
        assert_eq!(live_intact, 1);
    }

    #[test]
    fn delete_with_no_orphans_returns_zero() {
        let conn = setup_in_memory_db();
        insert_watch(&conn, "live");
        insert_queue_row(&conn, "q1", "live", "pending");

        let deleted = delete_orphan_rows(&conn, None).unwrap();
        assert_eq!(deleted, 0);
    }
}
