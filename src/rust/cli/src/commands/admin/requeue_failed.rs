//! Requeue-failed subcommand handler
//!
//! Resets `unified_queue` rows with `status='failed'` back to `pending`
//! when their `error_message` matches a substring. Useful after fixing
//! a transient bug that retry-exhausted a batch of items.
//!
//! Dry-run by default: prints sample rows. With `--apply`, updates the
//! rows in a single transaction (clearing `retry_count`, `error_message`,
//! `last_error_at`, `lease_until`, `worker_id`, and bumping `updated_at`).

use anyhow::{Context, Result};
use rusqlite::{Connection, OptionalExtension};

use crate::output;

/// One candidate row matched by `requeue-failed`.
#[derive(Debug, Clone)]
struct FailedRow {
    queue_id: String,
    item_type: String,
    op: String,
    tenant_id: String,
    file_path: Option<String>,
    retry_count: i64,
    error_message: Option<String>,
}

/// Open state.db in read-write mode. Mirrors `clean_orphan_queue_items`.
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

/// Reset retry-exhausted failed rows whose `error_message` contains the
/// given substring back to `pending`.
pub fn execute(reason_substring: String, max_rows: u64, apply: bool) -> Result<()> {
    output::section("Requeue Failed Items");
    output::kv("Reason filter", format!("'%{}%'", reason_substring));
    output::kv("Max rows", max_rows.to_string());
    if !apply {
        output::info("Dry run — no rows will be reset");
    }
    output::separator();

    if reason_substring.trim().is_empty() {
        anyhow::bail!("--reason-substring must not be empty");
    }
    if max_rows == 0 {
        anyhow::bail!("--max-rows must be greater than 0");
    }

    let conn = open_state_db_rw()?;

    let candidates = scan_candidates(&conn, &reason_substring, max_rows)?;
    let total_matching = count_matching(&conn, &reason_substring)?;

    if candidates.is_empty() {
        output::success("No failed rows match the given reason substring.");
        return Ok(());
    }

    print_candidates(&candidates, total_matching, max_rows);
    output::separator();

    if !apply {
        let limited_note = if total_matching > max_rows {
            format!(
                " (capped at --max-rows={}; total matching = {})",
                max_rows, total_matching
            )
        } else {
            String::new()
        };
        output::info(format!(
            "DRY RUN — would reset {} failed row(s){}. Re-run with --apply to commit.",
            candidates.len(),
            limited_note
        ));
        return Ok(());
    }

    let ids: Vec<String> = candidates.iter().map(|r| r.queue_id.clone()).collect();
    let updated = requeue_rows(&conn, &ids)?;
    output::success(format!(
        "Reset {} failed row(s) back to pending. The daemon picks them up on the next dequeue tick.",
        updated
    ));

    Ok(())
}

/// Fetch up to `max_rows` failed rows whose `error_message` matches the pattern.
fn scan_candidates(
    conn: &Connection,
    reason_substring: &str,
    max_rows: u64,
) -> Result<Vec<FailedRow>> {
    let pattern = like_pattern(reason_substring);
    let mut stmt = conn
        .prepare(
            "SELECT queue_id, item_type, op, tenant_id, file_path, retry_count, error_message \
             FROM unified_queue \
             WHERE status = 'failed' AND error_message LIKE ?1 \
             ORDER BY last_error_at DESC \
             LIMIT ?2",
        )
        .context("Failed to prepare candidate query")?;

    let rows = stmt
        .query_map(rusqlite::params![pattern, max_rows as i64], |row| {
            Ok(FailedRow {
                queue_id: row.get(0)?,
                item_type: row.get(1)?,
                op: row.get(2)?,
                tenant_id: row.get(3)?,
                file_path: row.get(4)?,
                retry_count: row.get(5)?,
                error_message: row.get(6)?,
            })
        })
        .context("Failed to query candidate rows")?;

    let mut out = Vec::new();
    for row in rows {
        out.push(row.context("Failed to read candidate row")?);
    }
    Ok(out)
}

/// Total number of failed rows matching the pattern (unbounded by limit).
fn count_matching(conn: &Connection, reason_substring: &str) -> Result<u64> {
    let pattern = like_pattern(reason_substring);
    let count: Option<i64> = conn
        .query_row(
            "SELECT COUNT(*) FROM unified_queue \
             WHERE status = 'failed' AND error_message LIKE ?1",
            rusqlite::params![pattern],
            |row| row.get(0),
        )
        .optional()
        .context("Failed to count matching rows")?;
    Ok(count.unwrap_or(0) as u64)
}

/// Wrap the substring in `%` for a SQL `LIKE` pattern.
fn like_pattern(substring: &str) -> String {
    format!("%{}%", substring)
}

/// Print summary + first 5 sample rows.
fn print_candidates(rows: &[FailedRow], total_matching: u64, max_rows: u64) {
    let shown = rows.len();
    output::warning(format!(
        "Matched {} failed row(s) (showing {}{}):",
        total_matching,
        shown.min(5),
        if total_matching > max_rows {
            format!(", capped at --max-rows={}", max_rows)
        } else {
            String::new()
        }
    ));

    for row in rows.iter().take(5) {
        let file = row.file_path.as_deref().unwrap_or("(no file_path)");
        let err = row
            .error_message
            .as_deref()
            .map(truncate_msg)
            .unwrap_or_else(|| "(no error_message)".to_string());
        output::kv(
            format!(
                "  [{} {}/{}] {} retries",
                row.tenant_id, row.item_type, row.op, row.retry_count
            ),
            format!("{} — {}", short_id(&row.queue_id), file),
        );
        output::kv("    error", err);
    }
}

/// Truncate a long error message for sample display.
fn truncate_msg(msg: &str) -> String {
    const MAX_LEN: usize = 120;
    if msg.len() <= MAX_LEN {
        msg.to_string()
    } else {
        let mut truncated: String = msg.chars().take(MAX_LEN).collect();
        truncated.push('…');
        truncated
    }
}

/// First 8 chars of a queue_id for compact display.
fn short_id(id: &str) -> String {
    id.chars().take(8).collect()
}

/// Reset the given queue_ids back to pending in one transaction.
///
/// Matches the SQL update used by `WriteActor::exec_retry_all`/`exec_retry_item`
/// — clears retry/lease state and bumps `updated_at`.
fn requeue_rows(conn: &Connection, queue_ids: &[String]) -> Result<u64> {
    if queue_ids.is_empty() {
        return Ok(0);
    }

    let tx = conn
        .unchecked_transaction()
        .context("Failed to begin transaction")?;

    let now = chrono::Utc::now()
        .format("%Y-%m-%dT%H:%M:%S%.3fZ")
        .to_string();

    let mut total = 0u64;
    // Chunk the update so we don't hit SQLite's parameter limit (999 by default).
    const CHUNK_SIZE: usize = 500;
    for chunk in queue_ids.chunks(CHUNK_SIZE) {
        let placeholders: String = (0..chunk.len())
            .map(|i| format!("?{}", i + 2))
            .collect::<Vec<_>>()
            .join(", ");

        let sql = format!(
            "UPDATE unified_queue \
             SET status='pending', retry_count=0, error_message=NULL, \
                 last_error_at=NULL, lease_until=NULL, worker_id=NULL, \
                 updated_at=?1 \
             WHERE queue_id IN ({})",
            placeholders
        );

        let mut stmt = tx.prepare(&sql).context("Failed to prepare update")?;
        let mut params: Vec<&dyn rusqlite::ToSql> = Vec::with_capacity(chunk.len() + 1);
        params.push(&now);
        for id in chunk {
            params.push(id);
        }
        let count = stmt
            .execute(params.as_slice())
            .context("Failed to reset failed rows")?;
        total += count as u64;
    }

    tx.commit().context("Failed to commit requeue")?;
    Ok(total)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn setup_in_memory_db() -> Connection {
        let conn = Connection::open_in_memory().unwrap();
        conn.execute_batch(
            "CREATE TABLE unified_queue (
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
                error_message TEXT,
                last_error_at TEXT,
                lease_until TEXT,
                worker_id TEXT,
                payload_json TEXT NOT NULL DEFAULT '{}',
                file_path TEXT
            );",
        )
        .unwrap();
        conn
    }

    fn insert_failed_row(
        conn: &Connection,
        queue_id: &str,
        tenant: &str,
        retries: i64,
        error: &str,
    ) {
        conn.execute(
            "INSERT INTO unified_queue \
             (queue_id, idempotency_key, item_type, op, tenant_id, collection, \
              status, created_at, updated_at, retry_count, error_message, \
              last_error_at, lease_until, worker_id, file_path) \
             VALUES (?1, ?2, 'file', 'add', ?3, 'projects', 'failed', \
                     '2026-01-01T00:00:00Z', '2026-01-01T00:00:00Z', \
                     ?4, ?5, '2026-01-01T00:00:00Z', '2026-01-01T01:00:00Z', \
                     'worker-1', '/tmp/foo.rs')",
            rusqlite::params![queue_id, queue_id, tenant, retries, error],
        )
        .unwrap();
    }

    fn insert_pending_row(conn: &Connection, queue_id: &str, tenant: &str) {
        conn.execute(
            "INSERT INTO unified_queue \
             (queue_id, idempotency_key, item_type, op, tenant_id, collection, \
              status, created_at, updated_at, retry_count) \
             VALUES (?1, ?2, 'file', 'add', ?3, 'projects', 'pending', \
                     '2026-01-01T00:00:00Z', '2026-01-01T00:00:00Z', 0)",
            rusqlite::params![queue_id, queue_id, tenant],
        )
        .unwrap();
    }

    #[test]
    fn scan_finds_only_matching_failed_rows() {
        let conn = setup_in_memory_db();
        insert_failed_row(&conn, "q1", "t1", 5, "tenant 'foo' not found");
        insert_failed_row(&conn, "q2", "t1", 5, "different error");
        insert_failed_row(&conn, "q3", "t2", 5, "tenant 'bar' not found");
        insert_pending_row(&conn, "q4", "t1"); // pending — excluded
        insert_failed_row(&conn, "q5", "t1", 5, "tenant 'baz' not found");

        let rows = scan_candidates(&conn, "tenant", 100).unwrap();
        assert_eq!(rows.len(), 3);
        assert_eq!(count_matching(&conn, "tenant").unwrap(), 3);
    }

    #[test]
    fn scan_respects_max_rows() {
        let conn = setup_in_memory_db();
        for i in 0..10 {
            insert_failed_row(&conn, &format!("q{i}"), "t1", 5, "matching error");
        }

        let rows = scan_candidates(&conn, "matching", 3).unwrap();
        assert_eq!(rows.len(), 3);
        // Total matching is still 10 even when limited.
        assert_eq!(count_matching(&conn, "matching").unwrap(), 10);
    }

    #[test]
    fn requeue_resets_status_and_clears_metadata() {
        let conn = setup_in_memory_db();
        insert_failed_row(&conn, "q1", "t1", 5, "tenant orphan");
        insert_failed_row(&conn, "q2", "t1", 3, "tenant orphan");
        insert_failed_row(&conn, "q3", "t2", 5, "different");

        let candidates = scan_candidates(&conn, "tenant", 100).unwrap();
        assert_eq!(candidates.len(), 2);

        let ids: Vec<String> = candidates.into_iter().map(|c| c.queue_id).collect();
        let updated = requeue_rows(&conn, &ids).unwrap();
        assert_eq!(updated, 2);

        // q1 and q2 should now be pending with cleared metadata
        for id in ["q1", "q2"] {
            let (status, retry_count, error, last_err, lease, worker): (
                String,
                i64,
                Option<String>,
                Option<String>,
                Option<String>,
                Option<String>,
            ) = conn
                .query_row(
                    "SELECT status, retry_count, error_message, last_error_at, \
                            lease_until, worker_id \
                     FROM unified_queue WHERE queue_id = ?1",
                    rusqlite::params![id],
                    |row| {
                        Ok((
                            row.get(0)?,
                            row.get(1)?,
                            row.get(2)?,
                            row.get(3)?,
                            row.get(4)?,
                            row.get(5)?,
                        ))
                    },
                )
                .unwrap();
            assert_eq!(status, "pending", "{id}");
            assert_eq!(retry_count, 0, "{id}");
            assert!(error.is_none(), "{id}");
            assert!(last_err.is_none(), "{id}");
            assert!(lease.is_none(), "{id}");
            assert!(worker.is_none(), "{id}");
        }

        // q3 (different error) remains failed
        let (status,): (String,) = conn
            .query_row(
                "SELECT status FROM unified_queue WHERE queue_id = 'q3'",
                [],
                |r| Ok((r.get(0)?,)),
            )
            .unwrap();
        assert_eq!(status, "failed");
    }

    #[test]
    fn requeue_empty_list_is_noop() {
        let conn = setup_in_memory_db();
        insert_failed_row(&conn, "q1", "t1", 5, "error");
        let updated = requeue_rows(&conn, &[]).unwrap();
        assert_eq!(updated, 0);
    }

    #[test]
    fn like_pattern_wraps_in_percent_signs() {
        assert_eq!(like_pattern("foo"), "%foo%");
        assert_eq!(like_pattern(""), "%%");
    }

    #[test]
    fn truncate_msg_caps_long_messages() {
        let long = "a".repeat(200);
        let truncated = truncate_msg(&long);
        assert!(truncated.ends_with("…"));
        assert!(truncated.chars().count() <= 121);
    }
}
