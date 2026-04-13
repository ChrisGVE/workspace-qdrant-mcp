//! Canonical data queries — single source of truth for CLI metrics.
//!
//! Every metric displayed by multiple commands must be queried through
//! this module to ensure consistency. All queries use SQLite directly
//! (no gRPC daemon metrics for aggregate numbers).

use anyhow::{Context, Result};
use rusqlite::{params, Connection};

// ── Queue Stats ──────────────────────────────────────────────────

/// Queue status breakdown from SQLite `unified_queue`.
#[derive(Debug, Default, Clone)]
pub struct QueueStats {
    pub pending: usize,
    pub in_progress: usize,
    pub done: usize,
    pub failed: usize,
}

impl QueueStats {
    pub fn total(&self) -> usize {
        self.pending + self.in_progress + self.done + self.failed
    }
}

/// Get queue stats from SQLite. Single source of truth — do NOT
/// use daemon gRPC metrics for queue counts.
pub fn get_queue_stats(conn: &Connection) -> Result<QueueStats> {
    let mut stats = QueueStats::default();
    let mut stmt = conn
        .prepare("SELECT status, COUNT(*) FROM unified_queue GROUP BY status")
        .context("Failed to query queue stats")?;

    let rows = stmt
        .query_map([], |row| {
            Ok((row.get::<_, String>(0)?, row.get::<_, usize>(1)?))
        })
        .context("Failed to read queue stats")?;

    for row in rows {
        let (status, count) = row.context("Failed to parse queue row")?;
        match status.as_str() {
            "pending" => stats.pending = count,
            "in_progress" => stats.in_progress = count,
            "done" => stats.done = count,
            "failed" => stats.failed = count,
            _ => {} // unknown status, ignore
        }
    }

    Ok(stats)
}

// ── Active Projects ──────────────────────────────────────────────

/// Basic project info from SQLite `watch_folders`.
#[derive(Debug, Clone)]
pub struct ProjectInfo {
    pub tenant_id: String,
    pub path: String,
    pub is_active: bool,
    pub watch_id: String,
}

/// Get all registered projects (parent watch folders in 'projects' collection).
pub fn get_projects(conn: &Connection) -> Result<Vec<ProjectInfo>> {
    let mut stmt = conn
        .prepare(
            "SELECT watch_id, tenant_id, path, COALESCE(is_active, 0) \
             FROM watch_folders \
             WHERE parent_watch_id IS NULL AND collection = 'projects' \
             ORDER BY is_active DESC, path ASC",
        )
        .context("Failed to query projects")?;

    let rows = stmt
        .query_map([], |row| {
            Ok(ProjectInfo {
                watch_id: row.get(0)?,
                tenant_id: row.get(1)?,
                path: row.get(2)?,
                is_active: row.get::<_, i64>(3)? > 0,
            })
        })
        .context("Failed to read projects")?;

    rows.collect::<Result<Vec<_>, _>>()
        .context("Failed to parse project rows")
}

/// Count active projects (enabled watch folders).
pub fn get_active_project_count(conn: &Connection) -> Result<usize> {
    conn.query_row(
        "SELECT COUNT(*) FROM watch_folders \
         WHERE parent_watch_id IS NULL AND collection = 'projects' AND is_active > 0",
        [],
        |row| row.get(0),
    )
    .context("Failed to count active projects")
}

// ── Document / Chunk Counts ──────────────────────────────────────

/// File and chunk counts for a tenant.
#[derive(Debug, Default, Clone)]
pub struct DocumentCounts {
    pub tracked_files: usize,
    pub chunk_count: usize,
}

/// Get tracked file and chunk counts for a specific tenant in a collection.
pub fn get_document_counts(
    conn: &Connection,
    tenant_id: &str,
    collection: &str,
) -> Result<DocumentCounts> {
    let result = conn.query_row(
        "SELECT COUNT(*), COALESCE(SUM(chunk_count), 0) \
         FROM tracked_files tf \
         JOIN watch_folders wf ON tf.watch_folder_id = wf.watch_id \
         WHERE wf.tenant_id = ?1 AND wf.collection = ?2",
        params![tenant_id, collection],
        |row| Ok((row.get::<_, usize>(0)?, row.get::<_, usize>(1)?)),
    );

    match result {
        Ok((files, chunks)) => Ok(DocumentCounts {
            tracked_files: files,
            chunk_count: chunks,
        }),
        Err(_) => Ok(DocumentCounts::default()),
    }
}

/// Get document counts for all tenants in a collection (single GROUP BY query).
pub fn get_all_document_counts(
    conn: &Connection,
    collection: &str,
) -> Result<std::collections::HashMap<String, DocumentCounts>> {
    use std::collections::HashMap;
    let mut stmt = conn
        .prepare(
            "SELECT wf.tenant_id, COUNT(tf.file_id), COALESCE(SUM(tf.chunk_count), 0) \
             FROM tracked_files tf \
             JOIN watch_folders wf ON tf.watch_folder_id = wf.watch_id \
             WHERE wf.collection = ?1 \
             GROUP BY wf.tenant_id",
        )
        .context("Failed to query document counts")?;

    let rows = stmt
        .query_map(params![collection], |row| {
            Ok((
                row.get::<_, String>(0)?,
                row.get::<_, usize>(1)?,
                row.get::<_, usize>(2)?,
            ))
        })
        .context("Failed to read document counts")?;

    let mut map = HashMap::new();
    for row in rows {
        let (tenant_id, files, chunks) = row.context("Failed to parse doc count row")?;
        map.insert(
            tenant_id,
            DocumentCounts {
                tracked_files: files,
                chunk_count: chunks,
            },
        );
    }
    Ok(map)
}

/// Get total document count across all tenants in a collection.
pub fn get_total_document_count(conn: &Connection, collection: &str) -> Result<usize> {
    Ok(conn
        .query_row(
            "SELECT COUNT(*) FROM tracked_files tf \
             JOIN watch_folders wf ON tf.watch_folder_id = wf.watch_id \
             WHERE wf.collection = ?1",
            params![collection],
            |row| row.get(0),
        )
        .unwrap_or(0))
}

// ── Language List ────────────────────────────────────────────────

/// Get languages for a tenant's tracked files, filtered to exclude
/// test fixtures and vendored code.
///
/// Languages must have at least 1% of total tracked files (minimum 3)
/// to be included. This filters out test fixture languages (e.g.,
/// `tests/language-support/` files) that inflate the language list.
pub fn get_languages(conn: &Connection, tenant_id: &str, collection: &str) -> Result<Vec<String>> {
    let mut stmt = conn
        .prepare(
            "SELECT tf.language, COUNT(*) as cnt \
             FROM tracked_files tf \
             JOIN watch_folders wf ON tf.watch_folder_id = wf.watch_id \
             WHERE wf.tenant_id = ?1 AND wf.collection = ?2 \
             AND tf.language IS NOT NULL AND tf.language != '' \
             GROUP BY tf.language \
             ORDER BY cnt DESC",
        )
        .context("Failed to query languages")?;

    let rows = stmt
        .query_map(params![tenant_id, collection], |row| {
            Ok((row.get::<_, String>(0)?, row.get::<_, usize>(1)?))
        })
        .context("Failed to read languages")?;

    let lang_counts: Vec<(String, usize)> = rows
        .collect::<Result<Vec<_>, _>>()
        .context("Failed to parse language rows")?;

    let total: usize = lang_counts.iter().map(|(_, c)| c).sum();
    let threshold = (total / 100).max(3); // 1% of total, minimum 3 files

    let mut languages: Vec<String> = lang_counts
        .into_iter()
        .filter(|(_, count)| *count >= threshold)
        .map(|(lang, _)| lang)
        .collect();

    languages.sort_unstable_by(|a, b| a.to_lowercase().cmp(&b.to_lowercase()));
    Ok(languages)
}

// ── Reconciliation Stats ─────────────────────────────────────────

/// File reconciliation breakdown for a project.
#[derive(Debug, Default, Clone)]
pub struct ReconcileStats {
    pub tracked_files: usize,
    pub chunk_count: usize,
    pub in_sync: usize,
    pub to_add: usize,
    pub to_update: usize,
    pub to_remove: usize,
}

/// Get full file stats including reconciliation for a project.
/// Combines document counts + reconciliation in one call.
pub fn get_project_file_stats(conn: &Connection, tenant_id: &str) -> Result<ReconcileStats> {
    let counts = get_document_counts(conn, tenant_id, "projects")?;

    let mut stats = ReconcileStats {
        tracked_files: counts.tracked_files,
        chunk_count: counts.chunk_count,
        ..Default::default()
    };

    // Get reconciliation breakdown
    let mut stmt = conn
        .prepare(
            "SELECT tf.reconcile_reason, COUNT(*) \
             FROM tracked_files tf \
             JOIN watch_folders wf ON tf.watch_folder_id = wf.watch_id \
             WHERE wf.tenant_id = ?1 AND wf.collection = 'projects' \
             AND tf.needs_reconcile = 1 \
             GROUP BY tf.reconcile_reason",
        )
        .context("Failed to query reconciliation stats")?;

    let rows = stmt
        .query_map(params![tenant_id], |row| {
            Ok((row.get::<_, Option<String>>(0)?, row.get::<_, usize>(1)?))
        })
        .context("Failed to read reconciliation stats")?;

    for row in rows {
        let (reason, count) = row.context("Failed to parse reconcile row")?;
        match reason.as_deref() {
            Some("new") | Some("added") => stats.to_add += count,
            Some("modified") | Some("updated") | Some("content_changed") => {
                stats.to_update += count
            }
            Some("deleted") | Some("removed") => stats.to_remove += count,
            _ => stats.to_update += count,
        }
    }

    let needs_reconcile = stats.to_add + stats.to_update + stats.to_remove;
    stats.in_sync = stats.tracked_files.saturating_sub(needs_reconcile);

    Ok(stats)
}

// ── Library Info ─────────────────────────────────────────────────

/// Basic library info from SQLite.
#[derive(Debug, Clone)]
pub struct LibraryInfo {
    pub watch_id: String,
    pub tenant_id: String,
    pub path: String,
    pub mode: String,
    pub enabled: bool,
    pub document_count: usize,
}

/// Get all registered libraries with document counts.
pub fn get_libraries(conn: &Connection) -> Result<Vec<LibraryInfo>> {
    let mut stmt = conn
        .prepare(
            "SELECT wf.watch_id, wf.tenant_id, wf.path, \
                    COALESCE(wf.library_mode, 'incremental'), \
                    COALESCE(wf.enabled, 1), \
                    COALESCE(tf_count.cnt, 0) \
             FROM watch_folders wf \
             LEFT JOIN ( \
                 SELECT watch_folder_id, COUNT(*) AS cnt \
                 FROM tracked_files GROUP BY watch_folder_id \
             ) tf_count ON tf_count.watch_folder_id = wf.watch_id \
             WHERE wf.collection = 'libraries' \
             ORDER BY wf.tenant_id",
        )
        .context("Failed to query libraries")?;

    let rows = stmt
        .query_map([], |row| {
            Ok(LibraryInfo {
                watch_id: row.get(0)?,
                tenant_id: row.get(1)?,
                path: row.get(2)?,
                mode: row.get(3)?,
                enabled: row.get::<_, i64>(4)? > 0,
                document_count: row.get(5)?,
            })
        })
        .context("Failed to read libraries")?;

    rows.collect::<Result<Vec<_>, _>>()
        .context("Failed to parse library rows")
}

// ── Collection Counts ────────────────────────────────────────────

/// Count of collections known to have data (from watch_folders).
pub fn get_active_collection_count(conn: &Connection) -> Result<usize> {
    conn.query_row(
        "SELECT COUNT(DISTINCT collection) FROM watch_folders",
        [],
        |row| row.get(0),
    )
    .context("Failed to count collections")
}

#[cfg(test)]
mod tests {
    use super::*;

    fn setup_test_db() -> Connection {
        let conn = Connection::open_in_memory().unwrap();
        conn.execute_batch(
            "CREATE TABLE watch_folders (
                watch_id TEXT PRIMARY KEY,
                tenant_id TEXT,
                path TEXT,
                collection TEXT,
                parent_watch_id TEXT,
                is_active INTEGER DEFAULT 1,
                enabled INTEGER DEFAULT 1,
                library_mode TEXT,
                is_paused INTEGER DEFAULT 0,
                is_archived INTEGER DEFAULT 0,
                git_remote_url TEXT,
                created_at TEXT,
                updated_at TEXT,
                last_scan TEXT,
                last_activity_at TEXT,
                follow_symlinks INTEGER DEFAULT 0,
                cleanup_on_disable INTEGER DEFAULT 0
            );
            CREATE TABLE tracked_files (
                file_id INTEGER PRIMARY KEY,
                watch_folder_id TEXT,
                file_path TEXT,
                language TEXT,
                chunk_count INTEGER DEFAULT 0,
                needs_reconcile INTEGER DEFAULT 0,
                reconcile_reason TEXT,
                tenant_id TEXT,
                collection TEXT
            );
            CREATE TABLE unified_queue (
                queue_id TEXT PRIMARY KEY,
                idempotency_key TEXT,
                item_type TEXT,
                op TEXT,
                collection TEXT,
                status TEXT,
                tenant_id TEXT,
                branch TEXT,
                payload_json TEXT,
                metadata TEXT,
                created_at TEXT,
                updated_at TEXT,
                lease_until TEXT,
                worker_id TEXT,
                retry_count INTEGER DEFAULT 0,
                error_message TEXT,
                last_error_at TEXT,
                file_path TEXT
            );",
        )
        .unwrap();
        conn
    }

    #[test]
    fn queue_stats_empty() {
        let conn = setup_test_db();
        let stats = get_queue_stats(&conn).unwrap();
        assert_eq!(stats.total(), 0);
    }

    #[test]
    fn queue_stats_with_data() {
        let conn = setup_test_db();
        conn.execute_batch(
            "INSERT INTO unified_queue (queue_id, status) VALUES ('q1', 'pending');
             INSERT INTO unified_queue (queue_id, status) VALUES ('q2', 'pending');
             INSERT INTO unified_queue (queue_id, status) VALUES ('q3', 'done');
             INSERT INTO unified_queue (queue_id, status) VALUES ('q4', 'failed');",
        )
        .unwrap();
        let stats = get_queue_stats(&conn).unwrap();
        assert_eq!(stats.pending, 2);
        assert_eq!(stats.done, 1);
        assert_eq!(stats.failed, 1);
        assert_eq!(stats.total(), 4);
    }

    #[test]
    fn active_projects_count() {
        let conn = setup_test_db();
        conn.execute_batch(
            "INSERT INTO watch_folders (watch_id, tenant_id, path, collection, is_active)
             VALUES ('w1', 't1', '/proj1', 'projects', 1);
             INSERT INTO watch_folders (watch_id, tenant_id, path, collection, is_active)
             VALUES ('w2', 't2', '/proj2', 'projects', 0);
             INSERT INTO watch_folders (watch_id, tenant_id, path, collection, is_active)
             VALUES ('w3', 't3', '/proj3', 'projects', 1);",
        )
        .unwrap();
        assert_eq!(get_active_project_count(&conn).unwrap(), 2);
        assert_eq!(get_projects(&conn).unwrap().len(), 3);
    }

    #[test]
    fn document_counts_per_tenant() {
        let conn = setup_test_db();
        conn.execute_batch(
            "INSERT INTO watch_folders (watch_id, tenant_id, path, collection)
             VALUES ('w1', 't1', '/proj1', 'projects');
             INSERT INTO tracked_files (file_id, watch_folder_id, chunk_count)
             VALUES (1, 'w1', 5);
             INSERT INTO tracked_files (file_id, watch_folder_id, chunk_count)
             VALUES (2, 'w1', 3);",
        )
        .unwrap();
        let counts = get_document_counts(&conn, "t1", "projects").unwrap();
        assert_eq!(counts.tracked_files, 2);
        assert_eq!(counts.chunk_count, 8);
    }

    #[test]
    fn languages_filters_low_count() {
        let conn = setup_test_db();
        conn.execute_batch(
            "INSERT INTO watch_folders (watch_id, tenant_id, path, collection)
             VALUES ('w1', 't1', '/proj1', 'projects');",
        )
        .unwrap();
        // Insert 100 Rust files, 50 Python files, and 2 Clojure files (test fixture)
        for i in 1..=100 {
            conn.execute(
                "INSERT INTO tracked_files (file_id, watch_folder_id, language) VALUES (?1, 'w1', 'Rust')",
                params![i],
            ).unwrap();
        }
        for i in 101..=150 {
            conn.execute(
                "INSERT INTO tracked_files (file_id, watch_folder_id, language) VALUES (?1, 'w1', 'Python')",
                params![i],
            ).unwrap();
        }
        // 2 Clojure files — below 1% threshold (152 total, threshold = max(1, 3) = 3)
        for i in 151..=152 {
            conn.execute(
                "INSERT INTO tracked_files (file_id, watch_folder_id, language) VALUES (?1, 'w1', 'Clojure')",
                params![i],
            ).unwrap();
        }
        let langs = get_languages(&conn, "t1", "projects").unwrap();
        // Clojure (2 files) should be filtered out; Rust and Python pass threshold
        assert_eq!(langs, vec!["Python", "Rust"]);
    }

    #[test]
    fn reconciliation_stats() {
        let conn = setup_test_db();
        conn.execute_batch(
            "INSERT INTO watch_folders (watch_id, tenant_id, path, collection)
             VALUES ('w1', 't1', '/proj1', 'projects');
             INSERT INTO tracked_files (file_id, watch_folder_id, needs_reconcile, reconcile_reason)
             VALUES (1, 'w1', 0, NULL);
             INSERT INTO tracked_files (file_id, watch_folder_id, needs_reconcile, reconcile_reason)
             VALUES (2, 'w1', 1, 'new');
             INSERT INTO tracked_files (file_id, watch_folder_id, needs_reconcile, reconcile_reason)
             VALUES (3, 'w1', 1, 'modified');
             INSERT INTO tracked_files (file_id, watch_folder_id, needs_reconcile, reconcile_reason)
             VALUES (4, 'w1', 1, 'deleted');
             INSERT INTO tracked_files (file_id, watch_folder_id, needs_reconcile, reconcile_reason)
             VALUES (5, 'w1', 0, NULL);",
        )
        .unwrap();
        let stats = get_project_file_stats(&conn, "t1").unwrap();
        assert_eq!(stats.tracked_files, 5);
        assert_eq!(stats.to_add, 1);
        assert_eq!(stats.to_update, 1);
        assert_eq!(stats.to_remove, 1);
        assert_eq!(stats.in_sync, 2);
    }

    #[test]
    fn libraries_with_counts() {
        let conn = setup_test_db();
        conn.execute_batch(
            "INSERT INTO watch_folders (watch_id, tenant_id, path, collection, library_mode)
             VALUES ('lib-docs', 'lib-docs', '/docs', 'libraries', 'sync');
             INSERT INTO tracked_files (file_id, watch_folder_id)
             VALUES (1, 'lib-docs');
             INSERT INTO tracked_files (file_id, watch_folder_id)
             VALUES (2, 'lib-docs');",
        )
        .unwrap();
        let libs = get_libraries(&conn).unwrap();
        assert_eq!(libs.len(), 1);
        assert_eq!(libs[0].document_count, 2);
        assert_eq!(libs[0].mode, "sync");
    }
}
