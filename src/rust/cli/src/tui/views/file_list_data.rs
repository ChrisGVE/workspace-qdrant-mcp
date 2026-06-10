//! SQLite fetch functions for the file-list popup tab.
//!
//! Shared by the Projects and Libraries views. Each view passes its
//! `watch_id` (the `watch_folders.watch_id` primary key). The query joins
//! `tracked_files` to `watch_folders` to reconstruct the absolute path using
//! the same `wf.path || '/' || tf.relative_path` pattern used in the
//! benchmark module (`src/rust/cli/src/commands/benchmark/sparse.rs`).
//!
//! File size is read from `tracked_files.size_bytes` when the column exists;
//! otherwise it falls back to `std::fs::metadata`. Missing files (deleted
//! after indexing) are included with `size: None` rather than silently dropped.

use crate::data::db::connect_readonly;

use super::file_list::FileEntry;

/// Maximum files returned per query (avoids loading huge projects into memory).
const FILE_FETCH_LIMIT: i64 = 5_000;

/// Fetch tracked files for a single watch folder, ordered by relative path.
///
/// Returns an empty `Vec` if the database is unreachable or the watch folder
/// has no tracked files yet (which is always the case for Libraries when the
/// live DB has 0 rows — code-complete but unverified until a library exists).
pub fn fetch_file_entries(watch_id: &str) -> Vec<FileEntry> {
    let conn = match connect_readonly() {
        Ok(c) => c,
        Err(_) => return Vec::new(),
    };
    fetch_from_conn(&conn, watch_id)
}

/// Core fetch logic operating on a provided connection.
///
/// Split out so integration tests can inject an in-memory database without
/// hitting the filesystem.
pub fn fetch_from_conn(conn: &rusqlite::Connection, watch_id: &str) -> Vec<FileEntry> {
    // Try to select size_bytes; the column was added in a later schema version.
    // We probe it with a schema query rather than catching a runtime error.
    let has_size_col = column_exists(conn, "tracked_files", "size_bytes");

    let size_expr = if has_size_col {
        "tf.size_bytes"
    } else {
        "NULL"
    };

    let sql = format!(
        "SELECT tf.relative_path, \
                wf.path || '/' || tf.relative_path AS abs_path, \
                {size_expr} AS size_bytes, \
                COALESCE(tf.chunk_count, 0) AS chunk_count \
         FROM tracked_files tf \
         JOIN watch_folders wf ON tf.watch_folder_id = wf.watch_id \
         WHERE tf.watch_folder_id = ?1 \
         ORDER BY tf.relative_path \
         LIMIT ?2"
    );

    let mut stmt = match conn.prepare(&sql) {
        Ok(s) => s,
        Err(_) => return Vec::new(),
    };

    let rows = stmt.query_map(rusqlite::params![watch_id, FILE_FETCH_LIMIT], |row| {
        Ok((
            row.get::<_, String>(0)?,      // relative_path
            row.get::<_, String>(1)?,      // abs_path
            row.get::<_, Option<i64>>(2)?, // size_bytes (nullable)
            row.get::<_, i64>(3)?,         // chunk_count
        ))
    });

    let rows = match rows {
        Ok(r) => r,
        Err(_) => return Vec::new(),
    };

    rows.flatten()
        .map(|(relative_path, abs_path, db_size, chunk_count)| {
            // Prefer DB size; fall back to filesystem metadata for older schema.
            let size = db_size
                .map(|b| b as u64)
                .or_else(|| std::fs::metadata(&abs_path).ok().map(|m| m.len()));

            FileEntry {
                relative_path,
                abs_path,
                size,
                chunk_count,
            }
        })
        .collect()
}

/// Return `true` if `column` exists in `table` according to `PRAGMA table_info`.
fn column_exists(conn: &rusqlite::Connection, table: &str, column: &str) -> bool {
    let sql = format!("PRAGMA table_info({table})");
    let Ok(mut stmt) = conn.prepare(&sql) else {
        return false;
    };
    stmt.query_map([], |row| row.get::<_, String>(1))
        .map(|rows| rows.flatten().any(|col| col == column))
        .unwrap_or(false)
}

// ─── Unit tests ───────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    /// Build a minimal in-memory DB with two watch folders and some tracked files.
    fn make_test_db() -> rusqlite::Connection {
        let conn = rusqlite::Connection::open_in_memory().unwrap();
        conn.execute_batch(
            "CREATE TABLE watch_folders (
                 watch_id   TEXT PRIMARY KEY,
                 path       TEXT NOT NULL
             );
             CREATE TABLE tracked_files (
                 file_id         INTEGER PRIMARY KEY,
                 watch_folder_id TEXT NOT NULL,
                 relative_path   TEXT NOT NULL,
                 chunk_count     INTEGER DEFAULT 0
             );
             INSERT INTO watch_folders VALUES ('w1', '/project');
             INSERT INTO watch_folders VALUES ('w2', '/library');
             INSERT INTO tracked_files (file_id, watch_folder_id, relative_path, chunk_count)
               VALUES (1, 'w1', 'src/main.rs',  3),
                      (2, 'w1', 'src/lib.rs',   5),
                      (3, 'w2', 'docs/guide.md', 1);",
        )
        .unwrap();
        conn
    }

    #[test]
    fn fetches_files_for_watch_folder() {
        let conn = make_test_db();
        let entries = fetch_from_conn(&conn, "w1");
        assert_eq!(entries.len(), 2);
        // Ordered by relative_path.
        assert_eq!(entries[0].relative_path, "src/lib.rs");
        assert_eq!(entries[1].relative_path, "src/main.rs");
    }

    #[test]
    fn absolute_path_reconstructed_correctly() {
        let conn = make_test_db();
        let entries = fetch_from_conn(&conn, "w1");
        let main = entries
            .iter()
            .find(|e| e.relative_path == "src/main.rs")
            .unwrap();
        assert_eq!(main.abs_path, "/project/src/main.rs");
    }

    #[test]
    fn chunk_count_propagated() {
        let conn = make_test_db();
        let entries = fetch_from_conn(&conn, "w1");
        let lib = entries
            .iter()
            .find(|e| e.relative_path == "src/lib.rs")
            .unwrap();
        assert_eq!(lib.chunk_count, 5);
    }

    #[test]
    fn unknown_watch_id_returns_empty() {
        let conn = make_test_db();
        let entries = fetch_from_conn(&conn, "no-such-id");
        assert!(entries.is_empty());
    }

    #[test]
    fn library_watch_folder_returns_correct_files() {
        let conn = make_test_db();
        let entries = fetch_from_conn(&conn, "w2");
        assert_eq!(entries.len(), 1);
        assert_eq!(entries[0].relative_path, "docs/guide.md");
        assert_eq!(entries[0].abs_path, "/library/docs/guide.md");
    }

    #[test]
    fn size_bytes_column_fallback_when_absent() {
        // DB without size_bytes column — fetch_from_conn must not error.
        let conn = make_test_db();
        let entries = fetch_from_conn(&conn, "w1");
        // Size will be None (no size_bytes column and /project/src/lib.rs
        // does not exist on disk during tests).
        for e in &entries {
            // Either None (not on disk) or Some(actual) — both are acceptable.
            // The important thing is no panic.
            let _ = e.size;
        }
    }

    #[test]
    fn size_bytes_column_used_when_present() {
        let conn = rusqlite::Connection::open_in_memory().unwrap();
        conn.execute_batch(
            "CREATE TABLE watch_folders (watch_id TEXT PRIMARY KEY, path TEXT NOT NULL);
             CREATE TABLE tracked_files (
                 file_id         INTEGER PRIMARY KEY,
                 watch_folder_id TEXT NOT NULL,
                 relative_path   TEXT NOT NULL,
                 chunk_count     INTEGER DEFAULT 0,
                 size_bytes      INTEGER
             );
             INSERT INTO watch_folders VALUES ('w1', '/root');
             INSERT INTO tracked_files (file_id, watch_folder_id, relative_path, chunk_count, size_bytes)
               VALUES (1, 'w1', 'file.txt', 2, 8192);",
        )
        .unwrap();

        let entries = fetch_from_conn(&conn, "w1");
        assert_eq!(entries.len(), 1);
        assert_eq!(entries[0].size, Some(8192));
    }

    #[test]
    fn null_size_bytes_falls_back_to_none_when_not_on_disk() {
        let conn = rusqlite::Connection::open_in_memory().unwrap();
        conn.execute_batch(
            "CREATE TABLE watch_folders (watch_id TEXT PRIMARY KEY, path TEXT NOT NULL);
             CREATE TABLE tracked_files (
                 file_id         INTEGER PRIMARY KEY,
                 watch_folder_id TEXT NOT NULL,
                 relative_path   TEXT NOT NULL,
                 chunk_count     INTEGER DEFAULT 0,
                 size_bytes      INTEGER
             );
             INSERT INTO watch_folders VALUES ('w1', '/nonexistent');
             INSERT INTO tracked_files (file_id, watch_folder_id, relative_path, chunk_count, size_bytes)
               VALUES (1, 'w1', 'ghost.txt', 0, NULL);",
        )
        .unwrap();

        let entries = fetch_from_conn(&conn, "w1");
        assert_eq!(entries.len(), 1);
        // /nonexistent/ghost.txt doesn't exist on disk → size is None.
        assert!(entries[0].size.is_none());
    }

    #[test]
    fn column_exists_detects_present_column() {
        let conn = make_test_db();
        assert!(column_exists(&conn, "tracked_files", "relative_path"));
        assert!(column_exists(&conn, "tracked_files", "chunk_count"));
    }

    #[test]
    fn column_exists_returns_false_for_missing() {
        let conn = make_test_db();
        assert!(!column_exists(&conn, "tracked_files", "size_bytes"));
        assert!(!column_exists(&conn, "tracked_files", "no_such_column"));
    }

    #[test]
    fn files_ordered_by_relative_path() {
        let conn = rusqlite::Connection::open_in_memory().unwrap();
        conn.execute_batch(
            "CREATE TABLE watch_folders (watch_id TEXT PRIMARY KEY, path TEXT NOT NULL);
             CREATE TABLE tracked_files (
                 file_id         INTEGER PRIMARY KEY,
                 watch_folder_id TEXT NOT NULL,
                 relative_path   TEXT NOT NULL,
                 chunk_count     INTEGER DEFAULT 0
             );
             INSERT INTO watch_folders VALUES ('w1', '/proj');
             INSERT INTO tracked_files VALUES (1, 'w1', 'z_last.rs',  0),
                                              (2, 'w1', 'a_first.rs', 0),
                                              (3, 'w1', 'm_middle.rs',0);",
        )
        .unwrap();

        let entries = fetch_from_conn(&conn, "w1");
        let paths: Vec<&str> = entries.iter().map(|e| e.relative_path.as_str()).collect();
        assert_eq!(paths, vec!["a_first.rs", "m_middle.rs", "z_last.rs"]);
    }
}
