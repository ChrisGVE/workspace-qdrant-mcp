//! Scratchpad mirror read queries — direct SQLite reads from `scratchpad_mirror`.
//!
//! SQL is verbatim from `scratchpad-mirror-queries.ts:81-95`.
//!
//! Schema: scratchpad_id, title, content, tags, tenant_id, created_at, updated_at

use rusqlite::params;
use rusqlite::Connection;

// ---------------------------------------------------------------------------
// Public types
// ---------------------------------------------------------------------------

/// A single row from the `scratchpad_mirror` table.
///
/// Mirrors `ScratchpadMirrorEntry` in `scratchpad-mirror-queries.ts`.
#[derive(Debug, Clone, PartialEq)]
pub struct ScratchpadMirrorEntry {
    pub scratchpad_id: String,
    pub title: Option<String>,
    pub content: String,
    /// JSON-encoded array of tags, e.g. `["rust","async"]`.
    pub tags: String,
    pub tenant_id: String,
    pub created_at: String,
    pub updated_at: String,
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/// List scratchpad entries from `scratchpad_mirror`, optionally filtered by
/// `tenant_id`.
///
/// SQL mirrors `scratchpad-mirror-queries.ts:81-95`:
/// ```sql
/// SELECT scratchpad_id, title, content, tags, tenant_id, created_at, updated_at
/// FROM scratchpad_mirror
/// [WHERE tenant_id = ?]
/// ORDER BY updated_at DESC LIMIT ?
/// ```
///
/// Returns an empty `Vec` when the connection is `None` or the table does not
/// exist.
pub fn list_scratchpad(
    conn: Option<&Connection>,
    tenant_id: Option<&str>,
    limit: usize,
) -> Vec<ScratchpadMirrorEntry> {
    let Some(conn) = conn else {
        return Vec::new();
    };

    let result: Result<Vec<ScratchpadMirrorEntry>, rusqlite::Error> = (|| {
        let limit_i64 = limit as i64;
        if let Some(tid) = tenant_id {
            let mut stmt = conn.prepare(
                "SELECT scratchpad_id, title, content, tags, tenant_id, created_at, updated_at \
                 FROM scratchpad_mirror \
                 WHERE tenant_id = ? \
                 ORDER BY updated_at DESC LIMIT ?",
            )?;
            let rows: Vec<ScratchpadMirrorEntry> = stmt
                .query_map(params![tid, limit_i64], map_row)?
                .collect::<Result<_, _>>()?;
            Ok(rows)
        } else {
            let mut stmt = conn.prepare(
                "SELECT scratchpad_id, title, content, tags, tenant_id, created_at, updated_at \
                 FROM scratchpad_mirror \
                 ORDER BY updated_at DESC LIMIT ?",
            )?;
            let rows: Vec<ScratchpadMirrorEntry> = stmt
                .query_map(params![limit_i64], map_row)?
                .collect::<Result<_, _>>()?;
            Ok(rows)
        }
    })();

    match result {
        Ok(entries) => entries,
        Err(e) if e.to_string().contains("no such table") => Vec::new(),
        Err(e) => {
            tracing::warn!("list_scratchpad query failed: {e}");
            Vec::new()
        }
    }
}

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

fn map_row(row: &rusqlite::Row<'_>) -> rusqlite::Result<ScratchpadMirrorEntry> {
    Ok(ScratchpadMirrorEntry {
        scratchpad_id: row.get(0)?,
        title: row.get(1)?,
        content: row.get(2)?,
        tags: row.get(3)?,
        tenant_id: row.get(4)?,
        created_at: row.get(5)?,
        updated_at: row.get(6)?,
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

    fn make_db(dir: &TempDir) -> (std::path::PathBuf, Connection) {
        let path = dir.path().join("state.db");
        let setup = Connection::open(&path).unwrap();
        setup
            .execute_batch(
                "PRAGMA journal_mode=WAL; PRAGMA synchronous=NORMAL;
                 CREATE TABLE scratchpad_mirror (
                     scratchpad_id TEXT PRIMARY KEY,
                     title         TEXT,
                     content       TEXT NOT NULL,
                     tags          TEXT NOT NULL DEFAULT '[]',
                     tenant_id     TEXT NOT NULL,
                     created_at    TEXT NOT NULL,
                     updated_at    TEXT NOT NULL
                 );",
            )
            .unwrap();
        drop(setup);
        let conn = Connection::open_with_flags(&path, OpenFlags::SQLITE_OPEN_READ_ONLY).unwrap();
        (path, conn)
    }

    fn insert_entry(
        path: &std::path::Path,
        id: &str,
        title: Option<&str>,
        content: &str,
        tags: &str,
        tenant_id: &str,
        updated_at: &str,
    ) {
        let setup = Connection::open(path).unwrap();
        setup
            .execute(
                "INSERT INTO scratchpad_mirror
                 (scratchpad_id, title, content, tags, tenant_id, created_at, updated_at)
                 VALUES (?1, ?2, ?3, ?4, ?5, '2024-01-01T00:00:00Z', ?6)",
                params![id, title, content, tags, tenant_id, updated_at],
            )
            .unwrap();
    }

    #[test]
    fn none_connection_returns_empty() {
        assert!(list_scratchpad(None, None, 100).is_empty());
    }

    #[test]
    fn empty_table_returns_empty() {
        let dir = TempDir::new().unwrap();
        let (_, conn) = make_db(&dir);
        assert!(list_scratchpad(Some(&conn), None, 100).is_empty());
    }

    #[test]
    fn lists_all_entries_no_filter() {
        let dir = TempDir::new().unwrap();
        let (path, conn) = make_db(&dir);
        drop(conn);
        insert_entry(
            &path,
            "s1",
            Some("Note A"),
            "content A",
            "[]",
            "tenant1",
            "2024-01-02T00:00:00Z",
        );
        insert_entry(
            &path,
            "s2",
            None,
            "content B",
            "[\"tag\"]",
            "tenant2",
            "2024-01-01T00:00:00Z",
        );

        let conn = Connection::open_with_flags(&path, OpenFlags::SQLITE_OPEN_READ_ONLY).unwrap();
        let result = list_scratchpad(Some(&conn), None, 100);
        assert_eq!(result.len(), 2);
        // ORDER BY updated_at DESC
        assert_eq!(result[0].scratchpad_id, "s1");
        assert_eq!(result[0].title, Some("Note A".to_string()));
        assert_eq!(result[1].title, None);
    }

    #[test]
    fn tenant_filter_works() {
        let dir = TempDir::new().unwrap();
        let (path, conn) = make_db(&dir);
        drop(conn);
        insert_entry(
            &path,
            "s1",
            None,
            "c",
            "[]",
            "tenant1",
            "2024-01-01T00:00:00Z",
        );
        insert_entry(
            &path,
            "s2",
            None,
            "c",
            "[]",
            "tenant2",
            "2024-01-01T00:00:00Z",
        );

        let conn = Connection::open_with_flags(&path, OpenFlags::SQLITE_OPEN_READ_ONLY).unwrap();
        let result = list_scratchpad(Some(&conn), Some("tenant1"), 100);
        assert_eq!(result.len(), 1);
        assert_eq!(result[0].scratchpad_id, "s1");
    }

    #[test]
    fn limit_is_respected() {
        let dir = TempDir::new().unwrap();
        let (path, conn) = make_db(&dir);
        drop(conn);
        for i in 0..5 {
            insert_entry(
                &path,
                &format!("s{i}"),
                None,
                "c",
                "[]",
                "tenant1",
                &format!("2024-01-0{}T00:00:00Z", i + 1),
            );
        }
        let conn = Connection::open_with_flags(&path, OpenFlags::SQLITE_OPEN_READ_ONLY).unwrap();
        let result = list_scratchpad(Some(&conn), None, 3);
        assert_eq!(result.len(), 3);
    }

    #[test]
    fn tags_field_preserved_as_json_string() {
        let dir = TempDir::new().unwrap();
        let (path, conn) = make_db(&dir);
        drop(conn);
        insert_entry(
            &path,
            "s1",
            None,
            "c",
            "[\"rust\",\"async\"]",
            "t1",
            "2024-01-01T00:00:00Z",
        );

        let conn = Connection::open_with_flags(&path, OpenFlags::SQLITE_OPEN_READ_ONLY).unwrap();
        let result = list_scratchpad(Some(&conn), None, 10);
        assert_eq!(result[0].tags, "[\"rust\",\"async\"]");
    }

    #[test]
    fn missing_table_returns_empty() {
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("empty.db");
        let setup = Connection::open(&path).unwrap();
        setup.execute_batch("PRAGMA journal_mode=WAL;").unwrap();
        drop(setup);
        let conn = Connection::open_with_flags(&path, OpenFlags::SQLITE_OPEN_READ_ONLY).unwrap();
        assert!(list_scratchpad(Some(&conn), None, 100).is_empty());
    }
}
