//! Rules mirror read queries — direct SQLite reads from `rules_mirror`.
//!
//! SQL is verbatim from `rules-mirror-queries.ts:81-103`.
//!
//! Schema note: `rules_mirror` columns are:
//!   rule_id, rule_text, scope, tenant_id, created_at, updated_at
//!
//! There is NO `priority`, `label`, `title`, or `tags` column (the task text
//! mentions ORDER BY priority — that is WRONG for this schema; the actual TS
//! uses `ORDER BY updatedAt DESC`).

use rusqlite::{params, Connection};

use wqm_common::constants::TENANT_GLOBAL;

use crate::sqlite::manager::StateManager;
use crate::tools::rules::RulesReader;

// ---------------------------------------------------------------------------
// Public types
// ---------------------------------------------------------------------------

/// A single row from the `rules_mirror` table.
///
/// Mirrors `RulesMirrorEntry` in `rules-mirror-queries.ts`.
#[derive(Debug, Clone, PartialEq)]
pub struct RulesMirrorEntry {
    pub rule_id: String,
    pub rule_text: String,
    pub scope: Option<String>,
    pub tenant_id: Option<String>,
    pub created_at: String,
    pub updated_at: String,
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/// List rules from `rules_mirror`, optionally filtered by scope / tenant.
///
/// Matches the filter logic in `rules-mirror-queries.ts:88-98`:
/// - `scope == "global"` → `(scope = 'global' OR scope IS NULL)`
/// - `scope == "project"` AND `tenant_id` is Some → `scope = 'project' AND tenant_id = ?`
/// - no filter → return all rows
///
/// Always appends `ORDER BY updated_at DESC LIMIT ?`.
/// Returns an empty `Vec` when the table does not exist.
pub fn list_rules(
    conn: Option<&Connection>,
    scope: Option<&str>,
    tenant_id: Option<&str>,
    limit: usize,
) -> Vec<RulesMirrorEntry> {
    let Some(conn) = conn else {
        return Vec::new();
    };

    let sql = build_list_sql(scope, tenant_id);

    let result: Result<Vec<RulesMirrorEntry>, rusqlite::Error> = (|| {
        let mut stmt = conn.prepare(&sql)?;
        let limit_i64 = limit as i64;

        let rows = match (scope, tenant_id) {
            (Some("project"), Some(tid)) => stmt.query_map(params![tid, limit_i64], map_row)?,
            _ => stmt.query_map(params![limit_i64], map_row)?,
        };

        let entries: Vec<RulesMirrorEntry> = rows.collect::<Result<_, _>>()?;
        Ok(entries)
    })();

    match result {
        Ok(entries) => entries,
        Err(e) if e.to_string().contains("no such table") => Vec::new(),
        Err(e) => {
            tracing::warn!("list_rules query failed: {e}");
            Vec::new()
        }
    }
}

// ---------------------------------------------------------------------------
// RulesReader impl for StateManager
// ---------------------------------------------------------------------------

/// Implement [`RulesReader`] for [`StateManager`] so the dispatcher can pass
/// the shared SQLite handle directly to `rules_tool` without an extra wrapper.
impl RulesReader for StateManager {
    fn list_from_mirror(
        &self,
        scope: Option<&str>,
        tenant_id: Option<&str>,
        limit: usize,
    ) -> Vec<RulesMirrorEntry> {
        list_rules(self.connection(), scope, tenant_id, limit)
    }
}

/// Implement [`RulesReader`] for [`crate::sqlite::SharedStateManager`].
///
/// `SharedStateManager` wraps `StateManager` in a `std::sync::Mutex`,
/// making it `Send + Sync`.  Locking here is always synchronous (no await
/// held), so there is no deadlock risk.
impl RulesReader for crate::sqlite::SharedStateManager {
    fn list_from_mirror(
        &self,
        scope: Option<&str>,
        tenant_id: Option<&str>,
        limit: usize,
    ) -> Vec<RulesMirrorEntry> {
        let guard = self.lock();
        list_rules(guard.connection(), scope, tenant_id, limit)
    }
}

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

fn build_list_sql(scope: Option<&str>, tenant_id: Option<&str>) -> String {
    let where_clause = match (scope, tenant_id) {
        (Some(s), _) if s == TENANT_GLOBAL => {
            " WHERE (scope = 'global' OR scope IS NULL)".to_string()
        }
        (Some("project"), Some(_)) => " WHERE scope = 'project' AND tenant_id = ?".to_string(),
        _ => String::new(),
    };

    format!(
        "SELECT rule_id, rule_text, scope, tenant_id, created_at, updated_at \
         FROM rules_mirror{where_clause} \
         ORDER BY updated_at DESC LIMIT ?",
    )
}

fn map_row(row: &rusqlite::Row<'_>) -> rusqlite::Result<RulesMirrorEntry> {
    Ok(RulesMirrorEntry {
        rule_id: row.get(0)?,
        rule_text: row.get(1)?,
        scope: row.get(2)?,
        tenant_id: row.get(3)?,
        created_at: row.get(4)?,
        updated_at: row.get(5)?,
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
                 CREATE TABLE rules_mirror (
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
        let conn = Connection::open_with_flags(&path, OpenFlags::SQLITE_OPEN_READ_ONLY).unwrap();
        (path, conn)
    }

    fn insert_rule(
        path: &std::path::Path,
        id: &str,
        text: &str,
        scope: Option<&str>,
        tenant: Option<&str>,
        updated_at: &str,
    ) {
        let setup = Connection::open(path).unwrap();
        setup
            .execute(
                "INSERT INTO rules_mirror (rule_id, rule_text, scope, tenant_id, created_at, updated_at)
                 VALUES (?1, ?2, ?3, ?4, '2024-01-01T00:00:00Z', ?5)",
                params![id, text, scope, tenant, updated_at],
            )
            .unwrap();
    }

    #[test]
    fn none_connection_returns_empty() {
        let result = list_rules(None, None, None, 50);
        assert!(result.is_empty());
    }

    #[test]
    fn empty_table_returns_empty() {
        let dir = TempDir::new().unwrap();
        let (_, conn) = make_db(&dir);
        let result = list_rules(Some(&conn), None, None, 50);
        assert!(result.is_empty());
    }

    #[test]
    fn lists_all_rules_no_filter() {
        let dir = TempDir::new().unwrap();
        let (path, conn) = make_db(&dir);
        drop(conn);
        insert_rule(
            &path,
            "r1",
            "rule one",
            Some("global"),
            None,
            "2024-01-02T00:00:00Z",
        );
        insert_rule(
            &path,
            "r2",
            "rule two",
            Some("project"),
            Some("proj1"),
            "2024-01-01T00:00:00Z",
        );

        let conn = Connection::open_with_flags(&path, OpenFlags::SQLITE_OPEN_READ_ONLY).unwrap();
        let result = list_rules(Some(&conn), None, None, 50);
        assert_eq!(result.len(), 2);
        // ORDER BY updated_at DESC: r1 should be first
        assert_eq!(result[0].rule_id, "r1");
    }

    #[test]
    fn global_scope_filter() {
        let dir = TempDir::new().unwrap();
        let (path, conn) = make_db(&dir);
        drop(conn);
        insert_rule(
            &path,
            "r1",
            "global rule",
            Some("global"),
            None,
            "2024-01-01T00:00:00Z",
        );
        insert_rule(
            &path,
            "r2",
            "null scope",
            None,
            None,
            "2024-01-01T00:00:00Z",
        );
        insert_rule(
            &path,
            "r3",
            "project rule",
            Some("project"),
            Some("p1"),
            "2024-01-01T00:00:00Z",
        );

        let conn = Connection::open_with_flags(&path, OpenFlags::SQLITE_OPEN_READ_ONLY).unwrap();
        let result = list_rules(Some(&conn), Some("global"), None, 50);
        // global scope matches scope='global' OR scope IS NULL → r1, r2
        assert_eq!(result.len(), 2);
        let ids: Vec<_> = result.iter().map(|r| r.rule_id.as_str()).collect();
        assert!(ids.contains(&"r1"));
        assert!(ids.contains(&"r2"));
    }

    #[test]
    fn project_scope_filter_by_tenant() {
        let dir = TempDir::new().unwrap();
        let (path, conn) = make_db(&dir);
        drop(conn);
        insert_rule(
            &path,
            "r1",
            "proj1 rule",
            Some("project"),
            Some("proj1"),
            "2024-01-01T00:00:00Z",
        );
        insert_rule(
            &path,
            "r2",
            "proj2 rule",
            Some("project"),
            Some("proj2"),
            "2024-01-01T00:00:00Z",
        );

        let conn = Connection::open_with_flags(&path, OpenFlags::SQLITE_OPEN_READ_ONLY).unwrap();
        let result = list_rules(Some(&conn), Some("project"), Some("proj1"), 50);
        assert_eq!(result.len(), 1);
        assert_eq!(result[0].rule_id, "r1");
    }

    #[test]
    fn limit_is_respected() {
        let dir = TempDir::new().unwrap();
        let (path, conn) = make_db(&dir);
        drop(conn);
        for i in 0..5 {
            insert_rule(
                &path,
                &format!("r{i}"),
                "text",
                None,
                None,
                &format!("2024-01-0{} 00:00:00", i + 1),
            );
        }
        let conn = Connection::open_with_flags(&path, OpenFlags::SQLITE_OPEN_READ_ONLY).unwrap();
        let result = list_rules(Some(&conn), None, None, 3);
        assert_eq!(result.len(), 3);
    }

    #[test]
    fn missing_table_returns_empty() {
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("empty.db");
        let setup = Connection::open(&path).unwrap();
        setup.execute_batch("PRAGMA journal_mode=WAL;").unwrap();
        drop(setup);
        let conn = Connection::open_with_flags(&path, OpenFlags::SQLITE_OPEN_READ_ONLY).unwrap();
        let result = list_rules(Some(&conn), None, None, 50);
        assert!(result.is_empty());
    }
}
