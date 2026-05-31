//! Project queries — reads from `watch_folders`.
//!
//! SQL is verbatim from `project-queries.ts` and `instance-queries.ts`.
//!
//! Schema note: PK is `watch_id` (NOT `id`). The join column is
//! `watch_folder_id = watch_id` — see the memory note about bug #76.

use rusqlite::params;
use rusqlite::Connection;

use wqm_common::constants::COLLECTION_PROJECTS;

// ---------------------------------------------------------------------------
// Public types
// ---------------------------------------------------------------------------

/// A registered project row from `watch_folders`.
///
/// Mirrors `RegisteredProject` in `types/state.ts`.
#[derive(Debug, Clone, PartialEq)]
pub struct RegisteredProject {
    pub project_id: String,
    pub project_path: String,
    pub git_remote_url: Option<String>,
    pub remote_hash: Option<String>,
    pub disambiguation_path: Option<String>,
    /// Last path segment of `project_path`.
    pub container_folder: String,
    pub is_active: bool,
    pub created_at: String,
    pub last_seen_at: Option<String>,
    pub last_activity_at: Option<String>,
}

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

const SELECT_FIELDS: &str = "SELECT tenant_id, path, git_remote_url, remote_hash, \
            disambiguation_path, is_active, created_at, updated_at, last_activity_at \
     FROM watch_folders";

fn map_row(row: &rusqlite::Row<'_>) -> rusqlite::Result<RegisteredProject> {
    let path: String = row.get(1)?;
    let container_folder = path
        .split('/')
        .filter(|s| !s.is_empty())
        .last()
        .unwrap_or(&path)
        .to_string();
    let is_active_int: i64 = row.get(5)?;
    Ok(RegisteredProject {
        project_id: row.get(0)?,
        project_path: path,
        git_remote_url: row.get(2)?,
        remote_hash: row.get(3)?,
        disambiguation_path: row.get(4)?,
        container_folder,
        is_active: is_active_int == 1,
        created_at: row.get(6)?,
        last_seen_at: row.get(7)?,
        last_activity_at: row.get(8)?,
    })
}

fn is_no_such_table(e: &rusqlite::Error) -> bool {
    e.to_string().contains("no such table")
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/// Get a project by filesystem path using longest-prefix matching.
///
/// SQL verbatim from `project-queries.ts:83-88`:
/// ```sql
/// SELECT … FROM watch_folders
/// WHERE collection = ? AND (? = path OR ? LIKE path || '/' || '%')
/// ORDER BY length(path) DESC
/// LIMIT 1
/// ```
pub fn get_project_by_path(
    conn: Option<&Connection>,
    project_path: &str,
) -> Option<RegisteredProject> {
    let conn = conn?;
    let sql = format!(
        "{SELECT_FIELDS} \
         WHERE collection = ? AND (? = path OR ? LIKE path || '/' || '%') \
         ORDER BY length(path) DESC \
         LIMIT 1"
    );
    match conn.query_row(
        &sql,
        params![COLLECTION_PROJECTS, project_path, project_path],
        map_row,
    ) {
        Ok(row) => Some(row),
        Err(rusqlite::Error::QueryReturnedNoRows) => None,
        Err(e) if is_no_such_table(&e) => None,
        Err(e) => {
            tracing::warn!("get_project_by_path failed: {e}");
            None
        }
    }
}

/// Get a project by its `tenant_id`.
///
/// SQL verbatim from `project-queries.ts:92-97`:
/// ```sql
/// SELECT … FROM watch_folders WHERE tenant_id = ? AND collection = ?
/// ```
pub fn get_project_by_id(conn: Option<&Connection>, project_id: &str) -> Option<RegisteredProject> {
    let conn = conn?;
    let sql = format!("{SELECT_FIELDS} WHERE tenant_id = ? AND collection = ?");
    match conn.query_row(&sql, params![project_id, COLLECTION_PROJECTS], map_row) {
        Ok(row) => Some(row),
        Err(rusqlite::Error::QueryReturnedNoRows) => None,
        Err(e) if is_no_such_table(&e) => None,
        Err(e) => {
            tracing::warn!("get_project_by_id failed: {e}");
            None
        }
    }
}

/// List all active projects ordered by `last_activity_at DESC`.
///
/// SQL verbatim from `project-queries.ts:152-157`:
/// ```sql
/// SELECT … FROM watch_folders
/// WHERE is_active = 1 AND collection = ?
/// ORDER BY last_activity_at DESC
/// ```
pub fn list_active_projects(conn: Option<&Connection>) -> Vec<RegisteredProject> {
    let Some(conn) = conn else {
        return Vec::new();
    };
    let sql = format!(
        "{SELECT_FIELDS} WHERE is_active = 1 AND collection = ? ORDER BY last_activity_at DESC"
    );
    let result: Result<Vec<RegisteredProject>, rusqlite::Error> = (|| {
        let mut stmt = conn.prepare(&sql)?;
        let rows: Vec<RegisteredProject> = stmt
            .query_map(params![COLLECTION_PROJECTS], map_row)?
            .collect::<Result<_, _>>()?;
        Ok(rows)
    })();
    match result {
        Ok(rows) => rows,
        Err(e) if is_no_such_table(&e) => Vec::new(),
        Err(e) => {
            tracing::warn!("list_active_projects failed: {e}");
            Vec::new()
        }
    }
}

/// Get the `watch_id` for a project by its `tenant_id`.
///
/// SQL verbatim from `instance-queries.ts:21-25`:
/// ```sql
/// SELECT watch_id FROM watch_folders
/// WHERE tenant_id = ? AND collection = 'projects' AND parent_watch_id IS NULL
/// LIMIT 1
/// ```
pub fn get_watch_folder_id_by_tenant(conn: Option<&Connection>, tenant_id: &str) -> Option<String> {
    let conn = conn?;
    match conn.query_row(
        "SELECT watch_id FROM watch_folders \
         WHERE tenant_id = ? AND collection = 'projects' AND parent_watch_id IS NULL \
         LIMIT 1",
        params![tenant_id],
        |row| row.get(0),
    ) {
        Ok(id) => Some(id),
        Err(rusqlite::Error::QueryReturnedNoRows) => None,
        Err(e) if is_no_such_table(&e) => None,
        Err(e) => {
            tracing::warn!("get_watch_folder_id_by_tenant failed: {e}");
            None
        }
    }
}

/// Get distinct `base_point` values for files under a watch folder.
///
/// SQL verbatim from `instance-queries.ts:49-67`.
pub fn get_active_base_points(
    conn: Option<&Connection>,
    watch_folder_id: &str,
    include_submodules: bool,
) -> Vec<String> {
    let Some(conn) = conn else {
        return Vec::new();
    };
    let result: Result<Vec<String>, rusqlite::Error> = if include_submodules {
        (|| {
            let mut stmt = conn.prepare(
                "SELECT DISTINCT base_point FROM tracked_files \
                 WHERE base_point IS NOT NULL AND ( \
                     watch_folder_id = ? \
                     OR watch_folder_id IN ( \
                         SELECT child_watch_id FROM watch_folder_submodules \
                         WHERE parent_watch_id = ? \
                     ) \
                 )",
            )?;
            let rows: Vec<String> = stmt
                .query_map(params![watch_folder_id, watch_folder_id], |row| {
                    row.get::<_, String>(0)
                })?
                .collect::<Result<_, _>>()?;
            Ok(rows)
        })()
    } else {
        (|| {
            let mut stmt = conn.prepare(
                "SELECT DISTINCT base_point FROM tracked_files \
                 WHERE base_point IS NOT NULL AND watch_folder_id = ?",
            )?;
            let rows: Vec<String> = stmt
                .query_map(params![watch_folder_id], |row| row.get::<_, String>(0))?
                .collect::<Result<_, _>>()?;
            Ok(rows)
        })()
    };
    match result {
        Ok(v) => v,
        Err(e) if is_no_such_table(&e) => Vec::new(),
        Err(e) => {
            tracing::warn!("get_active_base_points failed: {e}");
            Vec::new()
        }
    }
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
                 CREATE TABLE watch_folders (
                     watch_id            TEXT PRIMARY KEY,
                     tenant_id           TEXT NOT NULL,
                     path                TEXT NOT NULL,
                     collection          TEXT NOT NULL,
                     git_remote_url      TEXT,
                     remote_hash         TEXT,
                     disambiguation_path TEXT,
                     is_active           INTEGER NOT NULL DEFAULT 1,
                     parent_watch_id     TEXT,
                     submodule_path      TEXT,
                     created_at          TEXT NOT NULL DEFAULT (datetime('now')),
                     updated_at          TEXT,
                     last_activity_at    TEXT
                 );
                 CREATE TABLE tracked_files (
                     file_id         TEXT PRIMARY KEY,
                     watch_folder_id TEXT NOT NULL,
                     base_point      TEXT,
                     relative_path   TEXT NOT NULL,
                     file_type       TEXT,
                     language        TEXT,
                     extension       TEXT,
                     is_test         INTEGER NOT NULL DEFAULT 0,
                     branches        TEXT NOT NULL DEFAULT '[]',
                     component       TEXT
                 );",
            )
            .unwrap();
        drop(setup);
        let conn = Connection::open_with_flags(&path, OpenFlags::SQLITE_OPEN_READ_ONLY).unwrap();
        (path, conn)
    }

    fn insert_project(
        path: &std::path::Path,
        watch_id: &str,
        tenant_id: &str,
        proj_path: &str,
        is_active: i64,
        last_activity_at: Option<&str>,
    ) {
        let setup = Connection::open(path).unwrap();
        setup
            .execute(
                "INSERT INTO watch_folders
                 (watch_id, tenant_id, path, collection, is_active, last_activity_at)
                 VALUES (?1, ?2, ?3, 'projects', ?4, ?5)",
                params![watch_id, tenant_id, proj_path, is_active, last_activity_at],
            )
            .unwrap();
    }

    #[test]
    fn get_project_by_path_exact_match() {
        let dir = TempDir::new().unwrap();
        let (path, conn) = make_db(&dir);
        drop(conn);
        insert_project(&path, "w1", "t1", "/home/user/proj", 1, None);

        let conn = Connection::open_with_flags(&path, OpenFlags::SQLITE_OPEN_READ_ONLY).unwrap();
        let result = get_project_by_path(Some(&conn), "/home/user/proj");
        assert!(result.is_some());
        assert_eq!(result.unwrap().project_id, "t1");
    }

    #[test]
    fn get_project_by_path_prefix_match() {
        let dir = TempDir::new().unwrap();
        let (path, conn) = make_db(&dir);
        drop(conn);
        insert_project(&path, "w1", "t1", "/home/user/proj", 1, None);

        let conn = Connection::open_with_flags(&path, OpenFlags::SQLITE_OPEN_READ_ONLY).unwrap();
        let result = get_project_by_path(Some(&conn), "/home/user/proj/src/main.rs");
        assert!(result.is_some());
    }

    #[test]
    fn get_project_by_path_no_match() {
        let dir = TempDir::new().unwrap();
        let (path, conn) = make_db(&dir);
        drop(conn);
        let conn = Connection::open_with_flags(&path, OpenFlags::SQLITE_OPEN_READ_ONLY).unwrap();
        let result = get_project_by_path(Some(&conn), "/nonexistent");
        assert!(result.is_none());
    }

    #[test]
    fn get_project_by_id_found() {
        let dir = TempDir::new().unwrap();
        let (path, conn) = make_db(&dir);
        drop(conn);
        insert_project(&path, "w1", "tenant-abc", "/proj", 1, None);

        let conn = Connection::open_with_flags(&path, OpenFlags::SQLITE_OPEN_READ_ONLY).unwrap();
        let result = get_project_by_id(Some(&conn), "tenant-abc");
        assert_eq!(result.unwrap().project_path, "/proj");
    }

    #[test]
    fn list_active_projects_filters_inactive() {
        let dir = TempDir::new().unwrap();
        let (path, conn) = make_db(&dir);
        drop(conn);
        insert_project(
            &path,
            "w1",
            "t1",
            "/active",
            1,
            Some("2024-01-02T00:00:00Z"),
        );
        insert_project(
            &path,
            "w2",
            "t2",
            "/inactive",
            0,
            Some("2024-01-01T00:00:00Z"),
        );

        let conn = Connection::open_with_flags(&path, OpenFlags::SQLITE_OPEN_READ_ONLY).unwrap();
        let result = list_active_projects(Some(&conn));
        assert_eq!(result.len(), 1);
        assert_eq!(result[0].project_path, "/active");
    }

    #[test]
    fn get_watch_folder_id_by_tenant_found() {
        let dir = TempDir::new().unwrap();
        let (path, conn) = make_db(&dir);
        drop(conn);
        insert_project(&path, "wid-xyz", "tenant-xyz", "/p", 1, None);

        let conn = Connection::open_with_flags(&path, OpenFlags::SQLITE_OPEN_READ_ONLY).unwrap();
        let result = get_watch_folder_id_by_tenant(Some(&conn), "tenant-xyz");
        assert_eq!(result, Some("wid-xyz".to_string()));
    }

    #[test]
    fn container_folder_extracted_correctly() {
        let dir = TempDir::new().unwrap();
        let (path, conn) = make_db(&dir);
        drop(conn);
        insert_project(&path, "w1", "t1", "/home/user/my-project", 1, None);

        let conn = Connection::open_with_flags(&path, OpenFlags::SQLITE_OPEN_READ_ONLY).unwrap();
        let result = get_project_by_id(Some(&conn), "t1").unwrap();
        assert_eq!(result.container_folder, "my-project");
    }

    #[test]
    fn none_conn_returns_none_or_empty() {
        assert!(get_project_by_path(None, "/foo").is_none());
        assert!(get_project_by_id(None, "x").is_none());
        assert!(list_active_projects(None).is_empty());
        assert!(get_watch_folder_id_by_tenant(None, "x").is_none());
        assert!(get_active_base_points(None, "x", false).is_empty());
    }
}
