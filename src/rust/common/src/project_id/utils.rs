//! Free utility functions for project ID resolution

use std::path::Path;
use super::calculator::ProjectIdCalculator;

/// Detect git remote URL for a project using `git` CLI
///
/// Tries `origin` first, falls back to `upstream`. Returns `None` on failure.
pub fn detect_git_remote(project_root: &Path) -> Option<String> {
    for remote_name in &["origin", "upstream"] {
        if let Ok(output) = std::process::Command::new("git")
            .args(["-C", &project_root.to_string_lossy(), "remote", "get-url", remote_name])
            .output()
        {
            if output.status.success() {
                let url = String::from_utf8_lossy(&output.stdout).trim().to_string();
                if !url.is_empty() {
                    return Some(url);
                }
            }
        }
    }
    None
}

/// Calculate tenant ID for a project path (convenience function)
///
/// Combines `detect_git_remote()` + `ProjectIdCalculator::new().calculate()`.
pub fn calculate_tenant_id(project_root: &Path) -> String {
    let git_remote = detect_git_remote(project_root);
    let calculator = ProjectIdCalculator::new();
    calculator.calculate(project_root, git_remote.as_deref(), None)
}

/// Resolve a working directory to a registered project.
///
/// Looks up the `watch_folders` table for the longest matching path where
/// `cwd` equals or is a subdirectory of a registered project. Returns
/// `(tenant_id, path)` on success, or `None` if no match or on any error.
///
/// Opens the database read-only. Any failure (missing db, missing table,
/// query error) returns `None` silently — callers should degrade gracefully.
#[cfg(feature = "sqlite")]
pub fn resolve_path_to_project(db_path: &Path, cwd: &Path) -> Option<(String, String)> {
    use crate::schema::sqlite::watch_folders as wf;

    let cwd_str = cwd.to_str()?;

    let conn = rusqlite::Connection::open_with_flags(
        db_path,
        rusqlite::OpenFlags::SQLITE_OPEN_READ_ONLY | rusqlite::OpenFlags::SQLITE_OPEN_NO_MUTEX,
    )
    .ok()?;

    let sql = format!(
        "SELECT {tenant}, {path} FROM {table} \
         WHERE {collection} = 'projects' \
           AND (?1 = {path} OR ?1 LIKE {path} || '/' || '%') \
         ORDER BY length({path}) DESC \
         LIMIT 1",
        tenant = wf::TENANT_ID.name,
        path = wf::PATH.name,
        table = wf::TABLE.name,
        collection = wf::COLLECTION.name,
    );

    let mut stmt = conn.prepare(&sql).ok()?;
    stmt.query_row(rusqlite::params![cwd_str], |row| {
        Ok((row.get::<_, String>(0)?, row.get::<_, String>(1)?))
    })
    .ok()
}
