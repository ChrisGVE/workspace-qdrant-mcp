//! Query operations for `tracked_files`, `watch_folders` (submodules), and
//! `project_components`.
//!
//! SQL is verbatim from:
//!   - `tracked-files-queries/tracked-files.ts`
//!   - `tracked-files-queries/submodules.ts`
//!   - `tracked-files-queries/components.ts`

use rusqlite::{params, Connection};

use super::filters::{build_filter_clause, FilterParam, ListTrackedFilesOptions};

// ---------------------------------------------------------------------------
// Public types
// ---------------------------------------------------------------------------

/// A single row returned by `list_tracked_files`.
///
/// Mirrors `TrackedFileEntry` in `tracked-files-queries/tracked-files.ts`.
#[derive(Debug, Clone, PartialEq)]
pub struct TrackedFileEntry {
    pub relative_path: String,
    pub file_type: Option<String>,
    pub language: Option<String>,
    pub extension: Option<String>,
    pub is_test: bool,
}

/// A submodule row from `watch_folders`.
///
/// Mirrors `SubmoduleEntry` in `tracked-files-queries/submodules.ts`.
#[derive(Debug, Clone, PartialEq)]
pub struct SubmoduleEntry {
    pub submodule_path: String,
    pub repo_name: String,
}

/// A project component row from `project_components`.
///
/// Mirrors `ComponentEntry` in `tracked-files-queries/components.ts`.
#[derive(Debug, Clone, PartialEq)]
pub struct ComponentEntry {
    pub component_name: String,
    pub base_path: String,
    pub source: String,
}

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

fn is_no_such_table(e: &rusqlite::Error) -> bool {
    e.to_string().contains("no such table")
}

/// Extract repo name from a git remote URL.
///
/// Mirrors `extractRepoName` in `tracked-files-queries/submodules.ts:80-96`.
pub fn extract_repo_name(git_remote_url: Option<&str>, submodule_path: &str) -> String {
    if let Some(url) = git_remote_url {
        let cleaned = url.trim_end_matches('/').trim_end_matches(".git");
        if let Some(last) = cleaned.split('/').next_back() {
            // Handle git@host:user/repo format
            let colon_part = last.split(':').next_back().unwrap_or(last);
            if !colon_part.is_empty() {
                return colon_part.to_string();
            }
        }
    }
    // Fallback: last non-empty segment of submodule_path
    submodule_path
        .split('/')
        .rfind(|s| !s.is_empty())
        .unwrap_or(submodule_path)
        .to_string()
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/// List tracked files for a project with optional filtering.
///
/// SQL verbatim from `tracked-files-queries/tracked-files.ts:125-131`:
/// ```sql
/// SELECT relative_path, file_type, language, extension, is_test
/// FROM tracked_files
/// WHERE <conditions>
/// ORDER BY relative_path ASC
/// LIMIT ?
/// ```
pub fn list_tracked_files(
    conn: Option<&Connection>,
    options: &ListTrackedFilesOptions,
) -> Vec<TrackedFileEntry> {
    let Some(conn) = conn else {
        return Vec::new();
    };
    let clause = build_filter_clause(options);
    let limit = options.limit.unwrap_or(500) as i64;

    let where_str = clause.conditions.join(" AND ");
    let sql = format!(
        "SELECT relative_path, file_type, language, extension, is_test \
         FROM tracked_files \
         WHERE {where_str} \
         ORDER BY relative_path ASC \
         LIMIT ?"
    );

    let result: Result<Vec<TrackedFileEntry>, rusqlite::Error> = (|| {
        let mut stmt = conn.prepare(&sql)?;
        let mut all_params: Vec<FilterParam> = clause.params;
        all_params.push(FilterParam::Int(limit));
        let refs: Vec<&dyn rusqlite::types::ToSql> = all_params
            .iter()
            .map(|p| p as &dyn rusqlite::types::ToSql)
            .collect();
        let rows: Vec<TrackedFileEntry> = stmt
            .query_map(refs.as_slice(), |row| {
                let is_test_int: i64 = row.get(4)?;
                Ok(TrackedFileEntry {
                    relative_path: row.get(0)?,
                    file_type: row.get(1)?,
                    language: row.get(2)?,
                    extension: row.get(3)?,
                    is_test: is_test_int == 1,
                })
            })?
            .collect::<Result<_, _>>()?;
        Ok(rows)
    })();

    match result {
        Ok(v) => v,
        Err(e) if is_no_such_table(&e) => Vec::new(),
        Err(e) => {
            tracing::warn!("list_tracked_files failed: {e}");
            Vec::new()
        }
    }
}

/// Count tracked files matching the same filters (ignoring limit).
///
/// SQL verbatim from `tracked-files-queries/tracked-files.ts:158-163`:
/// ```sql
/// SELECT COUNT(*) as cnt FROM tracked_files WHERE <conditions>
/// ```
pub fn count_tracked_files(conn: Option<&Connection>, options: &ListTrackedFilesOptions) -> i64 {
    let Some(conn) = conn else {
        return 0;
    };
    let clause = build_filter_clause(options);
    let where_str = clause.conditions.join(" AND ");
    let sql = format!("SELECT COUNT(*) as cnt FROM tracked_files WHERE {where_str}");

    let result: Result<i64, rusqlite::Error> = (|| {
        let mut stmt = conn.prepare(&sql)?;
        let refs: Vec<&dyn rusqlite::types::ToSql> = clause
            .params
            .iter()
            .map(|p| p as &dyn rusqlite::types::ToSql)
            .collect();
        stmt.query_row(refs.as_slice(), |row| row.get(0))
    })();

    match result {
        Ok(n) => n,
        Err(e) if is_no_such_table(&e) => 0,
        Err(e) => {
            tracing::warn!("count_tracked_files failed: {e}");
            0
        }
    }
}

/// List submodules for a project.
///
/// SQL verbatim from `tracked-files-queries/submodules.ts:43-48`:
/// ```sql
/// SELECT submodule_path, git_remote_url
/// FROM watch_folders
/// WHERE parent_watch_id = ?
/// ORDER BY submodule_path ASC
/// ```
pub fn list_submodules(conn: Option<&Connection>, watch_folder_id: &str) -> Vec<SubmoduleEntry> {
    let Some(conn) = conn else {
        return Vec::new();
    };
    let result: Result<Vec<SubmoduleEntry>, rusqlite::Error> = (|| {
        let mut stmt = conn.prepare(
            "SELECT submodule_path, git_remote_url \
             FROM watch_folders \
             WHERE parent_watch_id = ? \
             ORDER BY submodule_path ASC",
        )?;
        let rows: Vec<SubmoduleEntry> = stmt
            .query_map(params![watch_folder_id], |row| {
                let submodule_path: Option<String> = row.get(0)?;
                let git_remote_url: Option<String> = row.get(1)?;
                Ok((submodule_path, git_remote_url))
            })?
            .filter_map(|r| r.ok())
            .filter_map(|(sp, url)| {
                let sp = sp?;
                let repo_name = extract_repo_name(url.as_deref(), &sp);
                Some(SubmoduleEntry {
                    submodule_path: sp,
                    repo_name,
                })
            })
            .collect();
        Ok(rows)
    })();

    match result {
        Ok(v) => v,
        Err(e) if is_no_such_table(&e) => Vec::new(),
        Err(e) => {
            tracing::warn!("list_submodules failed: {e}");
            Vec::new()
        }
    }
}

/// List project components from `project_components`.
///
/// SQL verbatim from `tracked-files-queries/components.ts:44-47`:
/// ```sql
/// SELECT component_name, base_path, source
/// FROM project_components
/// WHERE watch_folder_id = ?
/// ORDER BY component_name ASC
/// ```
pub fn list_project_components(
    conn: Option<&Connection>,
    watch_folder_id: &str,
) -> Vec<ComponentEntry> {
    let Some(conn) = conn else {
        return Vec::new();
    };
    let result: Result<Vec<ComponentEntry>, rusqlite::Error> = (|| {
        let mut stmt = conn.prepare(
            "SELECT component_name, base_path, source \
             FROM project_components \
             WHERE watch_folder_id = ? \
             ORDER BY component_name ASC",
        )?;
        let rows: Vec<ComponentEntry> = stmt
            .query_map(params![watch_folder_id], |row| {
                Ok(ComponentEntry {
                    component_name: row.get(0)?,
                    base_path: row.get(1)?,
                    source: row.get(2)?,
                })
            })?
            .collect::<Result<_, _>>()?;
        Ok(rows)
    })();

    match result {
        Ok(v) => v,
        Err(e) if is_no_such_table(&e) => Vec::new(),
        Err(e) => {
            tracing::warn!("list_project_components failed: {e}");
            Vec::new()
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
#[path = "queries_tests.rs"]
mod tests;
