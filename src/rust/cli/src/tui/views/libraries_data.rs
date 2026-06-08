//! Data types and SQLite fetching logic for the TUI library browser.
//!
//! Separated from the view module to keep both files under the 500-line limit
//! and to allow unit-testing data logic independently from rendering.

use std::collections::HashMap;

use crate::data::db::connect_readonly;

/// A single library row ready for display in the TUI list.
#[derive(Debug, Clone)]
pub struct LibraryRow {
    /// Watch folder ID (e.g., "lib-rust-docs").
    pub watch_id: String,
    /// Library tag/name (tenant_id).
    pub tag: String,
    /// Human-readable display name: the path's base folder, disambiguated as
    /// `parent/base` when two libraries share a base folder name.
    pub name: String,
    /// Path with home directory replaced by `~`.
    pub display_path: String,
    /// Whether the library watch is enabled.
    pub enabled: bool,
    /// Whether the library is actively being watched (is_active > 0).
    pub is_active: bool,
    /// Library mode: "sync" or "incremental".
    pub mode: String,
    /// Number of tracked documents for this library.
    pub doc_count: u64,
    /// Source marker for a library nested under a project: `P:<project-name>`.
    /// `None` for a top-level library.
    pub source: Option<String>,
}

/// Full detail of a single library for the popup view.
#[derive(Debug, Clone)]
pub struct LibraryDetail {
    pub watch_id: String,
    pub tag: String,
    pub display_path: String,
    pub enabled: bool,
    pub is_active: bool,
    pub mode: String,
    pub doc_count: u64,
    pub follow_symlinks: bool,
    pub cleanup_on_disable: bool,
    pub is_paused: bool,
    pub is_archived: bool,
    pub created_at: String,
    pub updated_at: String,
    pub last_scan: Option<String>,
    pub last_activity_at: Option<String>,
}

/// Fetch all library rows from SQLite.
pub fn fetch_library_rows() -> Vec<LibraryRow> {
    let conn = match connect_readonly() {
        Ok(c) => c,
        Err(_) => return Vec::new(),
    };

    let home_dir = home_prefix();

    let Ok(mut stmt) = conn.prepare(
        "SELECT wf.watch_id, wf.tenant_id, wf.path, wf.enabled, wf.is_active, \
         COALESCE(wf.library_mode, 'incremental'), \
         COALESCE(tf_count.cnt, 0), \
         parent.path, parent.collection \
         FROM watch_folders wf \
         LEFT JOIN ( \
             SELECT watch_folder_id, COUNT(*) AS cnt FROM tracked_files GROUP BY watch_folder_id \
         ) tf_count ON tf_count.watch_folder_id = wf.watch_id \
         LEFT JOIN watch_folders parent ON parent.watch_id = wf.parent_watch_id \
         WHERE wf.collection = 'libraries' \
         ORDER BY wf.tenant_id",
    ) else {
        return Vec::new();
    };

    // First pass: collect raw rows so we can disambiguate display names against
    // the full set of library paths before building the final rows.
    struct Raw {
        watch_id: String,
        tag: String,
        path: String,
        enabled: bool,
        is_active: bool,
        mode: String,
        doc_count: u64,
        parent_path: Option<String>,
        parent_collection: Option<String>,
    }

    let Ok(rows) = stmt.query_map([], |row| {
        let is_active_val: i64 = row.get(4)?;
        Ok(Raw {
            watch_id: row.get(0)?,
            tag: row.get(1)?,
            path: row.get(2)?,
            enabled: row.get(3)?,
            is_active: is_active_val > 0,
            mode: row.get(5)?,
            doc_count: row.get::<_, i64>(6).unwrap_or(0) as u64,
            parent_path: row.get(7)?,
            parent_collection: row.get(8)?,
        })
    }) else {
        return Vec::new();
    };

    let raws: Vec<Raw> = rows.flatten().collect();
    let names = library_display_names(&raws.iter().map(|r| r.path.clone()).collect::<Vec<_>>());

    raws.into_iter()
        .zip(names)
        .map(|(r, name)| LibraryRow {
            watch_id: r.watch_id,
            tag: r.tag,
            name,
            display_path: abbreviate_home(&r.path, &home_dir),
            enabled: r.enabled,
            is_active: r.is_active,
            mode: r.mode,
            doc_count: r.doc_count,
            source: project_source(&r.parent_path, &r.parent_collection),
        })
        .collect()
}

/// Build the `P:<project-name>` source marker for a library nested under a
/// project. Returns `None` for a top-level library or one whose parent is
/// itself a library.
fn project_source(
    parent_path: &Option<String>,
    parent_collection: &Option<String>,
) -> Option<String> {
    let path = parent_path.as_deref()?;
    // Only a non-library parent (i.e. a project) is marked.
    if parent_collection.as_deref() == Some("libraries") {
        return None;
    }
    let base = path
        .trim_end_matches('/')
        .rsplit('/')
        .find(|s| !s.is_empty())
        .unwrap_or(path);
    Some(format!("P:{base}"))
}

/// Split a path into its (optional parent component, base component).
fn name_parts(path: &str) -> (Option<String>, String) {
    let trimmed = path.trim_end_matches('/');
    let mut comps = trimmed.rsplit('/').filter(|s| !s.is_empty());
    let base = comps.next().unwrap_or(trimmed).to_string();
    let parent = comps.next().map(|s| s.to_string());
    (parent, base)
}

/// Compute display names for a set of library paths: each path's base folder,
/// disambiguated as `parent/base` when two or more paths share a base name.
pub fn library_display_names(paths: &[String]) -> Vec<String> {
    let parts: Vec<(Option<String>, String)> = paths.iter().map(|p| name_parts(p)).collect();
    let mut counts: HashMap<&str, usize> = HashMap::new();
    for (_, base) in &parts {
        *counts.entry(base.as_str()).or_default() += 1;
    }
    parts
        .iter()
        .map(|(parent, base)| {
            if counts.get(base.as_str()).copied().unwrap_or(0) > 1 {
                match parent {
                    Some(p) => format!("{p}/{base}"),
                    None => base.clone(),
                }
            } else {
                base.clone()
            }
        })
        .collect()
}

/// Fetch full detail for a single library by watch_id.
pub fn fetch_library_detail(watch_id: &str) -> Option<LibraryDetail> {
    let conn = connect_readonly().ok()?;
    let home_dir = home_prefix();

    let mut stmt = conn
        .prepare(
            "SELECT wf.watch_id, wf.tenant_id, wf.path, wf.enabled, wf.is_active, \
             COALESCE(wf.library_mode, 'incremental'), \
             wf.follow_symlinks, wf.cleanup_on_disable, wf.is_paused, wf.is_archived, \
             wf.created_at, wf.updated_at, wf.last_scan, wf.last_activity_at, \
             COALESCE(tf_count.cnt, 0) \
             FROM watch_folders wf \
             LEFT JOIN ( \
                 SELECT watch_folder_id, COUNT(*) AS cnt \
                 FROM tracked_files GROUP BY watch_folder_id \
             ) tf_count ON tf_count.watch_folder_id = wf.watch_id \
             WHERE wf.watch_id = ?1 AND wf.collection = 'libraries'",
        )
        .ok()?;

    stmt.query_row(rusqlite::params![watch_id], |row| {
        let path: String = row.get(2)?;
        let is_active_val: i64 = row.get(4)?;
        Ok(LibraryDetail {
            watch_id: row.get(0)?,
            tag: row.get(1)?,
            display_path: abbreviate_home(&path, &home_dir),
            enabled: row.get(3)?,
            is_active: is_active_val > 0,
            mode: row.get(5)?,
            follow_symlinks: row.get(6)?,
            cleanup_on_disable: row.get(7)?,
            is_paused: row.get(8)?,
            is_archived: row.get(9)?,
            created_at: row.get(10)?,
            updated_at: row.get(11)?,
            last_scan: row.get(12)?,
            last_activity_at: row.get(13)?,
            doc_count: row.get::<_, i64>(14).unwrap_or(0) as u64,
        })
    })
    .ok()
}

/// Get the home directory prefix for path abbreviation.
fn home_prefix() -> Option<String> {
    dirs::home_dir().map(|p| p.to_string_lossy().to_string())
}

/// Replace the home directory prefix with `~` for compact display.
fn abbreviate_home(path: &str, home: &Option<String>) -> String {
    if let Some(ref h) = home {
        if path.starts_with(h.as_str()) {
            return format!("~{}", &path[h.len()..]);
        }
    }
    path.to_string()
}

/// Format a library status as a human-readable label.
pub fn status_label(enabled: bool, is_active: bool) -> &'static str {
    if !enabled {
        "disabled"
    } else if is_active {
        "watching"
    } else {
        "stopped"
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn abbreviate_home_replaces_prefix() {
        let home = Some("/Users/alice".to_string());
        assert_eq!(
            abbreviate_home("/Users/alice/docs/lib", &home),
            "~/docs/lib"
        );
    }

    #[test]
    fn abbreviate_home_no_match() {
        let home = Some("/Users/alice".to_string());
        assert_eq!(abbreviate_home("/opt/data/lib", &home), "/opt/data/lib");
    }

    #[test]
    fn abbreviate_home_none() {
        assert_eq!(abbreviate_home("/some/path", &None), "/some/path");
    }

    #[test]
    fn status_label_disabled() {
        assert_eq!(status_label(false, false), "disabled");
        assert_eq!(status_label(false, true), "disabled");
    }

    #[test]
    fn status_label_watching() {
        assert_eq!(status_label(true, true), "watching");
    }

    #[test]
    fn status_label_stopped() {
        assert_eq!(status_label(true, false), "stopped");
    }

    #[test]
    fn library_row_fields() {
        let row = LibraryRow {
            watch_id: "lib-rust-docs".to_string(),
            tag: "rust-docs".to_string(),
            name: "rust".to_string(),
            display_path: "~/docs/rust".to_string(),
            enabled: true,
            is_active: true,
            mode: "sync".to_string(),
            doc_count: 42,
            source: None,
        };
        assert_eq!(row.tag, "rust-docs");
        assert_eq!(row.doc_count, 42);
        assert!(row.enabled);
    }

    #[test]
    fn display_names_disambiguate_collisions() {
        let paths = vec![
            "/home/u/docs/rust".to_string(),
            "/home/u/refs/rust".to_string(),
            "/home/u/python".to_string(),
        ];
        let names = library_display_names(&paths);
        // "rust" collides → prefixed with parent; "python" is unique.
        assert_eq!(names, vec!["docs/rust", "refs/rust", "python"]);
    }

    #[test]
    fn display_names_trailing_slash() {
        let paths = vec!["/home/u/lib/".to_string()];
        assert_eq!(library_display_names(&paths), vec!["lib"]);
    }

    #[test]
    fn project_source_marks_project_parent() {
        // Parent is a project → P:<project base name>.
        assert_eq!(
            project_source(&Some("/home/u/dev/myproj".into()), &Some("projects".into())),
            Some("P:myproj".to_string())
        );
        // No parent → no marker.
        assert_eq!(project_source(&None, &None), None);
        // Parent is itself a library → no marker.
        assert_eq!(
            project_source(&Some("/home/u/libs/x".into()), &Some("libraries".into())),
            None
        );
    }

    #[test]
    fn library_detail_fields() {
        let detail = LibraryDetail {
            watch_id: "lib-test".to_string(),
            tag: "test".to_string(),
            display_path: "/tmp/lib".to_string(),
            enabled: true,
            is_active: false,
            mode: "incremental".to_string(),
            doc_count: 10,
            follow_symlinks: false,
            cleanup_on_disable: true,
            is_paused: false,
            is_archived: false,
            created_at: "2025-01-01T00:00:00Z".to_string(),
            updated_at: "2025-01-01T00:00:00Z".to_string(),
            last_scan: Some("2025-01-01T00:00:00Z".to_string()),
            last_activity_at: None,
        };
        assert_eq!(detail.tag, "test");
        assert_eq!(detail.mode, "incremental");
        assert!(detail.cleanup_on_disable);
    }
}
