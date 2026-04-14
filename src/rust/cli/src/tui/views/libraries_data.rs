//! Data types and SQLite fetching logic for the TUI library browser.
//!
//! Separated from the view module to keep both files under the 500-line limit
//! and to allow unit-testing data logic independently from rendering.

use crate::data::db::connect_readonly;

/// A single library row ready for display in the TUI list.
#[derive(Debug, Clone)]
pub struct LibraryRow {
    /// Watch folder ID (e.g., "lib-rust-docs").
    pub watch_id: String,
    /// Library tag/name (tenant_id).
    pub tag: String,
    /// Absolute filesystem path.
    pub path: String,
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
}

/// Full detail of a single library for the popup view.
#[derive(Debug, Clone)]
pub struct LibraryDetail {
    pub watch_id: String,
    pub tag: String,
    pub path: String,
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
         COALESCE(tf_count.cnt, 0) \
         FROM watch_folders wf \
         LEFT JOIN ( \
             SELECT watch_folder_id, COUNT(*) AS cnt FROM tracked_files GROUP BY watch_folder_id \
         ) tf_count ON tf_count.watch_folder_id = wf.watch_id \
         WHERE wf.collection = 'libraries' \
         ORDER BY wf.tenant_id",
    ) else {
        return Vec::new();
    };

    let Ok(rows) = stmt.query_map([], |row| {
        let path: String = row.get(2)?;
        let is_active_val: i64 = row.get(4)?;
        Ok(LibraryRow {
            watch_id: row.get(0)?,
            tag: row.get(1)?,
            display_path: abbreviate_home(&path, &home_dir),
            path,
            enabled: row.get(3)?,
            is_active: is_active_val > 0,
            mode: row.get(5)?,
            doc_count: row.get::<_, i64>(6).unwrap_or(0) as u64,
        })
    }) else {
        return Vec::new();
    };

    rows.flatten().collect()
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
            path,
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
            path: "/Users/alice/docs/rust".to_string(),
            display_path: "~/docs/rust".to_string(),
            enabled: true,
            is_active: true,
            mode: "sync".to_string(),
            doc_count: 42,
        };
        assert_eq!(row.tag, "rust-docs");
        assert_eq!(row.doc_count, 42);
        assert!(row.enabled);
    }

    #[test]
    fn library_detail_fields() {
        let detail = LibraryDetail {
            watch_id: "lib-test".to_string(),
            tag: "test".to_string(),
            path: "/tmp/lib".to_string(),
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
