//! Popup data types, state, and fetch logic for dashboard cells.
//!
//! Rendering is in `dashboard_popup_ui.rs`.

use std::collections::HashMap;

use wqm_common::constants::TENANT_GLOBAL;

use crate::data::db::connect_readonly;

// Re-export draw function from the UI module.
pub use super::dashboard_popup_ui::draw_popup;

// ---------------------------------------------------------------------------
// Data types
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
pub struct FileDetailRow {
    pub prefix: String,
    pub rel_path: String,
    pub filename: String,
    pub chunk_count: i64,
    pub status: FileStatus,
}

#[derive(Debug, Clone, Copy)]
pub enum FileStatus {
    UpToDate,
    Pending,
    InProgress,
    Errored,
}

#[derive(Debug, Clone)]
pub struct ErrorDetail {
    pub collection_label: String,
    pub error_message: String,
    pub item_type: String,
    pub op: String,
    pub tenant_id: String,
    pub file_path: Option<String>,
    pub created_at: String,
    pub updated_at: String,
    pub retry_count: i32,
}

#[derive(Debug, Clone)]
pub struct NoteEntry {
    pub content: String,
    pub status: FileStatus,
}

// ---------------------------------------------------------------------------
// Popup state
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
pub enum PopupState {
    Project {
        name: String,
        is_active: bool,
        files: Vec<FileDetailRow>,
        scroll: usize,
    },
    Library {
        name: String,
        files: Vec<FileDetailRow>,
        scroll: usize,
    },
    Scratchpad {
        name: String,
        notes: Vec<NoteEntry>,
        scroll: usize,
    },
    Rules {
        name: String,
        notes: Vec<NoteEntry>,
        scroll: usize,
    },
    Error(ErrorDetail),
}

impl PopupState {
    pub fn scroll_down(&mut self) {
        match self {
            PopupState::Project { files, scroll, .. } => {
                if *scroll < files.len().saturating_sub(1) {
                    *scroll += 1;
                }
            }
            PopupState::Library { files, scroll, .. } => {
                if *scroll < files.len().saturating_sub(1) {
                    *scroll += 1;
                }
            }
            PopupState::Scratchpad { notes, scroll, .. }
            | PopupState::Rules { notes, scroll, .. } => {
                if *scroll < notes.len().saturating_sub(1) {
                    *scroll += 1;
                }
            }
            PopupState::Error(_) => {}
        }
    }

    pub fn scroll_up(&mut self) {
        match self {
            PopupState::Project { scroll, .. }
            | PopupState::Library { scroll, .. }
            | PopupState::Scratchpad { scroll, .. }
            | PopupState::Rules { scroll, .. } => {
                *scroll = scroll.saturating_sub(1);
            }
            PopupState::Error(_) => {}
        }
    }
}

// ---------------------------------------------------------------------------
// Fetch functions
// ---------------------------------------------------------------------------

pub fn fetch_project_popup(tenant_id: &str) -> Option<PopupState> {
    let conn = connect_readonly().ok()?;
    let name = fetch_project_name(&conn, tenant_id)?;
    let is_active = fetch_is_active(&conn, tenant_id);
    let files = fetch_file_details(&conn, tenant_id, "projects", true);
    Some(PopupState::Project {
        name,
        is_active,
        files,
        scroll: 0,
    })
}

pub fn fetch_library_popup(tenant_id: &str) -> Option<PopupState> {
    let conn = connect_readonly().ok()?;
    let files = fetch_file_details(&conn, tenant_id, "libraries", false);
    Some(PopupState::Library {
        name: tenant_id.to_string(),
        files,
        scroll: 0,
    })
}

pub fn fetch_scratchpad_popup(tenant_id: &str) -> Option<PopupState> {
    let conn = connect_readonly().ok()?;
    let name = resolve_display_name(&conn, tenant_id);
    let notes = fetch_queue_notes(&conn, tenant_id, "scratchpad");
    Some(PopupState::Scratchpad {
        name,
        notes,
        scroll: 0,
    })
}

pub fn fetch_rules_popup(tenant_id: &str) -> Option<PopupState> {
    let conn = connect_readonly().ok()?;
    let name = resolve_display_name(&conn, tenant_id);
    let notes = fetch_queue_notes(&conn, tenant_id, "rules");
    Some(PopupState::Rules {
        name,
        notes,
        scroll: 0,
    })
}

pub fn fetch_error_popup(queue_id: &str) -> Option<PopupState> {
    let conn = connect_readonly().ok()?;
    let tenant_names = fetch_tenant_name_map(&conn);

    let mut stmt = conn
        .prepare(
            "SELECT collection, tenant_id, error_message, item_type, op, \
         file_path, created_at, updated_at, retry_count \
         FROM unified_queue WHERE queue_id = ?1",
        )
        .ok()?;

    let detail = stmt
        .query_row(rusqlite::params![queue_id], |row| {
            let coll: String = row.get(0)?;
            let tid: String = row.get(1)?;
            let letter = super::dashboard_fetch::collection_letter(&coll);
            let tname = tenant_names.get(&tid).cloned().unwrap_or(tid.clone());
            Ok(ErrorDetail {
                collection_label: format!("[{}] {}", letter, tname),
                error_message: row.get::<_, String>(2).unwrap_or_default(),
                item_type: row.get(3)?,
                op: row.get(4)?,
                tenant_id: tid,
                file_path: row.get(5)?,
                created_at: row.get(6)?,
                updated_at: row.get(7)?,
                retry_count: row.get(8)?,
            })
        })
        .ok()?;

    Some(PopupState::Error(detail))
}

// ---------------------------------------------------------------------------
// SQLite helpers
// ---------------------------------------------------------------------------

fn fetch_project_name(conn: &rusqlite::Connection, tenant_id: &str) -> Option<String> {
    conn.query_row(
        "SELECT path FROM watch_folders WHERE tenant_id = ?1 \
         AND parent_watch_id IS NULL LIMIT 1",
        rusqlite::params![tenant_id],
        |row| {
            let path: String = row.get(0)?;
            Ok(super::dashboard_fetch::path_last_component(&path))
        },
    )
    .ok()
}

fn fetch_is_active(conn: &rusqlite::Connection, tenant_id: &str) -> bool {
    conn.query_row(
        "SELECT is_active FROM watch_folders WHERE tenant_id = ?1 \
         AND parent_watch_id IS NULL LIMIT 1",
        rusqlite::params![tenant_id],
        |row| row.get::<_, i64>(0),
    )
    .map(|v| v > 0)
    .unwrap_or(false)
}

fn resolve_display_name(conn: &rusqlite::Connection, tenant_id: &str) -> String {
    if tenant_id == TENANT_GLOBAL {
        TENANT_GLOBAL.to_string()
    } else {
        fetch_project_name(conn, tenant_id).unwrap_or(tenant_id.to_string())
    }
}

fn fetch_file_details(
    conn: &rusqlite::Connection,
    tenant_id: &str,
    collection: &str,
    include_workspace: bool,
) -> Vec<FileDetailRow> {
    let sql = "SELECT tf.relative_path, tf.chunk_count, wf.path AS wf_path \
               FROM tracked_files tf \
               JOIN watch_folders wf ON tf.watch_folder_id = wf.watch_id \
               WHERE wf.tenant_id = ?1 AND wf.collection = ?2 \
               ORDER BY tf.relative_path";

    let Ok(mut stmt) = conn.prepare(sql) else {
        return Vec::new();
    };
    let Ok(rows) = stmt.query_map(rusqlite::params![tenant_id, collection], |row| {
        Ok((
            row.get::<_, String>(0)?,
            row.get::<_, i64>(1).unwrap_or(0),
            row.get::<_, String>(2)?,
        ))
    }) else {
        return Vec::new();
    };

    let status_map = fetch_file_status_map(conn, tenant_id, collection);

    rows.flatten()
        .map(|(relative_path, chunks, wf_path)| {
            // unified_queue.file_path stores the watch-folder-relative path,
            // so tracked_files.relative_path is the matching lookup key.
            let rel = relative_path.as_str();
            let status = status_map
                .get(rel)
                .copied()
                .unwrap_or(FileStatus::UpToDate);
            let (dir, fname) = match rel.rsplit_once('/') {
                Some((d, f)) => (d.to_string(), f.to_string()),
                None => (String::new(), rel.to_string()),
            };
            let prefix = if include_workspace {
                format!(
                    "Wrk: {}",
                    super::dashboard_fetch::path_last_component(&wf_path)
                )
            } else {
                String::new()
            };
            FileDetailRow {
                prefix,
                rel_path: dir,
                filename: fname,
                chunk_count: chunks,
                status,
            }
        })
        .collect()
}

fn fetch_file_status_map(
    conn: &rusqlite::Connection,
    tenant_id: &str,
    collection: &str,
) -> HashMap<String, FileStatus> {
    let mut map = HashMap::new();
    let Ok(mut stmt) = conn.prepare(
        "SELECT file_path, status FROM unified_queue \
         WHERE tenant_id = ?1 AND collection = ?2 \
         AND file_path IS NOT NULL \
         AND status IN ('pending', 'in_progress', 'failed')",
    ) else {
        return map;
    };
    let Ok(rows) = stmt.query_map(rusqlite::params![tenant_id, collection], |row| {
        Ok((row.get::<_, String>(0)?, row.get::<_, String>(1)?))
    }) else {
        return map;
    };
    for row in rows.flatten() {
        let status = match row.1.as_str() {
            "pending" => FileStatus::Pending,
            "in_progress" => FileStatus::InProgress,
            "failed" => FileStatus::Errored,
            _ => FileStatus::UpToDate,
        };
        map.insert(row.0, status);
    }
    map
}

fn fetch_queue_notes(
    conn: &rusqlite::Connection,
    tenant_id: &str,
    collection: &str,
) -> Vec<NoteEntry> {
    let Ok(mut stmt) = conn.prepare(
        "SELECT COALESCE(payload_json, '{}'), status FROM unified_queue \
         WHERE tenant_id = ?1 AND collection = ?2 \
         ORDER BY created_at DESC LIMIT 100",
    ) else {
        return Vec::new();
    };
    let Ok(rows) = stmt.query_map(rusqlite::params![tenant_id, collection], |row| {
        Ok((row.get::<_, String>(0)?, row.get::<_, String>(1)?))
    }) else {
        return Vec::new();
    };
    rows.flatten()
        .map(|(payload, status_str)| {
            let content = extract_content(&payload);
            let status = match status_str.as_str() {
                "pending" => FileStatus::Pending,
                "in_progress" => FileStatus::InProgress,
                "failed" => FileStatus::Errored,
                _ => FileStatus::UpToDate,
            };
            NoteEntry { content, status }
        })
        .collect()
}

fn extract_content(payload_json: &str) -> String {
    serde_json::from_str::<serde_json::Value>(payload_json)
        .ok()
        .and_then(|v| v["content"].as_str().map(|s| s.to_string()))
        .unwrap_or_else(|| payload_json.to_string())
}

fn fetch_tenant_name_map(conn: &rusqlite::Connection) -> HashMap<String, String> {
    let mut map = HashMap::new();
    let Ok(mut stmt) =
        conn.prepare("SELECT tenant_id, path FROM watch_folders WHERE parent_watch_id IS NULL")
    else {
        return map;
    };
    let Ok(rows) = stmt.query_map([], |row| {
        Ok((row.get::<_, String>(0)?, row.get::<_, String>(1)?))
    }) else {
        return map;
    };
    for row in rows.flatten() {
        map.insert(row.0, super::dashboard_fetch::path_last_component(&row.1));
    }
    map
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn popup_state_scroll() {
        let mut p = PopupState::Project {
            name: "test".into(),
            is_active: true,
            files: vec![
                FileDetailRow {
                    prefix: "a".into(),
                    rel_path: "src".into(),
                    filename: "main.rs".into(),
                    chunk_count: 5,
                    status: FileStatus::UpToDate,
                },
                FileDetailRow {
                    prefix: "a".into(),
                    rel_path: "src".into(),
                    filename: "lib.rs".into(),
                    chunk_count: 3,
                    status: FileStatus::Pending,
                },
            ],
            scroll: 0,
        };
        p.scroll_down();
        assert!(matches!(&p, PopupState::Project { scroll: 1, .. }));
        p.scroll_down();
        assert!(matches!(&p, PopupState::Project { scroll: 1, .. }));
        p.scroll_up();
        assert!(matches!(&p, PopupState::Project { scroll: 0, .. }));
    }

    #[test]
    fn extract_content_from_json() {
        assert_eq!(extract_content(r#"{"content":"hello"}"#), "hello");
    }

    #[test]
    fn extract_content_fallback() {
        assert_eq!(extract_content("not json"), "not json");
    }
}
