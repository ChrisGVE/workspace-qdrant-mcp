//! Queue display types and formatting utilities

use chrono::{DateTime, Utc};
use colored::Colorize;
use serde::Serialize;
use tabled::Tabled;

use crate::output::ColumnHints;

/// Queue item for compact list display (no ID column)
#[derive(Debug, Tabled, Serialize)]
pub struct QueueListItemCompact {
    #[tabled(rename = "Project")]
    pub project: String,
    #[tabled(rename = "Object")]
    pub object: String,
    #[tabled(rename = "Type")]
    pub item_type: String,
    #[tabled(rename = "Op")]
    pub op: String,
    #[tabled(rename = "Status")]
    pub status: String,
    #[tabled(rename = "Age")]
    pub age: String,
    #[tabled(rename = "Retry")]
    pub retry_count: i32,
}

impl ColumnHints for QueueListItemCompact {
    fn content_columns() -> &'static [usize] {
        // Project (index 0) and Object (index 1) are content columns
        &[0, 1]
    }
}

/// Queue item for list display (with ID column, shown via --id flag)
#[derive(Debug, Tabled, Serialize)]
pub struct QueueListItem {
    #[tabled(rename = "ID")]
    pub queue_id: String,
    #[tabled(rename = "Project")]
    pub project: String,
    #[tabled(rename = "Object")]
    pub object: String,
    #[tabled(rename = "Type")]
    pub item_type: String,
    #[tabled(rename = "Op")]
    pub op: String,
    #[tabled(rename = "Status")]
    pub status: String,
    #[tabled(rename = "Age")]
    pub age: String,
    #[tabled(rename = "Retry")]
    pub retry_count: i32,
}

impl ColumnHints for QueueListItem {
    fn content_columns() -> &'static [usize] {
        // Project (index 1) and Object (index 2) are content columns
        &[1, 2]
    }
}

/// Queue item for list display with error column (shown when failed items exist, no ID)
#[derive(Debug, Tabled, Serialize)]
pub struct QueueListItemCompactWithError {
    #[tabled(rename = "Project")]
    pub project: String,
    #[tabled(rename = "Object")]
    pub object: String,
    #[tabled(rename = "Type")]
    pub item_type: String,
    #[tabled(rename = "Op")]
    pub op: String,
    #[tabled(rename = "Status")]
    pub status: String,
    #[tabled(rename = "Age")]
    pub age: String,
    #[tabled(rename = "Retry")]
    pub retry_count: i32,
    #[tabled(rename = "Error")]
    pub error_message: String,
}

impl ColumnHints for QueueListItemCompactWithError {
    fn content_columns() -> &'static [usize] {
        // Project (index 0), Object (index 1), and Error (index 7) are content columns
        &[0, 1, 7]
    }
}

/// Queue item for list display with error column and ID
#[derive(Debug, Tabled, Serialize)]
pub struct QueueListItemWithError {
    #[tabled(rename = "ID")]
    pub queue_id: String,
    #[tabled(rename = "Project")]
    pub project: String,
    #[tabled(rename = "Object")]
    pub object: String,
    #[tabled(rename = "Type")]
    pub item_type: String,
    #[tabled(rename = "Op")]
    pub op: String,
    #[tabled(rename = "Status")]
    pub status: String,
    #[tabled(rename = "Age")]
    pub age: String,
    #[tabled(rename = "Retry")]
    pub retry_count: i32,
    #[tabled(rename = "Error")]
    pub error_message: String,
}

impl ColumnHints for QueueListItemWithError {
    fn content_columns() -> &'static [usize] {
        // Project (index 1), Object (index 2), and Error (index 8) are content columns
        &[1, 2, 8]
    }
}

/// Queue item for verbose list display
#[derive(Debug, Tabled, Serialize)]
pub struct QueueListItemVerbose {
    #[tabled(rename = "ID")]
    pub queue_id: String,
    #[tabled(rename = "Idempotency Key")]
    pub idempotency_key: String,
    #[tabled(rename = "Project")]
    pub project: String,
    #[tabled(rename = "Object")]
    pub object: String,
    #[tabled(rename = "Type")]
    pub item_type: String,
    #[tabled(rename = "Op")]
    pub op: String,
    #[tabled(rename = "Collection")]
    pub collection: String,
    #[tabled(rename = "Status")]
    pub status: String,
    #[tabled(rename = "Created")]
    pub created_at: String,
    #[tabled(rename = "Retry")]
    pub retry_count: i32,
    #[tabled(rename = "Worker")]
    pub worker_id: String,
}

impl ColumnHints for QueueListItemVerbose {
    fn content_columns() -> &'static [usize] {
        // Project (index 2) and Object (index 3) are content columns
        &[2, 3]
    }
}

/// Queue item detail view
#[derive(Debug, Serialize)]
pub struct QueueDetailItem {
    pub queue_id: String,
    pub idempotency_key: String,
    pub item_type: String,
    pub op: String,
    pub tenant_id: String,
    pub collection: String,
    pub status: String,
    pub branch: String,
    pub payload_json: String,
    pub metadata: Option<String>,
    pub created_at: String,
    pub updated_at: String,
    pub lease_until: Option<String>,
    pub worker_id: Option<String>,
    pub retry_count: i32,
    pub error_message: Option<String>,
    pub last_error_at: Option<String>,
}

/// Queue statistics
#[derive(Debug, Serialize)]
pub struct QueueStatsSummary {
    pub total_items: i64,
    pub by_status: StatusBreakdown,
    pub oldest_pending_age_seconds: Option<f64>,
    pub oldest_pending_id: Option<String>,
    pub active_collections: i64,
    pub active_projects: i64,
}

/// Status breakdown
#[derive(Debug, Serialize, Default)]
pub struct StatusBreakdown {
    pub pending: i64,
    pub in_progress: i64,
    pub done: i64,
    pub failed: i64,
}

/// Format relative time from ISO timestamp.
///
/// Returns a compact, non-wrapping string like "22h", "3m", "5d".
pub fn format_relative_time(timestamp_str: &str) -> String {
    if let Ok(dt) = DateTime::parse_from_rfc3339(timestamp_str) {
        let now = Utc::now();
        let duration = now.signed_duration_since(dt.with_timezone(&Utc));

        let secs = duration.num_seconds();
        if secs < 0 {
            return "future".to_string();
        }

        if secs < 60 {
            format!("{}s", secs)
        } else if secs < 3600 {
            format!("{}m", secs / 60)
        } else if secs < 86400 {
            format!("{}h", secs / 3600)
        } else {
            format!("{}d", secs / 86400)
        }
    } else {
        "unknown".to_string()
    }
}

/// Format status with color
pub fn format_status(status: &str) -> String {
    match status {
        "pending" => "pending".yellow().to_string(),
        "in_progress" => "in_progress".blue().to_string(),
        "done" => "done".green().to_string(),
        "failed" => "failed".red().to_string(),
        _ => status.to_string(),
    }
}

/// Extract a human-readable object name from `payload_json` based on item type.
///
/// For file operations, returns the file basename (e.g. `main.rs`).
/// For folder operations, returns the folder name with a trailing `/`.
/// For URL operations, returns the URL.
/// For text/content operations, returns a truncated content preview.
/// Falls back to an empty string on parse failure.
pub fn extract_object(item_type: &str, payload_json: &str) -> String {
    let Ok(value) = serde_json::from_str::<serde_json::Value>(payload_json) else {
        return String::new();
    };

    match item_type {
        "file" => {
            if let Some(path) = value.get("file_path").and_then(|v| v.as_str()) {
                basename(path).to_string()
            } else {
                String::new()
            }
        }
        "folder" => {
            if let Some(path) = value.get("folder_path").and_then(|v| v.as_str()) {
                let name = basename(path);
                format!("{name}/")
            } else {
                String::new()
            }
        }
        "url" | "website" => value
            .get("url")
            .and_then(|v| v.as_str())
            .unwrap_or("")
            .to_string(),
        "text" | "doc" => {
            // Show title if available, otherwise truncated content
            if let Some(title) = value.get("title").and_then(|v| v.as_str()) {
                truncate_str(title, 40).to_string()
            } else if let Some(content) = value.get("content").and_then(|v| v.as_str()) {
                truncate_str(content, 40).to_string()
            } else {
                String::new()
            }
        }
        "tenant" | "collection" => {
            // Administrative operations -- no meaningful file object
            String::new()
        }
        _ => String::new(),
    }
}

/// Like [`extract_object`] but, for file and folder items, returns the path
/// relative to its project/library `root` rather than only the basename. Falls
/// back to the basename when the root is unknown or the path is not under it.
/// Non-path item types behave exactly like [`extract_object`].
pub fn extract_object_relative(item_type: &str, payload_json: &str, root: Option<&str>) -> String {
    let Ok(value) = serde_json::from_str::<serde_json::Value>(payload_json) else {
        return String::new();
    };
    match item_type {
        "file" => {
            relative_or_basename(value.get("file_path").and_then(|v| v.as_str()), root, false)
        }
        "folder" => relative_or_basename(
            value.get("folder_path").and_then(|v| v.as_str()),
            root,
            true,
        ),
        _ => extract_object(item_type, payload_json),
    }
}

/// Resolve a path to its root-relative form, or the basename when that is not
/// possible. Directories get a trailing `/`.
fn relative_or_basename(path: Option<&str>, root: Option<&str>, is_dir: bool) -> String {
    let Some(path) = path else {
        return String::new();
    };
    let rel = root
        .and_then(|r| relativize(path, r))
        .unwrap_or_else(|| basename(path).to_string());
    if is_dir {
        format!("{rel}/")
    } else {
        rel
    }
}

/// Strip the `root` prefix from `path`, returning the remainder without a
/// leading slash. `None` when `path` is not under `root` (or equals it).
fn relativize(path: &str, root: &str) -> Option<String> {
    let root = root.trim_end_matches('/');
    let stripped = path.strip_prefix(root)?.trim_start_matches('/');
    if stripped.is_empty() {
        None
    } else {
        Some(stripped.to_string())
    }
}

/// Return the last path component (file or directory name).
fn basename(path: &str) -> &str {
    path.rsplit('/').find(|s| !s.is_empty()).unwrap_or(path)
}

/// Truncate a string to `max_len` characters, appending "..." if truncated.
pub fn truncate_str(s: &str, max_len: usize) -> String {
    if s.chars().count() <= max_len {
        s.to_string()
    } else {
        let truncated: String = s.chars().take(max_len.saturating_sub(3)).collect();
        format!("{truncated}...")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn relative_object_strips_root() {
        let payload = r#"{"file_path":"/home/u/proj/src/main.rs"}"#;
        assert_eq!(
            extract_object_relative("file", payload, Some("/home/u/proj")),
            "src/main.rs"
        );
        // Trailing slash on the root is tolerated.
        assert_eq!(
            extract_object_relative("file", payload, Some("/home/u/proj/")),
            "src/main.rs"
        );
    }

    #[test]
    fn relative_object_falls_back_to_basename() {
        let payload = r#"{"file_path":"/home/u/proj/src/main.rs"}"#;
        // Unknown root → basename.
        assert_eq!(extract_object_relative("file", payload, None), "main.rs");
        // Path not under the given root → basename.
        assert_eq!(
            extract_object_relative("file", payload, Some("/other")),
            "main.rs"
        );
    }

    #[test]
    fn relative_folder_keeps_trailing_slash() {
        let payload = r#"{"folder_path":"/home/u/proj/src/util"}"#;
        assert_eq!(
            extract_object_relative("folder", payload, Some("/home/u/proj")),
            "src/util/"
        );
    }

    #[test]
    fn relative_object_passthrough_for_url() {
        let payload = r#"{"url":"https://example.com/x"}"#;
        assert_eq!(
            extract_object_relative("url", payload, Some("/root")),
            "https://example.com/x"
        );
    }
}
