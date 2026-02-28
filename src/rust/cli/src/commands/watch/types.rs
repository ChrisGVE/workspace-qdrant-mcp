//! Display types for watch command output

use serde::Serialize;
use tabled::Tabled;

use crate::output::ColumnHints;

/// Watch item for list display
#[derive(Debug, Tabled, Serialize)]
pub struct WatchListItem {
    #[tabled(rename = "Watch ID")]
    pub watch_id: String,
    #[tabled(rename = "Path")]
    pub path: String,
    #[tabled(rename = "Collection")]
    pub collection: String,
    #[tabled(rename = "Enabled")]
    pub enabled: String,
    #[tabled(rename = "Active")]
    pub is_active: String,
    #[tabled(rename = "Paused")]
    pub is_paused: String,
    #[tabled(rename = "Archived")]
    pub archived: String,
    #[tabled(rename = "Last Scan")]
    pub last_scan: String,
}

impl ColumnHints for WatchListItem {
    // Path(1) is content
    fn content_columns() -> &'static [usize] {
        &[1]
    }
}

/// Watch item for verbose list display
#[derive(Debug, Tabled, Serialize)]
pub struct WatchListItemVerbose {
    #[tabled(rename = "Watch ID")]
    pub watch_id: String,
    #[tabled(rename = "Path")]
    pub path: String,
    #[tabled(rename = "Collection")]
    pub collection: String,
    #[tabled(rename = "Tenant ID")]
    pub tenant_id: String,
    #[tabled(rename = "Enabled")]
    pub enabled: String,
    #[tabled(rename = "Active")]
    pub is_active: String,
    #[tabled(rename = "Paused")]
    pub is_paused: String,
    #[tabled(rename = "Archived")]
    pub archived: String,
    #[tabled(rename = "Git Remote")]
    pub git_remote_url: String,
    #[tabled(rename = "Library Mode")]
    pub library_mode: String,
    #[tabled(rename = "Last Scan")]
    pub last_scan: String,
}

impl ColumnHints for WatchListItemVerbose {
    // Path(1) is content
    fn content_columns() -> &'static [usize] {
        &[1]
    }
}

/// Watch item detail view
#[derive(Debug, Serialize)]
pub struct WatchDetailItem {
    pub watch_id: String,
    pub path: String,
    pub collection: String,
    pub tenant_id: String,
    pub git_remote_url: Option<String>,
    pub remote_hash: Option<String>,
    pub disambiguation_path: Option<String>,
    pub enabled: bool,
    pub is_active: bool,
    pub is_paused: bool,
    pub is_archived: bool,
    pub library_mode: Option<String>,
    pub follow_symlinks: bool,
    pub created_at: String,
    pub updated_at: String,
    pub last_scan: Option<String>,
    pub last_activity_at: Option<String>,
    pub parent_watch_id: Option<String>,
    pub submodule_path: Option<String>,
}
