//! Queue display types and formatting utilities

use chrono::{DateTime, Utc};
use colored::Colorize;
use serde::Serialize;
use tabled::Tabled;

use crate::output::ColumnHints;

/// Queue item for list display
#[derive(Debug, Tabled, Serialize)]
pub struct QueueListItem {
    #[tabled(rename = "ID")]
    pub queue_id: String,
    #[tabled(rename = "Type")]
    pub item_type: String,
    #[tabled(rename = "Op")]
    pub op: String,
    #[tabled(rename = "Collection")]
    pub collection: String,
    #[tabled(rename = "Status")]
    pub status: String,
    #[tabled(rename = "Age")]
    pub age: String,
    #[tabled(rename = "Retry")]
    pub retry_count: i32,
}

impl ColumnHints for QueueListItem {
    // All categorical
    fn content_columns() -> &'static [usize] {
        &[]
    }
}

/// Queue item for verbose list display
#[derive(Debug, Tabled, Serialize)]
pub struct QueueListItemVerbose {
    #[tabled(rename = "ID")]
    pub queue_id: String,
    #[tabled(rename = "Idempotency Key")]
    pub idempotency_key: String,
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
    // All categorical
    fn content_columns() -> &'static [usize] {
        &[]
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

/// Format relative time from ISO timestamp
pub fn format_relative_time(timestamp_str: &str) -> String {
    if let Ok(dt) = DateTime::parse_from_rfc3339(timestamp_str) {
        let now = Utc::now();
        let duration = now.signed_duration_since(dt.with_timezone(&Utc));

        let secs = duration.num_seconds();
        if secs < 0 {
            return "future".to_string();
        }

        if secs < 60 {
            format!("{}s ago", secs)
        } else if secs < 3600 {
            format!("{}m ago", secs / 60)
        } else if secs < 86400 {
            format!("{}h ago", secs / 3600)
        } else {
            format!("{}d ago", secs / 86400)
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
