//! Shared types and HTTP client helpers for backup commands.

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use tabled::Tabled;

use crate::output::{self, ColumnHints};

/// Get Qdrant URL from environment or default
pub fn qdrant_url() -> String {
    std::env::var("QDRANT_URL")
        .unwrap_or_else(|_| wqm_common::constants::DEFAULT_QDRANT_URL.to_string())
}

/// Get optional Qdrant API key
pub fn qdrant_api_key() -> Option<String> {
    std::env::var("QDRANT_API_KEY").ok()
}

/// Build a reqwest client with optional API key header
pub fn build_client() -> Result<reqwest::Client> {
    let mut headers = reqwest::header::HeaderMap::new();
    if let Some(key) = qdrant_api_key() {
        headers.insert(
            "api-key",
            reqwest::header::HeaderValue::from_str(&key).context("Invalid QDRANT_API_KEY value")?,
        );
    }
    reqwest::Client::builder()
        .default_headers(headers)
        .timeout(std::time::Duration::from_secs(300))
        .build()
        .context("Failed to build HTTP client")
}

/// Qdrant API response wrapper
#[derive(Debug, Deserialize)]
pub struct QdrantResponse<T> {
    #[allow(dead_code)]
    pub status: Option<String>,
    pub result: T,
    #[allow(dead_code)]
    pub time: Option<f64>,
}

/// Snapshot metadata from Qdrant
#[derive(Debug, Deserialize, Serialize, Clone)]
pub struct SnapshotInfo {
    pub name: String,
    pub size: i64,
    pub creation_time: Option<String>,
    pub checksum: Option<String>,
}

/// Displayable snapshot row for table output
#[derive(Tabled)]
pub struct SnapshotRow {
    #[tabled(rename = "Name")]
    pub name: String,
    #[tabled(rename = "Size")]
    pub size: String,
    #[tabled(rename = "Created")]
    pub created: String,
    #[tabled(rename = "Checksum")]
    pub checksum: String,
}

impl ColumnHints for SnapshotRow {
    // All categorical
    fn content_columns() -> &'static [usize] {
        &[]
    }
}

impl From<&SnapshotInfo> for SnapshotRow {
    fn from(s: &SnapshotInfo) -> Self {
        Self {
            name: s.name.clone(),
            size: output::format_bytes(s.size),
            created: s
                .creation_time
                .as_deref()
                .map(wqm_common::timestamp_fmt::format_local)
                .unwrap_or_else(|| "unknown".into()),
            checksum: s.checksum.clone().unwrap_or_else(|| "none".into()),
        }
    }
}

/// Collection list entry from Qdrant
#[derive(Debug, Deserialize)]
pub struct CollectionEntry {
    pub name: String,
}

/// Collections result from Qdrant
#[derive(Debug, Deserialize)]
pub struct CollectionsResult {
    pub collections: Vec<CollectionEntry>,
}
