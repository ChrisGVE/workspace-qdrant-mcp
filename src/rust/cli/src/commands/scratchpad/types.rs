//! Shared types and payload helpers for scratchpad commands

use serde::{Deserialize, Serialize};
use tabled::Tabled;

use crate::output::ColumnHints;

// ─── Qdrant scroll response types ──────────────────────────────────────────

#[derive(Deserialize)]
pub(super) struct ScrollResponse {
    pub result: ScrollResult,
}

#[derive(Deserialize)]
pub(super) struct ScrollResult {
    pub points: Vec<QdrantPoint>,
}

#[derive(Deserialize)]
pub(super) struct QdrantPoint {
    #[allow(dead_code)]
    pub id: serde_json::Value,
    pub payload: Option<serde_json::Value>,
}

// ─── Display types ─────────────────────────────────────────────────────────

#[derive(Tabled)]
pub(super) struct ScratchRow {
    #[tabled(rename = "Title")]
    pub title: String,
    #[tabled(rename = "Tenant")]
    pub tenant_id: String,
    #[tabled(rename = "Tags")]
    pub tags: String,
    #[tabled(rename = "Created")]
    pub created_at: String,
}

impl ColumnHints for ScratchRow {
    fn content_columns() -> &'static [usize] {
        &[0]
    }
}

#[derive(Tabled)]
pub(super) struct ScratchRowVerbose {
    #[tabled(rename = "Title")]
    pub title: String,
    #[tabled(rename = "Tenant")]
    pub tenant_id: String,
    #[tabled(rename = "Tags")]
    pub tags: String,
    #[tabled(rename = "Content")]
    pub content: String,
    #[tabled(rename = "Created")]
    pub created_at: String,
}

impl ColumnHints for ScratchRowVerbose {
    fn content_columns() -> &'static [usize] {
        &[0, 3]
    }
}

#[derive(Serialize)]
pub(super) struct ScratchJson {
    pub title: String,
    pub content: String,
    pub tenant_id: String,
    pub tags: Vec<String>,
    pub source_type: String,
    pub created_at: String,
}

// ─── Payload helpers ────────────────────────────────────────────────────────

pub(super) fn payload_str(payload: &serde_json::Value, key: &str) -> String {
    payload
        .get(key)
        .and_then(|v| v.as_str())
        .unwrap_or("")
        .to_string()
}

pub(super) fn payload_tags(payload: &serde_json::Value) -> Vec<String> {
    payload
        .get("tags")
        .and_then(|v| v.as_array())
        .map(|arr| {
            arr.iter()
                .filter_map(|v| v.as_str().map(String::from))
                .collect()
        })
        .or_else(|| {
            payload
                .get("tags")
                .and_then(|v| v.as_str())
                .map(|s| s.split(',').map(String::from).collect())
        })
        .unwrap_or_default()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_payload_str_missing() {
        let v = serde_json::json!({});
        assert_eq!(payload_str(&v, "missing"), "");
    }

    #[test]
    fn test_payload_str_present() {
        let v = serde_json::json!({"title": "hello"});
        assert_eq!(payload_str(&v, "title"), "hello");
    }

    #[test]
    fn test_payload_tags_array() {
        let v = serde_json::json!({"tags": ["a", "b"]});
        assert_eq!(payload_tags(&v), vec!["a", "b"]);
    }

    #[test]
    fn test_payload_tags_string() {
        let v = serde_json::json!({"tags": "a,b"});
        assert_eq!(payload_tags(&v), vec!["a", "b"]);
    }

    #[test]
    fn test_payload_tags_missing() {
        let v = serde_json::json!({});
        assert!(payload_tags(&v).is_empty());
    }
}
