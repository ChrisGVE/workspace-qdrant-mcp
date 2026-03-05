//! Shared types and helpers for rules subcommands
//!
//! Contains Qdrant REST client helpers, deserialization types,
//! display row types, and payload extraction utilities.

use std::collections::HashMap;
use std::path::Path;

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use tabled::Tabled;

use wqm_common::schema::qdrant::rules as rules_schema;
use wqm_common::schema::sqlite::watch_folders as wf_schema;

use crate::config::get_database_path_checked;
use crate::output::ColumnHints;

// ---- Qdrant REST helpers ----

/// Get Qdrant URL from environment or default
pub fn qdrant_url() -> String {
    std::env::var("QDRANT_URL")
        .unwrap_or_else(|_| wqm_common::constants::DEFAULT_QDRANT_URL.to_string())
}

/// Build a reqwest client with optional Qdrant API key header
pub fn build_qdrant_client() -> Result<reqwest::Client> {
    let mut headers = reqwest::header::HeaderMap::new();
    if let Ok(key) = std::env::var("QDRANT_API_KEY") {
        headers.insert(
            "api-key",
            reqwest::header::HeaderValue::from_str(&key).context("Invalid QDRANT_API_KEY value")?,
        );
    }
    reqwest::Client::builder()
        .default_headers(headers)
        .timeout(std::time::Duration::from_secs(30))
        .build()
        .context("Failed to build HTTP client")
}

// ---- Qdrant scroll response types ----

#[derive(Deserialize)]
pub struct ScrollResponse {
    pub result: ScrollResult,
}

#[derive(Deserialize)]
pub struct ScrollResult {
    pub points: Vec<QdrantPoint>,
}

#[derive(Deserialize)]
pub struct QdrantPoint {
    #[allow(dead_code)]
    pub id: serde_json::Value,
    pub payload: Option<serde_json::Value>,
}

// ---- Display types ----

/// Compact table row for default display
#[derive(Tabled)]
pub struct RuleRow {
    #[tabled(rename = "Label")]
    pub label: String,
    #[tabled(rename = "Title")]
    pub title: String,
    #[tabled(rename = "Scope")]
    pub scope: String,
    #[tabled(rename = "Priority")]
    pub priority: String,
    #[tabled(rename = "Created")]
    pub created_at: String,
}

impl ColumnHints for RuleRow {
    // Title(1) is content
    fn content_columns() -> &'static [usize] {
        &[1]
    }
}

/// Verbose table row with content
#[derive(Tabled)]
pub struct RuleRowVerbose {
    #[tabled(rename = "Label")]
    pub label: String,
    #[tabled(rename = "Title")]
    pub title: String,
    #[tabled(rename = "Scope")]
    pub scope: String,
    #[tabled(rename = "Priority")]
    pub priority: String,
    #[tabled(rename = "Tags")]
    pub tags: String,
    #[tabled(rename = "Content")]
    pub content: String,
    #[tabled(rename = "Created")]
    pub created_at: String,
}

impl ColumnHints for RuleRowVerbose {
    // Title(1), Content(5) are content
    fn content_columns() -> &'static [usize] {
        &[1, 5]
    }
}

/// Full rule data for JSON output
#[derive(Serialize)]
pub struct RuleJson {
    pub label: String,
    pub title: String,
    pub content: String,
    pub scope: String,
    pub project_id: Option<String>,
    pub source_type: String,
    pub priority: Option<u32>,
    pub tags: Vec<String>,
    pub created_at: String,
    pub updated_at: String,
}

// ---- Payload extraction helpers ----

/// Helper to extract a string from a JSON payload field
pub fn payload_str(payload: &serde_json::Value, key: &str) -> String {
    payload
        .get(key)
        .and_then(|v| v.as_str())
        .unwrap_or("")
        .to_string()
}

/// Helper to extract an optional u32 from a JSON payload field
pub fn payload_u32(payload: &serde_json::Value, key: &str) -> Option<u32> {
    payload.get(key).and_then(|v| v.as_u64().map(|n| n as u32))
}

/// Format a title with optional project name for project-scoped rules.
///
/// For rules with scope "project", appends project info in parenthesis
/// after the title text. Non-verbose shows just the project name;
/// verbose includes the tenant_id. Uses same-line format so tabled's
/// word wrapping handles layout naturally.
pub fn format_title_with_project(
    payload: &serde_json::Value,
    project_names: &HashMap<String, String>,
    verbose: bool,
) -> String {
    let title = payload_str(payload, rules_schema::TITLE.name);
    let scope = payload_str(payload, rules_schema::SCOPE.name);
    if scope == "project" {
        if let Some(pid) = payload
            .get(rules_schema::PROJECT_ID.name)
            .and_then(|v| v.as_str())
        {
            let name = project_names.get(pid);
            return match (name, verbose) {
                (Some(name), false) => format!("{} (project: {})", title, name),
                (Some(name), true) => format!("{} (project: {} / {})", title, name, pid),
                (None, _) => format!("{} (project: {})", title, pid),
            };
        }
    }
    title
}

/// Build a tenant_id -> project name mapping from watch_folders.
///
/// Extracts the last path component as the project name. Returns an
/// empty map if the database is unavailable.
pub fn load_project_names() -> HashMap<String, String> {
    let mut map = HashMap::new();
    let db_path = match get_database_path_checked() {
        Ok(p) => p,
        Err(_) => return map,
    };
    let conn = match rusqlite::Connection::open_with_flags(
        &db_path,
        rusqlite::OpenFlags::SQLITE_OPEN_READ_ONLY | rusqlite::OpenFlags::SQLITE_OPEN_NO_MUTEX,
    ) {
        Ok(c) => c,
        Err(_) => return map,
    };
    let sql = format!(
        "SELECT {}, {} FROM {} WHERE {} = 'projects'",
        wf_schema::TENANT_ID.name,
        wf_schema::PATH.name,
        wf_schema::TABLE.name,
        wf_schema::COLLECTION.name,
    );
    let mut stmt = match conn.prepare(&sql) {
        Ok(s) => s,
        Err(_) => return map,
    };
    let rows = stmt.query_map([], |row| {
        Ok((row.get::<_, String>(0)?, row.get::<_, String>(1)?))
    });
    if let Ok(rows) = rows {
        for row in rows.flatten() {
            let (tenant_id, path) = row;
            let name = Path::new(&path)
                .file_name()
                .map(|n| n.to_string_lossy().to_string())
                .unwrap_or_else(|| path.clone());
            map.insert(tenant_id, name);
        }
    }
    map
}

/// Normalize comma-separated values for table display.
///
/// Ensures a space after each comma so `keep_words()` can wrap at
/// word boundaries instead of breaking mid-word.
pub fn normalize_commas(s: &str) -> String {
    let mut result = String::with_capacity(s.len() + 10);
    let mut chars = s.chars().peekable();
    while let Some(c) = chars.next() {
        result.push(c);
        if c == ',' && chars.peek() != Some(&' ') {
            result.push(' ');
        }
    }
    result
}

/// Build a Qdrant filter from scope string
pub fn build_scope_filter(scope_str: &str) -> serde_json::Value {
    let mut must = Vec::new();

    if scope_str == "global" {
        must.push(serde_json::json!({
            "key": rules_schema::SCOPE.name,
            "match": { "value": "global" }
        }));
    } else if let Some(project_id) = scope_str.strip_prefix("project:") {
        must.push(serde_json::json!({
            "key": rules_schema::SCOPE.name,
            "match": { "value": "project" }
        }));
        must.push(serde_json::json!({
            "key": rules_schema::PROJECT_ID.name,
            "match": { "value": project_id }
        }));
    }

    serde_json::json!({ "must": must })
}

// ---- Tests ----

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_payload_str() {
        let payload = serde_json::json!({
            "label": "test-label",
            "missing": null,
        });
        assert_eq!(payload_str(&payload, "label"), "test-label");
        assert_eq!(payload_str(&payload, "nonexistent"), "");
        assert_eq!(payload_str(&payload, "missing"), "");
    }

    #[test]
    fn test_payload_u32() {
        let payload = serde_json::json!({
            "priority": 8,
            "zero": 0,
        });
        assert_eq!(payload_u32(&payload, "priority"), Some(8));
        assert_eq!(payload_u32(&payload, "zero"), Some(0));
        assert_eq!(payload_u32(&payload, "missing"), None);
    }

    #[test]
    fn test_scroll_response_deserialization() {
        let json = r#"{
            "result": {
                "points": [
                    {
                        "id": "abc-123",
                        "payload": {
                            "content": "test rule",
                            "label": "test-label",
                            "scope": "global",
                            "priority": 5,
                            "source_type": "rule",
                            "created_at": "2026-02-12T10:00:00.000Z"
                        }
                    }
                ],
                "next_page_offset": null
            },
            "status": "ok",
            "time": 0.001
        }"#;

        let response: ScrollResponse = serde_json::from_str(json).unwrap();
        assert_eq!(response.result.points.len(), 1);

        let payload = response.result.points[0].payload.as_ref().unwrap();
        assert_eq!(payload_str(payload, "label"), "test-label");
        assert_eq!(payload_str(payload, "content"), "test rule");
        assert_eq!(payload_u32(payload, "priority"), Some(5));
    }

    #[test]
    fn test_rule_json_serialization() {
        let rule = RuleJson {
            label: "test".to_string(),
            title: "Test Rule".to_string(),
            content: "test content".to_string(),
            scope: "global".to_string(),
            project_id: None,
            source_type: "rule".to_string(),
            priority: Some(5),
            tags: vec!["tag1".to_string(), "tag2".to_string()],
            created_at: "2026-02-12T10:00:00.000Z".to_string(),
            updated_at: "2026-02-12T10:00:00.000Z".to_string(),
        };
        let json = serde_json::to_string(&rule).unwrap();
        assert!(json.contains("\"label\":\"test\""));
        assert!(json.contains("\"priority\":5"));
        assert!(json.contains("tag1"));
    }

    #[test]
    fn test_normalize_commas_adds_spaces() {
        assert_eq!(normalize_commas("a,b,c"), "a, b, c");
        assert_eq!(normalize_commas("one,two,three"), "one, two, three");
    }

    #[test]
    fn test_normalize_commas_preserves_existing_spaces() {
        assert_eq!(normalize_commas("a, b, c"), "a, b, c");
        assert_eq!(normalize_commas("a,b, c"), "a, b, c");
    }

    #[test]
    fn test_normalize_commas_empty() {
        assert_eq!(normalize_commas(""), "");
        assert_eq!(normalize_commas("no-commas"), "no-commas");
    }

    #[test]
    fn test_format_title_global_scope() {
        let payload = serde_json::json!({
            "title": "Some Rule",
            "scope": "global",
        });
        let names = HashMap::new();
        assert_eq!(
            format_title_with_project(&payload, &names, false),
            "Some Rule"
        );
        assert_eq!(
            format_title_with_project(&payload, &names, true),
            "Some Rule"
        );
    }

    #[test]
    fn test_format_title_project_scope_with_name() {
        let payload = serde_json::json!({
            "title": "My Rule",
            "scope": "project",
            "project_id": "abc123",
        });
        let mut names = HashMap::new();
        names.insert("abc123".to_string(), "my-project".to_string());

        assert_eq!(
            format_title_with_project(&payload, &names, false),
            "My Rule (project: my-project)"
        );
        assert_eq!(
            format_title_with_project(&payload, &names, true),
            "My Rule (project: my-project / abc123)"
        );
    }

    #[test]
    fn test_format_title_project_scope_unknown_id() {
        let payload = serde_json::json!({
            "title": "My Rule",
            "scope": "project",
            "project_id": "unknown999",
        });
        let names = HashMap::new();
        // Falls back to showing just the tenant_id
        assert_eq!(
            format_title_with_project(&payload, &names, false),
            "My Rule (project: unknown999)"
        );
        assert_eq!(
            format_title_with_project(&payload, &names, true),
            "My Rule (project: unknown999)"
        );
    }

    #[test]
    fn test_build_scope_filter_global() {
        let filter = build_scope_filter("global");
        let must = filter["must"].as_array().unwrap();
        assert_eq!(must.len(), 1);
        assert_eq!(must[0]["key"], "scope");
        assert_eq!(must[0]["match"]["value"], "global");
    }

    #[test]
    fn test_build_scope_filter_project() {
        let filter = build_scope_filter("project:abc123");
        let must = filter["must"].as_array().unwrap();
        assert_eq!(must.len(), 2);
        assert_eq!(must[0]["key"], "scope");
        assert_eq!(must[0]["match"]["value"], "project");
        assert_eq!(must[1]["key"], "project_id");
        assert_eq!(must[1]["match"]["value"], "abc123");
    }
}
