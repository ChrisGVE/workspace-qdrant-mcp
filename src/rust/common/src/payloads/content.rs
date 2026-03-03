//! Payloads for text-based content: user content, scratchpad entries, memory rules

use serde::{Deserialize, Serialize};

/// Payload for text items (was "content" in old taxonomy)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ContentPayload {
    /// The actual text content
    pub content: String,
    /// Source type: scratchbook, mcp, clipboard
    pub source_type: String,
    /// Primary categorization tag
    #[serde(skip_serializing_if = "Option::is_none")]
    pub main_tag: Option<String>,
    /// Full hierarchical tag
    #[serde(skip_serializing_if = "Option::is_none")]
    pub full_tag: Option<String>,
}

/// Payload for scratchpad items (persistent LLM scratch space)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScratchpadPayload {
    /// The text content
    pub content: String,
    /// Optional title for the entry
    #[serde(skip_serializing_if = "Option::is_none")]
    pub title: Option<String>,
    /// Tags for categorization
    #[serde(default)]
    pub tags: Vec<String>,
    /// Source type (always "scratchpad")
    #[serde(default = "default_scratchpad_source")]
    pub source_type: String,
}

fn default_scratchpad_source() -> String {
    "scratchpad".to_string()
}

/// Payload for memory rule items (queued via MCP memory tool)
///
/// Memory rules have their own payload type because they carry metadata
/// (label, scope, title, tags, priority) that must be persisted in the
/// Qdrant point payload for filtering and display.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryPayload {
    /// Rule content text (required for add/update, optional for remove)
    #[serde(default)]
    pub content: String,
    /// Source type (always "memory_rule")
    #[serde(default)]
    pub source_type: String,
    /// Rule label (identifier, max 15 chars, e.g. "prefer-uv")
    #[serde(skip_serializing_if = "Option::is_none")]
    pub label: Option<String>,
    /// Action: add, update, remove
    #[serde(skip_serializing_if = "Option::is_none")]
    pub action: Option<String>,
    /// Scope: global or project
    #[serde(skip_serializing_if = "Option::is_none")]
    pub scope: Option<String>,
    /// Project ID for project-scoped rules
    #[serde(skip_serializing_if = "Option::is_none")]
    pub project_id: Option<String>,
    /// Rule title (max 50 chars)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub title: Option<String>,
    /// Tags for categorization
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tags: Option<Vec<String>>,
    /// Priority (higher = more important)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub priority: Option<u32>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_content_payload_serde() {
        let payload = ContentPayload {
            content: "test content".to_string(),
            source_type: "cli".to_string(),
            main_tag: Some("tag1".to_string()),
            full_tag: None,
        };
        let json = serde_json::to_string(&payload).unwrap();
        assert!(json.contains("test content"));
        assert!(json.contains("cli"));
        assert!(json.contains("tag1"));
        assert!(!json.contains("full_tag"));

        let back: ContentPayload = serde_json::from_str(&json).unwrap();
        assert_eq!(back.content, "test content");
    }

    #[test]
    fn test_memory_payload_full_serde() {
        let payload = MemoryPayload {
            content: "always use bun".to_string(),
            source_type: "memory_rule".to_string(),
            label: Some("prefer-bun".to_string()),
            action: Some("add".to_string()),
            scope: Some("global".to_string()),
            project_id: None,
            title: Some("Prefer bun over npm".to_string()),
            tags: Some(vec!["tooling".to_string(), "workflow".to_string()]),
            priority: Some(8),
        };
        let json = serde_json::to_string(&payload).unwrap();
        assert!(json.contains("prefer-bun"));
        assert!(json.contains("global"));
        assert!(json.contains("tooling"));
        assert!(!json.contains("project_id"));

        let back: MemoryPayload = serde_json::from_str(&json).unwrap();
        assert_eq!(back.label, Some("prefer-bun".to_string()));
        assert_eq!(
            back.tags,
            Some(vec!["tooling".to_string(), "workflow".to_string()])
        );
        assert_eq!(back.priority, Some(8));
    }

    #[test]
    fn test_memory_payload_minimal_serde() {
        let json = r#"{"content":"test rule","source_type":"memory_rule"}"#;
        let payload: MemoryPayload = serde_json::from_str(json).unwrap();
        assert_eq!(payload.content, "test rule");
        assert_eq!(payload.label, None);
        assert_eq!(payload.scope, None);
        assert_eq!(payload.tags, None);
    }

    #[test]
    fn test_memory_payload_from_mcp_json() {
        // Simulate the JSON the MCP server actually sends
        let json = r#"{
            "content": "deploy after build",
            "source_type": "memory_rule",
            "label": "deploy-after-build",
            "action": "add",
            "scope": "project",
            "project_id": "abc123",
            "title": "Deploy binaries after changes",
            "tags": ["workflow", "deployment"],
            "priority": 9
        }"#;
        let payload: MemoryPayload = serde_json::from_str(json).unwrap();
        assert_eq!(payload.scope, Some("project".to_string()));
        assert_eq!(payload.project_id, Some("abc123".to_string()));
        assert_eq!(payload.priority, Some(9));
    }

    #[test]
    fn test_memory_payload_remove_no_content() {
        // Remove action omits content — this must not fail deserialization
        let json = r#"{"action":"remove","label":"old-rule","source_type":"memory_rule"}"#;
        let payload: MemoryPayload = serde_json::from_str(json).unwrap();
        assert_eq!(payload.action, Some("remove".to_string()));
        assert_eq!(payload.label, Some("old-rule".to_string()));
        assert_eq!(payload.content, ""); // defaults to empty string

        // Even without source_type (MCP server may omit it for remove)
        let json2 = r#"{"action":"remove","label":"old-rule"}"#;
        let payload2: MemoryPayload = serde_json::from_str(json2).unwrap();
        assert_eq!(payload2.label, Some("old-rule".to_string()));
        assert_eq!(payload2.source_type, "");
    }

    #[test]
    fn test_scratchpad_payload_full_serde() {
        let payload = ScratchpadPayload {
            content: "design decision: use RRF for fusion".to_string(),
            title: Some("Search Architecture".to_string()),
            tags: vec!["architecture".to_string(), "search".to_string()],
            source_type: "scratchpad".to_string(),
        };
        let json = serde_json::to_string(&payload).unwrap();
        assert!(json.contains("design decision"));
        assert!(json.contains("Search Architecture"));
        assert!(json.contains("architecture"));

        let back: ScratchpadPayload = serde_json::from_str(&json).unwrap();
        assert_eq!(back.content, "design decision: use RRF for fusion");
        assert_eq!(back.title, Some("Search Architecture".to_string()));
        assert_eq!(back.tags, vec!["architecture", "search"]);
    }

    #[test]
    fn test_scratchpad_payload_minimal_serde() {
        let json = r#"{"content":"quick note"}"#;
        let payload: ScratchpadPayload = serde_json::from_str(json).unwrap();
        assert_eq!(payload.content, "quick note");
        assert_eq!(payload.title, None);
        assert!(payload.tags.is_empty());
        assert_eq!(payload.source_type, "scratchpad");
    }
}
