//! Types, input parsing, and result structs for the `rules` MCP tool.
//!
//! Mirrors `rules-types.ts:10-47` and `tool-builders/rules.ts`.

use serde::Serialize;

// ─────────────────────────────────────────────────────────────────────────────
// Input
// ─────────────────────────────────────────────────────────────────────────────

/// Parsed input for the rules tool.
#[derive(Debug)]
pub struct RulesInput {
    /// `"add"` | `"update"` | `"remove"` | `"list"`
    pub action: String,
    pub content: Option<String>,
    pub label: Option<String>,
    /// `"global"` | `"project"` (default `"project"`)
    pub scope: String,
    pub project_id: Option<String>,
    pub title: Option<String>,
    pub tags: Option<Vec<String>>,
    pub priority: Option<i64>,
    /// default 50 — rules-list.ts:109
    pub limit: usize,
}

impl RulesInput {
    /// Parse from the JSON arguments map.
    ///
    /// Mirrors `buildRuleOptions` in tool-builders/rules.ts and the
    /// action-dispatch in rules.ts:65-100.
    ///
    /// # Errors
    /// Returns `Err(message)` when `action` is missing or invalid.
    pub fn from_args(args: &serde_json::Map<String, serde_json::Value>) -> Result<Self, String> {
        let action = args.get("action").and_then(|v| v.as_str()).unwrap_or("");
        if !matches!(action, "add" | "update" | "remove" | "list") {
            return Err(format!("Invalid rules action: {action}"));
        }

        let content = args
            .get("content")
            .and_then(|v| v.as_str())
            .map(str::to_string);

        let label = args
            .get("label")
            .and_then(|v| v.as_str())
            .map(str::to_string);

        let scope = args
            .get("scope")
            .and_then(|v| v.as_str())
            .filter(|s| matches!(*s, "global" | "project"))
            .unwrap_or("project")
            .to_string();

        let project_id = args
            .get("projectId")
            .and_then(|v| v.as_str())
            .map(str::to_string);

        let title = args
            .get("title")
            .and_then(|v| v.as_str())
            .map(str::to_string);

        let tags: Option<Vec<String>> = args.get("tags").and_then(|v| v.as_array()).map(|arr| {
            arr.iter()
                .filter_map(|v| v.as_str().map(str::to_string))
                .collect()
        });

        // parseInt semantics for priority — accepts integer JSON number
        let priority = args.get("priority").and_then(|v| v.as_i64());

        let limit = args
            .get("limit")
            .and_then(|v| v.as_u64())
            .map(|n| n as usize)
            .unwrap_or(50);

        Ok(Self {
            action: action.to_string(),
            content,
            label,
            scope,
            project_id,
            title,
            tags,
            priority,
            limit,
        })
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Result types — field order MUST match TS `RuleResponse` declaration order
// (rules-types.ts:38-47)
//
// success → action → label? → rules? → similar_rules? → message? →
// fallback_mode? → queue_id?
//
// RuleItem field order (pointToRule, rules-list.ts:35-54):
// id → content → scope → label? → projectId? → title? → tags? →
// priority? → createdAt? → updatedAt? → similarity?
// ─────────────────────────────────────────────────────────────────────────────

/// A single rule in a list or duplicate result.
///
/// Mirrors TS `Rule` interface (rules-types.ts:13-24) and the key-emit order of
/// `pointToRule` in rules-list.ts:35-54:
///   id → content → scope → label? → projectId? → title? → tags? →
///   priority? → createdAt? → updatedAt? → similarity?
///
/// `content` and `scope` come BEFORE `label` — matches the order in which
/// `pointToRule` assigns them (lines 37-38 assign content+scope unconditionally,
/// then label at line 41 conditionally).
#[derive(Debug, Serialize)]
pub struct RuleItem {
    pub id: String,
    pub content: String,
    pub scope: String,
    #[serde(rename = "label", skip_serializing_if = "Option::is_none")]
    pub label: Option<String>,
    #[serde(rename = "projectId", skip_serializing_if = "Option::is_none")]
    pub project_id: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub title: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tags: Option<Vec<String>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub priority: Option<i64>,
    #[serde(rename = "createdAt", skip_serializing_if = "Option::is_none")]
    pub created_at: Option<String>,
    #[serde(rename = "updatedAt", skip_serializing_if = "Option::is_none")]
    pub updated_at: Option<String>,
    /// Duplicate similarity score — present only in `similar_rules` entries.
    /// Rounded to 3 decimals: `(score * 1000.0).round() / 1000.0`.
    /// (rules.ts:153)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub similarity: Option<f64>,
}

/// Rules tool response — mirrors TS `RuleResponse` (rules-types.ts:38-47).
///
/// Field order: success → action → label? → rules? → similar_rules? →
/// message? → fallback_mode? → queue_id?
#[derive(Debug, Serialize)]
pub struct RulesResponse {
    pub success: bool,
    pub action: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub label: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub rules: Option<Vec<RuleItem>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub similar_rules: Option<Vec<RuleItem>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub message: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub fallback_mode: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub queue_id: Option<String>,
}

impl RulesResponse {
    pub fn error(action: &str, message: impl Into<String>) -> Self {
        Self {
            success: false,
            action: action.to_string(),
            label: None,
            rules: None,
            similar_rules: None,
            message: Some(message.into()),
            fallback_mode: None,
            queue_id: None,
        }
    }
}
