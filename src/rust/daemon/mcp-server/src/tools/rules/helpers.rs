//! Shared helpers for the `rules` MCP tool.
//!
//! - `resolve_tenant`        — mirrors `resolveProjectScopeId` (rules-mutation-helpers.ts:116-142)
//! - `is_connectivity_error` — mirrors `isConnectivityError` (rules-mutation-helpers.ts:21-33)
//! - `queue_rule_op`         — mirrors `queueRuleOperation` (rules-mutation-helpers.ts:65-96)
//! - `upsert_mirror`         — mirrors `upsertMirror` (rules-mutation-helpers.ts:98-113)
//! - `build_add_metadata`    — mirrors `buildAddMetadata` (rules-mutation-helpers.ts:144-158)
//! - `build_update_metadata` — mirrors `buildUpdateMetadata` (rules-mutation-helpers.ts:220-234)
//! - `mirror_to_rule_item`   — mirrors `readRulesFromMirror` inner map (rules-list.ts:80-92)

use std::collections::HashMap;

use wqm_common::constants::TENANT_GLOBAL;
use wqm_common::timestamps::now_utc;

use crate::canonicalize::payload_builders::{build_rule_payload, RulePayloadInput};
use crate::sqlite::rules_mirror::RulesMirrorEntry;

use super::traits::RulesDaemon;
use super::types::RuleItem;

// ─────────────────────────────────────────────────────────────────────────────
// Connectivity error check — mirrors `isConnectivityError`
// in rules-mutation-helpers.ts:21-33
// ─────────────────────────────────────────────────────────────────────────────

pub fn is_connectivity_error(msg: &str) -> bool {
    msg.contains("UNAVAILABLE")
        || msg.contains("DEADLINE_EXCEEDED")
        || msg.contains("ECONNREFUSED")
        || msg.contains("connect ECONNREFUSED")
}

// ─────────────────────────────────────────────────────────────────────────────
// Resolve tenant ID (mirrors `resolveProjectScopeId` in
// rules-mutation-helpers.ts:116-142)
// ─────────────────────────────────────────────────────────────────────────────

/// Resolve the tenant ID for a scoped operation.
///
/// - `"global"` scope → `TENANT_GLOBAL`
/// - `"project"` scope with explicit `project_id` → that ID
/// - `"project"` scope with session fallback → session project ID
/// - `"project"` scope with nothing → `Err`
pub fn resolve_tenant(
    scope: &str,
    explicit_project_id: Option<&str>,
    session_project_id: Option<&str>,
) -> Result<Option<String>, String> {
    if scope == "global" {
        return Ok(None); // None means TENANT_GLOBAL
    }
    // project scope
    let resolved = explicit_project_id
        .filter(|s| !s.is_empty())
        .or(session_project_id.filter(|s| !s.is_empty()));
    if let Some(pid) = resolved {
        return Ok(Some(pid.to_string()));
    }
    Err(
        "Project-scoped rule requested but the current directory is not a registered project. \
         Run `wqm project watch <path>` first, or pass `projectId` explicitly, \
         or set `scope: \"global\"`."
            .to_string(),
    )
}

// ─────────────────────────────────────────────────────────────────────────────
// Queue helper — mirrors `queueRuleOperation` in rules-mutation-helpers.ts:65-96
// ─────────────────────────────────────────────────────────────────────────────

pub async fn queue_rule_op<D>(
    daemon: &mut D,
    action: &str,
    label: &str,
    content: Option<&str>,
    scope: &str,
    project_id: Option<&str>,
    title: Option<&str>,
    tags: Option<Vec<&str>>,
    priority: Option<i64>,
) -> Result<String, String>
where
    D: RulesDaemon,
{
    let queue_op = match action {
        "update" => "update",
        "remove" => "delete",
        _ => "add",
    };

    let payload_json = build_rule_payload(RulePayloadInput {
        action,
        label,
        content,
        scope: Some(scope),
        project_id,
        title,
        tags,
        priority,
    });

    let tenant_id = project_id.unwrap_or(TENANT_GLOBAL);

    daemon
        .enqueue_item(
            "text",
            queue_op,
            tenant_id,
            "rules",
            &payload_json,
            "main",
            Some("{\"source\":\"mcp_rules_tool\"}"),
        )
        .await
}

// ─────────────────────────────────────────────────────────────────────────────
// Mirror upsert helper
// ─────────────────────────────────────────────────────────────────────────────

pub async fn upsert_mirror<D>(
    daemon: &mut D,
    label: &str,
    content: &str,
    scope: &str,
    tenant_id: Option<&str>,
) where
    D: RulesDaemon,
{
    let now = now_utc();
    daemon
        .upsert_rule_mirror(
            label.to_string(),
            content.to_string(),
            Some(scope.to_string()),
            tenant_id.map(str::to_string),
            now.clone(),
            now,
        )
        .await;
}

// ─────────────────────────────────────────────────────────────────────────────
// Build metadata helpers — mirrors `buildAddMetadata` / `buildUpdateMetadata`
// in rules-mutation-helpers.ts:144-158 / :220-234
// ─────────────────────────────────────────────────────────────────────────────

pub fn build_add_metadata(
    label: &str,
    scope: &str,
    project_id: Option<&str>,
    title: Option<&str>,
    tags: Option<&[String]>,
    priority: Option<i64>,
) -> HashMap<String, String> {
    let mut m = HashMap::new();
    m.insert("scope".to_string(), scope.to_string());
    m.insert("rule_type".to_string(), "behavioral".to_string());
    m.insert("label".to_string(), label.to_string());
    if let Some(pid) = project_id {
        m.insert("project_id".to_string(), pid.to_string());
    }
    if let Some(t) = title {
        if !t.is_empty() {
            m.insert("title".to_string(), t.to_string());
        }
    }
    if let Some(ts) = tags {
        if !ts.is_empty() {
            m.insert("tags".to_string(), ts.join(","));
        }
    }
    if let Some(p) = priority {
        m.insert("priority".to_string(), p.to_string());
    }
    m
}

pub fn build_update_metadata(
    label: &str,
    scope: &str,
    project_id: Option<&str>,
    title: Option<&str>,
    tags: Option<&[String]>,
    priority: Option<i64>,
) -> HashMap<String, String> {
    let mut m = HashMap::new();
    m.insert("label".to_string(), label.to_string());
    m.insert("scope".to_string(), scope.to_string());
    if let Some(pid) = project_id {
        m.insert("project_id".to_string(), pid.to_string());
    }
    if let Some(t) = title {
        if !t.is_empty() {
            m.insert("title".to_string(), t.to_string());
        }
    }
    if let Some(ts) = tags {
        if !ts.is_empty() {
            m.insert("tags".to_string(), ts.join(","));
        }
    }
    if let Some(p) = priority {
        m.insert("priority".to_string(), p.to_string());
    }
    m
}

// ─────────────────────────────────────────────────────────────────────────────
// Mirror-row → RuleItem mapper
// ─────────────────────────────────────────────────────────────────────────────

/// Map a mirror entry to a `RuleItem`.
///
/// Mirrors `readRulesFromMirror` inner map in rules-list.ts:80-92.
pub fn mirror_to_rule_item(row: &RulesMirrorEntry) -> RuleItem {
    RuleItem {
        id: row.rule_id.clone(),
        label: None, // mirror schema has no separate label column
        content: row.rule_text.clone(),
        scope: row
            .scope
            .clone()
            .unwrap_or_else(|| TENANT_GLOBAL.to_string()),
        project_id: row.tenant_id.clone(),
        title: None,
        tags: None,
        priority: None,
        created_at: Some(row.created_at.clone()),
        updated_at: Some(row.updated_at.clone()),
        similarity: None,
        owner: Some(
            row.tenant_id
                .clone()
                .unwrap_or_else(|| TENANT_GLOBAL.to_string()),
        ),
    }
}
