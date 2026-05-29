//! `rules` MCP tool handler.
//!
//! Mirrors `src/typescript/mcp-server/src/tools/rules.ts`,
//! `rules-mutations.ts`, `rules-list.ts`, and `rules-mutation-helpers.ts`.
//!
//! # Action dispatch (rules.ts:65-100)
//!
//! | action   | Rust path     | TS path            |
//! |----------|---------------|--------------------|
//! | `"add"`  | `add_rule`    | `addRule`          |
//! | `"update"`| `update_rule`| `updateRule`       |
//! | `"remove"`| `remove_rule`| `removeRule`       |
//! | `"list"` | `list_rules`  | `listRules`        |
//!
//! # Result shapes (field order per TS `RuleResponse` declaration)
//!
//! All responses: `{ success, action, label?, rules?, similar_rules?, message?,
//!                   fallback_mode?, queue_id? }`
//! (rules-types.ts:38-47)
//!
//! # document_id asymmetry (rules-mutation-helpers.ts:194 / :279)
//!
//! - ADD: `randomUUID()` — fresh UUID each time
//! - UPDATE: stable label string
//!
//! # Mirror writes
//!
//! add/update: `upsert_rule_mirror` (fire-and-forget).
//! remove:     `delete_rule_mirror` (fire-and-forget).
//!
//! # Connectivity error handling
//!
//! When `ingest_text` fails with a connectivity error, the tool falls back to
//! `enqueue_item` (queue path) — matching TS `persistAddRule`/`persistUpdateRule`.
//! Non-connectivity errors are propagated as error_text.

use rmcp::model::CallToolResult;
use serde::Serialize;
use std::collections::HashMap;

use wqm_common::constants::TENANT_GLOBAL;
use wqm_common::timestamps::now_utc;

use crate::canonicalize::payload_builders::{build_rule_payload, RulePayloadInput};
use crate::sqlite::rules_mirror::RulesMirrorEntry;
use crate::tools::envelope::{error_text, ok_text};

// ─────────────────────────────────────────────────────────────────────────────
// Public injectable trait
// ─────────────────────────────────────────────────────────────────────────────

/// Abstraction over daemon I/O needed by the rules tool.
///
/// Injected by tests via a mock to avoid live gRPC dependencies.
pub trait RulesDaemon {
    /// Ingest text via DocumentService.
    ///
    /// Returns `Ok(true)` when the daemon accepted the text and `Ok(false)` on
    /// a soft failure (daemon returned success=false). Connectivity errors
    /// (`Err(is_connectivity_error=true, ...)`) trigger queue fallback.
    fn ingest_text(
        &mut self,
        content: String,
        collection_basename: String,
        tenant_id: String,
        document_id: String,
        metadata: HashMap<String, String>,
    ) -> impl std::future::Future<Output = Result<bool, (bool, String)>> + Send;

    /// Enqueue an item into the unified queue.
    ///
    /// Returns the `queue_id` assigned by the daemon.
    fn enqueue_item(
        &mut self,
        item_type: &str,
        op: &str,
        tenant_id: &str,
        collection: &str,
        payload_json: &str,
        branch: &str,
        metadata_json: Option<&str>,
    ) -> impl std::future::Future<Output = Result<String, String>> + Send;

    /// Upsert rule into mirror — fire-and-forget.
    ///
    /// Mirrors `upsertMirror` in rules-mutation-helpers.ts:98-113.
    fn upsert_rule_mirror(
        &mut self,
        rule_id: String,
        rule_text: String,
        scope: Option<String>,
        tenant_id: Option<String>,
        created_at: String,
        updated_at: String,
    ) -> impl std::future::Future<Output = ()> + Send;

    /// Delete rule from mirror — fire-and-forget.
    ///
    /// Mirrors `stateManager.deleteRulesMirror(label)` in rules-mutations.ts:122.
    fn delete_rule_mirror(
        &mut self,
        rule_id: String,
    ) -> impl std::future::Future<Output = ()> + Send;
}

/// Abstraction over the rules list read path.
///
/// Separated from `RulesDaemon` so the list operation can use a SQLite
/// connection for the mirror read without a live daemon.
pub trait RulesReader {
    /// Read rules from the SQLite mirror, matching TS `readRulesFromMirror`.
    fn list_from_mirror(
        &self,
        scope: Option<&str>,
        tenant_id: Option<&str>,
        limit: usize,
    ) -> Vec<RulesMirrorEntry>;
}

// ─────────────────────────────────────────────────────────────────────────────
// Blanket impl: DaemonClient → RulesDaemon
// ─────────────────────────────────────────────────────────────────────────────

impl RulesDaemon for crate::grpc::DaemonClient {
    async fn ingest_text(
        &mut self,
        content: String,
        collection_basename: String,
        tenant_id: String,
        document_id: String,
        metadata: HashMap<String, String>,
    ) -> Result<bool, (bool, String)> {
        crate::grpc::DaemonClient::ingest_text(
            self,
            content,
            collection_basename,
            tenant_id,
            document_id,
            metadata,
        )
        .await
        .map(|resp| resp.success)
        .map_err(|e| {
            let msg = e.message().to_string();
            let is_conn = is_connectivity_error(&msg);
            (is_conn, msg)
        })
    }

    async fn enqueue_item(
        &mut self,
        item_type: &str,
        op: &str,
        tenant_id: &str,
        collection: &str,
        payload_json: &str,
        branch: &str,
        metadata_json: Option<&str>,
    ) -> Result<String, String> {
        crate::grpc::DaemonClient::enqueue_item(
            self,
            item_type.to_string(),
            op.to_string(),
            tenant_id.to_string(),
            collection.to_string(),
            payload_json.to_string(),
            branch.to_string(),
            metadata_json.map(str::to_string),
        )
        .await
        .map(|r| r.queue_id)
        .map_err(|e| e.to_string())
    }

    async fn upsert_rule_mirror(
        &mut self,
        rule_id: String,
        rule_text: String,
        scope: Option<String>,
        tenant_id: Option<String>,
        created_at: String,
        updated_at: String,
    ) {
        let _ = crate::grpc::DaemonClient::upsert_rule_mirror(
            self, rule_id, rule_text, scope, tenant_id, created_at, updated_at,
        )
        .await;
    }

    async fn delete_rule_mirror(&mut self, rule_id: String) {
        let _ = crate::grpc::DaemonClient::delete_rule_mirror(self, rule_id).await;
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Connectivity error check — mirrors `isConnectivityError` in
// rules-mutation-helpers.ts:21-33
// ─────────────────────────────────────────────────────────────────────────────

fn is_connectivity_error(msg: &str) -> bool {
    msg.contains("UNAVAILABLE")
        || msg.contains("DEADLINE_EXCEEDED")
        || msg.contains("ECONNREFUSED")
        || msg.contains("connect ECONNREFUSED")
}

// ─────────────────────────────────────────────────────────────────────────────
// Input parsing
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
// ─────────────────────────────────────────────────────────────────────────────

/// A single rule in a list or duplicate result.
///
/// Mirrors TS `Rule` interface (rules-types.ts:13-24).
/// Field order: id → label? → content → scope → projectId? → title? → tags? →
/// priority? → createdAt? → updatedAt?
#[derive(Debug, Serialize)]
pub struct RuleItem {
    pub id: String,
    #[serde(rename = "label", skip_serializing_if = "Option::is_none")]
    pub label: Option<String>,
    pub content: String,
    pub scope: String,
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
    fn error(action: &str, message: impl Into<String>) -> Self {
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

// ─────────────────────────────────────────────────────────────────────────────
// Tool entry point
// ─────────────────────────────────────────────────────────────────────────────

/// Execute the `rules` tool.
///
/// Dispatches by `action`. An invalid `action` returns `error_text` matching
/// the TS dispatcher throw semantics (tool-dispatcher.ts:106-109).
pub async fn rules_tool<D, R>(
    input: RulesInput,
    daemon: &mut D,
    reader: &R,
    session_project_id: Option<&str>,
) -> CallToolResult
where
    D: RulesDaemon,
    R: RulesReader,
{
    match input.action.as_str() {
        "add" => add_rule(input, daemon, session_project_id).await,
        "update" => update_rule(input, daemon, session_project_id).await,
        "remove" => remove_rule(input, daemon, session_project_id).await,
        "list" => list_rules::<R>(input, reader, session_project_id).await,
        other => error_text(&format!("Unknown action: {other}")),
    }
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
fn resolve_tenant(
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

async fn queue_rule_op<D>(
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

async fn upsert_mirror<D>(
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

fn build_add_metadata(
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

fn build_update_metadata(
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
// add_rule — mirrors `persistAddRule` in rules-mutation-helpers.ts:177-218
// ─────────────────────────────────────────────────────────────────────────────

async fn add_rule<D>(
    input: RulesInput,
    daemon: &mut D,
    session_project_id: Option<&str>,
) -> CallToolResult
where
    D: RulesDaemon,
{
    let content = match input.content.as_deref() {
        Some(c) if !c.trim().is_empty() => c,
        _ => {
            return ok_text(&RulesResponse::error(
                "add",
                "Content is required for adding a rule",
            ));
        }
    };
    let label = match input.label.as_deref() {
        Some(l) if !l.trim().is_empty() => l.trim(),
        _ => {
            return ok_text(&RulesResponse::error(
                "add",
                "Label is required for adding a rule (max 15 chars, format: word-word-word, \
                 e.g. \"prefer-uv\", \"use-pytest\")",
            ));
        }
    };

    let resolved_tenant = match resolve_tenant(
        &input.scope,
        input.project_id.as_deref(),
        session_project_id,
    ) {
        Err(e) => return ok_text(&RulesResponse::error("add", e)),
        Ok(t) => t,
    };
    let tenant_id_str: Option<&str> = resolved_tenant.as_deref();

    let metadata = build_add_metadata(
        label,
        &input.scope,
        tenant_id_str,
        input.title.as_deref(),
        input.tags.as_deref(),
        input.priority,
    );

    // Try ingest_text first (direct path) — rules-mutation-helpers.ts:189-209
    let doc_id = uuid::Uuid::new_v4().to_string(); // ADD: randomUUID()
    let result = daemon
        .ingest_text(
            content.to_string(),
            "rules".to_string(),
            tenant_id_str.unwrap_or(TENANT_GLOBAL).to_string(),
            doc_id,
            metadata,
        )
        .await;

    match result {
        Ok(true) => {
            // Direct ingest succeeded
            upsert_mirror(daemon, label, content, &input.scope, tenant_id_str).await;
            ok_text(&RulesResponse {
                success: true,
                action: "add".to_string(),
                label: Some(label.to_string()),
                rules: None,
                similar_rules: None,
                message: Some("Rule added successfully".to_string()),
                fallback_mode: None,
                queue_id: None,
            })
        }
        Ok(false) | Err((true, _)) => {
            // Soft failure or connectivity error → queue fallback
            let tags_ref: Option<Vec<&str>> = input
                .tags
                .as_deref()
                .map(|ts| ts.iter().map(String::as_str).collect());
            let queue_result = queue_rule_op(
                daemon,
                "add",
                label,
                Some(content),
                &input.scope,
                tenant_id_str,
                input.title.as_deref(),
                tags_ref,
                input.priority,
            )
            .await;
            match queue_result {
                Err(e) => error_text(&format!("Failed to queue rule: {e}")),
                Ok(queue_id) => {
                    upsert_mirror(daemon, label, content, &input.scope, tenant_id_str).await;
                    ok_text(&RulesResponse {
                        success: true,
                        action: "add".to_string(),
                        label: Some(label.to_string()),
                        rules: None,
                        similar_rules: None,
                        message: Some("Rule queued for processing".to_string()),
                        fallback_mode: Some("unified_queue".to_string()),
                        queue_id: Some(queue_id),
                    })
                }
            }
        }
        Err((false, e)) => error_text(&format!("Failed to add rule: {e}")),
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// update_rule — mirrors `persistUpdateRule` in rules-mutation-helpers.ts:258-303
// ─────────────────────────────────────────────────────────────────────────────

async fn update_rule<D>(
    input: RulesInput,
    daemon: &mut D,
    session_project_id: Option<&str>,
) -> CallToolResult
where
    D: RulesDaemon,
{
    let label = match input.label.as_deref() {
        Some(l) if !l.is_empty() => l,
        _ => {
            return ok_text(&RulesResponse::error(
                "update",
                "Label is required for updating",
            ));
        }
    };
    let content = match input.content.as_deref() {
        Some(c) if !c.trim().is_empty() => c,
        _ => {
            return ok_text(&RulesResponse::error(
                "update",
                "Content is required for updating a rule",
            ));
        }
    };

    let resolved_tenant = match resolve_tenant(
        &input.scope,
        input.project_id.as_deref(),
        session_project_id,
    ) {
        Err(e) => {
            // match TS: `{ ...error, action: 'update' }` — rules-mutations.ts:81
            return ok_text(&RulesResponse::error("update", e));
        }
        Ok(t) => t,
    };
    let tenant_id_str: Option<&str> = resolved_tenant.as_deref();

    let metadata = build_update_metadata(
        label,
        &input.scope,
        tenant_id_str,
        input.title.as_deref(),
        input.tags.as_deref(),
        input.priority,
    );

    // UPDATE: document_id = label (stable) — rules-mutation-helpers.ts:279
    let result = daemon
        .ingest_text(
            content.to_string(),
            "rules".to_string(),
            tenant_id_str.unwrap_or(TENANT_GLOBAL).to_string(),
            label.to_string(), // stable label as doc ID
            metadata,
        )
        .await;

    match result {
        Ok(true) => {
            upsert_mirror(daemon, label, content, &input.scope, tenant_id_str).await;
            ok_text(&RulesResponse {
                success: true,
                action: "update".to_string(),
                label: Some(label.to_string()),
                rules: None,
                similar_rules: None,
                message: Some("Rule updated successfully".to_string()),
                fallback_mode: None,
                queue_id: None,
            })
        }
        Ok(false) | Err((true, _)) => {
            let tags_ref: Option<Vec<&str>> = input
                .tags
                .as_deref()
                .map(|ts| ts.iter().map(String::as_str).collect());
            let queue_result = queue_rule_op(
                daemon,
                "update",
                label,
                Some(content),
                &input.scope,
                tenant_id_str,
                input.title.as_deref(),
                tags_ref,
                input.priority,
            )
            .await;
            match queue_result {
                Err(e) => error_text(&format!("Failed to queue rule update: {e}")),
                Ok(queue_id) => {
                    upsert_mirror(daemon, label, content, &input.scope, tenant_id_str).await;
                    ok_text(&RulesResponse {
                        success: true,
                        action: "update".to_string(),
                        label: Some(label.to_string()),
                        rules: None,
                        similar_rules: None,
                        message: Some("Rule update queued for processing".to_string()),
                        fallback_mode: Some("unified_queue".to_string()),
                        queue_id: Some(queue_id),
                    })
                }
            }
        }
        Err((false, e)) => error_text(&format!("Failed to update rule: {e}")),
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// remove_rule — mirrors `removeRule` in rules-mutations.ts:96-132
// ─────────────────────────────────────────────────────────────────────────────

async fn remove_rule<D>(
    input: RulesInput,
    daemon: &mut D,
    session_project_id: Option<&str>,
) -> CallToolResult
where
    D: RulesDaemon,
{
    let label = match input.label.as_deref() {
        Some(l) if !l.is_empty() => l,
        _ => {
            return ok_text(&RulesResponse::error(
                "remove",
                "Label is required for removal",
            ));
        }
    };

    let resolved_tenant = match resolve_tenant(
        &input.scope,
        input.project_id.as_deref(),
        session_project_id,
    ) {
        Err(e) => return ok_text(&RulesResponse::error("remove", e)),
        Ok(t) => t,
    };
    let tenant_id_str: Option<&str> = resolved_tenant.as_deref();

    let queue_result = queue_rule_op(
        daemon,
        "remove",
        label,
        None, // no content for remove
        &input.scope,
        tenant_id_str,
        None,
        None,
        None,
    )
    .await;

    match queue_result {
        Err(e) => error_text(&format!("Failed to queue rule removal: {e}")),
        Ok(queue_id) => {
            // Fire-and-forget mirror delete — rules-mutations.ts:122
            daemon.delete_rule_mirror(label.to_string()).await;
            ok_text(&RulesResponse {
                success: true,
                action: "remove".to_string(),
                label: Some(label.to_string()),
                rules: None,
                similar_rules: None,
                message: Some("Rule removal queued for processing".to_string()),
                fallback_mode: Some("unified_queue".to_string()),
                queue_id: Some(queue_id),
            })
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// list_rules — mirrors `listRules` in rules-list.ts:103-135
// ─────────────────────────────────────────────────────────────────────────────

/// Map a mirror entry to a `RuleItem`.
///
/// Mirrors `readRulesFromMirror` inner map in rules-list.ts:80-92.
fn mirror_to_rule_item(row: &RulesMirrorEntry) -> RuleItem {
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
    }
}

async fn list_rules<R>(
    input: RulesInput,
    reader: &R,
    session_project_id: Option<&str>,
) -> CallToolResult
where
    R: RulesReader,
{
    // Resolve project_id for project-scope list — rules-list.ts:111-114
    let resolved_pid: Option<String> = if input.scope == "project" {
        input
            .project_id
            .as_deref()
            .filter(|s| !s.is_empty())
            .map(str::to_string)
            .or_else(|| session_project_id.map(str::to_string))
    } else {
        None
    };

    let scope_str: Option<&str> = Some(input.scope.as_str());
    let tenant_str: Option<&str> = resolved_pid.as_deref();

    let rows = reader.list_from_mirror(scope_str, tenant_str, input.limit);
    let rules: Vec<RuleItem> = rows.iter().map(mirror_to_rule_item).collect();
    let count = rules.len();

    ok_text(&RulesResponse {
        success: true,
        action: "list".to_string(),
        label: None,
        rules: Some(rules),
        similar_rules: None,
        message: Some(format!("Found {count} rule(s)")),
        fallback_mode: None,
        queue_id: None,
    })
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
#[path = "rules_tests.rs"]
mod tests;
