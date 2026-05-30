//! `add_rule`, `update_rule`, `remove_rule` mutations for the rules tool.
//!
//! ## add_rule execution order (mirrors rules.ts:65-93, rules-mutations.ts:25-42)
//!
//! 1. Extract content (required).
//! 2. Run dup-check via `find_similar_rules` **before** label/tenant checks —
//!    TS runs the dup-check at the `execute()` level (rules.ts:71-91) which is
//!    called before `addRule` (rules-mutations.ts:25-42), so a duplicate add
//!    with a bad label/scope still returns the dup-refusal, not a validation error.
//! 3. Check label (required).
//! 4. Resolve tenant.
//! 5. Ingest or queue.

use rmcp::model::CallToolResult;

use wqm_common::constants::TENANT_GLOBAL;

use crate::tools::envelope::{error_text, ok_text};

use super::helpers::{
    build_add_metadata, build_update_metadata, queue_rule_op, resolve_tenant, upsert_mirror,
};
use super::list::find_similar_rules;
use super::traits::{RulesDaemon, RulesQdrant};
use super::types::{RulesInput, RulesResponse};

// ─────────────────────────────────────────────────────────────────────────────
// add_rule — mirrors `persistAddRule` in rules-mutation-helpers.ts:177-218
//            with dup-check prepended from rules.ts:68-93
// ─────────────────────────────────────────────────────────────────────────────

/// Run the duplicate-detection step. Returns `Some(refusal)` when ≥1 similar
/// rule is found (the add must NOT proceed), `None` to allow the add. Mirrors
/// the `findSimilarRules` + refusal branch in rules.ts:71-91.
async fn add_dup_refusal<D, Q>(
    input: &RulesInput,
    daemon: &mut D,
    qdrant: &Q,
    content: &str,
    session_project_id: Option<&str>,
    duplication_threshold: f64,
) -> Option<CallToolResult>
where
    D: RulesDaemon,
    Q: RulesQdrant,
{
    let dup_scope: &str = input.scope.as_str();
    let dup_pid: Option<&str> = input
        .project_id
        .as_deref()
        .filter(|s| !s.is_empty())
        .or(session_project_id);
    let duplicates = find_similar_rules(
        daemon,
        qdrant,
        content,
        dup_scope,
        dup_pid,
        duplication_threshold,
    )
    .await;

    if duplicates.is_empty() {
        return None;
    }
    let count = duplicates.len();
    Some(ok_text(&RulesResponse {
        success: false,
        action: "add".to_string(),
        label: None,
        rules: None,
        similar_rules: Some(duplicates),
        message: Some(format!(
            "Found {count} similar rule(s). Review before adding to avoid duplication."
        )),
        fallback_mode: None,
        queue_id: None,
    }))
}

pub async fn add_rule<D, Q>(
    input: RulesInput,
    daemon: &mut D,
    qdrant: &Q,
    session_project_id: Option<&str>,
    duplication_threshold: f64,
) -> CallToolResult
where
    D: RulesDaemon,
    Q: RulesQdrant,
{
    // 1. Content required
    let content = match input.content.as_deref() {
        Some(c) if !c.trim().is_empty() => c,
        _ => {
            return ok_text(&RulesResponse::error(
                "add",
                "Content is required for adding a rule",
            ));
        }
    };

    // 2. Dup-check BEFORE label/tenant checks — rules.ts:71-91 runs this at
    //    the execute() level before delegating to addRule (rules-mutations.ts).
    //    Scope uses the raw input scope so an unresolvable-project add with
    //    duplicates still returns the dup-refusal.
    if let Some(refusal) = add_dup_refusal(
        &input,
        daemon,
        qdrant,
        content,
        session_project_id,
        duplication_threshold,
    )
    .await
    {
        return refusal;
    }

    // 3. Label required
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

    // 4-5. Resolve tenant, then persist (ingest-first with queue fallback).
    add_persist(&input, daemon, content, label, session_project_id).await
}

/// Resolve tenant, build metadata, and persist the rule via `ingest_text`
/// (direct path) with a queue fallback. Mirrors rules-mutation-helpers.ts:189-218.
async fn add_persist<D: RulesDaemon>(
    input: &RulesInput,
    daemon: &mut D,
    content: &str,
    label: &str,
    session_project_id: Option<&str>,
) -> CallToolResult {
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
            on_add_ingest_success(daemon, label, content, &input.scope, tenant_id_str).await
        }
        Ok(false) | Err((true, _)) => queue_add_fallback(daemon, label, content, input).await,
        Err((false, e)) => error_text(&format!("Failed to add rule: {e}")),
    }
}

/// Called when ingest succeeded directly — upsert mirror and return success.
async fn on_add_ingest_success<D: RulesDaemon>(
    daemon: &mut D,
    label: &str,
    content: &str,
    scope: &str,
    tenant_id_str: Option<&str>,
) -> CallToolResult {
    upsert_mirror(daemon, label, content, scope, tenant_id_str).await;
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

/// Queue fallback for add — soft failure or connectivity error.
///
/// Mirrors `buildAddQueueOp` + `persistAddRule` fallback path in
/// rules-mutation-helpers.ts:200-209.
async fn queue_add_fallback<D: RulesDaemon>(
    daemon: &mut D,
    label: &str,
    content: &str,
    input: &RulesInput,
) -> CallToolResult {
    let tenant_id_str: Option<&str> = input.project_id.as_deref().filter(|s| !s.is_empty());
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

// ─────────────────────────────────────────────────────────────────────────────
// update_rule — mirrors `persistUpdateRule` in rules-mutation-helpers.ts:258-303
// ─────────────────────────────────────────────────────────────────────────────

pub async fn update_rule<D, Q>(
    input: RulesInput,
    daemon: &mut D,
    _qdrant: &Q,
    session_project_id: Option<&str>,
) -> CallToolResult
where
    D: RulesDaemon,
    Q: RulesQdrant,
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
        Err(e) => return ok_text(&RulesResponse::error("update", e)),
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
            on_update_ingest_success(daemon, label, content, &input.scope, tenant_id_str).await
        }
        Ok(false) | Err((true, _)) => queue_update_fallback(daemon, label, content, &input).await,
        Err((false, e)) => error_text(&format!("Failed to update rule: {e}")),
    }
}

/// Called when update ingest succeeded directly — upsert mirror and return success.
async fn on_update_ingest_success<D: RulesDaemon>(
    daemon: &mut D,
    label: &str,
    content: &str,
    scope: &str,
    tenant_id_str: Option<&str>,
) -> CallToolResult {
    upsert_mirror(daemon, label, content, scope, tenant_id_str).await;
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

/// Queue fallback for update — soft failure or connectivity error.
///
/// Mirrors `buildUpdateQueueOp` + `persistUpdateRule` fallback path in
/// rules-mutation-helpers.ts:284-303.
async fn queue_update_fallback<D: RulesDaemon>(
    daemon: &mut D,
    label: &str,
    content: &str,
    input: &RulesInput,
) -> CallToolResult {
    let tenant_id_str: Option<&str> = input.project_id.as_deref().filter(|s| !s.is_empty());
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

// ─────────────────────────────────────────────────────────────────────────────
// remove_rule — mirrors `removeRule` in rules-mutations.ts:96-132
// ─────────────────────────────────────────────────────────────────────────────

pub async fn remove_rule<D, Q>(
    input: RulesInput,
    daemon: &mut D,
    _qdrant: &Q,
    session_project_id: Option<&str>,
) -> CallToolResult
where
    D: RulesDaemon,
    Q: RulesQdrant,
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
