//! `add_rule`, `update_rule`, `remove_rule` mutations for the rules tool.
//!
//! ## add_rule — dup-check BEFORE add (FIX 2)
//!
//! If `content` is non-empty, `find_similar_rules` is called first.
//! When duplicates are found the function returns early with:
//! `{success:false, action:'add', similar_rules:[…],
//!   message:"Found N similar rule(s). Review before adding to avoid duplication."}`
//! (rules.ts:83-90).
//!
//! Only when no duplicates are found does the function proceed to `ingest_text`.

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

    // ── FIX 2: duplicate detection BEFORE add (rules.ts:71-91) ───────────────
    // Only run when content is non-empty (already guaranteed above).
    let dup_scope: &str = input.scope.as_str();
    let duplicates = find_similar_rules(
        daemon,
        qdrant,
        content,
        dup_scope,
        tenant_id_str,
        duplication_threshold,
    )
    .await;

    if !duplicates.is_empty() {
        let count = duplicates.len();
        return ok_text(&RulesResponse {
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
        });
    }

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
