//! Rule management: reassign-scope and update (amend) (#122).
//!
//! Located at: `src/rust/cli/src/commands/rules/manage.rs`
//!
//! Both verbs go through the enqueue-only write path. Reassign is
//! add-under-new-scope followed by remove-under-old-scope (in that order,
//! so a failure between the two leaves a duplicate rather than a lost
//! rule); update uses the daemon's `action: "update"` (delete + re-insert
//! by label, tenant-scoped). The rule's current state is resolved
//! read-only from Qdrant by label.
//!
//! Neighbors: `mod.rs` (dispatch), `add.rs` (enqueue shape this mirrors),
//! `helpers.rs` (Qdrant client + payload extraction).

use anyhow::{Context, Result};
use wqm_common::constants::TENANT_GLOBAL;

use crate::grpc::ensure_daemon_available;
use crate::grpc::proto::{EnqueueItemRequest, QueueType, RefreshSignalRequest};
use crate::output;

use super::helpers::{build_qdrant_client, payload_str, payload_u32, qdrant_url, ScrollResponse};

/// Resolve a rule's full payload by exact label match.
async fn fetch_rule_by_label(label: &str) -> Result<serde_json::Value> {
    let client = build_qdrant_client()?;
    let url = format!("{}/collections/rules/points/scroll", qdrant_url());

    let body = serde_json::json!({
        "limit": 2,
        "with_payload": true,
        "filter": {
            "must": [{
                "key": "label",
                "match": { "value": label }
            }]
        }
    });

    let response = client
        .post(&url)
        .json(&body)
        .send()
        .await
        .context("Failed to connect to Qdrant")?;

    if !response.status().is_success() {
        anyhow::bail!("Qdrant request failed ({})", response.status());
    }

    let scroll: ScrollResponse = response
        .json()
        .await
        .context("Failed to parse Qdrant response")?;

    match scroll.result.points.len() {
        0 => anyhow::bail!("No rule found with label '{}'", label),
        1 => {}
        _ => anyhow::bail!(
            "Multiple rules share label '{}' (different scopes) — \
             remove the stale one first with: wqm rules remove --label {}",
            label,
            label
        ),
    }

    scroll.result.points[0]
        .payload
        .clone()
        .context("Rule has no payload data")
}

/// Enqueue one rules write item and nudge the daemon.
async fn enqueue_rules_item(op_payload: serde_json::Value, tenant_id: &str) -> Result<String> {
    let mut client = ensure_daemon_available().await?;

    let response = client
        .queue_write()
        .enqueue_item(EnqueueItemRequest {
            item_type: "text".to_string(),
            op: "add".to_string(),
            tenant_id: tenant_id.to_string(),
            collection: "rules".to_string(),
            payload_json: op_payload.to_string(),
            branch: "main".to_string(),
            metadata_json: None,
        })
        .await?
        .into_inner();

    let request = RefreshSignalRequest {
        queue_type: QueueType::IngestQueue as i32,
        lsp_languages: vec![],
        grammar_languages: vec![],
    };
    let _ = client.system().send_refresh_signal(request).await;

    Ok(response.queue_id)
}

/// Move a rule between global and project scope.
///
/// `to_project`: `Some(project)` → project scope; `None` → global scope.
pub async fn reassign_rule(label: &str, to_project: Option<String>) -> Result<()> {
    let payload = fetch_rule_by_label(label).await?;

    let old_scope = payload_str(&payload, "scope");
    let old_tenant = payload_str(&payload, "tenant_id");

    let (new_scope, new_project_id, new_tenant) = match &to_project {
        Some(p) => {
            let path = std::path::Path::new(p);
            let pid = if path.exists() {
                wqm_common::project_id::calculate_tenant_id(path)
            } else {
                p.clone()
            };
            ("project".to_string(), Some(pid.clone()), pid)
        }
        None => (TENANT_GLOBAL.to_string(), None, TENANT_GLOBAL.to_string()),
    };

    if new_tenant == old_tenant {
        output::info(format!("Rule already belongs to tenant '{}'", new_tenant));
        return Ok(());
    }

    output::section("Reassign Rule");
    output::kv("Label", label);
    output::kv("From", format!("{} ({})", old_scope, old_tenant));
    output::kv("To", format!("{} ({})", new_scope, new_tenant));
    output::separator();

    // 1. Add under the new scope (action "add" preserves all rule fields).
    let mut add_payload = serde_json::json!({
        "content": payload_str(&payload, "content"),
        "source_type": "rule",
        "label": label,
        "scope": new_scope,
        "action": "add",
    });
    if let Some(pid) = &new_project_id {
        add_payload["project_id"] = serde_json::json!(pid);
    }
    let title = payload_str(&payload, "title");
    if !title.is_empty() {
        add_payload["title"] = serde_json::json!(title);
    }
    if let Some(priority) = payload_u32(&payload, "priority") {
        add_payload["priority"] = serde_json::json!(priority);
    }
    let tags = payload_str(&payload, "tags");
    if !tags.is_empty() {
        add_payload["tags"] = serde_json::json!(tags.split(',').map(str::trim).collect::<Vec<_>>());
    }
    let add_id = enqueue_rules_item(add_payload, &new_tenant).await?;

    // 2. Remove under the old scope (tenant-scoped, cannot touch the new copy).
    let remove_payload = serde_json::json!({
        "content": "",
        "source_type": "rule",
        "label": label,
        "action": "remove",
    });
    let remove_id = enqueue_rules_item(remove_payload, &old_tenant).await?;

    output::success("Reassignment queued (add + remove)");
    output::kv("Add Queue ID", &add_id);
    output::kv("Remove Queue ID", &remove_id);
    Ok(())
}

/// Amend a rule in place: new content and/or title, same label and scope.
///
/// Uses the daemon's `action: "update"` handling (tenant-scoped delete by
/// label + re-insert), preserving scope, project binding, priority, and tags.
pub async fn update_rule(
    label: &str,
    content: Option<String>,
    title: Option<String>,
) -> Result<()> {
    if content.is_none() && title.is_none() {
        anyhow::bail!("Nothing to update: pass --content and/or --title");
    }

    let payload = fetch_rule_by_label(label).await?;
    let tenant_id = payload_str(&payload, "tenant_id");

    let new_content = content.unwrap_or_else(|| payload_str(&payload, "content"));
    let old_title = payload_str(&payload, "title");
    let new_title = title.unwrap_or(old_title);

    let mut update_payload = serde_json::json!({
        "content": new_content,
        "source_type": "rule",
        "label": label,
        "action": "update",
    });
    let scope = payload_str(&payload, "scope");
    if !scope.is_empty() {
        update_payload["scope"] = serde_json::json!(scope);
    }
    let project_id = payload_str(&payload, "project_id");
    if !project_id.is_empty() {
        update_payload["project_id"] = serde_json::json!(project_id);
    }
    if !new_title.is_empty() {
        update_payload["title"] = serde_json::json!(new_title);
    }
    if let Some(priority) = payload_u32(&payload, "priority") {
        update_payload["priority"] = serde_json::json!(priority);
    }
    let tags = payload_str(&payload, "tags");
    if !tags.is_empty() {
        update_payload["tags"] =
            serde_json::json!(tags.split(',').map(str::trim).collect::<Vec<_>>());
    }

    let queue_id = enqueue_rules_item(update_payload, &tenant_id).await?;

    output::section("Rule Update Queued");
    output::kv("Label", label);
    output::kv("Queue ID", &queue_id);
    Ok(())
}
