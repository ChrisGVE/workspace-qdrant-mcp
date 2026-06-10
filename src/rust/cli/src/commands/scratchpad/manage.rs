//! Scratchpad entry management: delete, update (amend), reassign-scope (#122).
//!
//! Located at: `src/rust/cli/src/commands/scratchpad/manage.rs`
//!
//! All mutations go through the enqueue-only write path: the CLI enqueues
//! `{item_type: "text", collection: "scratchpad"}` items and the daemon's
//! TextStrategy applies them (content-addressed identity; `old_content`
//! triggers in-place edit, reassign is add-under-new-tenant + delete-under-
//! old-tenant). Entries are resolved read-only from Qdrant by title match,
//! like `info`/`list`.
//!
//! Neighbors: `mod.rs` (dispatch), `add.rs` (the add path this mirrors),
//! `client.rs` (Qdrant client + tenant resolution), `types.rs` (payloads).

use anyhow::{Context, Result};

use crate::grpc::ensure_daemon_available;
use crate::grpc::proto::{EnqueueItemRequest, QueueType, RefreshSignalRequest};
use crate::output;

use super::client::{build_qdrant_client, qdrant_url, resolve_tenant_id};
use super::types::{payload_str, payload_tags, ScrollResponse};

/// A fully resolved scratchpad entry (current stored state).
pub(super) struct Entry {
    pub title: String,
    pub content: String,
    pub tags: Vec<String>,
    pub tenant_id: String,
    pub document_id: String,
}

impl Entry {
    /// Name the user must type in the destructive confirmation.
    fn confirm_name(&self) -> &str {
        if self.title.is_empty() {
            &self.document_id
        } else {
            &self.title
        }
    }
}

/// Resolve exactly one scratchpad entry by title (full-text match, same
/// matcher `info` uses). Zero matches or more than one is an error — a
/// destructive or mutating command must never guess.
pub(super) async fn resolve_entry(identifier: &str) -> Result<Entry> {
    let client = build_qdrant_client()?;
    let collection = wqm_common::constants::COLLECTION_SCRATCHPAD;
    let url = format!("{}/collections/{}/points/scroll", qdrant_url(), collection);

    let body = serde_json::json!({
        "limit": 2,
        "with_payload": true,
        "filter": {
            "must": [{
                "key": "title",
                "match": { "text": identifier }
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
        let status = response.status();
        if status.as_u16() == 404 {
            anyhow::bail!("Scratchpad collection does not exist yet.");
        }
        let text = response.text().await.unwrap_or_default();
        anyhow::bail!("Qdrant request failed ({}): {}", status, text);
    }

    let scroll: ScrollResponse = response
        .json()
        .await
        .context("Failed to parse Qdrant response")?;

    match scroll.result.points.len() {
        0 => anyhow::bail!("No scratchpad entry found matching '{}'", identifier),
        1 => {}
        _ => anyhow::bail!(
            "Multiple scratchpad entries match '{}' — refine the title",
            identifier
        ),
    }

    let payload = scroll.result.points[0]
        .payload
        .as_ref()
        .context("Entry has no payload data")?;

    Ok(Entry {
        title: payload_str(payload, "title"),
        content: payload_str(payload, "content"),
        tags: payload_tags(payload),
        tenant_id: payload_str(payload, "tenant_id"),
        document_id: payload_str(payload, "document_id"),
    })
}

/// Enqueue one scratchpad write item and nudge the daemon.
async fn enqueue_scratchpad_item(
    op: &str,
    tenant_id: &str,
    payload_json: String,
) -> Result<String> {
    let mut client = ensure_daemon_available().await?;

    let response = client
        .queue_write()
        .enqueue_item(EnqueueItemRequest {
            item_type: "text".to_string(),
            op: op.to_string(),
            tenant_id: tenant_id.to_string(),
            collection: wqm_common::constants::COLLECTION_SCRATCHPAD.to_string(),
            payload_json,
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

/// Delete one scratchpad entry (typed confirmation, #123 gate).
pub(super) async fn delete_entry(identifier: &str, yes: bool) -> Result<()> {
    let entry = resolve_entry(identifier).await?;

    output::section("Delete Scratchpad Entry");
    output::kv(
        "Title",
        if entry.title.is_empty() {
            "(untitled)"
        } else {
            &entry.title
        },
    );
    output::kv("Tenant", &entry.tenant_id);
    output::separator();

    if !yes {
        output::warning("This permanently deletes the entry and cannot be undone.");
        if !output::typed_confirm(entry.confirm_name()) {
            output::info("Aborted");
            return Ok(());
        }
    }

    let payload_json = serde_json::json!({
        "content": entry.content,
        "source_type": "scratchpad",
    })
    .to_string();

    let queue_id = enqueue_scratchpad_item("delete", &entry.tenant_id, payload_json).await?;
    output::success("Deletion queued");
    output::kv("Queue ID", &queue_id);
    Ok(())
}

/// Amend one scratchpad entry: new content, title, and/or tags.
///
/// Identity is content-addressed, so a content change writes the new point
/// first and the daemon removes the superseded one (`old_content`).
pub(super) async fn update_entry(
    identifier: &str,
    content: Option<String>,
    title: Option<String>,
    tags: Option<String>,
) -> Result<()> {
    if content.is_none() && title.is_none() && tags.is_none() {
        anyhow::bail!("Nothing to update: pass --content, --title, and/or --tags");
    }

    let entry = resolve_entry(identifier).await?;

    let new_content = content.unwrap_or_else(|| entry.content.clone());
    let new_title = title.or_else(|| {
        if entry.title.is_empty() {
            None
        } else {
            Some(entry.title.clone())
        }
    });
    let new_tags: Vec<String> = match tags {
        Some(t) => t
            .split(',')
            .map(|s| s.trim().to_string())
            .filter(|s| !s.is_empty())
            .collect(),
        None => entry.tags.clone(),
    };

    let mut payload = serde_json::json!({
        "content": new_content,
        "title": new_title,
        "tags": new_tags,
        "source_type": "scratchpad",
    });
    if new_content != entry.content {
        payload["old_content"] = serde_json::json!(entry.content);
    }

    let queue_id = enqueue_scratchpad_item("update", &entry.tenant_id, payload.to_string()).await?;

    output::section("Scratchpad Update Queued");
    output::kv("Queue ID", &queue_id);
    output::kv("Tenant", &entry.tenant_id);
    if let Some(t) = &new_title {
        output::kv("Title", t);
    }
    Ok(())
}

/// Reassign one entry's scope: move it to another project (or global).
///
/// Implemented as add-under-new-tenant followed by delete-under-old-tenant,
/// in that order, so a failure between the two leaves a duplicate (safe,
/// re-runnable) rather than a lost entry.
pub(super) async fn reassign_entry(identifier: &str, to_project: Option<String>) -> Result<()> {
    let entry = resolve_entry(identifier).await?;
    let new_tenant = resolve_tenant_id(to_project.as_deref())?;

    if new_tenant == entry.tenant_id {
        output::info(format!("Entry already belongs to tenant '{}'", new_tenant));
        return Ok(());
    }

    output::section("Reassign Scratchpad Entry");
    output::kv(
        "Title",
        if entry.title.is_empty() {
            "(untitled)"
        } else {
            &entry.title
        },
    );
    output::kv("From", &entry.tenant_id);
    output::kv("To", &new_tenant);
    output::separator();

    let add_payload = serde_json::json!({
        "content": entry.content,
        "title": if entry.title.is_empty() { None } else { Some(entry.title.clone()) },
        "tags": entry.tags,
        "source_type": "scratchpad",
    })
    .to_string();
    let add_id = enqueue_scratchpad_item("add", &new_tenant, add_payload).await?;

    let delete_payload = serde_json::json!({
        "content": entry.content,
        "source_type": "scratchpad",
    })
    .to_string();
    let delete_id = enqueue_scratchpad_item("delete", &entry.tenant_id, delete_payload).await?;

    output::success("Reassignment queued (add + delete)");
    output::kv("Add Queue ID", &add_id);
    output::kv("Delete Queue ID", &delete_id);
    Ok(())
}
