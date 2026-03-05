//! Rules reconciliation helpers for `rebuild_rules`.
//!
//! Extracted from system_service.rs to keep functions under 80 lines.

use std::collections::HashMap;
use std::sync::Arc;

use qdrant_client::qdrant::{value::Kind, Filter, RetrievedPoint};
use sqlx::SqlitePool;
use tracing::{info, warn};

use workspace_qdrant_core::StorageClient;

/// A single rule entry from Qdrant: (point_id, content, scope, tenant, updated_at).
pub(crate) type QdrantRuleEntry = (String, String, Option<String>, Option<String>, String);

/// Extract a string field from a Qdrant point payload.
pub(crate) fn extract_str(point: &RetrievedPoint, key: &str) -> Option<String> {
    point.payload.get(key).and_then(|v| {
        v.kind.as_ref().and_then(|k| match k {
            Kind::StringValue(s) => Some(s.clone()),
            _ => None,
        })
    })
}

/// Extract the point ID as a string from a Qdrant point.
pub(crate) fn extract_point_id_str(point: &RetrievedPoint) -> Option<String> {
    point.id.as_ref().and_then(|pid| {
        pid.point_id_options.as_ref().map(|opts| match opts {
            qdrant_client::qdrant::point_id::PointIdOptions::Uuid(u) => u.clone(),
            qdrant_client::qdrant::point_id::PointIdOptions::Num(n) => n.to_string(),
        })
    })
}

/// Step 1: Scroll all rules from Qdrant and index by label.
///
/// Returns `(by_label, unlabeled_count)`.
pub(crate) async fn scroll_qdrant_rules(
    storage: &Arc<StorageClient>,
) -> Result<(HashMap<String, Vec<QdrantRuleEntry>>, Vec<RetrievedPoint>), String> {
    match storage.collection_exists("rules").await {
        Ok(false) => {
            info!("[rebuild:rules] Rules collection does not exist — nothing to reconcile");
            return Err("no_collection".into());
        }
        Err(e) => return Err(format!("Failed to check rules collection: {}", e)),
        Ok(true) => {}
    }

    let all_points = storage
        .scroll_with_filter("rules", Filter::default(), 10000, None)
        .await
        .map_err(|e| format!("Failed to scroll rules from Qdrant: {}", e))?;

    Ok((index_points_by_label(&all_points), all_points))
}

/// Index Qdrant points by their label field.
fn index_points_by_label(points: &[RetrievedPoint]) -> HashMap<String, Vec<QdrantRuleEntry>> {
    let mut by_label: HashMap<String, Vec<QdrantRuleEntry>> = HashMap::new();

    for point in points {
        let point_id = match extract_point_id_str(point) {
            Some(id) => id,
            None => continue,
        };
        let content = extract_str(point, "content").unwrap_or_default();
        let scope = extract_str(point, "scope");
        let tenant = extract_str(point, "tenant_id");
        let updated_at = extract_str(point, "updated_at").unwrap_or_default();
        let label = extract_str(point, "label");

        match label {
            Some(l) if !l.is_empty() => {
                by_label
                    .entry(l)
                    .or_default()
                    .push((point_id, content, scope, tenant, updated_at));
            }
            _ => {
                warn!(
                    "[rebuild:rules] Qdrant rule point {} has no label — skipping",
                    point_id
                );
            }
        }
    }

    by_label
}

/// Step 2: Read all rules from SQLite rules_mirror.
pub(crate) async fn read_sqlite_rules(
    pool: &SqlitePool,
) -> Result<HashMap<String, (String, Option<String>, Option<String>)>, String> {
    let rows: Vec<(String, String, Option<String>, Option<String>)> =
        sqlx::query_as("SELECT rule_id, rule_text, scope, tenant_id FROM rules_mirror")
            .fetch_all(pool)
            .await
            .map_err(|e| format!("Failed to read rules_mirror: {}", e))?;

    Ok(rows
        .into_iter()
        .map(|(label, text, scope, tenant)| (label, (text, scope, tenant)))
        .collect())
}

/// Steps 3-4: Deduplicate labels and content, return point IDs to delete.
pub(crate) async fn deduplicate_rules(
    qdrant_by_label: &HashMap<String, Vec<QdrantRuleEntry>>,
    pool: &SqlitePool,
) -> (Vec<String>, u64, u64) {
    let mut ids_to_delete = Vec::new();
    let mut label_dups = 0u64;

    // Step 3: Deduplicate by label (keep newest)
    for (label, entries) in qdrant_by_label {
        if entries.len() > 1 {
            label_dups += 1;
            let mut sorted = entries.clone();
            sorted.sort_by(|a, b| b.4.cmp(&a.4));
            info!(
                "[rebuild:rules] Label '{}' has {} duplicates — keeping point {}",
                label,
                entries.len(),
                sorted[0].0
            );
            for stale in &sorted[1..] {
                ids_to_delete.push(stale.0.clone());
            }
        }
    }

    // Step 4: Deduplicate by content across labels
    let content_dups = dedup_by_content(qdrant_by_label, &mut ids_to_delete, pool).await;

    (ids_to_delete, label_dups, content_dups)
}

/// Detect duplicate content across different labels.
async fn dedup_by_content(
    qdrant_by_label: &HashMap<String, Vec<QdrantRuleEntry>>,
    ids_to_delete: &mut Vec<String>,
    pool: &SqlitePool,
) -> u64 {
    let mut content_map: HashMap<String, Vec<(String, String)>> = HashMap::new();
    let mut content_dups = 0u64;

    for (label, entries) in qdrant_by_label {
        if let Some(entry) = entries.first() {
            if !ids_to_delete.contains(&entry.0) {
                content_map
                    .entry(entry.1.clone())
                    .or_default()
                    .push((label.clone(), entry.0.clone()));
            }
        }
    }

    for entries in content_map.values() {
        if entries.len() > 1 {
            content_dups += 1;
            info!(
                "[rebuild:rules] Duplicate content across {} labels: {:?} — keeping '{}'",
                entries.len(),
                entries.iter().map(|(l, _)| l.as_str()).collect::<Vec<_>>(),
                entries[0].0
            );
            for dup in &entries[1..] {
                ids_to_delete.push(dup.1.clone());
                let _ = sqlx::query("DELETE FROM rules_mirror WHERE rule_id = ?1")
                    .bind(&dup.0)
                    .execute(pool)
                    .await;
            }
        }
    }

    content_dups
}

/// Build deduplicated Qdrant state: label -> (content, scope, tenant).
pub(crate) fn build_deduped_state(
    qdrant_by_label: &HashMap<String, Vec<QdrantRuleEntry>>,
    deleted_ids: &[String],
) -> HashMap<String, (String, Option<String>, Option<String>)> {
    let mut deduped = HashMap::new();
    for (label, entries) in qdrant_by_label {
        if let Some(winner) = entries.iter().find(|e| !deleted_ids.contains(&e.0)) {
            deduped.insert(
                label.clone(),
                (winner.1.clone(), winner.2.clone(), winner.3.clone()),
            );
        }
    }
    deduped
}

/// Steps 5-7: Bidirectional reconciliation between Qdrant and SQLite.
pub(crate) async fn reconcile_rules(
    pool: &SqlitePool,
    qdrant_deduped: &HashMap<String, (String, Option<String>, Option<String>)>,
    db_by_label: &HashMap<String, (String, Option<String>, Option<String>)>,
) -> (u64, u64, u64) {
    let now = wqm_common::timestamps::now_utc();
    let mut inserted = 0u64;
    let mut updated = 0u64;
    let mut enqueued = 0u64;

    // Step 5 & 7: Qdrant -> SQLite sync
    for (label, (q_content, q_scope, q_tenant)) in qdrant_deduped {
        match db_by_label.get(label) {
            None => {
                let _ = sqlx::query(
                    "INSERT OR IGNORE INTO rules_mirror \
                     (rule_id, rule_text, scope, tenant_id, created_at, updated_at) \
                     VALUES (?1, ?2, ?3, ?4, ?5, ?6)",
                )
                .bind(label)
                .bind(q_content)
                .bind(q_scope)
                .bind(q_tenant)
                .bind(&now)
                .bind(&now)
                .execute(pool)
                .await;
                inserted += 1;
            }
            Some((db_content, _, _)) if db_content != q_content => {
                let _ = sqlx::query(
                    "UPDATE rules_mirror SET rule_text = ?1, scope = ?2, \
                     tenant_id = ?3, updated_at = ?4 WHERE rule_id = ?5",
                )
                .bind(q_content)
                .bind(q_scope)
                .bind(q_tenant)
                .bind(&now)
                .bind(label)
                .execute(pool)
                .await;
                updated += 1;
            }
            _ => {}
        }
    }

    // Step 6: SQLite -> Qdrant enqueue
    for (label, (db_content, db_scope, db_tenant)) in db_by_label {
        if !qdrant_deduped.contains_key(label) {
            enqueue_rule_ingestion(pool, label, db_content, db_scope, db_tenant, &now).await;
            enqueued += 1;
        }
    }

    (inserted, updated, enqueued)
}

/// Enqueue a single rule for re-ingestion into Qdrant.
async fn enqueue_rule_ingestion(
    pool: &SqlitePool,
    label: &str,
    content: &str,
    scope: &Option<String>,
    tenant: &Option<String>,
    now: &str,
) {
    let tid = tenant.as_deref().unwrap_or("global");
    let payload = serde_json::json!({
        "content": content,
        "scope": scope,
        "label": label,
    });
    let payload_str = payload.to_string();
    let idem_key = wqm_common::hashing::compute_content_hash(&format!(
        "text|add|{}|rules|{}",
        tid, payload_str
    ));

    let _ = sqlx::query(
        "INSERT OR IGNORE INTO unified_queue \
         (queue_id, idempotency_key, item_type, op, tenant_id, collection, \
          priority, status, payload_json, created_at, updated_at) \
         VALUES (?1, ?2, 'text', 'add', ?3, 'rules', 8, 'pending', ?4, ?5, ?6)",
    )
    .bind(uuid::Uuid::new_v4().to_string())
    .bind(&idem_key[..32])
    .bind(tid)
    .bind(&payload_str)
    .bind(now)
    .bind(now)
    .execute(pool)
    .await;
}
