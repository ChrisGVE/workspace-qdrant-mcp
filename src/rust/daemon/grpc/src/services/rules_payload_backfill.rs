//! Backfill for legacy rules missing payload scope/label fields.
//!
//! Issue #58: rules stored via pre-schema code paths have the
//! `RULE label:X scope:Y ...` tokens embedded in `content` but not
//! in the Qdrant payload. `wqm rules list` / `wqm rules inject`
//! therefore skip them (they filter by `scope`/`label` payload
//! fields).
//!
//! This one-shot scrolls the `rules` collection, parses the legacy
//! header out of any point missing a `label`, and `set_payload`s the
//! recovered `label`, `scope`, `project_id`, `rule_type`, `priority`,
//! and cleaned `content` fields. It also refreshes the SQLite
//! `rules_mirror` row so downstream reconciliation sees a consistent
//! state.

use std::sync::Arc;

use qdrant_client::qdrant::{value::Kind, Filter, RetrievedPoint};
use sqlx::SqlitePool;
use tracing::{info, warn};

use workspace_qdrant_core::StorageClient;
use wqm_common::rules_legacy::{parse_rule_header, LegacyRuleHeader};

/// Summary of a backfill pass, returned to callers / logged.
#[derive(Debug, Default, Clone, Copy)]
pub struct BackfillStats {
    pub scanned: u64,
    pub backfilled: u64,
    pub already_ok: u64,
    pub unparseable: u64,
    pub errors: u64,
    pub mirror_upserts: u64,
}

/// Extract a string-valued payload field from a Qdrant point.
fn extract_str(point: &RetrievedPoint, key: &str) -> Option<String> {
    point.payload.get(key).and_then(|v| {
        v.kind.as_ref().and_then(|k| match k {
            Kind::StringValue(s) => Some(s.clone()),
            _ => None,
        })
    })
}

/// Extract the point ID as a UUID string.
fn extract_point_id_str(point: &RetrievedPoint) -> Option<String> {
    point.id.as_ref().and_then(|pid| {
        pid.point_id_options.as_ref().map(|opts| match opts {
            qdrant_client::qdrant::point_id::PointIdOptions::Uuid(u) => u.clone(),
            qdrant_client::qdrant::point_id::PointIdOptions::Num(n) => n.to_string(),
        })
    })
}

/// Decide whether a point needs backfilling.
///
/// A point is considered legacy when both:
/// 1. its payload has no non-empty `label` field, AND
/// 2. its `content` starts with the `RULE\n...---` marker.
fn needs_backfill(point: &RetrievedPoint, content: &str) -> bool {
    let has_label = extract_str(point, "label")
        .map(|s| !s.is_empty())
        .unwrap_or(false);
    !has_label && wqm_common::rules_legacy::is_legacy_rule_content(content)
}

/// Build the payload update map from a parsed legacy header.
pub(crate) fn build_payload_updates(
    header: &LegacyRuleHeader,
    now: &str,
) -> std::collections::HashMap<String, serde_json::Value> {
    let (scope, project_id) = header.split_scope();
    let mut map = std::collections::HashMap::new();

    if let Some(label) = header.label() {
        map.insert("label".to_string(), serde_json::json!(label));
    }
    map.insert("scope".to_string(), serde_json::json!(scope));
    if let Some(pid) = project_id {
        map.insert("project_id".to_string(), serde_json::json!(pid));
    }
    if let Some(rt) = header.rule_type() {
        map.insert("rule_type".to_string(), serde_json::json!(rt));
    }
    if let Some(p) = header.priority() {
        map.insert("priority".to_string(), serde_json::json!(p));
    }
    // Strip the header from content so the injected output is clean.
    map.insert("content".to_string(), serde_json::json!(header.body));
    map.insert("enabled".to_string(), serde_json::json!("true"));
    map.insert("updated_at".to_string(), serde_json::json!(now));
    map
}

/// Upsert the `rules_mirror` row with the recovered fields.
///
/// `tenant_id` falls back to `wqm_common::constants::TENANT_GLOBAL` when the
/// legacy payload had no explicit tenant.
async fn upsert_mirror_row(
    pool: &SqlitePool,
    label: &str,
    body: &str,
    scope: &str,
    tenant_id: &str,
    now: &str,
) -> Result<(), sqlx::Error> {
    sqlx::query(
        "INSERT INTO rules_mirror \
         (rule_id, rule_text, scope, tenant_id, created_at, updated_at) \
         VALUES (?1, ?2, ?3, ?4, ?5, ?5) \
         ON CONFLICT(rule_id) DO UPDATE SET \
           rule_text = excluded.rule_text, \
           scope = excluded.scope, \
           tenant_id = excluded.tenant_id, \
           updated_at = excluded.updated_at",
    )
    .bind(label)
    .bind(body)
    .bind(scope)
    .bind(tenant_id)
    .bind(now)
    .execute(pool)
    .await
    .map(|_| ())
}

/// Scroll the `rules` collection and backfill any legacy points.
///
/// Errors are logged and counted; the pass continues through the rest of
/// the collection so one malformed point never prevents the healthy ones
/// from being recovered.
pub async fn backfill_rules_payload(
    storage: &Arc<StorageClient>,
    pool: &SqlitePool,
) -> Result<BackfillStats, String> {
    match storage.collection_exists("rules").await {
        Ok(false) => {
            info!("[backfill:rules-payload] Rules collection does not exist — nothing to do");
            return Ok(BackfillStats::default());
        }
        Err(e) => return Err(format!("Failed to check rules collection: {}", e)),
        Ok(true) => {}
    }

    let points = storage
        .scroll_with_filter("rules", Filter::default(), 10000, None)
        .await
        .map_err(|e| format!("Failed to scroll rules: {}", e))?;

    let mut stats = BackfillStats {
        scanned: points.len() as u64,
        ..BackfillStats::default()
    };
    let now = wqm_common::timestamps::now_utc();

    for point in &points {
        backfill_single_point(point, storage, pool, &mut stats, &now).await;
    }

    info!(
        "[backfill:rules-payload] Complete: scanned={} backfilled={} already_ok={} \
         unparseable={} mirror_upserts={} errors={}",
        stats.scanned,
        stats.backfilled,
        stats.already_ok,
        stats.unparseable,
        stats.mirror_upserts,
        stats.errors
    );

    Ok(stats)
}

async fn backfill_single_point(
    point: &RetrievedPoint,
    storage: &Arc<StorageClient>,
    pool: &SqlitePool,
    stats: &mut BackfillStats,
    now: &str,
) {
    let content = extract_str(point, "content").unwrap_or_default();
    if !needs_backfill(point, &content) {
        stats.already_ok += 1;
        return;
    }

    let Some(header) = parse_rule_header(&content) else {
        stats.unparseable += 1;
        return;
    };
    let Some(label) = header.label().map(str::to_string) else {
        stats.unparseable += 1;
        return;
    };
    let Some(point_id) = extract_point_id_str(point) else {
        warn!("[backfill:rules-payload] Point has no ID — skipping");
        stats.errors += 1;
        return;
    };

    let updates = build_payload_updates(&header, now);
    let (scope_val, _) = header.split_scope();
    let tenant_id = extract_str(point, "tenant_id")
        .unwrap_or_else(|| wqm_common::constants::TENANT_GLOBAL.to_string());

    match storage
        .set_payload_on_point("rules", &point_id, updates)
        .await
    {
        Ok(()) => {
            stats.backfilled += 1;
            info!(
                "[backfill:rules-payload] Restored payload for label={} (point={})",
                label, point_id
            );
        }
        Err(e) => {
            warn!(
                "[backfill:rules-payload] set_payload failed for point={}: {}",
                point_id, e
            );
            stats.errors += 1;
            return;
        }
    }

    match upsert_mirror_row(pool, &label, &header.body, &scope_val, &tenant_id, now).await {
        Ok(()) => stats.mirror_upserts += 1,
        Err(e) => {
            warn!(
                "[backfill:rules-payload] mirror upsert failed for label={}: {}",
                label, e
            );
            stats.errors += 1;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const LEGACY: &str = "RULE\n\
label:use-common-crate\n\
type:constraint\n\
scope:project:4ed81466dec7\n\
priority:8\n\
---\n\
When introducing or modifying shared data structures, always use wqm-common.";

    #[test]
    fn build_payload_updates_project_scope() {
        let header = parse_rule_header(LEGACY).expect("parse");
        let updates = build_payload_updates(&header, "2026-04-20T12:00:00Z");

        assert_eq!(updates.get("label").unwrap(), "use-common-crate");
        assert_eq!(updates.get("scope").unwrap(), "project");
        assert_eq!(updates.get("project_id").unwrap(), "4ed81466dec7");
        assert_eq!(updates.get("rule_type").unwrap(), "constraint");
        assert_eq!(updates.get("priority").unwrap(), 8);
        assert_eq!(updates.get("enabled").unwrap(), "true");
        // Body stripped of RULE header.
        let content = updates.get("content").unwrap().as_str().unwrap();
        assert!(content.starts_with("When introducing or modifying shared"));
        assert!(!content.contains("label:use-common-crate"));
    }

    #[test]
    fn build_payload_updates_global_scope_has_no_project_id() {
        let src = "RULE\nlabel:foo\nscope:global\ntype:behavior\npriority:5\n---\nbody";
        let header = parse_rule_header(src).expect("parse");
        let updates = build_payload_updates(&header, "2026-04-20T12:00:00Z");

        assert_eq!(updates.get("scope").unwrap(), "global");
        assert!(!updates.contains_key("project_id"));
    }

    #[tokio::test]
    async fn upsert_mirror_row_inserts_then_updates() {
        let pool = sqlx::SqlitePool::connect(":memory:").await.unwrap();
        sqlx::query(
            "CREATE TABLE rules_mirror ( \
                rule_id TEXT PRIMARY KEY, \
                rule_text TEXT NOT NULL, \
                scope TEXT, \
                tenant_id TEXT, \
                created_at TEXT, \
                updated_at TEXT \
             )",
        )
        .execute(&pool)
        .await
        .unwrap();

        upsert_mirror_row(
            &pool,
            "foo",
            "hello",
            "global",
            "global",
            "2026-04-20T12:00:00Z",
        )
        .await
        .unwrap();
        upsert_mirror_row(
            &pool,
            "foo",
            "hello v2",
            "global",
            "global",
            "2026-04-20T13:00:00Z",
        )
        .await
        .unwrap();

        let (text, updated): (String, String) =
            sqlx::query_as("SELECT rule_text, updated_at FROM rules_mirror WHERE rule_id = 'foo'")
                .fetch_one(&pool)
                .await
                .unwrap();
        assert_eq!(text, "hello v2");
        assert_eq!(updated, "2026-04-20T13:00:00Z");
    }
}
