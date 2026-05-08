//! Rebuild handlers for storage-backed targets: rules, rules-payload, scratchpad.
//!
//! Each handler reconciles the Qdrant collection against the SQLite mirror
//! (or delegates to a dedicated reconciliation module).

use std::sync::Arc;
use tracing::{error, info};

/// Self-diagnosing rules reconciliation.
///
/// Compares Qdrant `rules` collection against SQLite `rules_mirror` and fixes
/// discrepancies in both directions. Deduplicates labels and content in Qdrant.
///
/// Steps: scroll Qdrant -> read SQLite -> dedup labels -> dedup content ->
/// bidirectional sync (Qdrant<->SQLite).
pub(super) async fn rebuild_rules(
    storage_client: Option<Arc<workspace_qdrant_core::StorageClient>>,
    db_pool: Option<&sqlx::SqlitePool>,
) {
    use crate::services::rules_rebuild;

    let start = std::time::Instant::now();
    let Some(pool) = db_pool else {
        error!("[rebuild:rules] Database pool not configured");
        return;
    };
    let Some(storage) = storage_client else {
        error!("[rebuild:rules] Storage client not configured");
        return;
    };

    // Step 1: Scroll Qdrant rules
    let (qdrant_by_label, all_points) = match rules_rebuild::scroll_qdrant_rules(&storage).await {
        Ok(result) => result,
        Err(e) => {
            if e != "no_collection" {
                error!("[rebuild:rules] {}", e);
            }
            return;
        }
    };
    info!(
        "[rebuild:rules] Found {} Qdrant rules ({} unique labels)",
        all_points.len(),
        qdrant_by_label.len()
    );

    // Step 2: Read SQLite rules_mirror
    let db_by_label = match rules_rebuild::read_sqlite_rules(pool).await {
        Ok(rules) => rules,
        Err(e) => {
            error!("[rebuild:rules] {}", e);
            return;
        }
    };
    info!(
        "[rebuild:rules] Found {} rules in SQLite rules_mirror",
        db_by_label.len()
    );

    // Steps 3-4: Deduplicate labels and content
    let (ids_to_delete, label_dups, content_dups) =
        rules_rebuild::deduplicate_rules(&qdrant_by_label, pool).await;

    let deleted_count = ids_to_delete.len() as u64;
    if !ids_to_delete.is_empty() {
        match storage.delete_points_by_ids("rules", &ids_to_delete).await {
            Ok(_) => info!(
                "[rebuild:rules] Deleted {} duplicate Qdrant points",
                deleted_count
            ),
            Err(e) => error!("[rebuild:rules] Failed to delete duplicate points: {}", e),
        }
    }

    // Build deduplicated state and reconcile
    let qdrant_deduped = rules_rebuild::build_deduped_state(&qdrant_by_label, &ids_to_delete);
    let (inserted, updated, enqueued) =
        rules_rebuild::reconcile_rules(pool, &qdrant_deduped, &db_by_label).await;

    info!(
        "[rebuild:rules] Reconciliation complete in {}ms: \
        qdrant_total={}, db_total={}, label_dups={}, content_dups={}, \
        deleted={}, mirror_inserted={}, mirror_updated={}, enqueued={}",
        start.elapsed().as_millis(),
        all_points.len(),
        db_by_label.len(),
        label_dups,
        content_dups,
        deleted_count,
        inserted,
        updated,
        enqueued
    );
}

/// Backfill payload scope/label/etc. fields from legacy `RULE`-headered
/// content (issue #58).
pub(super) async fn rebuild_rules_payload(
    storage_client: Option<Arc<workspace_qdrant_core::StorageClient>>,
    db_pool: Option<&sqlx::SqlitePool>,
) {
    use crate::services::rules_payload_backfill;

    let Some(pool) = db_pool else {
        error!("[rebuild:rules-payload] Database pool not configured");
        return;
    };
    let Some(storage) = storage_client else {
        error!("[rebuild:rules-payload] Storage client not configured");
        return;
    };

    match rules_payload_backfill::backfill_rules_payload(&storage, pool).await {
        Ok(stats) => info!(
            "[rebuild:rules-payload] scanned={} backfilled={} already_ok={} \
             unparseable={} mirror_upserts={} errors={}",
            stats.scanned,
            stats.backfilled,
            stats.already_ok,
            stats.unparseable,
            stats.mirror_upserts,
            stats.errors
        ),
        Err(e) => error!("[rebuild:rules-payload] {}", e),
    }
}

/// Reconcile scratchpad entries between SQLite mirror and Qdrant.
pub(super) async fn rebuild_scratchpad(
    storage_client: Option<Arc<workspace_qdrant_core::StorageClient>>,
    db_pool: Option<&sqlx::SqlitePool>,
) {
    use crate::services::scratchpad_rebuild;

    let start = std::time::Instant::now();
    let Some(pool) = db_pool else {
        error!("[rebuild:scratchpad] Database pool not configured");
        return;
    };
    let Some(storage) = storage_client else {
        error!("[rebuild:scratchpad] Storage client not configured");
        return;
    };

    let qdrant_entries = match scratchpad_rebuild::scroll_qdrant_scratchpad(&storage).await {
        Ok(entries) => entries,
        Err(e) => {
            if e != "no_collection" {
                error!("[rebuild:scratchpad] {}", e);
            }
            // Even without Qdrant collection, enqueue all mirror entries
            std::collections::HashMap::new()
        }
    };

    let mirror_entries = match scratchpad_rebuild::read_sqlite_scratchpad(pool).await {
        Ok(entries) => entries,
        Err(e) => {
            error!("[rebuild:scratchpad] {}", e);
            return;
        }
    };

    info!(
        "[rebuild:scratchpad] Qdrant={}, mirror={}",
        qdrant_entries.len(),
        mirror_entries.len()
    );

    let enqueued =
        scratchpad_rebuild::reconcile_scratchpad(pool, &qdrant_entries, &mirror_entries).await;

    info!(
        "[rebuild:scratchpad] Reconciliation complete in {}ms: enqueued={}",
        start.elapsed().as_millis(),
        enqueued
    );
}
