//! Enqueue side of ignore-file reconciliation.
//!
//! Turns the stale/missing diff computed by `ignore_sync` into
//! `file|delete` / `file|add` unified-queue items, batched as one SQLite
//! transaction per batch.

use std::sync::Arc;

use tracing::{debug, info, warn};
use wqm_common::paths::RelativePath;

use crate::queue_operations::QueueManager;
use crate::unified_queue_schema::{ItemType, QueueOperation};

use super::ignore_sync::ReconcileStats;

/// Enqueue delete + add operations for stale and missing files.
///
/// `stale` and `missing` carry relative paths (forward-slash, normalized).
/// The JSON payload's `file_path` field is the relative form — `FilePayload`
/// types it as a validating `RelativePath`, and the downstream strategy
/// reanchors via `RelativePath::to_absolute(watch_folder_root)`. The consumer
/// rejects anything `RelativePath::from_user_input` rejects (absolute input,
/// Windows drive-letter prefixes, `..` traversal) as a permanent
/// `InvalidPayload`, so `enqueue_ignore_ops` applies the same validation
/// before enqueueing and skips offenders with a warning.
pub(super) async fn enqueue_reconcile_ops(
    queue_manager: &Arc<QueueManager>,
    tenant_id: &str,
    collection: &str,
    stale: &[&String],
    missing: &[&String],
) -> Result<ReconcileStats, String> {
    let mut stats = ReconcileStats::default();

    stats.stale_deleted = enqueue_ignore_ops(
        queue_manager,
        tenant_id,
        collection,
        QueueOperation::Delete,
        stale,
        "ignore_rule_change",
    )
    .await;

    stats.missing_added = enqueue_ignore_ops(
        queue_manager,
        tenant_id,
        collection,
        QueueOperation::Add,
        missing,
        "ignore_reconciliation",
    )
    .await;

    info!(
        "[ignore_sync] {} — enqueued {} deletes, {} adds",
        tenant_id, stats.stale_deleted, stats.missing_added
    );

    Ok(stats)
}

/// Default batch size for ignore-sync enqueues.
///
/// Each batch is committed as a single SQLite transaction so we amortise
/// lock contention across hundreds of inserts. 500 is a balance between
/// commit latency and transaction size that matches the `#59` acceptance
/// criteria.
pub(super) const IGNORE_SYNC_BATCH_SIZE: usize = 500;

/// Enqueue `file_paths` as `(File, op)` items in batches using a single
/// SQLite transaction per batch. Progress is logged every batch so large
/// backfills are observable in the daemon log.
///
/// Paths that fail `RelativePath::from_user_input` are skipped (with a
/// warning) instead of enqueued: the consumer's `FilePayload`
/// deserialization applies the same rules, so such an item could never
/// parse and would sit in the queue as a permanently-failed poison row.
/// This is reachable with real on-disk data — a directory literally named
/// `C:` inside a project root (left behind by a tool writing un-translated
/// Windows paths on Linux) walks to a genuine relative path that the
/// drive-letter defense still rejects.
async fn enqueue_ignore_ops(
    queue_manager: &Arc<QueueManager>,
    tenant_id: &str,
    collection: &str,
    op: QueueOperation,
    file_paths: &[&String],
    reason: &str,
) -> u64 {
    let total = file_paths.len();
    if total == 0 {
        return 0;
    }

    let mut enqueued: u64 = 0;
    let mut skipped: u64 = 0;
    let op_label = match op {
        QueueOperation::Delete => "delete",
        QueueOperation::Add => "add",
        _ => "op",
    };

    for chunk in file_paths.chunks(IGNORE_SYNC_BATCH_SIZE) {
        let payloads: Vec<String> = chunk
            .iter()
            .filter_map(|rel_path| match RelativePath::from_user_input(rel_path) {
                Ok(rel) => Some(build_payload(op, &rel, reason)),
                Err(e) => {
                    skipped += 1;
                    // Per-path detail for the first few; the aggregate count
                    // is logged once after the loop.
                    if skipped <= 3 {
                        warn!(
                            "[ignore_sync] {} — {}: skipping unqueueable path: {}",
                            tenant_id, op_label, e
                        );
                    }
                    None
                }
            })
            .collect();

        if payloads.is_empty() {
            continue;
        }

        match queue_manager
            .enqueue_unified_batch(ItemType::File, op, tenant_id, collection, &payloads, None)
            .await
        {
            Ok(n) => {
                enqueued += n;
                debug!(
                    "[ignore_sync] {} — {}: batch committed {}/{} items ({} total enqueued)",
                    tenant_id,
                    op_label,
                    n,
                    payloads.len(),
                    enqueued
                );
            }
            Err(e) => warn!(
                "[ignore_sync] {} — {} batch failed (size={}): {}",
                tenant_id,
                op_label,
                payloads.len(),
                e
            ),
        }
    }

    if skipped > 0 {
        warn!(
            "[ignore_sync] {} — {}: skipped {}/{} path(s) that fail RelativePath validation",
            tenant_id, op_label, skipped, total
        );
    }

    enqueued
}

/// Build the `FilePayload`-shaped JSON for one validated relative path.
///
/// Serializing `rel.as_str()` (the normalized form) keeps the payload — and
/// therefore the idempotency key derived from it — stable regardless of how
/// the producer's walk spelled the path.
fn build_payload(op: QueueOperation, rel: &RelativePath, reason: &str) -> String {
    match op {
        QueueOperation::Delete => serde_json::json!({
            "file_path": rel.as_str(),
            "reason": reason,
        }),
        _ => serde_json::json!({
            "file_path": rel.as_str(),
            "source": reason,
        }),
    }
    .to_string()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::unified_queue_schema::FilePayload;

    /// Regression for the Finance poison-queue incident: a literal `C:`
    /// directory inside a Linux project root produced walk paths like
    /// `C:/Users/...` that the consumer's validating `RelativePath`
    /// deserialization rejects. The producer must skip them instead of
    /// enqueueing items that fail forever as `[permanent_data]`.
    #[tokio::test]
    async fn skips_paths_the_consumer_would_reject() {
        let pool = super::super::tests::create_test_pool().await;
        super::super::tests::setup_schema(&pool).await;
        let queue_manager = Arc::new(QueueManager::new(pool.clone()));

        let junk = "C:/Users/u/AppData/Local/Temp/proj-copy/lib/a.dart".to_string();
        let good = "lib/src/main.dart".to_string();
        let paths: Vec<&String> = vec![&junk, &good];

        let enqueued = enqueue_ignore_ops(
            &queue_manager,
            "tenant-x",
            "projects",
            QueueOperation::Add,
            &paths,
            "ignore_reconciliation",
        )
        .await;
        assert_eq!(enqueued, 1, "only the valid path is enqueued");

        let rows: Vec<(String,)> =
            sqlx::query_as("SELECT payload_json FROM unified_queue WHERE tenant_id = 'tenant-x'")
                .fetch_all(&pool)
                .await
                .unwrap();
        assert_eq!(rows.len(), 1);

        // Producer→consumer contract: the stored payload must round-trip
        // through the consumer's validating FilePayload deserialization.
        let parsed: FilePayload = serde_json::from_str(&rows[0].0).unwrap();
        assert_eq!(parsed.file_path.as_str(), "lib/src/main.dart");
    }

    #[tokio::test]
    async fn skips_unqueueable_delete_paths_too() {
        let pool = super::super::tests::create_test_pool().await;
        super::super::tests::setup_schema(&pool).await;
        let queue_manager = Arc::new(QueueManager::new(pool.clone()));

        let junk = "C:/Users/u/legacy-junk/b.dart".to_string();
        let paths: Vec<&String> = vec![&junk];

        let enqueued = enqueue_ignore_ops(
            &queue_manager,
            "tenant-y",
            "projects",
            QueueOperation::Delete,
            &paths,
            "ignore_rule_change",
        )
        .await;
        assert_eq!(enqueued, 0);

        let count: (i64,) =
            sqlx::query_as("SELECT COUNT(*) FROM unified_queue WHERE tenant_id = 'tenant-y'")
                .fetch_one(&pool)
                .await
                .unwrap();
        assert_eq!(count.0, 0, "nothing enqueued for the rejected path");
    }

    #[tokio::test]
    async fn enqueues_normalized_form_of_messy_relative_paths() {
        let pool = super::super::tests::create_test_pool().await;
        super::super::tests::setup_schema(&pool).await;
        let queue_manager = Arc::new(QueueManager::new(pool.clone()));

        let messy = "src/./foo//bar.rs".to_string();
        let paths: Vec<&String> = vec![&messy];

        let enqueued = enqueue_ignore_ops(
            &queue_manager,
            "tenant-z",
            "projects",
            QueueOperation::Add,
            &paths,
            "ignore_reconciliation",
        )
        .await;
        assert_eq!(enqueued, 1);

        let row: (String,) =
            sqlx::query_as("SELECT payload_json FROM unified_queue WHERE tenant_id = 'tenant-z'")
                .fetch_one(&pool)
                .await
                .unwrap();
        let parsed: FilePayload = serde_json::from_str(&row.0).unwrap();
        assert_eq!(parsed.file_path.as_str(), "src/foo/bar.rs");
    }
}
