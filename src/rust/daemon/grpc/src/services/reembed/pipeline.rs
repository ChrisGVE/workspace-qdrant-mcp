use std::sync::atomic::Ordering;
use std::time::{Duration, Instant};

use tonic::Status;
use tracing::info;
use uuid::Uuid;

use crate::proto::TriggerReembedResponse;

use super::context::{ReembedContext, CANONICAL_COLLECTIONS};
use super::enqueue::{enqueue_folder_scans, enqueue_rules_mirror, enqueue_scratchpad_mirror};
use super::recreator::{collection_reembed_idempotency_key, CollectionRecreator};

/// Pause the queue and poll until all in-progress items complete or timeout.
///
/// Sets `ctx.pause_flag` to `true` on entry. Clears it (back to `false`) on
/// both timeout and internal query error before returning `Err`.
async fn drain_to_quiescence(
    ctx: &ReembedContext,
    timeout: Duration,
    poll_interval: Duration,
) -> Result<(), Status> {
    let drain_started = Instant::now();
    loop {
        let in_flight: i64 = sqlx::query_scalar(
            "SELECT COUNT(*) FROM unified_queue \
             WHERE status = 'in_progress' \
             AND lease_until IS NOT NULL \
             AND lease_until > strftime('%Y-%m-%dT%H:%M:%fZ', 'now')",
        )
        .fetch_one(&ctx.pool)
        .await
        .map_err(|e| {
            ctx.pause_flag.store(false, Ordering::SeqCst);
            Status::internal(format!("drain query failed: {e}"))
        })?;

        if in_flight == 0 {
            return Ok(());
        }

        if drain_started.elapsed() >= timeout {
            ctx.pause_flag.store(false, Ordering::SeqCst);
            return Err(Status::failed_precondition(format!(
                "drain-to-quiescence timeout: {} items still in_progress after {}s; pause flag released",
                in_flight,
                timeout.as_secs()
            )));
        }

        tokio::time::sleep(poll_interval).await;
    }
}

/// Flush ALL queue rows for the canonical collections and clear vector-derived
/// SQLite state.
///
/// Returns the number of rows deleted for logging.
///
/// Deleting only `pending` rows (the historical behavior) left `done` rows in
/// place. Re-enqueue uses `INSERT OR IGNORE` on a globally-unique
/// `idempotency_key` (plus a status-agnostic composite UNIQUE index on
/// `(tenant, branch, collection, item_type, op, file_path)`), so a surviving
/// `done` `folder|scan` / `File|Add` row silently deduped the re-enqueue and
/// reembed processed nothing (#96). The queue is paused and drained to
/// quiescence before this runs, so removing every row for these collections
/// (regardless of status) is safe and frees the dedup keys.
async fn flush_and_clear_state(ctx: &ReembedContext) -> Result<u32, Status> {
    let stale_deleted = sqlx::query(
        "DELETE FROM unified_queue \
         WHERE collection IN ('projects','libraries','rules','scratchpad')",
    )
    .execute(&ctx.pool)
    .await
    .map_err(|e| Status::internal(format!("flush queue rows failed: {e}")))?;
    info!(
        rows = stale_deleted.rows_affected(),
        "reembed: flushed all queue rows for canonical collections"
    );

    let mut tx = ctx
        .pool
        .begin()
        .await
        .map_err(|e| Status::internal(format!("clear-state tx begin failed: {e}")))?;
    sqlx::query("DELETE FROM tag_hierarchy_edges")
        .execute(&mut *tx)
        .await
        .map_err(|e| Status::internal(format!("clear tag_hierarchy_edges: {e}")))?;
    sqlx::query("DELETE FROM canonical_tags")
        .execute(&mut *tx)
        .await
        .map_err(|e| Status::internal(format!("clear canonical_tags: {e}")))?;
    tx.commit()
        .await
        .map_err(|e| Status::internal(format!("clear-state tx commit failed: {e}")))?;

    Ok(stale_deleted.rows_affected() as u32)
}

/// Execute the full reembed flow.
///
/// Returns the populated [`TriggerReembedResponse`] on success or a
/// pre-mapped `tonic::Status` describing the failure mode (typically
/// `failed_precondition` for dim mismatch / drain timeout).
pub async fn execute_reembed<R: CollectionRecreator + ?Sized>(
    ctx: &ReembedContext,
    recreator: &R,
    drain_timeout: Duration,
    poll_interval: Duration,
) -> Result<TriggerReembedResponse, Status> {
    // ── 1. Pre-flight dim check ──────────────────────────────────────────
    let cfg_dim = ctx.settings.output_dim;
    let provider_dim = ctx.provider.output_dim();
    if cfg_dim != provider_dim {
        return Err(Status::failed_precondition(format!(
            "provider output_dim mismatch: settings.output_dim={} but provider.output_dim()={}",
            cfg_dim, provider_dim
        )));
    }

    // ── 2–3. Pause flag + drain to quiescence ───────────────────────────
    ctx.pause_flag.store(true, Ordering::SeqCst);
    info!("reembed: pause flag set; awaiting queue quiescence");
    drain_to_quiescence(ctx, drain_timeout, poll_interval).await?;

    // ── 4–5. Flush stale pending + clear vector-derived state ───────────
    flush_and_clear_state(ctx).await?;

    // ── 6. Recreate the four canonical collections at settings.output_dim
    //      while workers are still paused ────────────────────────────────
    let recreate_dim = cfg_dim as u64;
    for name in CANONICAL_COLLECTIONS {
        recreator.recreate(name, recreate_dim).await?;
    }

    // ── 7. Enqueue 4 collection-reembed traceability items ───────────────
    let now = wqm_common::timestamps::now_utc();
    for collection in CANONICAL_COLLECTIONS {
        let queue_id = Uuid::new_v4().to_string();
        let idem_key = collection_reembed_idempotency_key(collection);
        sqlx::query(
            "INSERT OR IGNORE INTO unified_queue \
             (queue_id, idempotency_key, item_type, op, tenant_id, collection, \
              status, payload_json, created_at, updated_at) \
             VALUES (?1, ?2, 'collection', 'reembed', '_system', ?3, 'pending', '{}', ?4, ?5)",
        )
        .bind(&queue_id)
        .bind(&idem_key)
        .bind(collection)
        .bind(&now)
        .bind(&now)
        .execute(&ctx.pool)
        .await
        .map_err(|e| Status::internal(format!("enqueue reembed/{collection}: {e}")))?;
    }

    // ── 8. Re-enqueue from watch_folders, rules_mirror, scratchpad_mirror
    let files_enqueued = enqueue_folder_scans(&ctx.pool, &now)
        .await
        .map_err(|e| Status::internal(format!("re-enqueue folder scans: {e}")))?;
    let rules_enqueued = enqueue_rules_mirror(&ctx.pool, &now)
        .await
        .map_err(|e| Status::internal(format!("re-enqueue rules_mirror: {e}")))?;
    let scratchpad_enqueued = enqueue_scratchpad_mirror(&ctx.pool, &now)
        .await
        .map_err(|e| Status::internal(format!("re-enqueue scratchpad_mirror: {e}")))?;

    // ── 9. Resume queue workers ──────────────────────────────────────────
    ctx.pause_flag.store(false, Ordering::SeqCst);
    info!(
        files = files_enqueued,
        rules = rules_enqueued,
        scratchpad = scratchpad_enqueued,
        "reembed: complete; pause flag cleared"
    );

    Ok(TriggerReembedResponse {
        files_enqueued,
        rules_enqueued,
        scratchpad_enqueued,
        message: format!(
            "reembed complete at output_dim={cfg_dim}: {files_enqueued} files, \
             {rules_enqueued} rules, {scratchpad_enqueued} scratchpad items re-enqueued"
        ),
    })
}
