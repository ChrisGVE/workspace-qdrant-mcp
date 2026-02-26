//! Tenant deletion cascade, document deletion, and tenant rename.
//!
//! Handles the surgical multi-step deletion of all data associated with a
//! tenant across all 4 canonical collections (projects, libraries, rules,
//! scratchpad), including Qdrant points, SQLite tracked_files, qdrant_chunks,
//! keywords, tags, FTS5 entries, and graph data.

use std::sync::Arc;

use tracing::{debug, info, warn};

use crate::context::ProcessingContext;
use crate::fts_batch_processor::{FtsBatchConfig, FtsBatchProcessor};
use crate::specs::parse_payload;
use crate::storage::StorageClient;
use crate::unified_queue_processor::{UnifiedProcessorError, UnifiedProcessorResult};
use crate::unified_queue_schema::{ProjectPayload, UnifiedQueueItem};
use wqm_common::constants::{
    COLLECTION_LIBRARIES, COLLECTION_PROJECTS, COLLECTION_RULES, COLLECTION_SCRATCHPAD,
};

/// Process delete tenant item -- surgically delete all data for a tenant.
///
/// Full deletion cascade across all 4 canonical collections:
/// 1. Purge pending queue items for this tenant
/// 2. Collect tracked files + qdrant_chunks point IDs, grouped by collection
/// 3. Delete from Qdrant in batches of point IDs (projects, libraries)
/// 4. Scroll + batch-delete from memory collection
/// 5. Scroll + batch-delete from scratchpad collection
/// 6. SQLite cleanup: qdrant_chunks, tracked_files, keywords, tags,
///    keyword_baskets, tag_hierarchy_edges, canonical_tags, watch_folders
/// 7. Final verification: count remaining tenant points in all 4 collections
pub(crate) async fn process_delete_tenant_item(
    ctx: &ProcessingContext,
    item: &UnifiedQueueItem,
) -> UnifiedProcessorResult<()> {
    const QDRANT_BATCH_SIZE: usize = 100;

    info!(
        "Processing surgical delete for tenant={} (queue_id={})",
        item.tenant_id, item.queue_id
    );

    let pool = ctx.queue_manager.pool();

    // -- Step 1: Purge queue entries for this tenant --
    purge_tenant_queue(ctx, item).await;

    // -- Step 2: Collect tracked files with point IDs, grouped by collection --
    let (points_by_collection, file_count) =
        collect_tracked_points(pool, &item.tenant_id).await?;

    info!(
        "Step 2: found {} tracked files across {} collections for tenant={}",
        file_count, points_by_collection.len(), item.tenant_id
    );

    // -- Step 3: Batch-delete tracked points from Qdrant by ID --
    let mut total_qdrant_deleted =
        delete_tracked_qdrant_points(ctx, &points_by_collection, QDRANT_BATCH_SIZE).await?;

    // -- Step 3b: Sweep libraries collection for orphaned/untracked points --
    total_qdrant_deleted += sweep_orphaned_library_points(
        ctx, &item.tenant_id, &points_by_collection, QDRANT_BATCH_SIZE,
    ).await?;

    // -- Steps 4-5: Handle memory (rules) and scratchpad collections --
    for (coll, step) in [(COLLECTION_RULES, 4u8), (COLLECTION_SCRATCHPAD, 5)] {
        total_qdrant_deleted +=
            delete_collection_points(ctx, coll, &item.tenant_id, QDRANT_BATCH_SIZE, step).await?;
    }

    // -- Step 6: SQLite cleanup in single transaction --
    let mut tx = pool.begin().await.map_err(|e| {
        UnifiedProcessorError::ProcessingFailed(format!(
            "Failed to begin delete transaction: {}", e
        ))
    })?;

    sqlite_cascade_delete(&mut tx, &item.tenant_id).await;

    tx.commit().await.map_err(|e| {
        UnifiedProcessorError::ProcessingFailed(format!(
            "Failed to commit delete transaction: {}", e
        ))
    })?;

    // -- Step 6i: FTS5 cleanup --
    cleanup_fts5(ctx, &item.tenant_id).await;

    // -- Step 6j: Graph cleanup (graph-rag Task 3) --
    cleanup_graph(ctx, &item.tenant_id).await;

    // -- Step 7: Final verification --
    verify_deletion(ctx, &item.tenant_id).await;

    info!(
        "Successfully deleted tenant={}: {} Qdrant points, {} tracked files (queue_id={})",
        item.tenant_id, total_qdrant_deleted, file_count, item.queue_id
    );

    Ok(())
}

/// Step 1: Purge pending queue items for this tenant.
async fn purge_tenant_queue(ctx: &ProcessingContext, item: &UnifiedQueueItem) {
    match ctx.queue_manager.purge_pending_for_tenant(&item.tenant_id, &item.queue_id).await {
        Ok(purged) if purged > 0 => {
            info!("Step 1: purged {} queue items for tenant={}", purged, item.tenant_id);
        }
        Err(e) => {
            warn!("Step 1: queue purge failed for tenant={}: {} (continuing)", item.tenant_id, e);
        }
        _ => {}
    }
}

/// Step 2: Collect tracked files with point IDs, grouped by collection.
///
/// Returns `(points_by_collection, unique_file_count)`.
async fn collect_tracked_points(
    pool: &sqlx::SqlitePool,
    tenant_id: &str,
) -> UnifiedProcessorResult<(std::collections::HashMap<String, Vec<String>>, usize)> {
    let file_data: Vec<(i64, String, String, Option<String>)> = sqlx::query_as(
        r#"SELECT tf.file_id, tf.file_path, tf.collection, qc.point_id
           FROM tracked_files tf
           JOIN watch_folders wf ON tf.watch_folder_id = wf.watch_id
           LEFT JOIN qdrant_chunks qc ON qc.file_id = tf.file_id
           WHERE wf.tenant_id = ?1"#,
    )
    .bind(tenant_id)
    .fetch_all(pool)
    .await
    .map_err(|e| {
        UnifiedProcessorError::ProcessingFailed(format!(
            "Failed to query tracked files for tenant={}: {}", tenant_id, e
        ))
    })?;

    let mut points_by_collection: std::collections::HashMap<String, Vec<String>> =
        std::collections::HashMap::new();
    let mut file_ids: std::collections::HashSet<i64> = std::collections::HashSet::new();

    for (file_id, _file_path, collection, point_id) in &file_data {
        file_ids.insert(*file_id);
        if let Some(pid) = point_id {
            points_by_collection.entry(collection.clone()).or_default().push(pid.clone());
        }
    }

    Ok((points_by_collection, file_ids.len()))
}

/// Step 3: Batch-delete tracked Qdrant points by ID.
async fn delete_tracked_qdrant_points(
    ctx: &ProcessingContext,
    points_by_collection: &std::collections::HashMap<String, Vec<String>>,
    batch_size: usize,
) -> UnifiedProcessorResult<u64> {
    let mut total_deleted = 0u64;

    for (collection, point_ids) in points_by_collection {
        if !ctx.storage_client.collection_exists(collection).await
            .map_err(|e| UnifiedProcessorError::Storage(e.to_string()))?
        {
            warn!("Step 3: collection '{}' does not exist, skipping", collection);
            continue;
        }

        for batch in point_ids.chunks(batch_size) {
            let deleted = ctx.storage_client
                .delete_points_by_ids(collection, &batch.to_vec())
                .await
                .map_err(|e| UnifiedProcessorError::Storage(e.to_string()))?;
            total_deleted += deleted;
        }

        info!("Step 3: deleted {} points from '{}' (tenant delete)", point_ids.len(), collection);
    }

    Ok(total_deleted)
}

/// Step 3b: Sweep libraries collection for orphaned/untracked points.
async fn sweep_orphaned_library_points(
    ctx: &ProcessingContext,
    tenant_id: &str,
    points_by_collection: &std::collections::HashMap<String, Vec<String>>,
    batch_size: usize,
) -> UnifiedProcessorResult<u64> {
    if points_by_collection.contains_key(COLLECTION_LIBRARIES) {
        return Ok(0);
    }
    if !ctx.storage_client.collection_exists(COLLECTION_LIBRARIES).await
        .map_err(|e| UnifiedProcessorError::Storage(e.to_string()))?
    {
        return Ok(0);
    }

    let library_ids = ctx.storage_client
        .scroll_point_ids_by_tenant(COLLECTION_LIBRARIES, tenant_id)
        .await
        .map_err(|e| UnifiedProcessorError::Storage(e.to_string()))?;

    if library_ids.is_empty() {
        return Ok(0);
    }

    for batch in library_ids.chunks(batch_size) {
        ctx.storage_client
            .delete_points_by_ids(COLLECTION_LIBRARIES, &batch.to_vec())
            .await
            .map_err(|e| UnifiedProcessorError::Storage(e.to_string()))?;
    }

    info!("Step 3b: deleted {} orphaned library points for tenant={}", library_ids.len(), tenant_id);
    Ok(library_ids.len() as u64)
}

/// Delete all points for a tenant from a specific collection (Steps 4, 5).
async fn delete_collection_points(
    ctx: &ProcessingContext,
    collection: &str,
    tenant_id: &str,
    batch_size: usize,
    step_num: u8,
) -> UnifiedProcessorResult<u64> {
    if !ctx.storage_client.collection_exists(collection).await
        .map_err(|e| UnifiedProcessorError::Storage(e.to_string()))?
    {
        return Ok(0);
    }

    let point_ids = ctx.storage_client
        .scroll_point_ids_by_tenant(collection, tenant_id)
        .await
        .map_err(|e| UnifiedProcessorError::Storage(e.to_string()))?;

    if point_ids.is_empty() {
        return Ok(0);
    }

    for batch in point_ids.chunks(batch_size) {
        ctx.storage_client
            .delete_points_by_ids(collection, &batch.to_vec())
            .await
            .map_err(|e| UnifiedProcessorError::Storage(e.to_string()))?;
    }

    info!("Step {}: deleted {} points from '{}' for tenant={}", step_num, point_ids.len(), collection, tenant_id);
    Ok(point_ids.len() as u64)
}

/// Step 6: Run the SQLite cascade delete inside a transaction.
///
/// Deletes (in order): qdrant_chunks, tracked_files, keywords, keyword_baskets,
/// tags, tag_hierarchy_edges, canonical_tags, watch_folders.
async fn sqlite_cascade_delete(
    tx: &mut sqlx::Transaction<'_, sqlx::Sqlite>,
    tenant_id: &str,
) {
    // Table deletion specs: (step label, SQL query, table name).
    // Order matters -- children before parents, watch_folders last.
    let cascade_steps: &[(&str, &str, &str)] = &[
        ("6a", r#"DELETE FROM qdrant_chunks WHERE file_id IN (
            SELECT tf.file_id FROM tracked_files tf
            JOIN watch_folders wf ON tf.watch_folder_id = wf.watch_id
            WHERE wf.tenant_id = ?1)"#, "qdrant_chunks"),
        ("6b", r#"DELETE FROM tracked_files WHERE watch_folder_id IN (
            SELECT watch_id FROM watch_folders WHERE tenant_id = ?1)"#, "tracked_files"),
        ("6c", "DELETE FROM keywords WHERE tenant_id = ?1", "keywords"),
        ("6d", "DELETE FROM keyword_baskets WHERE tenant_id = ?1", "keyword_baskets"),
        ("6e", "DELETE FROM tags WHERE tenant_id = ?1", "tags"),
        ("6f", "DELETE FROM tag_hierarchy_edges WHERE tenant_id = ?1", "tag_hierarchy_edges"),
        ("6g", "DELETE FROM canonical_tags WHERE tenant_id = ?1", "canonical_tags"),
        ("6h", "DELETE FROM watch_folders WHERE tenant_id = ?1", "watch_folders"),
    ];

    for (step, query, table_name) in cascade_steps {
        match sqlx::query(query).bind(tenant_id).execute(&mut **tx).await {
            Ok(result) if result.rows_affected() > 0 => {
                info!("Step {}: deleted {} {} for tenant={}", step, result.rows_affected(), table_name, tenant_id);
            }
            Err(e) => {
                debug!("Step {}: {} delete: {}", step, table_name, e);
            }
            _ => {}
        }
    }
}

/// Step 6i: FTS5 cleanup (non-fatal).
async fn cleanup_fts5(ctx: &ProcessingContext, tenant_id: &str) {
    if let Some(sdb) = &ctx.search_db {
        let processor = FtsBatchProcessor::new(sdb, FtsBatchConfig::default());
        match processor.delete_tenant(tenant_id).await {
            Ok(deleted) if deleted > 0 => {
                info!("Step 6i: deleted {} FTS5 code_lines for tenant={}", deleted, tenant_id);
            }
            Err(e) => {
                warn!("Step 6i: FTS5 cleanup failed for tenant={}: {} (non-fatal)", tenant_id, e);
            }
            _ => {}
        }
    }
}

/// Step 6j: Graph cleanup (non-fatal).
///
/// Graph data is in a separate database (graph.db), not in the SQLite
/// transaction. Failures are logged but do not fail the tenant deletion.
async fn cleanup_graph(ctx: &ProcessingContext, tenant_id: &str) {
    if let Some(ref graph_store) = ctx.graph_store {
        match graph_store.delete_tenant(tenant_id).await {
            Ok(deleted) if deleted > 0 => {
                info!("Step 6j: deleted {} graph items for tenant={}", deleted, tenant_id);
            }
            Err(e) => {
                warn!("Step 6j: graph cleanup failed for tenant={}: {} (non-fatal)", tenant_id, e);
            }
            _ => {}
        }
    }
}

/// Step 7: Final verification -- count remaining tenant points in all 4 collections.
async fn verify_deletion(ctx: &ProcessingContext, tenant_id: &str) {
    for coll in [COLLECTION_PROJECTS, COLLECTION_LIBRARIES, COLLECTION_RULES, COLLECTION_SCRATCHPAD] {
        if let Ok(true) = ctx.storage_client.collection_exists(coll).await {
            match ctx.storage_client.count_points(coll, Some(tenant_id)).await {
                Ok(remaining) if remaining > 0 => {
                    warn!("Step 7: ORPHAN POINTS: {} remaining in '{}' for tenant={}", remaining, coll, tenant_id);
                }
                Ok(_) => {
                    debug!("Step 7: verified 0 remaining in '{}' for tenant={}", coll, tenant_id);
                }
                Err(e) => {
                    warn!("Step 7: verification failed for '{}': {}", coll, e);
                }
            }
        }
    }
}

/// Process delete document item -- delete specific document by document_id.
pub(crate) async fn process_delete_document_item(
    item: &UnifiedQueueItem,
    storage_client: &Arc<StorageClient>,
) -> UnifiedProcessorResult<()> {
    info!("Processing delete document item: {}", item.queue_id);

    let payload = item.parse_delete_document_payload().map_err(|e| {
        UnifiedProcessorError::InvalidPayload(format!(
            "Failed to parse DeleteDocumentPayload: {}", e
        ))
    })?;

    if payload.document_id.trim().is_empty() {
        return Err(UnifiedProcessorError::InvalidPayload(
            "document_id must not be empty".to_string(),
        ));
    }

    if storage_client.collection_exists(&item.collection).await
        .map_err(|e| UnifiedProcessorError::Storage(e.to_string()))?
    {
        storage_client
            .delete_points_by_document_id(&item.collection, &payload.document_id)
            .await
            .map_err(|e| UnifiedProcessorError::Storage(e.to_string()))?;
    }

    info!("Successfully deleted document {} from {} (tenant={})", payload.document_id, item.collection, item.tenant_id);
    Ok(())
}

/// Process tenant rename item -- update tenant_id on all matching Qdrant points.
///
/// Uses `ProjectPayload` with `old_tenant_id` field.
pub(crate) async fn process_tenant_rename_item(
    item: &UnifiedQueueItem,
    storage_client: &Arc<StorageClient>,
) -> UnifiedProcessorResult<()> {
    let payload: ProjectPayload = parse_payload(item)?;

    let old_tenant = payload.old_tenant_id.as_deref().ok_or_else(|| {
        UnifiedProcessorError::InvalidPayload(
            "Missing old_tenant_id in tenant rename payload".to_string(),
        )
    })?;
    let new_tenant = &item.tenant_id;

    let reason = item.metadata.as_deref()
        .and_then(|m| serde_json::from_str::<serde_json::Value>(m).ok())
        .and_then(|v| v.get("reason").and_then(|r| r.as_str().map(String::from)))
        .unwrap_or_else(|| "unknown".to_string());

    info!("Processing tenant rename: {} -> {} in collection '{}' (reason: {})", old_tenant, new_tenant, item.collection, reason);

    use qdrant_client::qdrant::{Condition, Filter};
    let filter = Filter::must([Condition::matches("tenant_id", old_tenant.to_string())]);

    let mut new_payload = std::collections::HashMap::new();
    new_payload.insert("tenant_id".to_string(), serde_json::Value::String(new_tenant.to_string()));

    storage_client
        .set_payload_by_filter(&item.collection, filter, new_payload)
        .await
        .map_err(|e| UnifiedProcessorError::Storage(e.to_string()))?;

    info!("Successfully processed tenant rename {} -> {} in '{}'", old_tenant, new_tenant, item.collection);
    Ok(())
}
