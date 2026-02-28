//! Rebuild target helper functions for the RebuildIndex RPC
//!
//! Each function handles one rebuild target (tags, search, vocabulary,
//! keywords, rules, watch_folders). The `dispatch` function routes
//! by target name.

use std::sync::Arc;
use tracing::{info, warn, error};

/// Dispatch a rebuild operation to the appropriate handler.
pub(super) async fn dispatch(
    target: &str,
    hierarchy_builder: Option<Arc<workspace_qdrant_core::HierarchyBuilder>>,
    search_db: Option<Arc<workspace_qdrant_core::SearchDbManager>>,
    lexicon_manager: Option<Arc<workspace_qdrant_core::LexiconManager>>,
    storage_client: Option<Arc<workspace_qdrant_core::StorageClient>>,
    db_pool: Option<&sqlx::SqlitePool>,
    tenant_id: Option<&str>,
    collection: &str,
) {
    match target {
        "tags" => rebuild_tags(hierarchy_builder, tenant_id).await,
        "search" => rebuild_search(search_db).await,
        "vocabulary" => rebuild_vocabulary(lexicon_manager, db_pool, collection).await,
        "keywords" => rebuild_keywords(db_pool, tenant_id, collection).await,
        "rules" => rebuild_rules(storage_client, db_pool).await,
        "projects" => rebuild_watch_folders(db_pool, "projects", tenant_id).await,
        "libraries" => rebuild_watch_folders(db_pool, "libraries", tenant_id).await,
        "all" => {
            info!("Starting full rebuild (all targets)");
            rebuild_vocabulary(lexicon_manager, db_pool, collection).await;
            rebuild_search(search_db).await;
            rebuild_tags(hierarchy_builder, tenant_id).await;
            rebuild_keywords(db_pool, tenant_id, collection).await;
            rebuild_rules(storage_client, db_pool).await;
            rebuild_watch_folders(db_pool, "projects", tenant_id).await;
            rebuild_watch_folders(db_pool, "libraries", tenant_id).await;
            info!("Full rebuild complete (all targets)");
        }
        _ => {} // Validated by caller
    }
}

/// Rebuild canonical tag hierarchy.
async fn rebuild_tags(
    builder: Option<Arc<workspace_qdrant_core::HierarchyBuilder>>,
    tenant_id: Option<&str>,
) {
    let Some(builder) = builder else {
        error!(target = "tags", "Hierarchy builder not configured");
        return;
    };
    let start = std::time::Instant::now();
    if let Some(tid) = tenant_id {
        match builder.rebuild_tenant(tid).await {
            Ok(Some(r)) => {
                let total = r.level1_count + r.level2_count + r.level3_count;
                info!(target = "tags", tenant = tid, canonical_tags = total,
                    edges = r.edges_created, duration_ms = start.elapsed().as_millis() as u64,
                    "Tag hierarchy rebuild complete");
            }
            Ok(None) => info!(target = "tags", tenant = tid, "Skipped (too few tags)"),
            Err(e) => error!(target = "tags", tenant = tid, error = %e, "Tag hierarchy rebuild failed"),
        }
    } else {
        match builder.rebuild_all().await {
            Ok(r) => info!(target = "tags", tenants = r.tenants_processed,
                canonical_tags = r.total_canonical_tags, edges = r.total_edges,
                duration_ms = start.elapsed().as_millis() as u64,
                "Tag hierarchy rebuild complete (all tenants)"),
            Err(e) => error!(target = "tags", error = %e, "Tag hierarchy rebuild failed (all tenants)"),
        }
    }
}

/// Rebuild FTS5 search index.
async fn rebuild_search(
    search_db: Option<Arc<workspace_qdrant_core::SearchDbManager>>,
) {
    let Some(sdb) = search_db else {
        error!(target = "search", "SearchDbManager not configured");
        return;
    };
    let start = std::time::Instant::now();
    match sdb.rebuild_fts().await {
        Ok(()) => {
            if let Err(e) = sdb.optimize_fts().await {
                warn!(target = "search", error = %e, "FTS5 rebuilt but optimize failed");
            } else {
                info!(target = "search", duration_ms = start.elapsed().as_millis() as u64,
                    "FTS5 search index rebuilt and optimized");
            }
        }
        Err(e) => error!(target = "search", error = %e, "FTS5 rebuild failed"),
    }
}

/// Rebuild BM25 sparse vocabulary.
async fn rebuild_vocabulary(
    lexicon: Option<Arc<workspace_qdrant_core::LexiconManager>>,
    db_pool: Option<&sqlx::SqlitePool>,
    collection: &str,
) {
    let Some(lexicon) = lexicon else {
        error!(target = "vocabulary", "LexiconManager not configured");
        return;
    };
    let Some(pool) = db_pool else {
        error!(target = "vocabulary", "Database pool not configured");
        return;
    };

    let start = std::time::Instant::now();

    // Step 1: Cleanup junk terms
    let junk_removed = match lexicon.cleanup_junk_terms().await {
        Ok(n) => n,
        Err(e) => {
            error!(target = "vocabulary", error = %e, "Junk cleanup failed");
            return;
        }
    };

    // Step 2: Delete vocabulary for the collection and reset corpus stats
    let vocab_deleted = match sqlx::query(
        "DELETE FROM sparse_vocabulary WHERE collection = ?1",
    )
    .bind(collection)
    .execute(pool)
    .await
    {
        Ok(r) => r.rows_affected(),
        Err(e) => {
            error!(target = "vocabulary", error = %e, "Vocabulary delete failed");
            return;
        }
    };

    if let Err(e) = sqlx::query(
        "DELETE FROM corpus_statistics WHERE collection = ?1",
    )
    .bind(collection)
    .execute(pool)
    .await
    {
        error!(target = "vocabulary", error = %e, "Corpus stats delete failed");
        return;
    }

    // Step 3: Clear in-memory BM25 state
    lexicon.clear_all().await;

    info!(target = "vocabulary", vocab_deleted, junk_removed, collection,
        duration_ms = start.elapsed().as_millis() as u64,
        "Vocabulary cleared. Will rebuild incrementally on next processing.");
}

/// Re-extract keywords/tags by enqueuing uplift operations.
async fn rebuild_keywords(
    db_pool: Option<&sqlx::SqlitePool>,
    tenant_id: Option<&str>,
    collection: &str,
) {
    let Some(pool) = db_pool else {
        error!(target = "keywords", "Database pool not configured");
        return;
    };

    let start = std::time::Instant::now();

    // Fetch all files for the scope
    let files: Vec<(i64, String, String)> = if let Some(tid) = tenant_id {
        match sqlx::query_as::<_, (i64, String, String)>(
            "SELECT tf.file_id, tf.file_path, wf.tenant_id \
             FROM tracked_files tf JOIN watch_folders wf ON tf.watch_folder_id = wf.watch_id \
             WHERE wf.tenant_id = ?1 AND wf.collection = ?2",
        )
        .bind(tid)
        .bind(collection)
        .fetch_all(pool)
        .await
        {
            Ok(rows) => rows,
            Err(e) => {
                error!(target = "keywords", error = %e, "Failed to fetch files");
                return;
            }
        }
    } else {
        match sqlx::query_as::<_, (i64, String, String)>(
            "SELECT tf.file_id, tf.file_path, wf.tenant_id \
             FROM tracked_files tf JOIN watch_folders wf ON tf.watch_folder_id = wf.watch_id \
             WHERE wf.collection = ?1",
        )
        .bind(collection)
        .fetch_all(pool)
        .await
        {
            Ok(rows) => rows,
            Err(e) => {
                error!(target = "keywords", error = %e, "Failed to fetch files");
                return;
            }
        }
    };

    if files.is_empty() {
        info!(target = "keywords", collection, "No files found");
        return;
    }

    // Delete existing keyword/tag data for the scope
    if tenant_id.is_some() {
        // Tenant-scoped: delete by file IDs
        for (file_id, _, _) in &files {
            let _ = sqlx::query("DELETE FROM keywords WHERE file_id = ?1")
                .bind(file_id).execute(pool).await;
            let _ = sqlx::query("DELETE FROM tags WHERE file_id = ?1")
                .bind(file_id).execute(pool).await;
        }
    } else {
        // Collection-wide: bulk delete
        let _ = sqlx::query("DELETE FROM keywords WHERE collection = ?1")
            .bind(collection).execute(pool).await;
        let _ = sqlx::query("DELETE FROM tags WHERE collection = ?1")
            .bind(collection).execute(pool).await;
        let _ = sqlx::query("DELETE FROM keyword_baskets WHERE collection = ?1")
            .bind(collection).execute(pool).await;
    }

    // Enqueue uplift operations for all files
    let now = wqm_common::timestamps::now_utc();
    let mut enqueued = 0u64;
    for (_file_id, file_path, tid) in &files {
        let payload = serde_json::json!({ "file_path": file_path, "rebuild": true });
        let payload_str = payload.to_string();
        let idem_key = wqm_common::hashing::compute_content_hash(
            &format!("file|uplift|{}|{}|{}", tid, collection, payload_str),
        );

        let _ = sqlx::query(
            "INSERT OR IGNORE INTO unified_queue \
             (queue_id, idempotency_key, item_type, op, tenant_id, collection, priority, status, payload_json, created_at, updated_at) \
             VALUES (?1, ?2, 'file', 'uplift', ?3, ?4, 3, 'pending', ?5, ?6, ?7)",
        )
        .bind(uuid::Uuid::new_v4().to_string())
        .bind(&idem_key[..32])
        .bind(tid)
        .bind(collection)
        .bind(&payload_str)
        .bind(&now)
        .bind(&now)
        .execute(pool)
        .await;
        enqueued += 1;
    }

    info!(target = "keywords", enqueued, collection,
        duration_ms = start.elapsed().as_millis() as u64,
        "Cleared keyword/tag data and enqueued uplift operations");
}

/// Self-diagnosing rules reconciliation.
///
/// Compares Qdrant `rules` collection against SQLite `rules_mirror` and fixes
/// discrepancies in both directions. Deduplicates labels and content in Qdrant.
///
/// Steps: scroll Qdrant -> read SQLite -> dedup labels -> dedup content ->
/// bidirectional sync (Qdrant<->SQLite).
async fn rebuild_rules(
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
    info!("[rebuild:rules] Found {} Qdrant rules ({} unique labels)",
        all_points.len(), qdrant_by_label.len());

    // Step 2: Read SQLite rules_mirror
    let db_by_label = match rules_rebuild::read_sqlite_rules(pool).await {
        Ok(rules) => rules,
        Err(e) => {
            error!("[rebuild:rules] {}", e);
            return;
        }
    };
    info!("[rebuild:rules] Found {} rules in SQLite rules_mirror", db_by_label.len());

    // Steps 3-4: Deduplicate labels and content
    let (ids_to_delete, label_dups, content_dups) =
        rules_rebuild::deduplicate_rules(&qdrant_by_label, pool).await;

    let deleted_count = ids_to_delete.len() as u64;
    if !ids_to_delete.is_empty() {
        match storage.delete_points_by_ids("rules", &ids_to_delete).await {
            Ok(_) => info!("[rebuild:rules] Deleted {} duplicate Qdrant points", deleted_count),
            Err(e) => error!("[rebuild:rules] Failed to delete duplicate points: {}", e),
        }
    }

    // Build deduplicated state and reconcile
    let qdrant_deduped = rules_rebuild::build_deduped_state(&qdrant_by_label, &ids_to_delete);
    let (inserted, updated, enqueued) =
        rules_rebuild::reconcile_rules(pool, &qdrant_deduped, &db_by_label).await;

    info!("[rebuild:rules] Reconciliation complete in {}ms: \
        qdrant_total={}, db_total={}, label_dups={}, content_dups={}, \
        deleted={}, mirror_inserted={}, mirror_updated={}, enqueued={}",
        start.elapsed().as_millis(), all_points.len(), db_by_label.len(),
        label_dups, content_dups, deleted_count, inserted, updated, enqueued);
}

/// Rescan watch folders by enqueuing scan operations.
async fn rebuild_watch_folders(
    db_pool: Option<&sqlx::SqlitePool>,
    collection: &str,
    tenant_id: Option<&str>,
) {
    let Some(pool) = db_pool else {
        error!("[rebuild:{}] Database pool not configured", collection);
        return;
    };

    // Fetch watch folders for the given collection (and optional tenant)
    let folders: Vec<(String, String, String)> = if let Some(tid) = tenant_id {
        match sqlx::query_as::<_, (String, String, String)>(
            "SELECT watch_id, tenant_id, path FROM watch_folders \
             WHERE collection = ?1 AND tenant_id = ?2 AND enabled = 1",
        )
        .bind(collection)
        .bind(tid)
        .fetch_all(pool)
        .await
        {
            Ok(rows) => rows,
            Err(e) => {
                error!("[rebuild:{}] Failed to fetch watch folders: {}", collection, e);
                return;
            }
        }
    } else {
        match sqlx::query_as::<_, (String, String, String)>(
            "SELECT watch_id, tenant_id, path FROM watch_folders \
             WHERE collection = ?1 AND enabled = 1",
        )
        .bind(collection)
        .fetch_all(pool)
        .await
        {
            Ok(rows) => rows,
            Err(e) => {
                error!("[rebuild:{}] Failed to fetch watch folders: {}", collection, e);
                return;
            }
        }
    };

    if folders.is_empty() {
        info!("[rebuild:{}] No enabled watch folders found", collection);
        return;
    }

    // Enqueue scan operations for each watch folder
    let now = wqm_common::timestamps::now_utc();
    let mut enqueued = 0u64;
    for (watch_id, tid, path) in &folders {
        let payload = serde_json::json!({
            "path": path,
            "watch_id": watch_id,
            "rebuild": true,
        });
        let payload_str = payload.to_string();
        let idem_key = wqm_common::hashing::compute_content_hash(
            &format!("tenant|scan|{}|{}|{}", tid, collection, payload_str),
        );

        let _ = sqlx::query(
            "INSERT OR IGNORE INTO unified_queue \
             (queue_id, idempotency_key, item_type, op, tenant_id, collection, priority, status, payload_json, created_at, updated_at) \
             VALUES (?1, ?2, 'tenant', 'scan', ?3, ?4, 5, 'pending', ?5, ?6, ?7)",
        )
        .bind(uuid::Uuid::new_v4().to_string())
        .bind(&idem_key[..32])
        .bind(tid)
        .bind(collection)
        .bind(&payload_str)
        .bind(&now)
        .bind(&now)
        .execute(pool)
        .await;
        enqueued += 1;
    }

    info!("[rebuild:{}] Enqueued {} scan operations for watch folders", collection, enqueued);
}
