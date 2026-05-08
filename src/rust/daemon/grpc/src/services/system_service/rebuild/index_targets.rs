//! Rebuild handlers for index-type targets: tags, search (FTS5), vocabulary (BM25).

use std::sync::Arc;
use tracing::{error, info, warn};

/// Rebuild canonical tag hierarchy.
pub(super) async fn rebuild_tags(
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
                info!(
                    target = "tags",
                    tenant = tid,
                    canonical_tags = total,
                    edges = r.edges_created,
                    duration_ms = start.elapsed().as_millis() as u64,
                    "Tag hierarchy rebuild complete"
                );
            }
            Ok(None) => info!(target = "tags", tenant = tid, "Skipped (too few tags)"),
            Err(e) => {
                error!(target = "tags", tenant = tid, error = %e, "Tag hierarchy rebuild failed")
            }
        }
    } else {
        match builder.rebuild_all().await {
            Ok(r) => info!(
                target = "tags",
                tenants = r.tenants_processed,
                canonical_tags = r.total_canonical_tags,
                edges = r.total_edges,
                duration_ms = start.elapsed().as_millis() as u64,
                "Tag hierarchy rebuild complete (all tenants)"
            ),
            Err(e) => {
                error!(target = "tags", error = %e, "Tag hierarchy rebuild failed (all tenants)")
            }
        }
    }
}

/// Rebuild FTS5 search index.
pub(super) async fn rebuild_search(search_db: Option<Arc<workspace_qdrant_core::SearchDbManager>>) {
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
                info!(
                    target = "search",
                    duration_ms = start.elapsed().as_millis() as u64,
                    "FTS5 search index rebuilt and optimized"
                );
            }
        }
        Err(e) => error!(target = "search", error = %e, "FTS5 rebuild failed"),
    }
}

/// Rebuild BM25 sparse vocabulary.
pub(super) async fn rebuild_vocabulary(
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
    let vocab_deleted = match sqlx::query("DELETE FROM sparse_vocabulary WHERE collection = ?1")
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

    if let Err(e) = sqlx::query("DELETE FROM corpus_statistics WHERE collection = ?1")
        .bind(collection)
        .execute(pool)
        .await
    {
        error!(target = "vocabulary", error = %e, "Corpus stats delete failed");
        return;
    }

    // Step 3: Clear in-memory BM25 state
    lexicon.clear_all().await;

    info!(
        target = "vocabulary",
        vocab_deleted,
        junk_removed,
        collection,
        duration_ms = start.elapsed().as_millis() as u64,
        "Vocabulary cleared. Will rebuild incrementally on next processing."
    );
}
