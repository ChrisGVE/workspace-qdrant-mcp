//! Rebuild target helper functions for the RebuildIndex RPC
//!
//! Each sub-module handles a group of related rebuild targets.
//! The `dispatch` function routes by target name.

use std::sync::Arc;
use tracing::info;

mod folder_targets;
mod index_targets;
mod keyword_targets;
mod storage_targets;

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
    force: bool,
) {
    match target {
        "tags" => index_targets::rebuild_tags(hierarchy_builder, tenant_id).await,
        "search" => index_targets::rebuild_search(search_db).await,
        "vocabulary" => {
            index_targets::rebuild_vocabulary(lexicon_manager, db_pool, collection).await
        }
        "keywords" => keyword_targets::rebuild_keywords(db_pool, tenant_id, collection).await,
        "rules" => storage_targets::rebuild_rules(storage_client, db_pool).await,
        "rules-payload" => storage_targets::rebuild_rules_payload(storage_client, db_pool).await,
        "scratchpad" => storage_targets::rebuild_scratchpad(storage_client, db_pool).await,
        "projects" => folder_targets::rebuild_watch_folders(db_pool, "projects", tenant_id).await,
        "libraries" => folder_targets::rebuild_watch_folders(db_pool, "libraries", tenant_id).await,
        "components" => folder_targets::rebuild_components(db_pool, tenant_id, force).await,
        "all" => {
            info!("Starting full rebuild (all targets)");
            index_targets::rebuild_vocabulary(lexicon_manager, db_pool, collection).await;
            index_targets::rebuild_search(search_db).await;
            folder_targets::rebuild_components(db_pool, tenant_id, force).await;
            index_targets::rebuild_tags(hierarchy_builder, tenant_id).await;
            keyword_targets::rebuild_keywords(db_pool, tenant_id, collection).await;
            // Recover payload fields from any legacy RULE-headered content
            // before reconciliation scans by label.
            storage_targets::rebuild_rules_payload(storage_client.clone(), db_pool).await;
            storage_targets::rebuild_rules(storage_client.clone(), db_pool).await;
            storage_targets::rebuild_scratchpad(storage_client, db_pool).await;
            folder_targets::rebuild_watch_folders(db_pool, "projects", tenant_id).await;
            folder_targets::rebuild_watch_folders(db_pool, "libraries", tenant_id).await;
            info!("Full rebuild complete (all targets)");
        }
        _ => {} // Validated by caller
    }
}
