//! Capability upgrade trigger: enqueues File→Uplift items for files
//! that can benefit from newly available grammar or LSP capabilities.

use sqlx::SqlitePool;
use tracing::{info, warn};

use crate::queue_operations::QueueManager;
use crate::tracked_files_schema::{self, UpgradeReason};
use crate::unified_queue_schema::{ItemType, QueueOperation};

/// Enqueue File→Uplift items for files needing a capability upgrade.
///
/// Called when:
/// - A grammar download completes (files can now get semantic chunks)
/// - An LSP server starts (files can now get LSP enrichment)
/// - A retry sweep finds files with failed enrichment
///
/// Returns the number of files enqueued for re-processing.
pub async fn trigger_capability_upgrade(
    pool: &SqlitePool,
    queue_manager: &QueueManager,
    tenant_id: &str,
    reason: UpgradeReason,
    language: Option<&str>,
) -> u32 {
    let files = match tracked_files_schema::get_files_needing_upgrade(
        pool, tenant_id, reason, language,
    )
    .await
    {
        Ok(f) => f,
        Err(e) => {
            warn!(
                tenant_id = tenant_id,
                reason = reason.as_str(),
                error = %e,
                "Failed to query files needing capability upgrade"
            );
            return 0;
        }
    };

    if files.is_empty() {
        return 0;
    }

    let total = files.len();
    let mut enqueued = 0u32;

    for (file_id, file_path, branch, collection) in &files {
        let payload = serde_json::json!({
            "file_path": file_path,
            "upgrade_reason": reason.as_str(),
        })
        .to_string();

        let metadata = serde_json::json!({
            "file_id": file_id,
            "upgrade_reason": reason.as_str(),
        })
        .to_string();

        match queue_manager
            .enqueue_unified(
                ItemType::File,
                QueueOperation::Uplift,
                tenant_id,
                collection,
                &payload,
                Some(branch.as_str()),
                Some(&metadata),
            )
            .await
        {
            Ok((_, true)) => enqueued += 1,
            Ok((_, false)) => {
                // Deduplicated — already enqueued
            }
            Err(e) => {
                warn!(
                    file_path = file_path,
                    error = %e,
                    "Failed to enqueue capability upgrade for file"
                );
            }
        }
    }

    info!(
        tenant_id = tenant_id,
        reason = reason.as_str(),
        language = language.unwrap_or("all"),
        "Capability upgrade: enqueued {}/{} files for re-processing",
        enqueued,
        total
    );

    enqueued
}
