//! Library rescan subcommand

use anyhow::Result;
use wqm_common::constants::COLLECTION_LIBRARIES;

use super::helpers::{signal_daemon_ingest_queue, DEFAULT_LIBRARY_PATTERNS};
use crate::data::db::connect_readonly;
use crate::grpc::ensure_daemon_available;
use crate::grpc::proto::EnqueueItemRequest;
use crate::output;

/// Rescan and re-ingest a library
pub async fn execute(tag: &str, force: bool) -> Result<()> {
    output::section(format!("Rescan Library: {}", tag));

    let conn = connect_readonly()?;
    let watch_id = format!("lib-{}", tag);

    // Look up library path from watch_folders
    let result: Result<(String, Option<String>), _> = conn.query_row(
        "SELECT path, library_mode FROM watch_folders \
         WHERE watch_id = ? AND collection = 'libraries'",
        [&watch_id],
        |row| Ok((row.get::<_, String>(0)?, row.get::<_, Option<String>>(1)?)),
    );

    let (lib_path, mode) = match result {
        Ok(r) => r,
        Err(_) => {
            output::error(format!("Library '{}' not found", tag));
            output::info("List libraries with: wqm library list");
            return Ok(());
        }
    };

    output::kv("Tag", tag);
    output::kv("Path", &lib_path);
    output::kv("Mode", mode.as_deref().unwrap_or("incremental"));
    output::kv("Force", force.to_string());
    output::separator();

    // Enqueue a folder scan for the library via gRPC
    let mut client = ensure_daemon_available().await?;

    let payload_json = serde_json::json!({
        "folder_path": lib_path,
        "recursive": true,
        "patterns": DEFAULT_LIBRARY_PATTERNS,
    })
    .to_string();

    match client
        .queue_write()
        .enqueue_item(EnqueueItemRequest {
            item_type: "folder".to_string(),
            op: "scan".to_string(),
            tenant_id: tag.to_string(),
            collection: COLLECTION_LIBRARIES.to_string(),
            payload_json,
            branch: String::new(),
            metadata_json: None,
        })
        .await
    {
        Ok(resp) => {
            let inner = resp.into_inner();
            if inner.is_new {
                output::success("Library rescan queued for processing");
            } else {
                output::info("Library rescan already queued");
            }
        }
        Err(e) => {
            output::warning(format!("Could not queue rescan: {}", e));
        }
    }

    // Signal daemon to process immediately
    signal_daemon_ingest_queue().await;

    Ok(())
}
