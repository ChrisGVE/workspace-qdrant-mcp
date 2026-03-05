//! Library rescan subcommand

use anyhow::Result;
use wqm_common::constants::COLLECTION_LIBRARIES;

use super::helpers::{open_db, signal_daemon_ingest_queue, DEFAULT_LIBRARY_PATTERNS};
use crate::output;
use crate::queue::{ItemType, QueueOperation, UnifiedQueueClient};

/// Rescan and re-ingest a library
pub async fn execute(tag: &str, force: bool) -> Result<()> {
    output::section(format!("Rescan Library: {}", tag));

    let conn = open_db()?;
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

    // Enqueue a folder scan for the library
    match UnifiedQueueClient::connect() {
        Ok(client) => {
            let payload_json = serde_json::json!({
                "folder_path": lib_path,
                "recursive": true,
                "patterns": DEFAULT_LIBRARY_PATTERNS,
            })
            .to_string();

            match client.enqueue(
                ItemType::Folder,
                QueueOperation::Scan,
                tag,
                COLLECTION_LIBRARIES,
                &payload_json,
                "",
                None,
            ) {
                Ok(result) => {
                    if result.was_duplicate {
                        output::info("Library rescan already queued");
                    } else {
                        output::success("Library rescan queued for processing");
                    }
                }
                Err(e) => {
                    output::warning(format!("Could not queue rescan: {}", e));
                }
            }
        }
        Err(e) => {
            output::warning(format!(
                "Could not connect to queue: {}. Start daemon first.",
                e
            ));
        }
    }

    // Signal daemon to process immediately
    signal_daemon_ingest_queue().await;

    Ok(())
}
