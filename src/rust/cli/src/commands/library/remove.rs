//! Library remove subcommand

use std::io::{self, Write};

use anyhow::{Context, Result};
use wqm_common::constants::COLLECTION_LIBRARIES;

use super::helpers::open_db;
use crate::grpc::client::DaemonClient;
use crate::grpc::proto::{QueueType, RefreshSignalRequest};
use crate::output;
use crate::queue::{ItemType, QueueOperation, UnifiedQueueClient};

/// Remove a library (deletes watch config AND queues vector deletion)
pub async fn execute(tag: &str, skip_confirm: bool) -> Result<()> {
    output::section(format!("Remove Library: {}", tag));

    let watch_id = format!("lib-{}", tag);
    let conn = open_db()?;

    let exists: bool = conn
        .query_row(
            "SELECT 1 FROM watch_folders WHERE watch_id = ?",
            [&watch_id],
            |_| Ok(true),
        )
        .unwrap_or(false);

    if !exists {
        output::error(format!(
            "Library '{}' not found (watch_id: {})",
            tag, watch_id
        ));
        return Ok(());
    }

    // Confirm deletion unless --yes flag
    if !skip_confirm {
        output::warning(format!(
            "This will delete ALL vectors for library '{}' from Qdrant.",
            tag
        ));
        output::warning("This action cannot be undone.");
        output::info("");
        print!("Continue? (y/N): ");
        io::stdout().flush()?;

        let mut input = String::new();
        io::stdin().read_line(&mut input)?;
        if !input.trim().eq_ignore_ascii_case("y") {
            output::info("Cancelled.");
            return Ok(());
        }
    }

    output::separator();
    delete_watch_config(&conn, tag, &watch_id)?;
    enqueue_vector_deletion(tag);
    signal_daemon(tag).await;

    output::separator();
    output::success(format!("Library '{}' removed", tag));

    Ok(())
}

/// Delete watch folder config from SQLite, cleaning up child records atomically
fn delete_watch_config(conn: &rusqlite::Connection, tag: &str, watch_id: &str) -> Result<()> {
    output::info("Removing watch configuration...");

    // Run all deletes in a single transaction with FK checks off so child-record
    // cleanup is atomic with the parent delete — prevents races with the daemon.
    conn.execute_batch("PRAGMA foreign_keys = OFF;")
        .context("Failed to disable FK enforcement")?;

    let result = (|| -> Result<(usize, usize)> {
        let tx = conn
            .unchecked_transaction()
            .context("Failed to begin transaction")?;

        let cancelled = tx
            .execute(
                "DELETE FROM unified_queue WHERE tenant_id = ?1 AND collection = 'libraries'",
                [tag],
            )
            .context("Failed to cancel queue items")?;

        tx.execute(
            "DELETE FROM project_components WHERE watch_folder_id = ?1",
            [watch_id],
        )
        .context("Failed to delete project components")?;

        tx.execute(
            "DELETE FROM tracked_files WHERE watch_folder_id = ?1",
            [watch_id],
        )
        .context("Failed to delete tracked files")?;

        let deleted = tx
            .execute("DELETE FROM watch_folders WHERE watch_id = ?1", [watch_id])
            .context("Failed to delete watch folder config")?;

        tx.commit()
            .context("Failed to commit removal transaction")?;
        Ok((cancelled, deleted))
    })();

    conn.execute_batch("PRAGMA foreign_keys = ON;")
        .context("Failed to re-enable FK enforcement")?;

    let (cancelled, deleted) = result?;

    if cancelled > 0 {
        output::info(format!("Cancelled {} pending queue items", cancelled));
    }
    if deleted > 0 {
        output::success(format!("Removed watch config for '{}'", tag));
    }

    Ok(())
}

/// Enqueue deletion of vectors from Qdrant
fn enqueue_vector_deletion(tag: &str) {
    output::info("Queueing vector deletion...");

    let collection = COLLECTION_LIBRARIES;
    let payload_json = serde_json::json!({
        "tenant_id_to_delete": tag
    })
    .to_string();

    match UnifiedQueueClient::connect() {
        Ok(client) => {
            match client.enqueue(
                ItemType::Tenant,
                QueueOperation::Delete,
                tag,
                collection,
                &payload_json,
                "",
                None,
            ) {
                Ok(result) => {
                    if result.was_duplicate {
                        output::info("Vector deletion already queued (duplicate)");
                    } else {
                        output::success(format!(
                            "Vector deletion queued (queue_id: {})",
                            result.queue_id
                        ));
                    }
                }
                Err(e) => {
                    output::warning(format!("Could not queue vector deletion: {}", e));
                    print_manual_delete_instructions(tag, collection);
                }
            }
        }
        Err(e) => {
            output::warning(format!("Could not connect to queue: {}", e));
            output::info("Vectors will need to be deleted manually or when daemon starts.");
        }
    }
}

/// Print manual curl instructions for deleting vectors
fn print_manual_delete_instructions(tag: &str, collection: &str) {
    output::info("You may need to manually delete vectors from Qdrant:");
    output::info(format!(
        "  curl -X POST 'http://localhost:6333/collections/{}/points/delete' \\",
        collection
    ));
    output::info("    -H 'Content-Type: application/json' \\");
    output::info(format!(
        "    -d '{{\"filter\": {{\"must\": [{{\"key\": \"library_name\", \
         \"match\": {{\"value\": \"{}\"}}}}]}}}}'",
        tag
    ));
}

/// Signal daemon if available
async fn signal_daemon(tag: &str) {
    if let Ok(mut client) = DaemonClient::connect_default().await {
        output::separator();
        output::info("Signaling daemon...");
        let request = RefreshSignalRequest {
            queue_type: QueueType::WatchedFolders as i32,
            lsp_languages: vec![],
            grammar_languages: vec![],
        };
        if client.system().send_refresh_signal(request).await.is_ok() {
            output::success("Daemon notified - will process deletion shortly");
        }
    } else {
        output::separator();
        output::info(format!(
            "Daemon not running. Vector deletion for '{}' will occur when daemon starts.",
            tag
        ));
    }
}
