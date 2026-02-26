//! Library watch subcommand

use std::path::PathBuf;

use anyhow::{Context, Result};
use wqm_common::constants::COLLECTION_LIBRARIES;
use wqm_common::timestamps;

use crate::grpc::client::DaemonClient;
use crate::grpc::proto::{QueueType, RefreshSignalRequest};
use crate::output;
use crate::queue::{UnifiedQueueClient, ItemType, QueueOperation};
use super::helpers::{open_db, mode_description, LibraryMode, DEFAULT_LIBRARY_PATTERNS};

/// Watch a library path for changes
pub async fn execute(
    tag: &str,
    path: &PathBuf,
    patterns: &[String],
    mode: LibraryMode,
) -> Result<()> {
    output::section(format!("Watch Library: {}", tag));

    // Validate path exists
    if !path.exists() {
        output::error(format!("Path does not exist: {}", path.display()));
        return Ok(());
    }

    let abs_path = path.canonicalize()
        .context("Could not resolve absolute path")?;

    let conn = open_db()?;
    let watch_id = format!("lib-{}", tag);
    let now = timestamps::now_utc();
    let abs_path_str = abs_path.to_string_lossy().to_string();

    // Check if library already exists
    let exists: bool = conn.query_row(
        "SELECT 1 FROM watch_folders WHERE watch_id = ?",
        [&watch_id],
        |_| Ok(true),
    ).unwrap_or(false);

    if exists {
        // Enable watching on existing library
        conn.execute(
            "UPDATE watch_folders SET enabled = 1, library_mode = ?, path = ?, \
             updated_at = ?, last_activity_at = ? WHERE watch_id = ?",
            rusqlite::params![&mode.to_string(), &abs_path_str, &now, &now, &watch_id],
        ).context("Failed to enable watch")?;
        output::success(format!("Library '{}' watching enabled", tag));
    } else {
        // Insert new library with watching enabled
        conn.execute(
            "INSERT INTO watch_folders \
             (watch_id, path, collection, tenant_id, library_mode, enabled, is_active, \
              follow_symlinks, cleanup_on_disable, created_at, updated_at, last_activity_at) \
             VALUES (?1, ?2, 'libraries', ?3, ?4, 1, 0, 0, 0, ?5, ?5, ?5)",
            rusqlite::params![&watch_id, &abs_path_str, tag, &mode.to_string(), &now],
        ).context("Failed to insert library watch")?;
        output::success(format!("Library '{}' added and watching enabled", tag));
    }

    output::kv("  Tag", tag);
    output::kv("  Path", &abs_path_str);
    output::kv("  Mode", &format!("{} ({})", mode, mode_description(mode)));

    // Use user-provided patterns or defaults
    let effective_patterns: Vec<String> = if patterns.is_empty() {
        DEFAULT_LIBRARY_PATTERNS.iter().map(|s| s.to_string()).collect()
    } else {
        patterns.to_vec()
    };

    output::kv("  Patterns", &format!("{}", effective_patterns.len()));

    enqueue_scan(tag, &abs_path_str, &effective_patterns);
    signal_daemon(tag).await;

    Ok(())
}

/// Enqueue a folder scan for the library
fn enqueue_scan(tag: &str, abs_path_str: &str, effective_patterns: &[String]) {
    match UnifiedQueueClient::connect() {
        Ok(client) => {
            let payload_json = serde_json::json!({
                "folder_path": abs_path_str,
                "recursive": true,
                "patterns": effective_patterns,
            }).to_string();

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
                        output::info("Library scan already queued");
                    } else {
                        output::success("Library scan queued for ingestion");
                    }
                }
                Err(e) => {
                    output::warning(format!("Could not queue library scan: {}", e));
                }
            }
        }
        Err(e) => {
            output::warning(format!(
                "Could not connect to queue: {}. Daemon will scan on next poll.",
                e
            ));
        }
    }
}

/// Signal daemon to start watching
async fn signal_daemon(tag: &str) {
    if let Ok(mut client) = DaemonClient::connect_default().await {
        let request = RefreshSignalRequest {
            queue_type: QueueType::WatchedFolders as i32,
            lsp_languages: vec![],
            grammar_languages: vec![],
        };
        if client.system().send_refresh_signal(request).await.is_ok() {
            output::success("Daemon notified - it will start watching shortly");
        }
    } else {
        output::warning(format!(
            "Daemon not running - start it to begin watching '{}'",
            tag
        ));
    }
}
