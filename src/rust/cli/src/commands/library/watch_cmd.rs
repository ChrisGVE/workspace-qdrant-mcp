//! Library watch subcommand

use std::path::PathBuf;

use anyhow::{Context, Result};
use wqm_common::constants::COLLECTION_LIBRARIES;

use super::helpers::{mode_description, LibraryMode, DEFAULT_LIBRARY_PATTERNS};
use crate::grpc::ensure_daemon_available;
use crate::grpc::proto::{
    EnqueueItemRequest, QueueType, RefreshSignalRequest, WatchLibraryRequest,
};
use crate::output;
use crate::output::style::home_to_tilde;

/// Watch a library path for changes
pub async fn execute(
    tag: &str,
    path: &PathBuf,
    patterns: &[String],
    mode: LibraryMode,
) -> Result<()> {
    output::section(format!("Watch Library: {}", tag));

    if !path.exists() {
        output::error(format!("Path does not exist: {}", path.display()));
        return Ok(());
    }

    let abs_path = path
        .canonicalize()
        .context("Could not resolve absolute path")?;
    let abs_path_str = abs_path.to_string_lossy().to_string();

    let mut client = ensure_daemon_available().await?;

    let response = client
        .library_write()
        .watch_library(WatchLibraryRequest {
            tag: tag.to_string(),
            path: abs_path_str.clone(),
            mode: mode.to_string(),
            patterns: patterns.to_vec(),
        })
        .await?
        .into_inner();

    output::success(&response.message);
    output::kv("  Tag", tag);
    output::kv("  Path", home_to_tilde(&abs_path_str));
    output::kv("  Mode", format!("{} ({})", mode, mode_description(mode)));

    let effective_patterns: Vec<String> = if patterns.is_empty() {
        DEFAULT_LIBRARY_PATTERNS
            .iter()
            .map(|s| s.to_string())
            .collect()
    } else {
        patterns.to_vec()
    };
    output::kv("  Patterns", format!("{}", effective_patterns.len()));

    // Enqueue folder scan via daemon
    let payload_json = serde_json::json!({
        "folder_path": abs_path_str,
        "recursive": true,
        "patterns": effective_patterns,
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
                output::success("Library scan queued for ingestion");
            } else {
                output::info("Library scan already queued");
            }
        }
        Err(e) => {
            output::warning(format!("Could not queue library scan: {}", e));
        }
    }

    // Signal daemon to start watching
    if let Ok(_resp) = client
        .system()
        .send_refresh_signal(RefreshSignalRequest {
            queue_type: QueueType::WatchedFolders as i32,
            lsp_languages: vec![],
            grammar_languages: vec![],
        })
        .await
    {
        output::success("Daemon notified - it will start watching shortly");
    }

    Ok(())
}
