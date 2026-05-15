//! Library watch subcommand

use std::path::PathBuf;

use anyhow::Result;
use wqm_common::constants::COLLECTION_LIBRARIES;

use super::helpers::{
    canonical_from_cli_path, mode_description, LibraryMode, DEFAULT_LIBRARY_PATTERNS,
};
use crate::grpc::ensure_daemon_available;
use crate::grpc::proto::{
    EnqueueItemRequest, QueueType, RefreshSignalRequest, WatchLibraryRequest,
};
use crate::output;
use crate::output::style::home_to_tilde;

async fn enqueue_scan_and_notify(
    client: &mut crate::grpc::client::DaemonClient,
    tag: &str,
    abs_path_str: &str,
    patterns: &[String],
) {
    // Library root scans omit folder_path: the daemon anchors to the
    // library's watch_folder root (looked up at processing time) rather
    // than embedding an absolute path that would violate the
    // RelativePath contract.
    let _ = abs_path_str;
    let payload_json = serde_json::json!({
        "recursive": true,
        "patterns": patterns,
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
}

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

    let abs_path = canonical_from_cli_path(path)?;
    let abs_path_str = abs_path.into_string();

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

    enqueue_scan_and_notify(&mut client, tag, &abs_path_str, &effective_patterns).await;

    Ok(())
}
