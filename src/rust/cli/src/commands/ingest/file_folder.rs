//! File and folder ingest subcommand handlers

use std::path::PathBuf;

use anyhow::{Context, Result};

use crate::grpc::client::DaemonClient;
use crate::grpc::proto::{QueueType, RefreshSignalRequest};
use crate::output;

pub async fn ingest_file(
    path: &PathBuf,
    collection: Option<String>,
    tag: Option<String>,
) -> Result<()> {
    output::section("Ingest File");

    // Validate path exists
    if !path.exists() {
        output::error(format!("File does not exist: {}", path.display()));
        return Ok(());
    }

    let abs_path = path
        .canonicalize()
        .context("Could not resolve absolute path")?;

    output::kv("File", abs_path.display().to_string());
    if let Some(c) = &collection {
        output::kv("Collection", c);
    }
    if let Some(t) = &tag {
        output::kv("Library Tag", t);
    }
    output::separator();

    // File ingestion goes through the daemon's queue
    match DaemonClient::connect_default().await {
        Ok(mut client) => {
            output::info("Signaling daemon to queue file for ingestion...");

            let request = RefreshSignalRequest {
                queue_type: QueueType::IngestQueue as i32,
                lsp_languages: vec![],
                grammar_languages: vec![],
            };

            match client.system().send_refresh_signal(request).await {
                Ok(_) => {
                    output::success("File queued for ingestion");
                    output::info("The daemon will process the file and add it to Qdrant.");
                    output::info("Check status with: wqm ingest status");
                }
                Err(e) => {
                    output::error(format!("Failed to queue file: {}", e));
                }
            }
        }
        Err(_) => {
            output::error("Daemon not running. Start with: wqm service start");
        }
    }

    Ok(())
}

pub async fn ingest_folder(
    path: &PathBuf,
    collection: Option<String>,
    tag: Option<String>,
    patterns: &[String],
    limit: Option<usize>,
) -> Result<()> {
    output::section("Ingest Folder");

    // Validate path exists
    if !path.exists() {
        output::error(format!("Folder does not exist: {}", path.display()));
        return Ok(());
    }

    if !path.is_dir() {
        output::error(format!("Path is not a directory: {}", path.display()));
        return Ok(());
    }

    let abs_path = path
        .canonicalize()
        .context("Could not resolve absolute path")?;

    output::kv("Folder", abs_path.display().to_string());
    if let Some(c) = &collection {
        output::kv("Collection", c);
    }
    if let Some(t) = &tag {
        output::kv("Library Tag", t);
    }
    if !patterns.is_empty() {
        output::kv("Patterns", patterns.join(", "));
    }
    if let Some(l) = limit {
        output::kv("Limit", l.to_string());
    }
    output::separator();

    // Folder ingestion - recommend setting up a watch
    output::info("For bulk folder ingestion, consider setting up a watch:");
    output::info(format!(
        "  wqm library watch {} {}",
        tag.as_deref().unwrap_or("mylib"),
        abs_path.display()
    ));
    output::separator();

    match DaemonClient::connect_default().await {
        Ok(mut client) => {
            output::info("Signaling daemon to scan folder...");

            let request = RefreshSignalRequest {
                queue_type: QueueType::IngestQueue as i32,
                lsp_languages: vec![],
                grammar_languages: vec![],
            };

            if client.system().send_refresh_signal(request).await.is_ok() {
                output::success("Folder scan requested");
            }
        }
        Err(_) => {
            output::warning("Daemon not running - files will be processed when daemon starts");
        }
    }

    Ok(())
}
