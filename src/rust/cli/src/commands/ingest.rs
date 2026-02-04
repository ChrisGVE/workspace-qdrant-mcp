//! Ingest command - document ingestion
//!
//! Phase 2 MEDIUM priority command for document processing.
//! Subcommands: file, folder, text, status
//!
//! Ingestion routes through the daemon which handles embedding and Qdrant writes.

use std::path::PathBuf;

use anyhow::{Context, Result};
use clap::{Args, Subcommand};

use crate::grpc::client::DaemonClient;
use crate::grpc::proto::{IngestTextRequest, QueueType, RefreshSignalRequest};
use crate::output::{self, ServiceStatus};
use crate::queue::{UnifiedQueueClient, ContentPayload as QueueContentPayload};

/// Ingest command arguments
#[derive(Args)]
pub struct IngestArgs {
    #[command(subcommand)]
    command: IngestCommand,
}

/// Ingest subcommands
#[derive(Subcommand)]
enum IngestCommand {
    /// Ingest a single file
    File {
        /// Path to file
        path: PathBuf,

        /// Target collection (optional, auto-detected)
        #[arg(short, long)]
        collection: Option<String>,

        /// Library tag (for library ingestion)
        #[arg(short, long)]
        tag: Option<String>,
    },

    /// Ingest a folder recursively
    Folder {
        /// Path to folder
        path: PathBuf,

        /// Target collection (optional, auto-detected)
        #[arg(short, long)]
        collection: Option<String>,

        /// Library tag (for library ingestion)
        #[arg(short, long)]
        tag: Option<String>,

        /// File patterns to include
        #[arg(short = 'p', long)]
        patterns: Vec<String>,

        /// Maximum files to process
        #[arg(short = 'n', long)]
        limit: Option<usize>,
    },

    /// Ingest raw text content
    Text {
        /// Text content to ingest
        content: String,

        /// Target collection
        #[arg(short, long)]
        collection: String,

        /// Document title/identifier
        #[arg(short, long)]
        title: Option<String>,
    },

    /// Show ingestion queue status
    Status {
        /// Show detailed queue items
        #[arg(short, long)]
        verbose: bool,
    },
}

/// Execute ingest command
pub async fn execute(args: IngestArgs) -> Result<()> {
    match args.command {
        IngestCommand::File {
            path,
            collection,
            tag,
        } => ingest_file(&path, collection, tag).await,
        IngestCommand::Folder {
            path,
            collection,
            tag,
            patterns,
            limit,
        } => ingest_folder(&path, collection, tag, &patterns, limit).await,
        IngestCommand::Text {
            content,
            collection,
            title,
        } => ingest_text(&content, &collection, title).await,
        IngestCommand::Status { verbose } => ingest_status(verbose).await,
    }
}

async fn ingest_file(
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

    output::kv("File", &abs_path.display().to_string());
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

async fn ingest_folder(
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

    output::kv("Folder", &abs_path.display().to_string());
    if let Some(c) = &collection {
        output::kv("Collection", c);
    }
    if let Some(t) = &tag {
        output::kv("Library Tag", t);
    }
    if !patterns.is_empty() {
        output::kv("Patterns", &patterns.join(", "));
    }
    if let Some(l) = limit {
        output::kv("Limit", &l.to_string());
    }
    output::separator();

    // Folder ingestion - recommend setting up a watch
    output::info("For bulk folder ingestion, consider setting up a watch:");
    output::info(&format!(
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

async fn ingest_text(content: &str, collection: &str, title: Option<String>) -> Result<()> {
    output::section("Ingest Text");

    let preview = if content.len() > 50 {
        format!("{}...", &content[..50])
    } else {
        content.to_string()
    };

    output::kv("Content", &preview);
    output::kv("Collection", collection);
    if let Some(t) = &title {
        output::kv("Title", t);
    }
    output::separator();

    match DaemonClient::connect_default().await {
        Ok(mut client) => {
            output::info("Ingesting text via daemon...");

            let request = IngestTextRequest {
                content: content.to_string(),
                collection_basename: collection.to_string(),
                tenant_id: String::new(), // Auto-detected by daemon
                document_id: title.clone(),
                metadata: std::collections::HashMap::new(),
                chunk_text: true,
            };

            match client.document().ingest_text(request).await {
                Ok(response) => {
                    let result = response.into_inner();
                    if result.success {
                        output::success("Text ingested successfully");
                        output::kv("Document ID", &result.document_id);
                        output::kv("Chunks Created", &result.chunks_created.to_string());
                    } else {
                        output::error(format!("Ingestion failed: {}", result.error_message));
                    }
                }
                Err(e) => {
                    output::error(format!("Failed to ingest text: {}", e));
                    // Try unified queue fallback
                    try_queue_fallback_text(content, collection, &title).await?;
                }
            }
        }
        Err(_) => {
            // Daemon not running - use unified queue fallback (Task 37.10)
            output::warning("Daemon not running, using unified queue fallback");
            try_queue_fallback_text(content, collection, &title).await?;
        }
    }

    Ok(())
}

/// Fallback to unified queue when daemon is unavailable (Task 37.10)
async fn try_queue_fallback_text(
    content: &str,
    collection: &str,
    _title: &Option<String>,
) -> Result<()> {
    output::info("Enqueueing to unified_queue for later processing...");

    match UnifiedQueueClient::connect() {
        Ok(queue_client) => {
            // Create content payload
            let payload = QueueContentPayload {
                content: content.to_string(),
                source_type: "cli".to_string(),
                main_tag: None,
                full_tag: None,
            };

            // Get tenant_id from current directory (auto-detect project)
            let tenant_id = detect_tenant_id();
            let branch = detect_branch();

            match queue_client.enqueue_content(
                &tenant_id,
                collection,
                &payload,
                8, // Default priority for CLI content
                &branch,
            ) {
                Ok(result) => {
                    if result.was_duplicate {
                        output::warning("Content already queued (duplicate)");
                        output::kv("Idempotency Key", &result.idempotency_key);
                    } else {
                        output::success("Content queued for processing");
                        output::kv("Queue ID", &result.queue_id);
                        output::kv("Status", "pending");
                        output::kv("Fallback Mode", "unified_queue");
                    }
                    output::separator();
                    output::info("The content will be processed when the daemon starts.");
                    output::info("Check status with: wqm status queue");
                }
                Err(e) => {
                    output::error(format!("Failed to enqueue content: {}", e));
                }
            }
        }
        Err(e) => {
            output::error(format!("Failed to connect to queue database: {}", e));
            output::info("Ensure the workspace-qdrant directory exists.");
        }
    }

    Ok(())
}

/// Detect tenant_id from current working directory
fn detect_tenant_id() -> String {
    // Try to get project root from git
    if let Ok(output) = std::process::Command::new("git")
        .args(["rev-parse", "--show-toplevel"])
        .output()
    {
        if output.status.success() {
            if let Ok(path) = String::from_utf8(output.stdout) {
                let path = path.trim();
                // Normalize to a tenant_id
                return path.replace(['/', '\\', ' '], "_")
                    .trim_start_matches('_')
                    .to_string();
            }
        }
    }

    // Fallback to current directory
    std::env::current_dir()
        .map(|p| p.to_string_lossy().replace(['/', '\\', ' '], "_")
            .trim_start_matches('_')
            .to_string())
        .unwrap_or_else(|_| "default".to_string())
}

/// Detect current git branch
fn detect_branch() -> String {
    if let Ok(output) = std::process::Command::new("git")
        .args(["rev-parse", "--abbrev-ref", "HEAD"])
        .output()
    {
        if output.status.success() {
            if let Ok(branch) = String::from_utf8(output.stdout) {
                return branch.trim().to_string();
            }
        }
    }
    "main".to_string()
}

async fn ingest_status(verbose: bool) -> Result<()> {
    output::section("Ingestion Status");

    match DaemonClient::connect_default().await {
        Ok(mut client) => {
            output::status_line("Daemon", ServiceStatus::Healthy);

            match client.system().get_metrics(()).await {
                Ok(response) => {
                    let metrics = response.into_inner();

                    let mut pending = 0.0;
                    let mut processed = 0.0;
                    let mut failed = 0.0;

                    for metric in &metrics.metrics {
                        match metric.name.as_str() {
                            "queue_pending" => pending = metric.value,
                            "queue_processed" => processed = metric.value,
                            "queue_failed" => failed = metric.value,
                            _ => {}
                        }
                    }

                    output::separator();
                    output::kv("Pending", &(pending as i64).to_string());
                    output::kv("Processed", &(processed as i64).to_string());
                    output::kv("Failed", &(failed as i64).to_string());

                    if verbose {
                        output::separator();
                        output::info("Queue details stored in SQLite unified_queue table:");
                        let db_path = dirs::data_local_dir()
                            .map(|p| p.join("workspace-qdrant/state.db"))
                            .unwrap_or_default();
                        output::info(&format!(
                            "  sqlite3 {} 'SELECT * FROM unified_queue LIMIT 20'",
                            db_path.display()
                        ));
                    }
                }
                Err(e) => {
                    output::error(format!("Failed to get queue status: {}", e));
                }
            }
        }
        Err(_) => {
            output::status_line("Daemon", ServiceStatus::Unhealthy);
            output::error("Daemon not running");
        }
    }

    Ok(())
}
