//! Shared helpers for library subcommands

use std::path::PathBuf;

use anyhow::{Context, Result};
use clap::ValueEnum;
use rusqlite::Connection;

use crate::config::get_database_path;
use crate::grpc::client::DaemonClient;
use crate::grpc::proto::{QueueType, RefreshSignalRequest};
use crate::output;

/// Library sync mode controlling how file deletions are handled
#[derive(Debug, Clone, Copy, Default, ValueEnum)]
pub enum LibraryMode {
    /// Mirror mode: Delete vectors when source files are removed
    Sync,
    /// Append-only mode: Never delete vectors, only add/update (default)
    #[default]
    Incremental,
}

impl std::fmt::Display for LibraryMode {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            LibraryMode::Sync => write!(f, "sync"),
            LibraryMode::Incremental => write!(f, "incremental"),
        }
    }
}

/// Default file patterns for library collections.
/// Covers all supported document formats in `AllowedExtensions::library_extensions`.
pub const DEFAULT_LIBRARY_PATTERNS: &[&str] = &[
    "*.pdf", "*.epub", "*.docx", "*.pptx", "*.ppt", "*.pages", "*.key", "*.odt", "*.odp", "*.ods",
    "*.rtf", "*.doc", "*.md", "*.txt", "*.html", "*.htm",
];

/// Get SQLite database path (canonical: ~/.workspace-qdrant/state.db)
pub fn get_db_path() -> Result<PathBuf> {
    get_database_path().map_err(|e| anyhow::anyhow!("{}", e))
}

/// Open a connection to the state database with WAL mode
pub fn open_db() -> Result<Connection> {
    let db_path = get_db_path()?;
    if !db_path.exists() {
        anyhow::bail!(
            "Database not found at {}. Run daemon first: wqm service start",
            db_path.display()
        );
    }
    let conn = Connection::open(&db_path).context("Failed to open state database")?;
    conn.execute_batch("PRAGMA journal_mode=WAL; PRAGMA synchronous=NORMAL;")
        .context("Failed to set SQLite pragmas")?;
    Ok(conn)
}

/// Returns a human-readable description of the library mode
pub fn mode_description(mode: LibraryMode) -> &'static str {
    match mode {
        LibraryMode::Sync => "deletes vectors when files are removed",
        LibraryMode::Incremental => "append-only, never deletes vectors",
    }
}

/// Supported document extensions mapped to their source_format and document_type family.
/// Returns `(source_format, document_type)` where document_type is "page_based" or "stream_based".
pub fn classify_document_extension(ext: &str) -> Option<(&'static str, &'static str)> {
    match ext {
        // Page-based documents
        "pdf" => Some(("pdf", "page_based")),
        "docx" => Some(("docx", "page_based")),
        "doc" => Some(("doc", "page_based")),
        "pptx" => Some(("pptx", "page_based")),
        "ppt" => Some(("ppt", "page_based")),
        "pages" => Some(("pages", "page_based")),
        "key" => Some(("key", "page_based")),
        "odp" => Some(("odp", "page_based")),
        "odt" => Some(("odt", "page_based")),
        "ods" => Some(("ods", "page_based")),
        "rtf" => Some(("rtf", "page_based")),
        // Stream-based documents
        "epub" => Some(("epub", "stream_based")),
        "html" | "htm" => Some(("html", "stream_based")),
        "md" | "markdown" => Some(("markdown", "stream_based")),
        "txt" => Some(("text", "stream_based")),
        _ => None,
    }
}

/// Signal the daemon to reload watched folders configuration.
/// Logs success/failure but does not propagate errors.
pub async fn signal_daemon_watch_folders() {
    if let Ok(mut client) = DaemonClient::connect_default().await {
        let request = RefreshSignalRequest {
            queue_type: QueueType::WatchedFolders as i32,
            lsp_languages: vec![],
            grammar_languages: vec![],
        };
        if client.system().send_refresh_signal(request).await.is_ok() {
            output::success("Daemon notified of configuration change");
        }
    }
}

/// Signal the daemon to process the ingest queue.
/// Logs success/failure but does not propagate errors.
pub async fn signal_daemon_ingest_queue() {
    if let Ok(mut client) = DaemonClient::connect_default().await {
        let request = RefreshSignalRequest {
            queue_type: QueueType::IngestQueue as i32,
            lsp_languages: vec![],
            grammar_languages: vec![],
        };
        if client.system().send_refresh_signal(request).await.is_ok() {
            output::success("Daemon notified - ingestion will begin shortly");
        }
    } else {
        output::info("Daemon not running - will be processed when daemon starts");
    }
}
