//! Shared helpers for library subcommands

use anyhow::{Context, Result};
use clap::ValueEnum;
use wqm_common::paths::CanonicalPath;

use crate::grpc::proto::{QueueType, RefreshSignalRequest};
use crate::output;

/// Build a [`CanonicalPath`] from a CLI path argument.
///
/// Applies the nine syntactic-canonical rules (spec §3.1) and never
/// calls `std::fs::canonicalize` (rule 7). Relative inputs are
/// absolutized by joining onto the process CWD — no symlink
/// resolution.
pub fn canonical_from_cli_path(path: &std::path::Path) -> Result<CanonicalPath> {
    let s = path.to_str().context("Path contains invalid UTF-8")?;
    if let Ok(cp) = CanonicalPath::from_user_input(s) {
        return Ok(cp);
    }
    let cwd = std::env::current_dir().context("Could not determine current directory")?;
    let joined = cwd.join(path);
    let joined_str = joined
        .to_str()
        .context("Path contains invalid UTF-8 after CWD join")?;
    CanonicalPath::from_user_input(joined_str)
        .map_err(|e| anyhow::anyhow!("Could not normalize path: {e}"))
}

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

/// Signal the daemon to process the ingest queue.
/// Logs success/failure but does not propagate errors.
pub async fn signal_daemon_ingest_queue() {
    if let Ok(mut client) = crate::grpc::connect_default().await {
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
