//! Collections command - Qdrant collection management
//!
//! Provides per-collection reset and listing of Qdrant collections.
//! Subcommands: list, reset
//!
//! Note: Tenant rename moved to `admin rename-tenant` command.

mod list;
mod reset;

use anyhow::Result;
use clap::{Args, Subcommand};

pub use list::list_collections;
pub use reset::reset_collections;

/// Canonical collection names (validated against wqm-common constants)
pub(crate) const VALID_COLLECTIONS: &[&str] = &[
    wqm_common::constants::COLLECTION_PROJECTS,
    wqm_common::constants::COLLECTION_LIBRARIES,
    wqm_common::constants::COLLECTION_RULES,
    wqm_common::constants::COLLECTION_SCRATCHPAD,
];

/// Get Qdrant URL from environment or default
pub(crate) fn qdrant_url() -> String {
    std::env::var("QDRANT_URL")
        .unwrap_or_else(|_| wqm_common::constants::DEFAULT_QDRANT_URL.to_string())
}

/// Get optional Qdrant API key
pub(crate) fn qdrant_api_key() -> Option<String> {
    std::env::var("QDRANT_API_KEY").ok()
}

/// Build a reqwest client with optional API key header
pub(crate) fn build_client() -> Result<reqwest::Client> {
    use anyhow::Context as _;
    let mut headers = reqwest::header::HeaderMap::new();
    if let Some(key) = qdrant_api_key() {
        headers.insert(
            "api-key",
            reqwest::header::HeaderValue::from_str(&key)
                .context("Invalid QDRANT_API_KEY value")?,
        );
    }
    reqwest::Client::builder()
        .default_headers(headers)
        .timeout(std::time::Duration::from_secs(60))
        .build()
        .context("Failed to build HTTP client")
}

/// Collections command arguments
#[derive(Args)]
pub struct CollectionsArgs {
    #[command(subcommand)]
    command: CollectionsCommand,
}

/// Collections subcommands
#[derive(Subcommand)]
enum CollectionsCommand {
    /// List Qdrant collections
    List {
        /// Output as JSON
        #[arg(long)]
        json: bool,

        /// Script-friendly space-separated output (no ANSI, one row per line)
        #[arg(long, conflicts_with = "json")]
        script: bool,

        /// Omit the header row (requires --script)
        #[arg(long, requires = "script")]
        no_headers: bool,
    },

    /// Reset (delete and recreate) specific collection(s)
    Reset {
        /// Collection name(s) to reset (projects, libraries, rules, scratchpad)
        #[arg(required = true)]
        names: Vec<String>,

        /// Also clean related pending/failed queue items from SQLite
        #[arg(long)]
        include_queue: bool,

        /// Skip confirmation prompts
        #[arg(short, long)]
        yes: bool,
    },
}

/// Execute collections command
pub async fn execute(args: CollectionsArgs) -> Result<()> {
    match args.command {
        CollectionsCommand::List { json, script, no_headers } => {
            list_collections(json, script, no_headers).await
        }
        CollectionsCommand::Reset {
            names,
            include_queue,
            yes,
        } => reset_collections(names, include_queue, yes).await,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_valid_collections() {
        assert!(VALID_COLLECTIONS.contains(&"projects"));
        assert!(VALID_COLLECTIONS.contains(&"libraries"));
        assert!(VALID_COLLECTIONS.contains(&"rules"));
        assert!(VALID_COLLECTIONS.contains(&"scratchpad"));
        assert!(!VALID_COLLECTIONS.contains(&"invalid"));
    }

    #[test]
    fn test_qdrant_url_default() {
        // When QDRANT_URL is not set, should use default
        std::env::remove_var("QDRANT_URL");
        let url = qdrant_url();
        assert!(url.starts_with("http"));
        assert!(url.contains("6333"));
    }
}
