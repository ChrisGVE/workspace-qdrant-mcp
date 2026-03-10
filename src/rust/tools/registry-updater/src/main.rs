//! Registry Updater — automated language registry generation tool.
//!
//! Scrapes upstream language data sources (GitHub Linguist, tree-sitter-grammars,
//! nvim-treesitter, mason-registry, Microsoft LSP list, langserver.org, tree-sitter
//! wiki) and generates a unified `language_registry.yaml` file.
//!
//! # Usage
//!
//! ```bash
//! cargo run --bin registry-updater -- --output language_registry.yaml
//! cargo run --bin registry-updater -- --dry-run
//! cargo run --bin registry-updater -- --sources linguist,ts-grammars,nvim-treesitter
//! ```

mod merger;
mod query_parser;
mod scraper;
mod validator;

use std::path::PathBuf;

use anyhow::Result;
use clap::Parser;

/// Automated language registry updater.
///
/// Scrapes upstream language data sources and generates a unified
/// language_registry.yaml with 200+ language definitions.
#[derive(Parser, Debug)]
#[command(name = "registry-updater", version, about)]
struct Cli {
    /// Output file path for the generated YAML.
    /// If not specified, prints to stdout.
    #[arg(short, long)]
    output: Option<PathBuf>,

    /// Dry-run mode: fetch and merge but only print summary statistics.
    #[arg(long, default_value_t = false)]
    dry_run: bool,

    /// Comma-separated list of sources to fetch from.
    /// Available: linguist, ts-grammars, nvim-treesitter, mason,
    ///            microsoft-lsp, langserver-org, ts-wiki
    /// Default: all sources.
    #[arg(long, value_delimiter = ',')]
    sources: Option<Vec<String>>,

    /// Path to the current language_registry.yaml for diff comparison.
    #[arg(long)]
    current: Option<PathBuf>,

    /// GitHub API token for higher rate limits (optional).
    #[arg(long, env = "GITHUB_TOKEN")]
    github_token: Option<String>,

    /// Log level (trace, debug, info, warn, error).
    #[arg(long, default_value = "info")]
    log_level: String,
}

#[tokio::main]
async fn main() -> Result<()> {
    let cli = Cli::parse();

    // Initialize logging
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| cli.log_level.parse().unwrap_or_default()),
        )
        .init();

    tracing::info!("Registry updater starting");

    // Determine which sources to fetch
    let source_filter = cli.sources.as_deref();

    // Phase 1: Scrape all upstream sources
    let scraped = scraper::scrape_all(source_filter, cli.github_token.as_deref()).await?;

    tracing::info!(
        languages = scraped.languages.len(),
        grammars = scraped.grammars.len(),
        lsp_servers = scraped.lsp_servers.len(),
        "Scraping complete"
    );

    // Phase 2: Merge scraped data into unified definitions
    let merged = merger::merge_all(&scraped)?;

    tracing::info!(total_languages = merged.len(), "Merge complete");

    // Phase 3: Diff against current registry if provided
    if let Some(ref current_path) = cli.current {
        let diff = merger::diff_with_current(current_path, &merged)?;
        tracing::info!(
            added = diff.added,
            removed = diff.removed,
            modified = diff.modified,
            "Diff against current registry"
        );
    }

    // Phase 4: Validate
    let issues = validator::validate_definitions(&merged);
    let errors: Vec<_> = issues
        .iter()
        .filter(|i| i.severity == validator::Severity::Error)
        .collect();
    if !errors.is_empty() {
        validator::print_report(&issues);
        anyhow::bail!("{} validation errors found", errors.len());
    }
    if !issues.is_empty() {
        validator::print_report(&issues);
    }

    // Phase 5: Output
    if cli.dry_run {
        print_summary(&merged);
    } else {
        let yaml = merger::serialize_definitions(&merged)?;
        match cli.output {
            Some(ref path) => {
                std::fs::write(path, &yaml)?;
                tracing::info!(path = %path.display(), "Wrote registry YAML");
            }
            None => {
                print!("{yaml}");
            }
        }
    }

    Ok(())
}

/// Print a summary of the merged registry.
fn print_summary(
    definitions: &[workspace_qdrant_core::language_registry::types::LanguageDefinition],
) {
    let with_grammar = definitions.iter().filter(|d| d.has_grammar()).count();
    let with_lsp = definitions.iter().filter(|d| d.has_lsp()).count();
    let with_patterns = definitions.iter().filter(|d| d.has_semantic_patterns()).count();

    println!("Registry Update Summary");
    println!("=======================");
    println!("Total languages:      {}", definitions.len());
    println!("With grammar sources: {with_grammar}");
    println!("With LSP servers:     {with_lsp}");
    println!("With semantic patterns: {with_patterns}");
    println!();

    // List languages by type
    use workspace_qdrant_core::language_registry::types::LanguageType;
    let programming = definitions
        .iter()
        .filter(|d| d.language_type == LanguageType::Programming)
        .count();
    let data = definitions
        .iter()
        .filter(|d| d.language_type == LanguageType::Data)
        .count();
    let markup = definitions
        .iter()
        .filter(|d| d.language_type == LanguageType::Markup)
        .count();
    let prose = definitions
        .iter()
        .filter(|d| d.language_type == LanguageType::Prose)
        .count();

    println!("By type:");
    println!("  Programming: {programming}");
    println!("  Data:        {data}");
    println!("  Markup:      {markup}");
    println!("  Prose:       {prose}");
}
