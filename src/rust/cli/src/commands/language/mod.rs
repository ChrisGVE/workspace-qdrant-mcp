//! Language command - consolidated LSP and Tree-sitter management
//!
//! Unified command for managing language support including LSP servers
//! and Tree-sitter grammars.
//! Subcommands: list, ts-install, ts-remove, lsp-install, lsp-remove, status

use anyhow::Result;
use clap::{Args, Subcommand};

pub(crate) mod helpers;
mod list;
mod lsp;
mod status;
mod treesitter;
mod warm;

/// Language command arguments
#[derive(Args)]
pub struct LanguageArgs {
    #[command(subcommand)]
    command: LanguageCommand,
}

/// Language subcommands
#[derive(Subcommand)]
enum LanguageCommand {
    /// List available languages with LSP/grammar support status
    List {
        /// Show only installed components
        #[arg(short, long)]
        installed: bool,

        /// Filter by category (programming, markup, config, data)
        #[arg(short, long)]
        category: Option<String>,

        /// Show detailed information
        #[arg(short, long)]
        verbose: bool,
    },

    /// List Tree-sitter grammars with cache status
    #[command(name = "ts-list")]
    TsList {
        /// Show all available grammars including uncached
        #[arg(short, long)]
        all: bool,
    },

    /// Install Tree-sitter grammar for a language
    #[command(name = "ts-install")]
    TsInstall {
        /// Language to install grammar for (e.g., rust, python)
        language: String,

        /// Force reinstall even if already cached
        #[arg(short, long)]
        force: bool,
    },

    /// Remove Tree-sitter grammar for a language
    #[command(name = "ts-remove")]
    TsRemove {
        /// Language to remove (or 'all' for all grammars)
        language: String,
    },

    /// Install LSP server for a language (shows installation guide)
    #[command(name = "lsp-install")]
    LspInstall {
        /// Language (rust, python, typescript, go, java, c, cpp)
        language: String,
    },

    /// Remove LSP server for a language (shows removal guide)
    #[command(name = "lsp-remove")]
    LspRemove {
        /// Language to remove (rust, python, typescript, go, java, c, cpp)
        language: String,
    },

    /// Show language support status (LSP + grammar availability)
    Status {
        /// Specific language to check (or all if omitted)
        language: Option<String>,

        /// Show detailed status information
        #[arg(short, long)]
        verbose: bool,
    },

    /// Show detailed information about a specific language
    Info {
        /// Language to inspect (e.g., rust, python, go)
        language: String,
    },

    /// Search available Tree-sitter grammars for a language
    #[command(name = "ts-search")]
    TsSearch {
        /// Language to search grammars for
        language: String,
    },

    /// List all LSP servers across all languages
    #[command(name = "lsp-list")]
    LspList {
        /// Show all servers including those not detected
        #[arg(short, long)]
        all: bool,
    },

    /// Search available LSP servers for a language
    #[command(name = "lsp-search")]
    LspSearch {
        /// Language to search LSP servers for
        language: String,
    },

    /// Pre-warm tree-sitter grammars for a project
    Warm {
        /// Project directory to scan (default: current directory)
        #[arg(short, long)]
        project: Option<String>,

        /// Comma-separated list of languages to warm (skips project scan)
        #[arg(short, long)]
        languages: Option<String>,

        /// Force re-download even if already cached
        #[arg(short, long)]
        force: bool,
    },

    /// Refresh language registry from upstream providers
    Refresh,
}

/// Execute language command
pub async fn execute(args: LanguageArgs) -> Result<()> {
    match args.command {
        LanguageCommand::List {
            installed,
            category,
            verbose,
        } => list::list_languages(installed, category, verbose).await,
        LanguageCommand::TsList { all } => treesitter::ts_list(all).await,
        LanguageCommand::TsInstall { language, force } => {
            treesitter::ts_install(&language, force).await
        }
        LanguageCommand::TsRemove { language } => treesitter::ts_remove(&language).await,
        LanguageCommand::LspInstall { language } => lsp::lsp_install(&language).await,
        LanguageCommand::LspRemove { language } => lsp::lsp_remove(&language).await,
        LanguageCommand::Status { language, verbose } => {
            status::language_status(language, verbose).await
        }
        LanguageCommand::Info { language } => status::language_info(&language).await,
        LanguageCommand::TsSearch { language } => treesitter::ts_search(&language).await,
        LanguageCommand::LspList { all } => lsp::lsp_list(all).await,
        LanguageCommand::LspSearch { language } => lsp::lsp_search(&language).await,
        LanguageCommand::Warm {
            project,
            languages,
            force,
        } => warm::warm(project.as_deref(), languages.as_deref(), force).await,
        LanguageCommand::Refresh => status::language_refresh().await,
    }
}
