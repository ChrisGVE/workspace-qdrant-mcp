//! Language command - consolidated LSP and Tree-sitter management
//!
//! Unified command for managing language support including LSP servers
//! and Tree-sitter grammars.
//! Subcommands: list, ts-install, ts-remove, lsp-install, lsp-remove, status

use anyhow::Result;
use clap::{Args, Subcommand};

mod helpers;
mod list;
mod lsp;
mod status;
mod treesitter;

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
}

/// Execute language command
pub async fn execute(args: LanguageArgs) -> Result<()> {
    match args.command {
        LanguageCommand::List {
            installed,
            category,
            verbose,
        } => list::list_languages(installed, category, verbose).await,
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
    }
}
