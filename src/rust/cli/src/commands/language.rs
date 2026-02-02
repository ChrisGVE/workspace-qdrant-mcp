//! Language command - LSP + grammar tools
//!
//! Phase 2 MEDIUM priority command for language tools.
//! Merged from old lsp and grammar commands.
//! Subcommands: list, status, install, restart, diagnose

use anyhow::Result;
use clap::{Args, Subcommand};

use crate::grpc::client::DaemonClient;
use crate::grpc::proto::{QueueType, RefreshSignalRequest};
use crate::output::{self, ServiceStatus};

/// Language command arguments
#[derive(Args)]
pub struct LanguageArgs {
    #[command(subcommand)]
    command: LanguageCommand,
}

/// Language subcommands
#[derive(Subcommand)]
enum LanguageCommand {
    /// List available languages and their LSP/grammar support
    List {
        /// Show only installed
        #[arg(short, long)]
        installed: bool,

        /// Filter by category (programming, markup, config, data)
        #[arg(short, long)]
        category: Option<String>,
    },

    /// Show language support status
    Status {
        /// Specific language to check
        language: Option<String>,
    },

    /// Install language support
    Install {
        /// Language to install
        language: String,

        /// Skip LSP installation
        #[arg(long)]
        no_lsp: bool,

        /// Skip grammar installation
        #[arg(long)]
        no_grammar: bool,
    },

    /// Restart language services
    Restart {
        /// Specific language to restart (or all if omitted)
        language: Option<String>,
    },

    /// Diagnose language support issues
    Diagnose {
        /// Language to diagnose
        language: String,
    },
}

/// Execute language command
pub async fn execute(args: LanguageArgs) -> Result<()> {
    match args.command {
        LanguageCommand::List {
            installed,
            category,
        } => list_languages(installed, category).await,
        LanguageCommand::Status { language } => language_status(language).await,
        LanguageCommand::Install {
            language,
            no_lsp,
            no_grammar,
        } => install_language(&language, no_lsp, no_grammar).await,
        LanguageCommand::Restart { language } => restart_language(language).await,
        LanguageCommand::Diagnose { language } => diagnose_language(&language).await,
    }
}

async fn list_languages(installed: bool, category: Option<String>) -> Result<()> {
    output::section("Language Support");

    if installed {
        output::info("Showing installed languages only");
    }
    if let Some(cat) = &category {
        output::kv("Category", cat);
    }
    output::separator();

    // Language support info comes from daemon
    match DaemonClient::connect_default().await {
        Ok(mut client) => {
            output::status_line("Daemon", ServiceStatus::Healthy);

            match client.system().get_status(()).await {
                Ok(_response) => {
                    output::separator();
                    output::info("Language support configured via research/languages/:");
                    output::info("  - LSP servers in research/languages/lsp/");
                    output::info("  - Tree-sitter grammars in research/languages/tree-sitter/");
                    output::separator();

                    output::info("Common languages with full support:");
                    let languages = [
                        ("Python", "pyright", "tree-sitter-python"),
                        ("JavaScript", "typescript-language-server", "tree-sitter-javascript"),
                        ("TypeScript", "typescript-language-server", "tree-sitter-typescript"),
                        ("Rust", "rust-analyzer", "tree-sitter-rust"),
                        ("Go", "gopls", "tree-sitter-go"),
                        ("Java", "jdtls", "tree-sitter-java"),
                    ];

                    for (lang, lsp, grammar) in languages {
                        output::kv(&format!("  {}", lang), &format!("{} + {}", lsp, grammar));
                    }
                }
                Err(e) => {
                    output::warning(format!("Could not get status: {}", e));
                }
            }
        }
        Err(_) => {
            output::status_line("Daemon", ServiceStatus::Unhealthy);
            output::warning("Cannot check installed languages without daemon");
        }
    }

    Ok(())
}

async fn language_status(language: Option<String>) -> Result<()> {
    output::section("Language Status");

    match &language {
        Some(lang) => output::kv("Language", lang),
        None => output::info("Checking all languages..."),
    }
    output::separator();

    match DaemonClient::connect_default().await {
        Ok(mut client) => {
            match client.system().health(()).await {
                Ok(response) => {
                    let health = response.into_inner();

                    // Look for LSP and grammar components
                    for comp in &health.components {
                        if comp.component_name.contains("lsp")
                            || comp.component_name.contains("grammar")
                            || comp.component_name.contains("language")
                        {
                            let status = ServiceStatus::from_proto(comp.status);
                            output::status_line(&comp.component_name, status);
                            if !comp.message.is_empty() {
                                output::kv("  Message", &comp.message);
                            }
                        }
                    }
                }
                Err(e) => {
                    output::error(format!("Failed to get health: {}", e));
                }
            }
        }
        Err(_) => {
            output::error("Daemon not running");
        }
    }

    Ok(())
}

async fn install_language(language: &str, no_lsp: bool, no_grammar: bool) -> Result<()> {
    output::section(format!("Install Language: {}", language));

    output::kv("Language", language);
    output::kv("Install LSP", &(!no_lsp).to_string());
    output::kv("Install Grammar", &(!no_grammar).to_string());
    output::separator();

    output::info("Language installation typically involves:");

    if !no_lsp {
        output::info(&format!("  1. Installing LSP server for {}", language));
        output::info("     - Check package manager (npm, pip, cargo, etc.)");
        output::info("     - Verify binary is in PATH");
    }

    if !no_grammar {
        output::info(&format!(
            "  2. Compiling tree-sitter grammar for {}",
            language
        ));
        output::info("     - Grammar sources in research/languages/tree-sitter/");
        output::info("     - Compiled .so files in shared location");
    }

    output::separator();
    output::info("After installation, signal daemon to reload:");

    match DaemonClient::connect_default().await {
        Ok(mut client) => {
            let request = RefreshSignalRequest {
                queue_type: QueueType::ToolsAvailable as i32,
                lsp_languages: vec![language.to_string()],
                grammar_languages: vec![language.to_string()],
            };

            match client.system().send_refresh_signal(request).await {
                Ok(_) => {
                    output::success("Daemon notified of new language support");
                }
                Err(e) => {
                    output::warning(format!("Could not notify daemon: {}", e));
                }
            }
        }
        Err(_) => {
            output::info("  wqm service restart (to reload language support)");
        }
    }

    Ok(())
}

async fn restart_language(language: Option<String>) -> Result<()> {
    output::section("Restart Language Services");

    match &language {
        Some(lang) => output::kv("Language", lang),
        None => output::info("Restarting all language services..."),
    }
    output::separator();

    match DaemonClient::connect_default().await {
        Ok(mut client) => {
            let request = RefreshSignalRequest {
                queue_type: QueueType::ToolsAvailable as i32,
                lsp_languages: language.clone().map_or(vec![], |l| vec![l.clone()]),
                grammar_languages: language.map_or(vec![], |l| vec![l]),
            };

            match client.system().send_refresh_signal(request).await {
                Ok(_) => {
                    output::success("Language services restart requested");
                }
                Err(e) => {
                    output::error(format!("Failed to restart: {}", e));
                }
            }
        }
        Err(_) => {
            output::error("Daemon not running");
        }
    }

    Ok(())
}

async fn diagnose_language(language: &str) -> Result<()> {
    output::section(format!("Diagnose: {}", language));

    output::info("Checking language support configuration...");
    output::separator();

    // Check for common issues
    output::info("Diagnostic checks:");

    output::info(&format!("  1. LSP server for {}:", language));
    output::info("     - Is LSP binary installed?");
    output::info("     - Is it in PATH?");
    output::info("     - Does it start without errors?");

    output::info(&format!("  2. Tree-sitter grammar for {}:", language));
    output::info("     - Is grammar compiled?");
    output::info("     - Is .so file loadable?");

    output::info("  3. File associations:");
    output::info("     - Are file extensions mapped correctly?");

    output::separator();
    output::info("Manual diagnosis commands:");
    output::info(&format!(
        "  which {}-language-server  # Check LSP binary",
        language
    ));
    output::info("  ls ~/.local/share/tree-sitter/  # Check grammar files");

    Ok(())
}
