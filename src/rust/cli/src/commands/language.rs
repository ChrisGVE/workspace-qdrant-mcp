//! Language command - consolidated LSP and Tree-sitter management
//!
//! Unified command for managing language support including LSP servers
//! and Tree-sitter grammars.
//! Subcommands: list, ts-install, ts-remove, lsp-install, lsp-remove, status

use anyhow::{anyhow, Result};
use clap::{Args, Subcommand};
use colored::Colorize;
use std::process::Command;

use crate::config;
use crate::grpc::client::DaemonClient;
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
}

/// Execute language command
pub async fn execute(args: LanguageArgs) -> Result<()> {
    match args.command {
        LanguageCommand::List {
            installed,
            category,
            verbose,
        } => list_languages(installed, category, verbose).await,
        LanguageCommand::TsInstall { language, force } => ts_install(&language, force).await,
        LanguageCommand::TsRemove { language } => ts_remove(&language).await,
        LanguageCommand::LspInstall { language } => lsp_install(&language).await,
        LanguageCommand::LspRemove { language } => lsp_remove(&language).await,
        LanguageCommand::Status { language, verbose } => language_status(language, verbose).await,
    }
}

/// List available languages with support status
async fn list_languages(installed: bool, category: Option<String>, verbose: bool) -> Result<()> {
    use workspace_qdrant_core::config::GrammarConfig;
    use workspace_qdrant_core::tree_sitter::{GrammarManager, GrammarStatus, StaticLanguageProvider};

    output::section("Language Support");

    if installed {
        output::info("Showing installed components only");
    }
    if let Some(cat) = &category {
        output::kv("Category", cat);
    }
    output::separator();

    // Check daemon connection
    let daemon_connected = DaemonClient::connect_default().await.is_ok();
    if daemon_connected {
        output::status_line("Daemon", ServiceStatus::Healthy);
    } else {
        output::status_line("Daemon", ServiceStatus::Unhealthy);
    }
    output::separator();

    // Show LSP servers available
    println!("{}", "LSP Servers".cyan().bold());
    let servers = detect_available_servers();
    if servers.is_empty() {
        if !installed {
            println!("  (none detected on PATH)");
        }
    } else {
        for (language, server_name, path) in servers {
            if verbose {
                println!("  {} {} - {} ({})", "✓".green(), language, server_name, path);
            } else {
                println!("  {} {} - {}", "✓".green(), language, server_name);
            }
        }
    }
    println!();

    // Show Tree-sitter grammars
    println!("{}", "Tree-sitter Grammars".cyan().bold());

    // Static grammars
    let static_langs = StaticLanguageProvider::SUPPORTED_LANGUAGES;
    if !static_langs.is_empty() {
        println!("  {}", "Static (bundled):".dimmed());
        for lang in static_langs {
            println!("    {} {}", "✓".green(), lang);
        }
    }

    // Cached grammars
    let config = GrammarConfig::default();
    let manager = GrammarManager::new(config);

    match manager.cached_languages() {
        Ok(cached) if !cached.is_empty() => {
            println!("  {}", "Cached:".dimmed());
            for lang in &cached {
                let status = manager.grammar_status(lang);
                let status_icon = match status {
                    GrammarStatus::Loaded => "✓".green(),
                    GrammarStatus::Cached => "●".blue(),
                    GrammarStatus::NeedsDownload => "↓".yellow(),
                    GrammarStatus::NotAvailable => "✗".red(),
                    GrammarStatus::IncompatibleVersion => "!".yellow(),
                };
                print!("    {} {}", status_icon, lang);
                if verbose {
                    let info = manager.grammar_info(lang);
                    if let Some(meta) = info.metadata {
                        print!(" (v{}, ts {})", meta.grammar_version, meta.tree_sitter_version);
                    }
                }
                println!();
            }
        }
        Ok(_) if !installed => println!("  Cached: (none)"),
        _ => {}
    }

    if !installed {
        output::separator();
        output::info("Install components with:");
        output::info("  wqm language lsp-install <language>  # LSP server");
        output::info("  wqm language ts-install <language>   # Tree-sitter grammar");
    }

    Ok(())
}

/// Install a Tree-sitter grammar
async fn ts_install(language: &str, force: bool) -> Result<()> {
    use workspace_qdrant_core::config::GrammarConfig;
    use workspace_qdrant_core::tree_sitter::GrammarManager;

    output::section(format!("Installing Tree-sitter Grammar: {}", language));

    let mut config = GrammarConfig::default();
    config.auto_download = true;

    let mut manager = GrammarManager::new(config);

    // Check if already cached
    if !force && manager.cache_paths().grammar_exists(language) {
        output::warning("Grammar already cached. Use --force to reinstall.");
        return Ok(());
    }

    // If force, clear existing cache first
    if force {
        match manager.clear_cache(language) {
            Ok(true) => output::info("Cleared existing cache"),
            Ok(false) => {}
            Err(e) => output::warning(format!("Could not clear cache: {}", e)),
        }
    }

    // Attempt to download and load
    output::info("Downloading grammar...");
    match manager.get_grammar(language).await {
        Ok(_) => {
            output::success("Grammar installed successfully");

            // Verify compatibility
            let info = manager.grammar_info(language);
            if let Some(compat) = info.compatibility {
                if compat.is_compatible() {
                    output::success("Version compatible");
                } else {
                    output::warning("Version may have compatibility issues");
                }
            }
        }
        Err(e) => {
            return Err(anyhow!("Failed to install grammar: {}", e));
        }
    }

    Ok(())
}

/// Remove a Tree-sitter grammar
async fn ts_remove(language: &str) -> Result<()> {
    use workspace_qdrant_core::config::GrammarConfig;
    use workspace_qdrant_core::tree_sitter::GrammarManager;

    output::section(format!("Removing Tree-sitter Grammar: {}", language));

    let config = GrammarConfig::default();
    let manager = GrammarManager::new(config);

    if language == "all" {
        match manager.clear_all_cache() {
            Ok(count) => {
                output::success(format!("Removed {} grammar(s) from cache", count));
            }
            Err(e) => {
                return Err(anyhow!("Failed to clear cache: {}", e));
            }
        }
    } else {
        match manager.clear_cache(language) {
            Ok(true) => {
                output::success("Grammar removed from cache");
            }
            Ok(false) => {
                output::warning("Grammar not found in cache");
            }
            Err(e) => {
                return Err(anyhow!("Failed to remove grammar: {}", e));
            }
        }
    }

    Ok(())
}

/// Show installation guide for an LSP server
async fn lsp_install(language: &str) -> Result<()> {
    // Capture PATH on first install command (for LSP discovery)
    match config::capture_user_path() {
        Ok(true) => {
            output::success("Captured user PATH for LSP discovery");
            output::info("Daemon will use this PATH to find LSP servers.");
            output::separator();
        }
        Ok(false) => {} // PATH already stored
        Err(e) => {
            output::warning(format!("Could not capture PATH: {}", e));
        }
    }

    output::section(format!("Installing {} Language Server", language));

    match language.to_lowercase().as_str() {
        "rust" => {
            output::info("rust-analyzer - The official Rust language server");
            output::separator();
            output::info("Installation options:");
            output::info("  1. Via rustup (recommended):");
            output::info("     rustup component add rust-analyzer");
            output::info("");
            output::info("  2. Via package manager:");
            output::info("     macOS: brew install rust-analyzer");
            output::info("     Arch:  pacman -S rust-analyzer");
            output::info("");
            output::info("  3. Manual download:");
            output::info("     https://github.com/rust-lang/rust-analyzer/releases");
        }
        "python" => {
            output::info("Python language servers (in order of preference):");
            output::separator();
            output::info("  1. ruff-lsp (fast, recommended for linting):");
            output::info("     pip install ruff-lsp");
            output::info("     # or: uv tool install ruff");
            output::info("");
            output::info("  2. pylsp (python-lsp-server):");
            output::info("     pip install python-lsp-server");
            output::info("");
            output::info("  3. pyright (static type checker):");
            output::info("     npm install -g pyright");
            output::info("     # or: pip install pyright");
        }
        "typescript" | "javascript" | "ts" | "js" => {
            output::info("typescript-language-server - For TypeScript and JavaScript");
            output::separator();
            output::info("Installation:");
            output::info("  npm install -g typescript-language-server typescript");
            output::info("");
            output::info("Note: Requires Node.js to be installed");
        }
        "go" | "golang" => {
            output::info("gopls - The official Go language server");
            output::separator();
            output::info("Installation:");
            output::info("  go install golang.org/x/tools/gopls@latest");
            output::info("");
            output::info("Make sure $GOPATH/bin is in your PATH");
        }
        "java" => {
            output::info("Eclipse JDT Language Server (jdtls)");
            output::separator();
            output::info("Installation options:");
            output::info("  1. Via package manager:");
            output::info("     macOS: brew install jdtls");
            output::info("");
            output::info("  2. Manual download:");
            output::info("     https://download.eclipse.org/jdtls/");
            output::info("");
            output::info("Note: Requires JDK 17+ to be installed");
        }
        "c" | "cpp" | "c++" => {
            output::info("clangd - LLVM-based language server for C/C++");
            output::separator();
            output::info("Installation:");
            output::info("  macOS: brew install llvm");
            output::info("  Ubuntu: apt install clangd");
            output::info("  Arch: pacman -S clang");
            output::info("");
            output::info("Alternative: ccls");
            output::info("  brew install ccls");
        }
        _ => {
            output::warning(format!("Unknown language: {}", language));
            output::info("Supported languages: rust, python, typescript, javascript, go, java, c, cpp");
        }
    }

    Ok(())
}

/// Show removal guide for an LSP server
async fn lsp_remove(language: &str) -> Result<()> {
    output::section(format!("Remove {} Language Server", language));

    let lsp_info = get_lsp_server_info(language);

    if lsp_info.is_none() {
        output::warning(format!("Unknown language: {}", language));
        output::info("Supported languages: rust, python, typescript, javascript, go, java, c, cpp");
        return Ok(());
    }

    let (server_name, executables) = lsp_info.unwrap();
    let installed_path = executables.iter().find_map(|exe| which_cmd(exe));

    match installed_path {
        Some(path) => {
            output::kv("Server", server_name);
            output::kv("Path", &path);
            output::separator();
            output::info("LSP servers are typically managed by package managers.");
            output::info("To remove, use the appropriate package manager:");
            output::separator();

            match language.to_lowercase().as_str() {
                "rust" => {
                    output::info("  rustup component remove rust-analyzer");
                    output::info("  # or: brew uninstall rust-analyzer");
                }
                "python" => {
                    output::info("  pip uninstall ruff-lsp python-lsp-server pyright");
                    output::info("  # or: uv tool uninstall ruff");
                }
                "typescript" | "javascript" | "ts" | "js" => {
                    output::info("  npm uninstall -g typescript-language-server");
                }
                "go" | "golang" => {
                    output::info("  rm $(which gopls)");
                }
                "java" => {
                    output::info("  brew uninstall jdtls");
                }
                "c" | "cpp" | "c++" => {
                    output::info("  brew uninstall llvm  # for clangd");
                    output::info("  # or: apt remove clangd");
                }
                _ => {}
            }
        }
        None => {
            output::info(format!("{} language server is not installed", language));
        }
    }

    Ok(())
}

/// Show language support status
async fn language_status(language: Option<String>, verbose: bool) -> Result<()> {
    output::section("Language Support Status");

    if let Some(lang) = &language {
        output::kv("Language", lang);
    } else {
        output::info("Checking all languages...");
    }
    output::separator();

    // Check specific language or all
    let languages_to_check: Vec<&str> = match &language {
        Some(l) => vec![l.as_str()],
        None => vec!["rust", "python", "typescript", "go", "java", "c", "cpp"],
    };

    for lang in languages_to_check {
        println!("{}", format!("  {}", lang).cyan().bold());

        // Check LSP
        let lsp_info = get_lsp_server_info(lang);
        if let Some((server_name, executables)) = lsp_info {
            let lsp_path = executables.iter().find_map(|exe| which_cmd(exe));
            match lsp_path {
                Some(path) => {
                    print!("    LSP: {} {}", "✓".green(), server_name);
                    if verbose {
                        print!(" ({})", path);
                    }
                    println!();
                }
                None => {
                    println!("    LSP: {} not installed", "✗".red());
                }
            }
        } else {
            println!("    LSP: {} not supported", "?".yellow());
        }

        // Check Tree-sitter grammar
        use workspace_qdrant_core::config::GrammarConfig;
        use workspace_qdrant_core::tree_sitter::{GrammarManager, GrammarStatus};

        let config = GrammarConfig::default();
        let manager = GrammarManager::new(config);
        let grammar_status = manager.grammar_status(lang);

        match grammar_status {
            GrammarStatus::Loaded => {
                println!("    Grammar: {} loaded", "✓".green());
            }
            GrammarStatus::Cached => {
                println!("    Grammar: {} cached", "●".blue());
            }
            GrammarStatus::NeedsDownload => {
                println!("    Grammar: {} available (needs download)", "↓".yellow());
            }
            GrammarStatus::NotAvailable => {
                println!("    Grammar: {} not available", "✗".red());
            }
            GrammarStatus::IncompatibleVersion => {
                println!("    Grammar: {} incompatible version", "!".yellow());
            }
        }

        println!();
    }

    // Check daemon language components if connected
    match DaemonClient::connect_default().await {
        Ok(mut client) => {
            output::separator();
            output::info("Daemon Language Components:");
            match client.system().health(()).await {
                Ok(response) => {
                    let health = response.into_inner();
                    for comp in &health.components {
                        if comp.component_name.contains("lsp")
                            || comp.component_name.contains("grammar")
                            || comp.component_name.contains("language")
                        {
                            let status = ServiceStatus::from_proto(comp.status);
                            output::status_line(&format!("  {}", comp.component_name), status);
                            if !comp.message.is_empty() && verbose {
                                output::kv("    Message", &comp.message);
                            }
                        }
                    }
                }
                Err(e) => {
                    output::warning(format!("Could not get daemon status: {}", e));
                }
            }
        }
        Err(_) => {
            output::separator();
            output::warning("Daemon not running - some status info unavailable");
        }
    }

    Ok(())
}

// Helper functions

/// Get LSP server info for a language
fn get_lsp_server_info(language: &str) -> Option<(&'static str, Vec<&'static str>)> {
    match language.to_lowercase().as_str() {
        "rust" => Some(("rust-analyzer", vec!["rust-analyzer"])),
        "python" => Some((
            "ruff-lsp/pylsp/pyright",
            vec!["ruff-lsp", "ruff", "pylsp", "pyright", "pyright-langserver"],
        )),
        "typescript" | "javascript" | "ts" | "js" => Some((
            "typescript-language-server",
            vec!["typescript-language-server"],
        )),
        "go" | "golang" => Some(("gopls", vec!["gopls"])),
        "java" => Some(("jdtls", vec!["jdtls"])),
        "c" | "cpp" | "c++" => Some(("clangd/ccls", vec!["clangd", "ccls"])),
        _ => None,
    }
}

/// Detect available language servers on PATH
fn detect_available_servers() -> Vec<(String, String, String)> {
    let servers_to_check = vec![
        ("Rust", vec!["rust-analyzer"]),
        (
            "Python",
            vec!["ruff-lsp", "ruff", "pylsp", "pyright", "pyright-langserver"],
        ),
        ("TypeScript/JavaScript", vec!["typescript-language-server"]),
        ("Go", vec!["gopls"]),
        ("Java", vec!["jdtls"]),
        ("C/C++", vec!["clangd", "ccls"]),
    ];

    let mut found = Vec::new();

    for (language, executables) in servers_to_check {
        for exe in executables {
            if let Some(path) = which_cmd(exe) {
                found.push((language.to_string(), exe.to_string(), path));
                break; // Only report first found for each language
            }
        }
    }

    found
}

/// Find executable on PATH
fn which_cmd(name: &str) -> Option<String> {
    match Command::new("which").arg(name).output() {
        Ok(output) if output.status.success() => {
            let path = String::from_utf8_lossy(&output.stdout).trim().to_string();
            if !path.is_empty() {
                return Some(path);
            }
        }
        _ => {}
    }

    // Fallback: use which crate
    match which::which(name) {
        Ok(path) => Some(path.display().to_string()),
        Err(_) => None,
    }
}
