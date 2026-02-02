//! LSP management commands
//!
//! Commands for managing Language Server Protocol (LSP) integration.
//! Subcommands: list, status, restart, install, remove, check, diagnose, path

use anyhow::Result;
use clap::{Args, Subcommand};
use std::process::Command;

use crate::config;
use crate::output;

/// LSP command arguments
#[derive(Args)]
pub struct LspArgs {
    #[command(subcommand)]
    command: LspCommand,
}

/// LSP subcommands
#[derive(Subcommand)]
enum LspCommand {
    /// List detected available language servers on PATH
    List,

    /// Show active LSP servers for a project
    Status {
        /// Project ID (12-char hex identifier)
        project_id: String,
    },

    /// Restart LSP server(s) for a project
    Restart {
        /// Project ID
        project_id: String,

        /// Language to restart (restart all if not specified)
        language: Option<String>,
    },

    /// Show installation guide for a language server
    Install {
        /// Language (rust, python, typescript, go, java, c, cpp)
        language: String,
    },

    /// Remove an LSP server installation
    Remove {
        /// Language to remove (rust, python, typescript, go, java, c, cpp)
        language: String,

        /// Force removal without confirmation
        #[arg(short, long)]
        force: bool,
    },

    /// Check which LSP servers are available on PATH
    Check,

    /// Diagnose language support issues
    Diagnose {
        /// Language to diagnose
        language: String,
    },

    /// Show or update stored PATH for LSP discovery
    Path {
        /// Update stored PATH with current environment PATH
        #[arg(short, long)]
        update: bool,

        /// Show current system PATH (not stored PATH)
        #[arg(short, long)]
        current: bool,
    },
}

/// Execute LSP command
pub async fn execute(args: LspArgs) -> Result<()> {
    match args.command {
        LspCommand::List => list().await,
        LspCommand::Status { project_id } => status(&project_id).await,
        LspCommand::Restart { project_id, language } => restart(&project_id, language.as_deref()).await,
        LspCommand::Install { language } => install(&language).await,
        LspCommand::Remove { language, force } => remove(&language, force).await,
        LspCommand::Check => check().await,
        LspCommand::Diagnose { language } => diagnose(&language).await,
        LspCommand::Path { update, current } => path(update, current).await,
    }
}

/// List detected available language servers
async fn list() -> Result<()> {
    output::section("Available Language Servers");

    let servers = detect_available_servers();

    if servers.is_empty() {
        output::warning("No language servers detected on PATH");
        output::info("Use 'wqm lsp install <language>' for installation instructions");
    } else {
        for (language, server_name, path) in servers {
            output::kv(&language, &format!("{} ({})", server_name, path));
        }
    }

    Ok(())
}

/// Show active LSP servers for a project (requires daemon)
async fn status(project_id: &str) -> Result<()> {
    output::section(format!("LSP Status for Project: {}", project_id));

    // TODO: Implement when LSP status gRPC is added to ProjectService
    output::info("LSP status query requires daemon connection.");
    output::info("This feature will show:");
    output::info("  - Active language servers for the project");
    output::info("  - Server health status and restart count");
    output::info("  - Last activity timestamp");
    output::separator();
    output::warning("Not yet implemented - requires LSP gRPC extension");

    Ok(())
}

/// Restart LSP server(s) for a project (requires daemon)
async fn restart(project_id: &str, language: Option<&str>) -> Result<()> {
    match language {
        Some(lang) => {
            output::section(format!("Restarting {} LSP for: {}", lang, project_id));
        }
        None => {
            output::section(format!("Restarting all LSP servers for: {}", project_id));
        }
    }

    // TODO: Implement when LSP restart gRPC is added to ProjectService
    output::warning("Not yet implemented - requires LSP gRPC extension");
    output::info("When implemented, this will:");
    output::info("  - Stop the specified language server(s)");
    output::info("  - Start fresh server instance(s)");
    output::info("  - Reset restart count for the server(s)");

    Ok(())
}

/// Show installation guide for a language server
async fn install(language: &str) -> Result<()> {
    // Capture PATH on first install command (for LSP discovery)
    match config::capture_user_path() {
        Ok(true) => {
            output::success("Captured user PATH for LSP discovery");
            output::info("Daemon will use this PATH to find LSP servers.");
            output::separator();
        }
        Ok(false) => {
            // PATH already stored, no action needed
        }
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

/// Check which LSP servers are available on PATH
async fn check() -> Result<()> {
    output::section("LSP Server Check");

    let checks = vec![
        ("Rust", "rust-analyzer", vec!["rust-analyzer"]),
        ("Python", "ruff-lsp", vec!["ruff-lsp", "ruff"]),
        ("Python", "pylsp", vec!["pylsp"]),
        ("Python", "pyright", vec!["pyright", "pyright-langserver"]),
        ("TypeScript/JS", "typescript-language-server", vec!["typescript-language-server"]),
        ("Go", "gopls", vec!["gopls"]),
        ("Java", "jdtls", vec!["jdtls"]),
        ("C/C++", "clangd", vec!["clangd"]),
        ("C/C++", "ccls", vec!["ccls"]),
    ];

    let mut found_any = false;

    for (language, server_name, executables) in checks {
        let found = executables.iter().any(|exe| which_cmd(exe).is_some());

        if found {
            output::success(format!("{}: {} found", language, server_name));
            found_any = true;
        } else {
            output::status_line(
                &format!("{}: {}", language, server_name),
                crate::output::ServiceStatus::Unhealthy,
            );
        }
    }

    if !found_any {
        output::separator();
        output::warning("No language servers found on PATH");
        output::info("Use 'wqm lsp install <language>' for installation instructions");
    }

    Ok(())
}

/// Remove an LSP server installation
async fn remove(language: &str, force: bool) -> Result<()> {
    output::section(format!("Remove {} Language Server", language));

    // Map language to LSP server info
    let lsp_info = get_lsp_server_info(language);

    if lsp_info.is_none() {
        output::warning(format!("Unknown language: {}", language));
        output::info("Supported languages: rust, python, typescript, javascript, go, java, c, cpp");
        return Ok(());
    }

    let (server_name, executables) = lsp_info.unwrap();

    // Check if installed
    let installed_path = executables.iter().find_map(|exe| which_cmd(exe));

    match installed_path {
        Some(path) => {
            output::kv("Server", server_name);
            output::kv("Path", &path);
            output::separator();

            if !force {
                output::warning("LSP servers are typically managed by package managers.");
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

                output::separator();
                output::info("Use --force to skip this message in scripts.");
            } else {
                output::info("Use package manager to uninstall:");
                output::info(&format!("  Server at: {}", path));
            }
        }
        None => {
            output::info(format!("{} language server is not installed", language));
        }
    }

    Ok(())
}

/// Diagnose language support issues
async fn diagnose(language: &str) -> Result<()> {
    output::section(format!("Diagnose: {}", language));

    output::info("Checking language support configuration...");
    output::separator();

    // Get LSP info
    let lsp_info = get_lsp_server_info(language);

    if lsp_info.is_none() {
        output::warning(format!("Unknown language: {}", language));
        output::info("Supported languages: rust, python, typescript, javascript, go, java, c, cpp");
        return Ok(());
    }

    let (server_name, executables) = lsp_info.unwrap();

    // Check 1: LSP binary
    output::info(&format!("1. LSP server ({}):", server_name));
    let installed_path = executables.iter().find_map(|exe| which_cmd(exe));

    match &installed_path {
        Some(path) => {
            output::success(format!("   ✓ Found at: {}", path));
        }
        None => {
            output::error(format!("   ✗ Not found (checked: {})", executables.join(", ")));
            output::info(&format!("   → Run: wqm lsp install {}", language));
        }
    }

    // Check 2: Try to invoke LSP
    if let Some(path) = &installed_path {
        output::info("2. Server invocation test:");
        match Command::new(path).arg("--version").output() {
            Ok(output) if output.status.success() => {
                let version = String::from_utf8_lossy(&output.stdout).trim().to_string();
                output::success(format!("   ✓ Version: {}", version.lines().next().unwrap_or("unknown")));
            }
            Ok(output) => {
                // Some LSPs don't support --version, try --help
                if !output.stderr.is_empty() {
                    output::info("   ○ Server responds (no version info available)");
                } else {
                    output::warning("   ? Could not determine version");
                }
            }
            Err(e) => {
                output::error(format!("   ✗ Failed to run: {}", e));
            }
        }
    }

    // Check 3: Grammar support
    output::info(&format!("3. Tree-sitter grammar for {}:", language));
    output::info("   → Grammar support is built into the daemon");
    output::info("   → Use 'wqm grammar list' to check available grammars");

    output::separator();
    output::info("Additional diagnostics:");
    output::info("  wqm lsp check      - Check all available LSP servers");
    output::info("  wqm grammar list   - List available Tree-sitter grammars");
    output::info("  wqm admin health   - Check daemon health");

    Ok(())
}

/// Show or update stored PATH for LSP discovery
async fn path(update: bool, current: bool) -> Result<()> {
    output::section("LSP PATH Configuration");

    if update {
        // Update stored PATH with current environment PATH
        let current_path = config::get_current_path();
        output::kv("Current PATH", &truncate_path(&current_path, 100));
        output::separator();

        match config::write_user_path(&current_path) {
            Ok(()) => {
                output::success("Stored PATH updated");
                output::info("Daemon will use this PATH to find LSP servers on next restart.");
            }
            Err(e) => {
                output::error(format!("Failed to update PATH: {}", e));
            }
        }
    } else if current {
        // Show current system PATH
        let current_path = config::get_current_path();
        output::kv("Current System PATH", "");
        output::separator();
        for (i, segment) in current_path.split(':').enumerate() {
            if !segment.is_empty() {
                output::info(&format!("  {}. {}", i + 1, segment));
            }
        }
    } else {
        // Show stored PATH
        match config::read_user_path() {
            Some(stored_path) => {
                output::kv("Stored PATH", "");
                output::separator();
                for (i, segment) in stored_path.split(':').enumerate() {
                    if !segment.is_empty() {
                        output::info(&format!("  {}. {}", i + 1, segment));
                    }
                }
                output::separator();
                output::info("This PATH is used by the daemon to find LSP servers.");
                output::info("Use 'wqm lsp path --update' to refresh from current environment.");
            }
            None => {
                output::warning("No PATH stored yet");
                output::info("PATH will be captured automatically on first 'wqm lsp install' command.");
                output::info("Or use 'wqm lsp path --update' to store current PATH now.");
            }
        }
    }

    Ok(())
}

/// Truncate PATH for display
fn truncate_path(path: &str, max_len: usize) -> String {
    if path.len() <= max_len {
        path.to_string()
    } else {
        format!("{}...", &path[..max_len])
    }
}

/// Get LSP server info for a language
fn get_lsp_server_info(language: &str) -> Option<(&'static str, Vec<&'static str>)> {
    match language.to_lowercase().as_str() {
        "rust" => Some(("rust-analyzer", vec!["rust-analyzer"])),
        "python" => Some(("ruff-lsp/pylsp/pyright", vec!["ruff-lsp", "ruff", "pylsp", "pyright", "pyright-langserver"])),
        "typescript" | "javascript" | "ts" | "js" => Some(("typescript-language-server", vec!["typescript-language-server"])),
        "go" | "golang" => Some(("gopls", vec!["gopls"])),
        "java" => Some(("jdtls", vec!["jdtls"])),
        "c" | "cpp" | "c++" => Some(("clangd/ccls", vec!["clangd", "ccls"])),
        _ => None,
    }
}

/// Detect available language servers
fn detect_available_servers() -> Vec<(String, String, String)> {
    let servers_to_check = vec![
        ("Rust", vec!["rust-analyzer"]),
        ("Python", vec!["ruff-lsp", "ruff", "pylsp", "pyright", "pyright-langserver"]),
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
    // Try the `which` command
    match Command::new("which").arg(name).output() {
        Ok(output) if output.status.success() => {
            let path = String::from_utf8_lossy(&output.stdout).trim().to_string();
            if !path.is_empty() {
                return Some(path);
            }
        }
        _ => {}
    }

    // Fallback: try which crate-style lookup
    match which::which(name) {
        Ok(path) => Some(path.display().to_string()),
        Err(_) => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_which_cmd_finds_common_tools() {
        // Test with common tools that should exist on most systems
        // At least one of these should exist
        let common_tools = ["ls", "cat", "echo"];
        let found = common_tools.iter().any(|tool| which_cmd(tool).is_some());
        assert!(found, "Expected at least one common tool to be found");
    }

    #[test]
    fn test_which_cmd_returns_none_for_nonexistent() {
        let result = which_cmd("this-command-definitely-does-not-exist-12345");
        assert!(result.is_none());
    }

    #[test]
    fn test_detect_available_servers_returns_vec() {
        // This test just verifies the function runs without panic
        let servers = detect_available_servers();
        // Result depends on what's installed, just verify it returns without error
        // The function should always return a valid Vec (possibly empty)
        let _ = servers; // Silence unused warning
    }
}
