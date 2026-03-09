//! `language lsp-install`, `language lsp-remove`, and `language lsp-search` subcommands

use anyhow::Result;
use colored::Colorize;

use crate::config;
use crate::output;

use super::helpers::{find_language, load_definitions, which_cmd, which_cmd_detailed};

/// Smart LSP install: checks existing installations, filters by available managers.
pub async fn lsp_install(language: &str) -> Result<()> {
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

    let def = match find_language(language) {
        Some(d) => d,
        None => {
            output::warning(format!("Unknown language: {language}"));
            output::info("Use 'wqm language list' to see all known languages.");
            output::info("");
            output::info("If your language has an LSP server, install it manually and");
            output::info("ensure the binary is on your PATH. The daemon will detect it");
            output::info("automatically.");
            return Ok(());
        }
    };

    output::section(format!("LSP Server: {}", def.language));

    if def.lsp_servers.is_empty() {
        output::warning(format!("No known LSP server for: {}", language));
        output::info("Use 'wqm language list' to see languages with LSP support.");
        return Ok(());
    }

    // Check if any server is already installed
    for server in &def.lsp_servers {
        if let Some(result) = which_cmd_detailed(&server.binary) {
            output::success(format!("{} is already installed", server.name));
            output::kv("  Binary", &server.binary);
            output::kv("  Path", &result.path.display().to_string());
            output::kv("  Source", &result.source.to_string());
            return Ok(());
        }
    }

    // Not installed — show install options filtered by available package managers
    output::info("No LSP server found. Install options:");
    output::separator();

    let mut option_num = 0;

    for server in &def.lsp_servers {
        if server.install_methods.is_empty() {
            continue;
        }

        let available_methods: Vec<_> = server
            .install_methods
            .iter()
            .filter(|m| which_cmd(&m.manager).is_some())
            .collect();

        let unavailable_methods: Vec<_> = server
            .install_methods
            .iter()
            .filter(|m| which_cmd(&m.manager).is_none())
            .collect();

        if !available_methods.is_empty() {
            println!("  {} ({})", server.name.bold(), server.binary);
            for method in &available_methods {
                option_num += 1;
                println!("    {}. {} {}", option_num, "▸".green(), method.command);
            }
            if !unavailable_methods.is_empty() {
                for method in &unavailable_methods {
                    println!(
                        "       {} {} (requires {})",
                        "▸".dimmed(),
                        method.command.dimmed(),
                        method.manager.dimmed()
                    );
                }
            }
            println!();
        } else {
            // No available managers — show all methods dimmed
            println!("  {} ({})", server.name.bold(), server.binary);
            for method in &server.install_methods {
                println!(
                    "    {} {} (requires {})",
                    "▸".dimmed(),
                    method.command.dimmed(),
                    method.manager
                );
            }
            println!();
        }
    }

    if option_num == 0 {
        output::info("No package managers found for the available install methods.");
        output::info("Install the LSP server manually and ensure it's on your PATH.");
    }

    Ok(())
}

/// Show removal guide for an LSP server.
pub async fn lsp_remove(language: &str) -> Result<()> {
    output::section(format!("Remove {} Language Server", language));

    let def = match find_language(language) {
        Some(d) if !d.lsp_servers.is_empty() => d,
        _ => {
            output::warning(format!("No known LSP server for: {}", language));
            output::info("Use 'wqm language list' to see languages with LSP support.");
            return Ok(());
        }
    };

    // Find installed server
    let installed = def
        .lsp_servers
        .iter()
        .find_map(|s| which_cmd_detailed(&s.binary).map(|r| (s, r)));

    match installed {
        Some((server, result)) => {
            output::kv("Server", &server.name);
            output::kv("Binary", &server.binary);
            output::kv("Path", &result.path.display().to_string());
            output::kv("Source", &result.source.to_string());
            output::separator();
            output::info("To remove, use the appropriate package manager:");
            output::separator();

            for method in &server.install_methods {
                let available = which_cmd(&method.manager).is_some();
                let marker = if available {
                    "▸".green().to_string()
                } else {
                    "▸".dimmed().to_string()
                };
                // Infer uninstall from install command
                let uninstall = infer_uninstall(&method.command);
                println!("  {} {}", marker, uninstall);
            }
        }
        None => {
            output::info(format!("{} language server is not installed", def.language));
        }
    }

    Ok(())
}

/// List all LSP servers across all registered languages.
pub async fn lsp_list(show_all: bool) -> Result<()> {
    output::section("LSP Servers");

    let defs = load_definitions();

    println!(
        "  {:<16} {:<25} {:<14} {:<10} {}",
        "Language".bold(),
        "Server".bold(),
        "Binary".bold(),
        "Source".bold(),
        "Status".bold()
    );
    println!("  {}", "─".repeat(75).dimmed());

    let mut shown = 0;
    let mut detected_count = 0;

    for def in &defs {
        if def.lsp_servers.is_empty() {
            continue;
        }

        for server in &def.lsp_servers {
            let result = which_cmd_detailed(&server.binary);
            let is_detected = result.is_some();

            if !show_all && !is_detected {
                continue;
            }

            let (source, status) = match &result {
                Some(r) => (
                    r.source.to_string(),
                    format!("{} {}", "✓".green(), r.path.display()),
                ),
                None => ("—".to_string(), "not found".dimmed().to_string()),
            };

            println!(
                "  {:<16} {:<25} {:<14} {:<10} {}",
                def.language, server.name, server.binary, source, status
            );

            shown += 1;
            if is_detected {
                detected_count += 1;
            }
        }
    }

    println!();
    if show_all {
        output::kv("Detected", format!("{detected_count}/{shown}"));
    } else {
        output::kv("Detected", format!("{detected_count}"));
        output::info("Use --all to show servers not yet installed.");
    }

    Ok(())
}

/// Search available LSP servers for a language.
pub async fn lsp_search(language: &str) -> Result<()> {
    output::section(format!("LSP Server Search: {}", language));

    let def = match find_language(language) {
        Some(d) => d,
        None => {
            output::warning(format!("Unknown language: {language}"));
            output::info("Use 'wqm language list' to see all known languages.");
            return Ok(());
        }
    };

    output::kv("Language", &def.language);
    output::separator();

    if def.lsp_servers.is_empty() {
        output::info("No LSP servers configured in registry for this language.");
        return Ok(());
    }

    println!(
        "  {:<25} {:<22} {:<10} {}",
        "Server".bold(),
        "Binary".bold(),
        "Priority".bold(),
        "Status".bold()
    );
    println!("  {}", "─".repeat(70).dimmed());

    for server in &def.lsp_servers {
        let detected = which_cmd_detailed(&server.binary);
        let status = match &detected {
            Some(r) => format!("{} {} ({})", "Found".green(), r.path.display(), r.source),
            None => "Not found".red().to_string(),
        };

        println!(
            "  {:<25} {:<22} {:<10} {}",
            server.name, server.binary, server.priority, status
        );

        // Show install methods for missing servers
        if detected.is_none() && !server.install_methods.is_empty() {
            for method in &server.install_methods {
                let available = which_cmd(&method.manager).is_some();
                let marker = if available {
                    "▸".green().to_string()
                } else {
                    "▸".dimmed().to_string()
                };
                let avail_note = if available { " (available)" } else { "" };
                println!(
                    "    {} {}: {}{}",
                    marker, method.manager, method.command, avail_note
                );
            }
        }
    }

    println!();
    output::info(format!("Install with: wqm language lsp-install {language}"));

    Ok(())
}

// ─── Helpers ─────────────────────────────────────────────────────────────────

/// Infer an uninstall command from an install command.
fn infer_uninstall(install_cmd: &str) -> String {
    let cmd = install_cmd
        .replace("pip install", "pip uninstall")
        .replace("npm install -g", "npm uninstall -g")
        .replace("brew install", "brew uninstall")
        .replace("cargo install", "cargo uninstall")
        .replace("gem install", "gem uninstall")
        .replace("go install", "# remove binary from $GOPATH/bin:")
        .replace("composer global require", "composer global remove")
        .replace("rustup component add", "rustup component remove")
        .replace("apt install", "apt remove")
        .replace("pacman -S", "pacman -R")
        .replace("dnf install", "dnf remove");

    if cmd == install_cmd {
        // No substitution matched
        format!("# reverse of: {install_cmd}")
    } else {
        cmd
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_infer_uninstall_pip() {
        assert_eq!(infer_uninstall("pip install ruff"), "pip uninstall ruff");
    }

    #[test]
    fn test_infer_uninstall_npm() {
        assert_eq!(
            infer_uninstall("npm install -g typescript-language-server"),
            "npm uninstall -g typescript-language-server"
        );
    }

    #[test]
    fn test_infer_uninstall_brew() {
        assert_eq!(
            infer_uninstall("brew install rust-analyzer"),
            "brew uninstall rust-analyzer"
        );
    }

    #[test]
    fn test_infer_uninstall_unknown() {
        let result = infer_uninstall("some-custom-installer foo");
        assert!(result.starts_with("# reverse of:"));
    }
}
