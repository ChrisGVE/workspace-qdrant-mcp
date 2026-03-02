//! Language diagnostics subcommand
//!
//! Checks LSP server availability, tree-sitter grammar presence,
//! daemon language support status, and file extension mappings.

use anyhow::Result;
use std::process::Command;

use crate::grpc::client::DaemonClient;
use crate::output::{self, ServiceStatus};

/// Diagnose language support issues.
pub async fn diagnose_language(language: &str, verbose: bool) -> Result<()> {
    output::section(format!("Language Diagnostics: {}", language));

    output::info("Running diagnostic checks...");
    output::separator();

    // 1. Check LSP server
    let lsp_found = check_lsp_server(language, verbose);

    output::separator();

    // 2. Check Tree-sitter grammar
    let grammar_found = check_tree_sitter_grammar(language, verbose);

    output::separator();

    // 3. Check daemon language support status
    check_daemon_language_support(language, verbose).await;

    output::separator();

    // 4. File extension mapping
    show_extension_mapping(language);

    output::separator();
    show_diagnostic_summary(language, lsp_found, grammar_found);

    Ok(())
}

/// Check for LSP server binaries on PATH.
fn check_lsp_server(language: &str, verbose: bool) -> bool {
    output::info("1. LSP Server Check");

    let lang_lower = language.to_lowercase();
    let fallback_lsp = format!("{}-language-server", lang_lower);
    let lsp_binaries: Vec<&str> = match lang_lower.as_str() {
        "python" => vec!["pyright", "pylsp", "pyright-langserver"],
        "javascript" | "typescript" | "js" | "ts" => {
            vec!["typescript-language-server", "tsserver"]
        }
        "rust" => vec!["rust-analyzer"],
        "go" | "golang" => vec!["gopls"],
        "java" => vec!["jdtls", "java-language-server"],
        "c" | "cpp" | "c++" => vec!["clangd", "ccls"],
        "ruby" => vec!["solargraph", "ruby-lsp"],
        "php" => vec!["phpactor", "psalm-language-server", "intelephense"],
        _ => vec![fallback_lsp.as_str()],
    };

    let mut lsp_found = false;
    for binary in &lsp_binaries {
        match which::which(binary) {
            Ok(path) => {
                output::success(format!("  Found: {} at {}", binary, path.display()));
                lsp_found = true;

                if verbose {
                    // Try to get version
                    if let Ok(ver_output) = Command::new(binary).arg("--version").output() {
                        if ver_output.status.success() {
                            let version = String::from_utf8_lossy(&ver_output.stdout);
                            output::kv("    Version", version.trim());
                        }
                    }
                }
                break;
            }
            Err(_) => {
                if verbose {
                    output::info(format!("  Not found: {}", binary));
                }
            }
        }
    }

    if !lsp_found {
        output::warning(format!("  No LSP server found for {}", language));
        show_lsp_install_suggestions(language);
    }

    lsp_found
}

/// Print LSP install suggestions for known languages.
fn show_lsp_install_suggestions(language: &str) {
    output::info("  Install suggestions:");
    match language.to_lowercase().as_str() {
        "python" => output::info("    npm install -g pyright"),
        "javascript" | "typescript" => {
            output::info("    npm install -g typescript-language-server typescript");
        }
        "rust" => output::info("    rustup component add rust-analyzer"),
        "go" => output::info("    go install golang.org/x/tools/gopls@latest"),
        "java" => output::info("    brew install jdtls"),
        "ruby" => output::info("    gem install ruby-lsp"),
        "php" => output::info("    composer global require phpactor/phpactor"),
        "shell" | "bash" => output::info("    npm install -g bash-language-server"),
        "html" => output::info("    npm install -g vscode-langservers-extracted"),
        _ => output::info(&format!(
            "    Search for {}-language-server",
            language
        )),
    }
}

/// Check tree-sitter grammar availability in standard paths.
fn check_tree_sitter_grammar(language: &str, _verbose: bool) -> bool {
    output::info("2. Tree-sitter Grammar Check");

    let grammar_paths = vec![
        dirs::data_local_dir()
            .map(|d| d.join("tree-sitter/lib"))
            .unwrap_or_default(),
        dirs::home_dir()
            .map(|h| h.join(".local/share/tree-sitter/lib"))
            .unwrap_or_default(),
        std::path::PathBuf::from("/usr/local/lib/tree-sitter"),
    ];

    let lang_lower = language.to_lowercase();
    let grammar_name = format!("{}.so", lang_lower);
    let alt_grammar_name = format!("tree-sitter-{}.so", lang_lower);
    let dylib_name = format!("{}.dylib", lang_lower);
    let alt_dylib_name = format!("tree-sitter-{}.dylib", lang_lower);

    let mut grammar_found = false;
    for base_path in &grammar_paths {
        if !base_path.exists() {
            continue;
        }

        for name in [&grammar_name, &alt_grammar_name, &dylib_name, &alt_dylib_name] {
            let grammar_path = base_path.join(name);
            if grammar_path.exists() {
                output::success(format!("  Found: {}", grammar_path.display()));
                grammar_found = true;
                break;
            }
        }
        if grammar_found {
            break;
        }
    }

    if !grammar_found {
        output::warning(format!("  No tree-sitter grammar found for {}", language));
        output::info("  Grammar search paths:");
        for path in &grammar_paths {
            output::info(&format!("    - {}", path.display()));
        }
        output::info("  Install with: wqm language install <language>");
    }

    grammar_found
}

/// Check daemon language support status via gRPC health endpoint.
async fn check_daemon_language_support(language: &str, verbose: bool) {
    output::info("3. Daemon Language Support");

    let lang_lower = language.to_lowercase();

    match DaemonClient::connect_default().await {
        Ok(mut client) => {
            match client.system().health(()).await {
                Ok(response) => {
                    let health = response.into_inner();

                    for comp in &health.components {
                        if comp.component_name.to_lowercase().contains(&lang_lower)
                            || comp.component_name.contains("lsp")
                            || comp.component_name.contains("grammar")
                        {
                            let status = ServiceStatus::from_proto(comp.status);
                            output::status_line(&format!("  {}", comp.component_name), status);
                            if !comp.message.is_empty() && verbose {
                                output::kv("    Details", &comp.message);
                            }
                        }
                    }
                }
                Err(e) => {
                    output::warning(format!("  Could not get daemon status: {}", e));
                }
            }
        }
        Err(_) => {
            output::warning("  Daemon not running - cannot check language support status");
        }
    }
}

/// Show file extension mapping for the language.
fn show_extension_mapping(language: &str) {
    output::info("4. File Extension Mapping");

    let extensions = match language.to_lowercase().as_str() {
        "python" => vec![".py", ".pyi", ".pyw"],
        "javascript" => vec![".js", ".mjs", ".cjs"],
        "typescript" => vec![".ts", ".mts", ".cts", ".tsx"],
        "rust" => vec![".rs"],
        "go" => vec![".go"],
        "java" => vec![".java"],
        "c" => vec![".c", ".h"],
        "cpp" | "c++" => vec![".cpp", ".hpp", ".cc", ".hh", ".cxx", ".hxx"],
        "ruby" => vec![".rb", ".rake", ".gemspec"],
        "php" => vec![".php", ".phtml"],
        "shell" | "bash" | "sh" => vec![".sh", ".bash"],
        "html" | "htm" => vec![".html", ".htm"],
        _ => vec![],
    };

    if !extensions.is_empty() {
        output::kv("  Extensions", &extensions.join(", "));
    } else {
        output::info(&format!(
            "  Unknown file extensions for {}",
            language
        ));
    }
}

/// Show diagnostic summary based on what was found.
fn show_diagnostic_summary(language: &str, lsp_found: bool, grammar_found: bool) {
    output::info("Diagnostic Summary:");

    if lsp_found && grammar_found {
        output::success(format!(
            "  {} support appears fully configured",
            language
        ));
    } else if lsp_found {
        output::warning(format!(
            "  {} has LSP but missing grammar",
            language
        ));
    } else if grammar_found {
        output::warning(format!(
            "  {} has grammar but missing LSP",
            language
        ));
    } else {
        output::error(format!(
            "  {} support not configured",
            language
        ));
    }
}
