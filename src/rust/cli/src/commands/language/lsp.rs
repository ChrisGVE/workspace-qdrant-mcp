//! `language lsp-install` and `language lsp-remove` subcommands

use anyhow::Result;

use crate::config;
use crate::output;

use super::helpers::{get_lsp_server_info, which_cmd};

/// Show installation guide for an LSP server.
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

    output::section(format!("Installing {} Language Server", language));
    print_install_instructions(language);
    Ok(())
}

/// Show removal guide for an LSP server.
pub async fn lsp_remove(language: &str) -> Result<()> {
    output::section(format!("Remove {} Language Server", language));

    let lsp_info = get_lsp_server_info(language);

    if lsp_info.is_none() {
        output::warning(format!("No known LSP server for: {}", language));
        output::info(
            "Known languages: rust, python, typescript, javascript, go, java, c, cpp, ruby, php, shell, html",
        );
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
            print_remove_instructions(language);
        }
        None => {
            output::info(format!("{} language server is not installed", language));
        }
    }

    Ok(())
}

// ─── Helpers ─────────────────────────────────────────────────────────────────

fn print_install_instructions(language: &str) {
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
        "ruby" | "rb" => {
            output::info("Ruby language servers:");
            output::separator();
            output::info("  1. ruby-lsp (recommended, by Shopify):");
            output::info("     gem install ruby-lsp");
            output::info("");
            output::info("  2. solargraph:");
            output::info("     gem install solargraph");
        }
        "php" => {
            output::info("PHP language servers:");
            output::separator();
            output::info("  1. phpactor:");
            output::info("     composer global require phpactor/phpactor");
            output::info("");
            output::info("  2. intelephense:");
            output::info("     npm install -g intelephense");
        }
        "shell" | "bash" | "sh" => {
            output::info("bash-language-server - For Bash/Shell scripts");
            output::separator();
            output::info("Installation:");
            output::info("  npm install -g bash-language-server");
        }
        "html" | "htm" => {
            output::info("vscode-html-languageserver - For HTML");
            output::separator();
            output::info("Installation:");
            output::info("  npm install -g vscode-langservers-extracted");
        }
        _ => {
            output::warning(format!("No known LSP server for: {}", language));
            output::info(
                "Known languages: rust, python, typescript, javascript, go, java, c, cpp, ruby, php, shell, html",
            );
            output::info("");
            output::info("If your language has an LSP server, install it manually and");
            output::info("ensure the binary is on your PATH. The daemon will detect it");
            output::info("automatically via PATH scanning.");
        }
    }
}

fn print_remove_instructions(language: &str) {
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
        "ruby" | "rb" => {
            output::info("  gem uninstall ruby-lsp solargraph");
        }
        "php" => {
            output::info("  composer global remove phpactor/phpactor");
            output::info("  # or: npm uninstall -g intelephense");
        }
        "shell" | "bash" | "sh" => {
            output::info("  npm uninstall -g bash-language-server");
        }
        "html" | "htm" => {
            output::info("  npm uninstall -g vscode-langservers-extracted");
        }
        _ => {}
    }
}
