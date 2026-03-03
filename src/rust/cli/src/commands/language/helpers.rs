//! Shared helper functions for language command submodules

use std::process::Command;

/// Get LSP server info for a language.
/// Returns (server display name, list of executable names to probe).
pub fn get_lsp_server_info(language: &str) -> Option<(&'static str, Vec<&'static str>)> {
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
        "ruby" | "rb" => Some(("ruby-lsp/solargraph", vec!["ruby-lsp", "solargraph"])),
        "php" => Some(("phpactor/intelephense", vec!["phpactor", "intelephense"])),
        "shell" | "bash" | "sh" => Some(("bash-language-server", vec!["bash-language-server"])),
        "html" | "htm" => {
            Some(("vscode-html-languageserver", vec!["vscode-html-language-server"]))
        }
        _ => None,
    }
}

/// Detect available language servers on PATH.
/// Returns a list of (language label, executable name, resolved path).
pub fn detect_available_servers() -> Vec<(String, String, String)> {
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
        ("Ruby", vec!["ruby-lsp", "solargraph"]),
        ("PHP", vec!["phpactor", "intelephense"]),
        ("Shell", vec!["bash-language-server"]),
        ("HTML", vec!["vscode-html-language-server"]),
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

/// Find an executable on PATH using `which` then the `which` crate as fallback.
pub fn which_cmd(name: &str) -> Option<String> {
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
