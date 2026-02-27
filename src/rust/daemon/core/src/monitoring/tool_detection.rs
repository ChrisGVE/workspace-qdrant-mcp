//! Tool detection for tree-sitter parsers and LSP servers.
//!
//! Pure Rust tool detection with no external dependencies:
//! - Tree-sitter: Checks for `lib<language>.so/dylib/dll` in standard locations
//! - LSP servers: Scans PATH for `*-language-server` executables

use std::path::{Path, PathBuf};

/// Find tree-sitter parser for a language.
///
/// Searches standard locations for tree-sitter dynamic libraries.
pub fn find_tree_sitter_parser(language: &str) -> Option<String> {
    let search_paths = get_tree_sitter_search_paths();

    #[cfg(target_os = "macos")]
    let lib_patterns = vec![
        format!("libtree-sitter-{}.dylib", language),
        format!("tree-sitter-{}.dylib", language),
    ];

    #[cfg(target_os = "linux")]
    let lib_patterns = vec![
        format!("libtree-sitter-{}.so", language),
        format!("tree-sitter-{}.so", language),
    ];

    #[cfg(target_os = "windows")]
    let lib_patterns = vec![
        format!("tree-sitter-{}.dll", language),
        format!("libtree-sitter-{}.dll", language),
    ];

    for search_path in search_paths {
        for pattern in &lib_patterns {
            let lib_path = search_path.join(pattern);
            if lib_path.exists() && lib_path.is_file() {
                return Some(lib_path.to_string_lossy().to_string());
            }
        }
    }

    None
}

/// Get standard tree-sitter search paths.
pub fn get_tree_sitter_search_paths() -> Vec<PathBuf> {
    let mut paths = Vec::new();

    if let Ok(home) = std::env::var("HOME") {
        let home_path = PathBuf::from(&home);
        paths.push(home_path.join(".tree-sitter/bin"));
        paths.push(PathBuf::from(&home).join(".local/lib"));
    }

    #[cfg(unix)]
    {
        paths.push(PathBuf::from("/usr/local/lib"));
        paths.push(PathBuf::from("/usr/lib"));
        paths.push(PathBuf::from("/opt/homebrew/lib"));
    }

    if let Ok(cwd) = std::env::current_dir() {
        paths.push(cwd.join("lib"));
    }

    paths
}

/// Find LSP server for a language.
///
/// Searches PATH for language server executables.
pub fn find_lsp_server(language: &str) -> Option<String> {
    let server_names = get_lsp_server_names(language);

    if let Ok(path_var) = std::env::var("PATH") {
        for path_dir in path_var.split(':') {
            let path = Path::new(path_dir);
            if !path.exists() || !path.is_dir() {
                continue;
            }

            for server_name in &server_names {
                let executable = path.join(server_name);

                #[cfg(unix)]
                {
                    if executable.exists() && executable.is_file() {
                        use std::os::unix::fs::PermissionsExt;
                        if let Ok(metadata) = std::fs::metadata(&executable) {
                            let permissions = metadata.permissions();
                            if permissions.mode() & 0o111 != 0 {
                                return Some(executable.to_string_lossy().to_string());
                            }
                        }
                    }
                }

                #[cfg(not(unix))]
                {
                    if executable.exists() && executable.is_file() {
                        return Some(executable.to_string_lossy().to_string());
                    }
                }
            }
        }
    }

    None
}

/// Get LSP server name patterns for a language.
pub fn get_lsp_server_names(language: &str) -> Vec<String> {
    match language {
        "rust" => vec!["rust-analyzer".to_string(), "rls".to_string()],
        "python" => vec![
            "pylsp".to_string(),
            "pyls".to_string(),
            "pyright-langserver".to_string(),
        ],
        "javascript" | "typescript" => vec![
            "typescript-language-server".to_string(),
            "tsserver".to_string(),
        ],
        "go" => vec!["gopls".to_string()],
        "java" => vec!["jdtls".to_string(), "java-language-server".to_string()],
        "c" | "cpp" => vec!["clangd".to_string(), "ccls".to_string()],
        "ruby" => vec!["solargraph".to_string()],
        "php" => vec!["phpactor".to_string(), "intelephense".to_string()],
        _ => vec![
            format!("{}-language-server", language),
            format!("{}-lsp", language),
            format!("{}ls", language),
        ],
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_get_lsp_server_names() {
        let names = get_lsp_server_names("rust");
        assert!(names.contains(&"rust-analyzer".to_string()));

        let names = get_lsp_server_names("python");
        assert!(names.contains(&"pylsp".to_string()));

        let names = get_lsp_server_names("unknown");
        assert!(names.contains(&"unknown-language-server".to_string()));
    }

    #[test]
    fn test_get_tree_sitter_search_paths() {
        let paths = get_tree_sitter_search_paths();
        assert!(!paths.is_empty());

        if std::env::var("HOME").is_ok() {
            assert!(paths
                .iter()
                .any(|p| p.to_string_lossy().contains(".tree-sitter")));
        }
    }
}
