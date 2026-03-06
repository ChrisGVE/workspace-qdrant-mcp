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
        "ada" => vec!["ada_language_server"],
        "bash" | "shell" | "sh" => vec!["bash-language-server"],
        "c" | "cpp" | "c_sharp" => vec!["clangd", "ccls"],
        "clojure" => vec!["clojure-lsp"],
        "css" => vec!["css-languageserver", "vscode-css-language-server"],
        "dart" => vec!["dart", "dart language-server"],
        "elixir" => vec!["elixir-ls", "nextls", "lexical"],
        "elm" => vec!["elm-language-server"],
        "erlang" => vec!["elp", "erlang_ls"],
        "fortran" => vec!["fortls"],
        "go" => vec!["gopls"],
        "haskell" => vec![
            "haskell-language-server-wrapper",
            "haskell-language-server",
            "hls",
        ],
        "html" => vec!["html-languageserver", "vscode-html-language-server"],
        "java" => vec!["jdtls", "java-language-server"],
        "javascript" | "typescript" | "tsx" | "jsx" => {
            vec!["typescript-language-server", "tsserver", "deno"]
        }
        "json" => vec!["vscode-json-language-server"],
        "julia" => vec!["julia"],
        "kotlin" => vec!["kotlin-language-server"],
        "latex" | "tex" => vec!["texlab", "digestif"],
        "lisp" | "commonlisp" | "common_lisp" => vec!["cl-lsp"],
        "lua" => vec!["lua-language-server"],
        "markdown" => vec!["marksman"],
        "nim" => vec!["nimlangserver", "nimlsp"],
        "nix" => vec!["nil", "nixd", "rnix-lsp"],
        "ocaml" => vec!["ocamllsp"],
        "odin" => vec!["ols"],
        "pascal" | "objectpascal" => vec!["pasls", "pascal-language-server"],
        "perl" => vec!["perlnavigator", "perl-languageserver"],
        "php" => vec!["phpactor", "intelephense"],
        "python" => vec![
            "pylsp",
            "pyright-langserver",
            "pyls",
            "ruff-lsp",
            "jedi-language-server",
        ],
        "r" => vec!["R", "languageserver"],
        "ruby" => vec!["solargraph", "ruby-lsp", "steep"],
        "rust" => vec!["rust-analyzer"],
        "scala" => vec!["metals"],
        "scheme" => vec!["scheme-langserver"],
        "sql" => vec!["sqls", "sql-language-server"],
        "swift" => vec!["sourcekit-lsp"],
        "toml" => vec!["taplo"],
        "vala" => vec!["vala-language-server"],
        "vue" => vec!["vue-language-server", "vls"],
        "yaml" => vec!["yaml-language-server"],
        "zig" => vec!["zls"],
        // Fallback: try common naming patterns
        _ => {
            return vec![
                format!("{language}-language-server"),
                format!("{language}-lsp"),
                format!("{language}ls"),
            ]
        }
    }
    .into_iter()
    .map(|s| s.to_string())
    .collect()
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

        let names = get_lsp_server_names("lua");
        assert!(names.contains(&"lua-language-server".to_string()));

        let names = get_lsp_server_names("swift");
        assert!(names.contains(&"sourcekit-lsp".to_string()));

        let names = get_lsp_server_names("zig");
        assert!(names.contains(&"zls".to_string()));

        let names = get_lsp_server_names("shell");
        assert!(names.contains(&"bash-language-server".to_string()));

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
