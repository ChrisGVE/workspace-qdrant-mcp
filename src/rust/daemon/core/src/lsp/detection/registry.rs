//! Known LSP Server Registry
//!
//! Contains the catalog of known LSP server configurations with their
//! executables, supported languages, default capabilities, and priorities.

use std::collections::HashMap;

use super::{ServerCapabilities, ServerTemplate};
use crate::lsp::Language;

/// Build the catalog of known LSP server configurations.
///
/// Returns a map from server name to its template configuration.
/// Servers are organized by language with priority rankings
/// (lower number = higher priority).
pub(crate) fn build_known_servers() -> HashMap<&'static str, ServerTemplate> {
    let mut servers = HashMap::new();

    register_python_servers(&mut servers);
    register_rust_servers(&mut servers);
    register_typescript_servers(&mut servers);
    register_json_servers(&mut servers);
    register_c_cpp_servers(&mut servers);
    register_go_servers(&mut servers);
    register_java_servers(&mut servers);
    register_ruby_servers(&mut servers);
    register_php_servers(&mut servers);
    register_shell_servers(&mut servers);
    register_html_servers(&mut servers);

    servers
}

/// Register Python LSP server templates
fn register_python_servers(servers: &mut HashMap<&'static str, ServerTemplate>) {
    servers.insert(
        "ruff-lsp",
        ServerTemplate {
            executable: "ruff-lsp",
            languages: &[Language::Python],
            capabilities: ServerCapabilities {
                text_document_sync: true,
                completion: true,
                hover: true,
                definition: true,
                references: true,
                document_symbol: true,
                workspace_symbol: true,
                code_action: true,
                document_formatting: true,
                document_range_formatting: true,
                document_on_type_formatting: false,
                diagnostics: true,
                ..Default::default()
            },
            priority: 1,
            version_args: &["--version"],
        },
    );

    servers.insert(
        "pylsp",
        ServerTemplate {
            executable: "pylsp",
            languages: &[Language::Python],
            capabilities: ServerCapabilities {
                text_document_sync: true,
                completion: true,
                hover: true,
                signature_help: true,
                definition: true,
                references: true,
                document_highlight: true,
                document_symbol: true,
                workspace_symbol: true,
                code_action: true,
                document_formatting: true,
                rename: true,
                diagnostics: true,
                ..Default::default()
            },
            priority: 2,
            version_args: &["--version"],
        },
    );

    servers.insert(
        "pyright-langserver",
        ServerTemplate {
            executable: "pyright-langserver",
            languages: &[Language::Python],
            capabilities: ServerCapabilities {
                text_document_sync: true,
                completion: true,
                hover: true,
                signature_help: true,
                definition: true,
                references: true,
                document_highlight: true,
                document_symbol: true,
                workspace_symbol: true,
                rename: true,
                diagnostics: true,
                ..Default::default()
            },
            priority: 3,
            version_args: &["--version"],
        },
    );
}

/// Register Rust LSP server templates
fn register_rust_servers(servers: &mut HashMap<&'static str, ServerTemplate>) {
    servers.insert(
        "rust-analyzer",
        ServerTemplate {
            executable: "rust-analyzer",
            languages: &[Language::Rust],
            capabilities: ServerCapabilities {
                text_document_sync: true,
                completion: true,
                hover: true,
                signature_help: true,
                definition: true,
                references: true,
                document_highlight: true,
                document_symbol: true,
                workspace_symbol: true,
                code_action: true,
                code_lens: true,
                document_formatting: true,
                document_range_formatting: false,
                document_on_type_formatting: false,
                rename: true,
                folding_range: true,
                selection_range: true,
                semantic_tokens: true,
                diagnostics: true,
            },
            priority: 1,
            version_args: &["--version"],
        },
    );
}

/// Register TypeScript/JavaScript LSP server templates
fn register_typescript_servers(servers: &mut HashMap<&'static str, ServerTemplate>) {
    servers.insert(
        "typescript-language-server",
        ServerTemplate {
            executable: "typescript-language-server",
            languages: &[Language::TypeScript, Language::JavaScript],
            capabilities: ServerCapabilities {
                text_document_sync: true,
                completion: true,
                hover: true,
                signature_help: true,
                definition: true,
                references: true,
                document_highlight: true,
                document_symbol: true,
                workspace_symbol: true,
                code_action: true,
                document_formatting: true,
                rename: true,
                diagnostics: true,
                ..Default::default()
            },
            priority: 1,
            version_args: &["--version"],
        },
    );
}

/// Register JSON LSP server templates
fn register_json_servers(servers: &mut HashMap<&'static str, ServerTemplate>) {
    servers.insert(
        "vscode-json-languageserver",
        ServerTemplate {
            executable: "vscode-json-languageserver",
            languages: &[Language::Json],
            capabilities: ServerCapabilities {
                text_document_sync: true,
                completion: true,
                hover: true,
                document_symbol: true,
                document_formatting: true,
                document_range_formatting: true,
                document_on_type_formatting: false,
                diagnostics: true,
                ..Default::default()
            },
            priority: 1,
            version_args: &["--version"],
        },
    );
}

/// Register C/C++ LSP server templates
fn register_c_cpp_servers(servers: &mut HashMap<&'static str, ServerTemplate>) {
    servers.insert(
        "clangd",
        ServerTemplate {
            executable: "clangd",
            languages: &[Language::C, Language::Cpp],
            capabilities: ServerCapabilities {
                text_document_sync: true,
                completion: true,
                hover: true,
                signature_help: true,
                definition: true,
                references: true,
                document_highlight: true,
                document_symbol: true,
                workspace_symbol: true,
                code_action: true,
                document_formatting: true,
                rename: true,
                semantic_tokens: true,
                diagnostics: true,
                ..Default::default()
            },
            priority: 1,
            version_args: &["--version"],
        },
    );

    servers.insert(
        "ccls",
        ServerTemplate {
            executable: "ccls",
            languages: &[Language::C, Language::Cpp],
            capabilities: ServerCapabilities {
                text_document_sync: true,
                completion: true,
                hover: true,
                definition: true,
                references: true,
                document_symbol: true,
                workspace_symbol: true,
                selection_range: false,
                diagnostics: true,
                ..Default::default()
            },
            priority: 2,
            version_args: &["--version"],
        },
    );
}

/// Register Go LSP server templates
fn register_go_servers(servers: &mut HashMap<&'static str, ServerTemplate>) {
    servers.insert(
        "gopls",
        ServerTemplate {
            executable: "gopls",
            languages: &[Language::Go],
            capabilities: ServerCapabilities {
                text_document_sync: true,
                completion: true,
                hover: true,
                signature_help: true,
                definition: true,
                references: true,
                document_highlight: true,
                document_symbol: true,
                workspace_symbol: true,
                code_action: true,
                code_lens: true,
                document_formatting: true,
                document_range_formatting: false,
                document_on_type_formatting: false,
                rename: true,
                folding_range: true,
                selection_range: false,
                semantic_tokens: true,
                diagnostics: true,
            },
            priority: 1,
            version_args: &["version"],
        },
    );
}

/// Register Java LSP server templates
fn register_java_servers(servers: &mut HashMap<&'static str, ServerTemplate>) {
    servers.insert(
        "jdtls",
        ServerTemplate {
            executable: "jdtls",
            languages: &[Language::Java],
            capabilities: ServerCapabilities {
                text_document_sync: true,
                completion: true,
                hover: true,
                signature_help: true,
                definition: true,
                references: true,
                document_highlight: true,
                document_symbol: true,
                workspace_symbol: true,
                code_action: true,
                code_lens: true,
                document_formatting: true,
                rename: true,
                diagnostics: true,
                ..Default::default()
            },
            priority: 1,
            version_args: &["--version"],
        },
    );
}

/// Register Ruby LSP server templates
fn register_ruby_servers(servers: &mut HashMap<&'static str, ServerTemplate>) {
    servers.insert(
        "ruby-lsp",
        ServerTemplate {
            executable: "ruby-lsp",
            languages: &[Language::Ruby],
            capabilities: ServerCapabilities {
                text_document_sync: true,
                completion: true,
                hover: true,
                definition: true,
                references: true,
                document_symbol: true,
                diagnostics: true,
                ..Default::default()
            },
            priority: 1,
            version_args: &["--version"],
        },
    );

    servers.insert(
        "solargraph",
        ServerTemplate {
            executable: "solargraph",
            languages: &[Language::Ruby],
            capabilities: ServerCapabilities {
                text_document_sync: true,
                completion: true,
                hover: true,
                definition: true,
                references: true,
                document_symbol: true,
                workspace_symbol: true,
                code_action: true,
                document_formatting: true,
                diagnostics: true,
                ..Default::default()
            },
            priority: 2,
            version_args: &["--version"],
        },
    );
}

/// Register PHP LSP server templates
fn register_php_servers(servers: &mut HashMap<&'static str, ServerTemplate>) {
    servers.insert(
        "phpactor",
        ServerTemplate {
            executable: "phpactor",
            languages: &[Language::Php],
            capabilities: ServerCapabilities {
                text_document_sync: true,
                completion: true,
                hover: true,
                definition: true,
                references: true,
                document_symbol: true,
                workspace_symbol: true,
                code_action: true,
                rename: true,
                diagnostics: true,
                ..Default::default()
            },
            priority: 1,
            version_args: &["--version"],
        },
    );

    servers.insert(
        "intelephense",
        ServerTemplate {
            executable: "intelephense",
            languages: &[Language::Php],
            capabilities: ServerCapabilities {
                text_document_sync: true,
                completion: true,
                hover: true,
                signature_help: true,
                definition: true,
                references: true,
                document_highlight: true,
                document_symbol: true,
                workspace_symbol: true,
                rename: true,
                diagnostics: true,
                ..Default::default()
            },
            priority: 2,
            version_args: &["--version"],
        },
    );
}

/// Register Shell/Bash LSP server templates
fn register_shell_servers(servers: &mut HashMap<&'static str, ServerTemplate>) {
    servers.insert(
        "bash-language-server",
        ServerTemplate {
            executable: "bash-language-server",
            languages: &[Language::Shell],
            capabilities: ServerCapabilities {
                text_document_sync: true,
                completion: true,
                hover: true,
                definition: true,
                references: true,
                document_symbol: true,
                document_highlight: true,
                diagnostics: true,
                ..Default::default()
            },
            priority: 1,
            version_args: &["--version"],
        },
    );
}

/// Register HTML LSP server templates
fn register_html_servers(servers: &mut HashMap<&'static str, ServerTemplate>) {
    servers.insert(
        "vscode-html-languageserver",
        ServerTemplate {
            executable: "vscode-html-languageserver",
            languages: &[Language::Html],
            capabilities: ServerCapabilities {
                text_document_sync: true,
                completion: true,
                hover: true,
                document_symbol: true,
                document_formatting: true,
                document_range_formatting: true,
                diagnostics: true,
                ..Default::default()
            },
            priority: 1,
            version_args: &["--version"],
        },
    );
}
