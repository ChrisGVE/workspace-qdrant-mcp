//! LSP Server Detection Module
//!
//! This module handles automatic detection of LSP servers available on the system
//! through PATH scanning and capability discovery.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::process::Command;
use serde::{Deserialize, Serialize};
use tracing::{debug, info, warn};
use which::which;

use crate::lsp::{Language, LspResult};

/// Information about a detected LSP server
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DetectedServer {
    /// Server executable name
    pub name: String,
    /// Full path to the executable
    pub path: PathBuf,
    /// Languages supported by this server
    pub languages: Vec<Language>,
    /// Server version if detectable
    pub version: Option<String>,
    /// Server capabilities
    pub capabilities: ServerCapabilities,
    /// Priority ranking for this server type
    pub priority: u8,
}

/// LSP server capabilities
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ServerCapabilities {
    /// Supports text document synchronization
    pub text_document_sync: bool,
    /// Supports completion
    pub completion: bool,
    /// Supports hover information
    pub hover: bool,
    /// Supports signature help
    pub signature_help: bool,
    /// Supports go to definition
    pub definition: bool,
    /// Supports find references
    pub references: bool,
    /// Supports document highlighting
    pub document_highlight: bool,
    /// Supports document symbols
    pub document_symbol: bool,
    /// Supports workspace symbols
    pub workspace_symbol: bool,
    /// Supports code actions
    pub code_action: bool,
    /// Supports code lens
    pub code_lens: bool,
    /// Supports document formatting
    pub document_formatting: bool,
    /// Supports document range formatting
    pub document_range_formatting: bool,
    /// Supports document on type formatting
    pub document_on_type_formatting: bool,
    /// Supports renaming
    pub rename: bool,
    /// Supports folding ranges
    pub folding_range: bool,
    /// Supports selection ranges
    pub selection_range: bool,
    /// Supports semantic tokens
    pub semantic_tokens: bool,
    /// Supports diagnostics
    pub diagnostics: bool,
}

impl Default for ServerCapabilities {
    fn default() -> Self {
        Self {
            text_document_sync: true,
            completion: true,
            hover: true,
            signature_help: false,
            definition: true,
            references: true,
            document_highlight: false,
            document_symbol: true,
            workspace_symbol: true,
            code_action: false,
            code_lens: false,
            document_formatting: false,
            document_range_formatting: false,
            document_on_type_formatting: false,
            rename: false,
            folding_range: false,
            selection_range: false,
            semantic_tokens: false,
            diagnostics: true,
        }
    }
}

/// LSP server detector that scans the system for available servers
pub struct LspServerDetector {
    /// Known LSP server configurations
    known_servers: HashMap<&'static str, ServerTemplate>,
}

/// Template for known LSP server configurations
#[derive(Debug, Clone)]
struct ServerTemplate {
    /// Executable name to look for
    executable: &'static str,
    /// Languages this server supports
    languages: &'static [Language],
    /// Default capabilities
    capabilities: ServerCapabilities,
    /// Priority (lower = higher priority)
    priority: u8,
    /// Version detection command arguments
    version_args: &'static [&'static str],
}

impl LspServerDetector {
    /// Create a new LSP server detector
    pub fn new() -> Self {
        let mut known_servers = HashMap::new();

        // Python LSP servers
        known_servers.insert("ruff-lsp", ServerTemplate {
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
            priority: 1, // High priority for Python
            version_args: &["--version"],
        });

        known_servers.insert("pylsp", ServerTemplate {
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
        });

        known_servers.insert("pyright-langserver", ServerTemplate {
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
        });

        // Rust LSP server
        known_servers.insert("rust-analyzer", ServerTemplate {
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
        });

        // TypeScript/JavaScript LSP servers
        known_servers.insert("typescript-language-server", ServerTemplate {
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
        });

        known_servers.insert("vscode-json-languageserver", ServerTemplate {
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
        });

        // C/C++ LSP servers
        known_servers.insert("clangd", ServerTemplate {
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
        });

        known_servers.insert("ccls", ServerTemplate {
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
        });

        // Go LSP server
        known_servers.insert("gopls", ServerTemplate {
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
        });

        Self { known_servers }
    }

    /// Detect all available LSP servers on the system
    pub async fn detect_servers(&self) -> LspResult<Vec<DetectedServer>> {
        info!("Starting LSP server detection");
        let mut detected = Vec::new();

        for (name, template) in &self.known_servers {
            debug!("Looking for LSP server: {}", name);
            
            match self.detect_server(name, template).await {
                Ok(Some(server)) => {
                    info!("Detected LSP server: {} at {}", server.name, server.path.display());
                    detected.push(server);
                }
                Ok(None) => {
                    debug!("LSP server not found: {}", name);
                }
                Err(e) => {
                    warn!("Error detecting LSP server {}: {}", name, e);
                }
            }
        }

        // Sort by priority and language coverage
        detected.sort_by(|a, b| {
            a.priority.cmp(&b.priority)
                .then_with(|| b.languages.len().cmp(&a.languages.len()))
        });

        info!("Detected {} LSP servers", detected.len());
        Ok(detected)
    }

    /// Detect a specific LSP server
    async fn detect_server(
        &self,
        name: &str,
        template: &ServerTemplate,
    ) -> LspResult<Option<DetectedServer>> {
        // Try to find the executable in PATH
        let path = match which(template.executable) {
            Ok(path) => path,
            Err(_) => return Ok(None),
        };

        // Verify the executable is actually executable
        if !self.is_executable(&path).await? {
            return Ok(None);
        }

        // Try to get version information
        let version = self.get_server_version(&path, template.version_args).await;

        let detected = DetectedServer {
            name: name.to_string(),
            path,
            languages: template.languages.to_vec(),
            version,
            capabilities: template.capabilities.clone(),
            priority: template.priority,
        };

        Ok(Some(detected))
    }

    /// Check if a path is an executable file
    async fn is_executable(&self, path: &Path) -> LspResult<bool> {
        let metadata = tokio::fs::metadata(path).await?;
        
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            let permissions = metadata.permissions();
            Ok(metadata.is_file() && (permissions.mode() & 0o111) != 0)
        }
        
        #[cfg(windows)]
        {
            Ok(metadata.is_file())
        }
    }

    /// Try to get version information from an LSP server
    async fn get_server_version(&self, path: &Path, version_args: &[&str]) -> Option<String> {
        let output = Command::new(path)
            .args(version_args)
            .output()
            .ok()?;

        if output.status.success() {
            let stdout = String::from_utf8_lossy(&output.stdout);
            let stderr = String::from_utf8_lossy(&output.stderr);
            
            // Try stdout first, then stderr
            let version_text = if !stdout.trim().is_empty() {
                stdout.trim()
            } else {
                stderr.trim()
            };

            if !version_text.is_empty() {
                Some(self.extract_version_number(version_text))
            } else {
                None
            }
        } else {
            None
        }
    }

    /// Extract version number from version output
    fn extract_version_number(&self, text: &str) -> String {
        // Look for common version patterns
        let version_patterns = [
            regex::Regex::new(r"(\d+\.\d+(?:\.\d+)?)").unwrap(),
            regex::Regex::new(r"version\s+(\d+\.\d+(?:\.\d+)?)").unwrap(),
            regex::Regex::new(r"v(\d+\.\d+(?:\.\d+)?)").unwrap(),
        ];

        for pattern in &version_patterns {
            if let Some(captures) = pattern.captures(text) {
                if let Some(version) = captures.get(1) {
                    return version.as_str().to_string();
                }
            }
        }

        // If no pattern matches, return the first line cleaned up
        text.lines()
            .next()
            .unwrap_or(text)
            .trim()
            .to_string()
    }

    /// Get servers for a specific language, sorted by priority
    pub fn get_servers_for_language(&self, language: &Language) -> Vec<&str> {
        let mut servers: Vec<(&str, u8)> = self.known_servers
            .iter()
            .filter_map(|(name, template)| {
                if template.languages.contains(language) {
                    Some((name.as_ref(), template.priority))
                } else {
                    None
                }
            })
            .collect();

        servers.sort_by_key(|(_, priority)| *priority);
        servers.into_iter().map(|(name, _)| name).collect()
    }

    /// Check if a specific server is known
    pub fn is_known_server(&self, name: &str) -> bool {
        self.known_servers.contains_key(name)
    }

    /// Get template for a known server
    pub fn get_server_template(&self, name: &str) -> Option<&ServerTemplate> {
        self.known_servers.get(name)
    }
}

impl Default for LspServerDetector {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_server_detector_creation() {
        let detector = LspServerDetector::new();
        assert!(!detector.known_servers.is_empty());
    }

    #[test]
    fn test_get_servers_for_language() {
        let detector = LspServerDetector::new();
        let python_servers = detector.get_servers_for_language(&Language::Python);
        assert!(!python_servers.is_empty());
        
        // Should be sorted by priority
        let priorities: Vec<u8> = python_servers.iter()
            .map(|name| detector.get_server_template(name).unwrap().priority)
            .collect();
        
        for i in 1..priorities.len() {
            assert!(priorities[i-1] <= priorities[i]);
        }
    }

    #[test]
    fn test_extract_version_number() {
        let detector = LspServerDetector::new();
        
        assert_eq!(detector.extract_version_number("rust-analyzer 0.3.1"), "0.3.1");
        assert_eq!(detector.extract_version_number("version 2.1.0"), "2.1.0");
        assert_eq!(detector.extract_version_number("v1.2.3"), "1.2.3");
        assert_eq!(detector.extract_version_number("Some program 1.0"), "1.0");
    }

    #[test]
    fn test_is_known_server() {
        let detector = LspServerDetector::new();
        assert!(detector.is_known_server("rust-analyzer"));
        assert!(detector.is_known_server("ruff-lsp"));
        assert!(!detector.is_known_server("unknown-server"));
    }

    #[tokio::test]
    async fn test_detect_servers() {
        let detector = LspServerDetector::new();
        // This test will depend on what's installed on the system
        let result = detector.detect_servers().await;
        assert!(result.is_ok());
    }
}