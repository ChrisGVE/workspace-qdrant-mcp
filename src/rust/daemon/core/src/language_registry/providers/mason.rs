//! mason.nvim registry provider for LSP server metadata.
//!
//! Fetches LSP server installation information from the mason-registry.
//! This is the primary source for discovering LSP server binaries and
//! their installation methods across package managers.

use std::collections::HashMap;
use std::sync::Mutex;

use async_trait::async_trait;
use serde::Deserialize;

use crate::error::DaemonError;
use crate::language_registry::provider::LanguageSourceProvider;
use crate::language_registry::types::{
    GrammarEntry, InstallMethod, LanguageEntry, LspEntry, LspServerEntry, ProviderData,
};

const REGISTRY_URL: &str =
    "https://raw.githubusercontent.com/mason-org/mason-registry/main/registry.json";

/// mason-registry entry structure (simplified).
#[derive(Debug, Deserialize)]
struct MasonPackage {
    name: String,
    #[serde(default)]
    categories: Vec<String>,
    #[serde(default)]
    languages: Vec<String>,
    #[serde(default)]
    source: MasonSource,
}

/// Mason package source.
#[derive(Debug, Default, Deserialize)]
struct MasonSource {
    id: Option<String>,
    #[serde(flatten)]
    _extra: HashMap<String, serde_json::Value>,
}

/// Provider that fetches LSP metadata from the mason.nvim registry.
///
/// Priority 30 — lower than Linguist and nvim-treesitter, as mason
/// is primarily an installation tool rather than a source of truth.
pub struct MasonProvider {
    url: String,
    cached_data: Mutex<Option<Vec<LspEntry>>>,
}

impl MasonProvider {
    /// Create a new provider with the default registry URL.
    pub fn new() -> Self {
        Self {
            url: REGISTRY_URL.to_string(),
            cached_data: Mutex::new(None),
        }
    }

    /// Create a provider with a custom URL (for testing).
    pub fn with_url(url: String) -> Self {
        Self {
            url,
            cached_data: Mutex::new(None),
        }
    }

    /// Parse mason registry JSON into LSP entries.
    fn parse_registry(json: &str) -> Result<Vec<LspEntry>, DaemonError> {
        let packages: Vec<MasonPackage> = serde_json::from_str(json)
            .map_err(|e| DaemonError::Other(format!("Failed to parse mason registry: {e}")))?;

        let mut entries = Vec::new();

        for pkg in &packages {
            // Only interested in LSP servers
            if !pkg.categories.iter().any(|c| c == "LSP") {
                continue;
            }

            // Extract install method from source ID
            let install_methods = Self::infer_install_methods(&pkg.name, &pkg.source);

            let binary = pkg.name.clone();

            for language in &pkg.languages {
                let lang_id = language.to_lowercase().replace(' ', "-");

                entries.push(LspEntry {
                    language: lang_id,
                    server: LspServerEntry {
                        name: pkg.name.clone(),
                        binary: binary.clone(),
                        args: Vec::new(),
                        priority: 50, // Default priority for mason-discovered servers
                        install_methods: install_methods.clone(),
                    },
                });
            }
        }

        entries.sort_by(|a, b| a.language.cmp(&b.language));
        Ok(entries)
    }

    /// Infer installation methods from package metadata.
    fn infer_install_methods(name: &str, source: &MasonSource) -> Vec<InstallMethod> {
        let mut methods = Vec::new();

        if let Some(ref id) = source.id {
            // Source IDs follow patterns like "pkg:npm/...", "pkg:pypi/...", "pkg:cargo/..."
            if id.starts_with("pkg:npm/") {
                methods.push(InstallMethod {
                    manager: "npm".to_string(),
                    command: format!("npm install -g {name}"),
                });
            } else if id.starts_with("pkg:pypi/") {
                methods.push(InstallMethod {
                    manager: "pip".to_string(),
                    command: format!("pip install {name}"),
                });
            } else if id.starts_with("pkg:cargo/") {
                methods.push(InstallMethod {
                    manager: "cargo".to_string(),
                    command: format!("cargo install {name}"),
                });
            } else if id.starts_with("pkg:gem/") {
                methods.push(InstallMethod {
                    manager: "gem".to_string(),
                    command: format!("gem install {name}"),
                });
            } else if id.starts_with("pkg:golang/") {
                methods.push(InstallMethod {
                    manager: "go".to_string(),
                    command: format!("go install {name}@latest"),
                });
            } else if id.starts_with("pkg:github/") {
                methods.push(InstallMethod {
                    manager: "github".to_string(),
                    command: format!("Download from {id}"),
                });
            }
        }

        methods
    }
}

#[async_trait]
impl LanguageSourceProvider for MasonProvider {
    fn name(&self) -> &str {
        "mason"
    }

    fn priority(&self) -> u8 {
        30
    }

    fn last_updated(&self) -> Option<chrono::DateTime<chrono::Utc>> {
        None
    }

    async fn fetch_languages(&self) -> Result<Vec<LanguageEntry>, DaemonError> {
        Ok(Vec::new())
    }

    async fn fetch_grammars(&self) -> Result<Vec<GrammarEntry>, DaemonError> {
        Ok(Vec::new())
    }

    async fn fetch_lsp_servers(&self) -> Result<Vec<LspEntry>, DaemonError> {
        if let Some(ref cached) = *self.cached_data.lock().unwrap() {
            return Ok(cached.clone());
        }

        let client = reqwest::Client::builder()
            .user_agent("workspace-qdrant-mcp")
            .build()
            .map_err(|e| DaemonError::Other(format!("HTTP client error: {e}")))?;

        let response = client
            .get(&self.url)
            .send()
            .await
            .map_err(|e| DaemonError::Other(format!("Failed to fetch mason registry: {e}")))?;

        if !response.status().is_success() {
            return Err(DaemonError::Other(format!(
                "Mason registry fetch failed: {}",
                response.status()
            )));
        }

        let body = response
            .text()
            .await
            .map_err(|e| DaemonError::Other(format!("Failed to read mason response: {e}")))?;

        let entries = Self::parse_registry(&body)?;
        *self.cached_data.lock().unwrap() = Some(entries.clone());
        Ok(entries)
    }

    async fn refresh(&self) -> Result<ProviderData, DaemonError> {
        *self.cached_data.lock().unwrap() = None;
        Ok(ProviderData {
            languages: Vec::new(),
            grammars: Vec::new(),
            lsp_servers: self.fetch_lsp_servers().await?,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const SAMPLE_REGISTRY: &str = r#"[
  {
    "name": "rust-analyzer",
    "categories": ["LSP"],
    "languages": ["Rust"],
    "source": {
      "id": "pkg:github/rust-lang/rust-analyzer"
    }
  },
  {
    "name": "pyright",
    "categories": ["LSP"],
    "languages": ["Python"],
    "source": {
      "id": "pkg:npm/@anthropic-ai/pyright"
    }
  },
  {
    "name": "prettier",
    "categories": ["Formatter"],
    "languages": ["JavaScript", "TypeScript"],
    "source": {
      "id": "pkg:npm/prettier"
    }
  },
  {
    "name": "gopls",
    "categories": ["LSP"],
    "languages": ["Go"],
    "source": {
      "id": "pkg:golang/golang.org/x/tools/gopls"
    }
  }
]"#;

    #[test]
    fn test_parse_registry() {
        let entries = MasonProvider::parse_registry(SAMPLE_REGISTRY).unwrap();

        // Only LSP entries should be included (prettier is Formatter)
        assert_eq!(entries.len(), 3);

        let rust = entries.iter().find(|e| e.language == "rust").unwrap();
        assert_eq!(rust.server.name, "rust-analyzer");
        assert_eq!(rust.server.install_methods.len(), 1);
        assert_eq!(rust.server.install_methods[0].manager, "github");

        let python = entries.iter().find(|e| e.language == "python").unwrap();
        assert_eq!(python.server.name, "pyright");
        assert_eq!(python.server.install_methods[0].manager, "npm");

        let go = entries.iter().find(|e| e.language == "go").unwrap();
        assert_eq!(go.server.name, "gopls");
        assert_eq!(go.server.install_methods[0].manager, "go");
    }

    #[test]
    fn test_formatter_excluded() {
        let entries = MasonProvider::parse_registry(SAMPLE_REGISTRY).unwrap();
        assert!(!entries.iter().any(|e| e.server.name == "prettier"));
    }

    #[test]
    fn test_provider_metadata() {
        let provider = MasonProvider::new();
        assert_eq!(provider.name(), "mason");
        assert_eq!(provider.priority(), 30);
        assert!(provider.is_enabled());
    }
}
