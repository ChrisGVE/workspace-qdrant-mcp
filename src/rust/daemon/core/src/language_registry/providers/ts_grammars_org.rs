//! tree-sitter-grammars organization provider.
//!
//! Discovers curated grammars from the tree-sitter-grammars GitHub
//! organization. These are the highest quality tier.

use std::sync::Mutex;

use async_trait::async_trait;
use serde::Deserialize;

use crate::error::DaemonError;
use crate::language_registry::provider::LanguageSourceProvider;
use crate::language_registry::types::{
    GrammarEntry, GrammarQuality, LanguageEntry, LspEntry, ProviderData,
};

const GITHUB_API_URL: &str = "https://api.github.com/orgs/tree-sitter-grammars/repos";

/// GitHub API repository entry (minimal fields).
#[derive(Debug, Deserialize)]
struct GithubRepo {
    name: String,
    full_name: String,
    archived: Option<bool>,
}

/// Provider discovering curated grammars from tree-sitter-grammars org.
///
/// Priority 15 — between Linguist (10) and nvim-treesitter (20).
/// Grammars from this org are marked `GrammarQuality::Curated`.
pub struct TreeSitterGrammarsOrgProvider {
    api_url: String,
    cached_data: Mutex<Option<Vec<GrammarEntry>>>,
}

impl TreeSitterGrammarsOrgProvider {
    /// Create a new provider with the default GitHub API URL.
    pub fn new() -> Self {
        Self {
            api_url: GITHUB_API_URL.to_string(),
            cached_data: Mutex::new(None),
        }
    }

    /// Create a provider with a custom API URL (for testing).
    pub fn with_url(url: String) -> Self {
        Self {
            api_url: url,
            cached_data: Mutex::new(None),
        }
    }

    /// Parse GitHub API response into grammar entries.
    fn parse_repos(json: &str) -> Result<Vec<GrammarEntry>, DaemonError> {
        let repos: Vec<GithubRepo> = serde_json::from_str(json)
            .map_err(|e| DaemonError::Other(format!("Failed to parse GitHub API: {e}")))?;

        let mut entries = Vec::new();

        for repo in repos {
            if repo.archived.unwrap_or(false) {
                continue;
            }

            // Extract language from "tree-sitter-{language}" pattern
            let language = match repo.name.strip_prefix("tree-sitter-") {
                Some(lang) if !lang.is_empty() => lang.to_string(),
                _ => continue, // Skip non-grammar repos
            };

            entries.push(GrammarEntry {
                language,
                repo: repo.full_name,
                quality: GrammarQuality::Curated,
                has_cpp_scanner: false,
                src_subdir: None,
                symbol_name: None,
                archive_branch: None,
            });
        }

        entries.sort_by(|a, b| a.language.cmp(&b.language));
        Ok(entries)
    }
}

#[async_trait]
impl LanguageSourceProvider for TreeSitterGrammarsOrgProvider {
    fn name(&self) -> &str {
        "tree-sitter-grammars"
    }

    fn priority(&self) -> u8 {
        15
    }

    fn last_updated(&self) -> Option<chrono::DateTime<chrono::Utc>> {
        None
    }

    async fn fetch_languages(&self) -> Result<Vec<LanguageEntry>, DaemonError> {
        Ok(Vec::new())
    }

    async fn fetch_grammars(&self) -> Result<Vec<GrammarEntry>, DaemonError> {
        if let Some(ref cached) = *self.cached_data.lock().unwrap() {
            return Ok(cached.clone());
        }

        let client = reqwest::Client::builder()
            .user_agent("workspace-qdrant-mcp")
            .build()
            .map_err(|e| DaemonError::Other(format!("HTTP client error: {e}")))?;

        // Paginate through all repos (GitHub default is 30 per page)
        let mut all_entries = Vec::new();
        let mut page = 1;

        loop {
            let url = format!("{}?per_page=100&page={}", self.api_url, page);
            let response = client.get(&url).send().await.map_err(|e| {
                DaemonError::Other(format!("GitHub API request failed: {e}"))
            })?;

            if response.status() == reqwest::StatusCode::FORBIDDEN {
                tracing::warn!("GitHub API rate limit hit for tree-sitter-grammars org");
                break;
            }

            if !response.status().is_success() {
                return Err(DaemonError::Other(format!(
                    "GitHub API failed: {}",
                    response.status()
                )));
            }

            let body = response.text().await.map_err(|e| {
                DaemonError::Other(format!("Failed to read GitHub API response: {e}"))
            })?;

            let entries = Self::parse_repos(&body)?;
            if entries.is_empty() {
                break;
            }
            all_entries.extend(entries);
            page += 1;

            // Safety limit
            if page > 10 {
                break;
            }
        }

        all_entries.sort_by(|a, b| a.language.cmp(&b.language));
        *self.cached_data.lock().unwrap() = Some(all_entries.clone());
        Ok(all_entries)
    }

    async fn fetch_lsp_servers(&self) -> Result<Vec<LspEntry>, DaemonError> {
        Ok(Vec::new())
    }

    async fn refresh(&self) -> Result<ProviderData, DaemonError> {
        *self.cached_data.lock().unwrap() = None;
        Ok(ProviderData {
            languages: Vec::new(),
            grammars: self.fetch_grammars().await?,
            lsp_servers: Vec::new(),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const SAMPLE_REPOS: &str = r#"[
  {
    "name": "tree-sitter-markdown",
    "full_name": "tree-sitter-grammars/tree-sitter-markdown",
    "archived": false
  },
  {
    "name": "tree-sitter-yaml",
    "full_name": "tree-sitter-grammars/tree-sitter-yaml",
    "archived": false
  },
  {
    "name": ".github",
    "full_name": "tree-sitter-grammars/.github",
    "archived": false
  },
  {
    "name": "tree-sitter-old-grammar",
    "full_name": "tree-sitter-grammars/tree-sitter-old-grammar",
    "archived": true
  }
]"#;

    #[test]
    fn test_parse_repos() {
        let entries = TreeSitterGrammarsOrgProvider::parse_repos(SAMPLE_REPOS).unwrap();
        assert_eq!(entries.len(), 2);

        let md = entries.iter().find(|e| e.language == "markdown").unwrap();
        assert_eq!(md.repo, "tree-sitter-grammars/tree-sitter-markdown");
        assert_eq!(md.quality, GrammarQuality::Curated);

        let yaml = entries.iter().find(|e| e.language == "yaml").unwrap();
        assert_eq!(yaml.repo, "tree-sitter-grammars/tree-sitter-yaml");

        // .github repo and archived repo should be filtered out
        assert!(!entries.iter().any(|e| e.language == ".github"));
        assert!(!entries.iter().any(|e| e.language == "old-grammar"));
    }

    #[test]
    fn test_provider_metadata() {
        let provider = TreeSitterGrammarsOrgProvider::new();
        assert_eq!(provider.name(), "tree-sitter-grammars");
        assert_eq!(provider.priority(), 15);
        assert!(provider.is_enabled());
    }
}
