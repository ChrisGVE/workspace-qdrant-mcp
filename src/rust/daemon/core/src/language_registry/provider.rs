//! Pluggable provider abstraction for language data sources.
//!
//! Each provider implements `LanguageSourceProvider` to contribute language
//! metadata from an upstream source of truth (e.g., GitHub Linguist,
//! nvim-treesitter, mason.nvim registry).

use async_trait::async_trait;

use crate::error::DaemonError;

use super::types::{GrammarEntry, LanguageEntry, LspEntry, ProviderData};

/// A pluggable source of language metadata.
///
/// Providers fetch language identity, grammar repositories, and LSP server
/// information from upstream sources. Each provider has a priority that
/// determines merge precedence (lower number = higher priority).
///
/// # Merge Strategy
///
/// When multiple providers contribute data for the same language:
/// - Language identity: the provider with the lowest `priority()` wins
/// - Grammar sources: merged and deduplicated by repo, sorted by quality tier
/// - LSP servers: merged and deduplicated by binary name
/// - User overrides always win over any provider data
///
/// # Adding New Providers
///
/// Implement this trait and register via `LanguageRegistry::register_provider()`.
/// No changes to core code are needed.
#[async_trait]
pub trait LanguageSourceProvider: Send + Sync {
    /// Provider name (e.g., "linguist", "nvim-treesitter", "mason").
    fn name(&self) -> &str;

    /// Merge priority — lower number means higher priority.
    fn priority(&self) -> u8;

    /// When this provider's data was last successfully fetched.
    fn last_updated(&self) -> Option<chrono::DateTime<chrono::Utc>>;

    /// Whether this provider is currently enabled.
    fn is_enabled(&self) -> bool {
        true
    }

    /// Fetch language identity entries from the upstream source.
    async fn fetch_languages(&self) -> Result<Vec<LanguageEntry>, DaemonError>;

    /// Fetch grammar repository entries from the upstream source.
    async fn fetch_grammars(&self) -> Result<Vec<GrammarEntry>, DaemonError>;

    /// Fetch LSP server entries from the upstream source.
    async fn fetch_lsp_servers(&self) -> Result<Vec<LspEntry>, DaemonError>;

    /// Re-fetch all data from the upstream source.
    ///
    /// This is the primary refresh mechanism. Implementations should update
    /// their internal cache and `last_updated` timestamp.
    async fn refresh(&self) -> Result<ProviderData, DaemonError>;
}

/// Configuration for a provider instance.
#[derive(Debug, Clone)]
pub struct ProviderConfig {
    /// Whether this provider is enabled.
    pub enabled: bool,
    /// Base URL for fetching data (provider-specific).
    pub url: Option<String>,
    /// Cache TTL in seconds before data is considered stale.
    pub cache_ttl_secs: u64,
}

impl Default for ProviderConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            url: None,
            cache_ttl_secs: 604800, // 1 week
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Mutex;

    /// Mock provider for testing the trait contract.
    struct MockProvider {
        name: String,
        priority: u8,
        languages: Vec<LanguageEntry>,
        grammars: Vec<GrammarEntry>,
        lsp_servers: Vec<LspEntry>,
        refresh_count: Mutex<u32>,
    }

    impl MockProvider {
        fn new(name: &str, priority: u8) -> Self {
            Self {
                name: name.to_string(),
                priority,
                languages: Vec::new(),
                grammars: Vec::new(),
                lsp_servers: Vec::new(),
                refresh_count: Mutex::new(0),
            }
        }

        fn with_languages(mut self, languages: Vec<LanguageEntry>) -> Self {
            self.languages = languages;
            self
        }
    }

    #[async_trait]
    impl LanguageSourceProvider for MockProvider {
        fn name(&self) -> &str {
            &self.name
        }

        fn priority(&self) -> u8 {
            self.priority
        }

        fn last_updated(&self) -> Option<chrono::DateTime<chrono::Utc>> {
            None
        }

        async fn fetch_languages(&self) -> Result<Vec<LanguageEntry>, DaemonError> {
            Ok(self.languages.clone())
        }

        async fn fetch_grammars(&self) -> Result<Vec<GrammarEntry>, DaemonError> {
            Ok(self.grammars.clone())
        }

        async fn fetch_lsp_servers(&self) -> Result<Vec<LspEntry>, DaemonError> {
            Ok(self.lsp_servers.clone())
        }

        async fn refresh(&self) -> Result<ProviderData, DaemonError> {
            *self.refresh_count.lock().unwrap() += 1;
            Ok(ProviderData {
                languages: self.languages.clone(),
                grammars: self.grammars.clone(),
                lsp_servers: self.lsp_servers.clone(),
            })
        }
    }

    #[tokio::test]
    async fn test_provider_trait_contract() {
        use crate::language_registry::types::LanguageType;

        let provider = MockProvider::new("test", 5).with_languages(vec![LanguageEntry {
            name: "Rust".to_string(),
            id: "rust".to_string(),
            aliases: vec!["rs".to_string()],
            extensions: vec![".rs".to_string()],
            language_type: LanguageType::Programming,
        }]);

        assert_eq!(provider.name(), "test");
        assert_eq!(provider.priority(), 5);
        assert!(provider.is_enabled());
        assert!(provider.last_updated().is_none());

        let langs = provider.fetch_languages().await.unwrap();
        assert_eq!(langs.len(), 1);
        assert_eq!(langs[0].id, "rust");
    }

    #[tokio::test]
    async fn test_provider_refresh_returns_all_data() {
        let provider = MockProvider::new("test", 1);
        let data = provider.refresh().await.unwrap();

        assert!(data.languages.is_empty());
        assert!(data.grammars.is_empty());
        assert!(data.lsp_servers.is_empty());
        assert_eq!(*provider.refresh_count.lock().unwrap(), 1);
    }

    #[tokio::test]
    async fn test_priority_ordering() {
        let high = MockProvider::new("high", 1);
        let low = MockProvider::new("low", 10);

        assert!(high.priority() < low.priority());
    }
}
