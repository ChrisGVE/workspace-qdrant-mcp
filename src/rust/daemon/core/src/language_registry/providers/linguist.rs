//! GitHub Linguist language metadata provider.
//!
//! Fetches canonical language identity data from the Linguist YAML file.
//! This is the authoritative source for language names, extensions, aliases,
//! and type classification.

use std::collections::HashMap;
use std::sync::Mutex;

use async_trait::async_trait;
use serde::Deserialize;

use crate::error::DaemonError;
use crate::language_registry::provider::LanguageSourceProvider;
use crate::language_registry::types::{
    GrammarEntry, LanguageEntry, LanguageType, LspEntry, ProviderData,
};

const LINGUIST_URL: &str =
    "https://raw.githubusercontent.com/github-linguist/linguist/master/lib/linguist/languages.yml";

/// Raw Linguist YAML entry structure.
#[derive(Debug, Deserialize)]
struct LinguistLanguage {
    #[serde(rename = "type")]
    language_type: Option<String>,
    aliases: Option<Vec<String>>,
    extensions: Option<Vec<String>>,
    #[serde(rename = "tm_scope")]
    _tm_scope: Option<String>,
    #[serde(rename = "ace_mode")]
    _ace_mode: Option<String>,
    #[serde(rename = "language_id")]
    _language_id: Option<u64>,
    // Accept and ignore any other fields
    #[serde(flatten)]
    _extra: HashMap<String, serde_yaml_ng::Value>,
}

/// Provider that fetches language metadata from GitHub Linguist.
///
/// Priority 10 (high — Linguist is authoritative for language identity).
pub struct LinguistProvider {
    url: String,
    last_etag: Mutex<Option<String>>,
    cached_data: Mutex<Option<Vec<LanguageEntry>>>,
}

impl LinguistProvider {
    /// Create a new Linguist provider with the default URL.
    pub fn new() -> Self {
        Self {
            url: LINGUIST_URL.to_string(),
            last_etag: Mutex::new(None),
            cached_data: Mutex::new(None),
        }
    }

    /// Create a provider with a custom URL (for testing).
    pub fn with_url(url: String) -> Self {
        Self {
            url,
            last_etag: Mutex::new(None),
            cached_data: Mutex::new(None),
        }
    }

    /// Parse Linguist YAML into language entries.
    fn parse_languages(yaml: &str) -> Result<Vec<LanguageEntry>, DaemonError> {
        let raw: HashMap<String, LinguistLanguage> = serde_yaml_ng::from_str(yaml)
            .map_err(|e| DaemonError::Other(format!("Failed to parse Linguist YAML: {e}")))?;

        let mut entries = Vec::with_capacity(raw.len());

        for (name, lang) in raw {
            let language_type = match lang.language_type.as_deref() {
                Some("programming") => LanguageType::Programming,
                Some("markup") => LanguageType::Markup,
                Some("data") => LanguageType::Data,
                Some("prose") => LanguageType::Prose,
                _ => LanguageType::Data, // Default for unknown types
            };

            let id = name.to_lowercase().replace(' ', "-");

            entries.push(LanguageEntry {
                name: name.clone(),
                id,
                aliases: lang.aliases.unwrap_or_default(),
                extensions: lang.extensions.unwrap_or_default(),
                language_type,
            });
        }

        entries.sort_by(|a, b| a.name.cmp(&b.name));
        Ok(entries)
    }
}

#[async_trait]
impl LanguageSourceProvider for LinguistProvider {
    fn name(&self) -> &str {
        "linguist"
    }

    fn priority(&self) -> u8 {
        10
    }

    fn last_updated(&self) -> Option<chrono::DateTime<chrono::Utc>> {
        None
    }

    async fn fetch_languages(&self) -> Result<Vec<LanguageEntry>, DaemonError> {
        // Return cached data if available
        if let Some(ref cached) = *self.cached_data.lock().unwrap() {
            return Ok(cached.clone());
        }

        let client = reqwest::Client::new();
        let mut request = client.get(&self.url);

        // Use ETag for conditional requests
        if let Some(ref etag) = *self.last_etag.lock().unwrap() {
            request = request.header("If-None-Match", etag.as_str());
        }

        let response = request
            .send()
            .await
            .map_err(|e| DaemonError::Other(format!("Failed to fetch Linguist data: {e}")))?;

        if response.status() == reqwest::StatusCode::NOT_MODIFIED {
            // Data hasn't changed, use cache
            if let Some(ref cached) = *self.cached_data.lock().unwrap() {
                return Ok(cached.clone());
            }
        }

        if !response.status().is_success() {
            return Err(DaemonError::Other(format!(
                "Linguist fetch failed with status: {}",
                response.status()
            )));
        }

        // Store ETag for future conditional requests
        if let Some(etag) = response.headers().get("etag") {
            if let Ok(etag_str) = etag.to_str() {
                *self.last_etag.lock().unwrap() = Some(etag_str.to_string());
            }
        }

        let body = response
            .text()
            .await
            .map_err(|e| DaemonError::Other(format!("Failed to read Linguist response: {e}")))?;

        let entries = Self::parse_languages(&body)?;
        *self.cached_data.lock().unwrap() = Some(entries.clone());
        Ok(entries)
    }

    async fn fetch_grammars(&self) -> Result<Vec<GrammarEntry>, DaemonError> {
        // Linguist doesn't provide grammar metadata
        Ok(Vec::new())
    }

    async fn fetch_lsp_servers(&self) -> Result<Vec<LspEntry>, DaemonError> {
        // Linguist doesn't provide LSP metadata
        Ok(Vec::new())
    }

    async fn refresh(&self) -> Result<ProviderData, DaemonError> {
        // Clear cache to force re-fetch
        *self.cached_data.lock().unwrap() = None;
        Ok(ProviderData {
            languages: self.fetch_languages().await?,
            grammars: Vec::new(),
            lsp_servers: Vec::new(),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const SAMPLE_YAML: &str = r#"
Rust:
  type: programming
  aliases:
    - rs
  extensions:
    - ".rs"
  tm_scope: source.rust
  ace_mode: rust
  language_id: 327

Python:
  type: programming
  aliases:
    - py
    - python3
  extensions:
    - ".py"
    - ".pyi"
    - ".pyw"
  tm_scope: source.python
  ace_mode: python
  language_id: 303

Markdown:
  type: prose
  aliases:
    - md
  extensions:
    - ".md"
    - ".markdown"
  tm_scope: text.html.markdown
  ace_mode: markdown
  language_id: 222

JSON:
  type: data
  extensions:
    - ".json"
  tm_scope: source.json
  ace_mode: json
  language_id: 174
"#;

    #[test]
    fn test_parse_languages() {
        let entries = LinguistProvider::parse_languages(SAMPLE_YAML).unwrap();
        assert_eq!(entries.len(), 4);

        let rust = entries.iter().find(|e| e.id == "rust").unwrap();
        assert_eq!(rust.name, "Rust");
        assert_eq!(rust.aliases, vec!["rs"]);
        assert_eq!(rust.extensions, vec![".rs"]);
        assert_eq!(rust.language_type, LanguageType::Programming);

        let python = entries.iter().find(|e| e.id == "python").unwrap();
        assert_eq!(python.aliases, vec!["py", "python3"]);
        assert_eq!(python.extensions.len(), 3);

        let markdown = entries.iter().find(|e| e.id == "markdown").unwrap();
        assert_eq!(markdown.language_type, LanguageType::Prose);

        let json = entries.iter().find(|e| e.id == "json").unwrap();
        assert_eq!(json.language_type, LanguageType::Data);
        assert!(json.aliases.is_empty());
    }

    #[test]
    fn test_provider_metadata() {
        let provider = LinguistProvider::new();
        assert_eq!(provider.name(), "linguist");
        assert_eq!(provider.priority(), 10);
        assert!(provider.is_enabled());
        assert!(provider.last_updated().is_none());
    }

    #[test]
    fn test_language_id_normalization() {
        let yaml = r#"
C Sharp:
  type: programming
  extensions:
    - ".cs"
  language_id: 42
"#;
        let entries = LinguistProvider::parse_languages(yaml).unwrap();
        assert_eq!(entries[0].id, "c-sharp");
    }
}
