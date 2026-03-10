//! Language registry — merges and caches language definitions from providers.
//!
//! The registry loads language definitions from multiple sources in priority
//! order: user local overrides > provider-fetched data > bundled defaults.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use tokio::sync::RwLock;

use crate::error::DaemonError;

use super::provider::LanguageSourceProvider;
use super::types::{GrammarSourceEntry, LanguageDefinition, LanguageMap, SourceMetadata};

/// The central language registry.
///
/// Holds merged language definitions from all registered providers plus
/// user-local overrides. Thread-safe via `RwLock` for concurrent read access.
pub struct LanguageRegistry {
    /// Merged language definitions keyed by lowercase language ID.
    languages: Arc<RwLock<LanguageMap>>,
    /// Registered providers sorted by priority (lowest first).
    providers: Vec<Box<dyn LanguageSourceProvider>>,
    /// Directory for user-local language override YAML files.
    user_dir: Option<PathBuf>,
}

impl LanguageRegistry {
    /// Create a new empty registry.
    pub fn new() -> Self {
        Self {
            languages: Arc::new(RwLock::new(HashMap::new())),
            providers: Vec::new(),
            user_dir: None,
        }
    }

    /// Set the directory for user-local language override files.
    pub fn with_user_dir(mut self, dir: PathBuf) -> Self {
        self.user_dir = Some(dir);
        self
    }

    /// Register a provider. Providers are sorted by priority on registration.
    pub fn register_provider(&mut self, provider: Box<dyn LanguageSourceProvider>) {
        self.providers.push(provider);
        self.providers.sort_by_key(|p| p.priority());
    }

    /// Load and merge all data from registered providers + user overrides.
    ///
    /// This is the primary initialization method. Call once at startup,
    /// then use `refresh()` for incremental updates.
    pub async fn load(&self) -> Result<(), DaemonError> {
        let mut merged: LanguageMap = HashMap::new();

        // Load from providers in priority order (lowest priority first,
        // so higher-priority providers overwrite).
        let mut sorted_providers: Vec<&dyn LanguageSourceProvider> =
            self.providers.iter().map(|p| p.as_ref()).collect();
        sorted_providers.sort_by_key(|p| std::cmp::Reverse(p.priority()));

        for provider in sorted_providers {
            if !provider.is_enabled() {
                continue;
            }
            let data = match provider.refresh().await {
                Ok(d) => d,
                Err(e) => {
                    tracing::warn!(
                        provider = provider.name(),
                        error = %e,
                        "Failed to load from provider, skipping"
                    );
                    continue;
                }
            };

            // Merge language entries
            for lang_entry in &data.languages {
                let id = lang_entry.id.to_lowercase();
                let def = merged
                    .entry(id.clone())
                    .or_insert_with(|| LanguageDefinition {
                        language: lang_entry.name.clone(),
                        aliases: Vec::new(),
                        extensions: Vec::new(),
                        language_type: lang_entry.language_type,
                        grammar: Default::default(),
                        semantic_patterns: None,
                        lsp_servers: Vec::new(),
                        sources: SourceMetadata::default(),
                    });

                // Higher priority provider overwrites identity fields
                def.language = lang_entry.name.clone();
                if !lang_entry.aliases.is_empty() {
                    def.aliases = lang_entry.aliases.clone();
                }
                if !lang_entry.extensions.is_empty() {
                    def.extensions = lang_entry.extensions.clone();
                }
                def.language_type = lang_entry.language_type;
                def.sources.language = Some(provider.name().to_string());
            }

            // Merge grammar entries
            for grammar in &data.grammars {
                let id = grammar.language.to_lowercase();
                let def = merged
                    .entry(id.clone())
                    .or_insert_with(|| LanguageDefinition {
                        language: grammar.language.clone(),
                        aliases: Vec::new(),
                        extensions: Vec::new(),
                        language_type: super::types::LanguageType::Programming,
                        grammar: Default::default(),
                        semantic_patterns: None,
                        lsp_servers: Vec::new(),
                        sources: SourceMetadata::default(),
                    });

                // Add grammar source if not already present (dedup by repo)
                let already_has = def.grammar.sources.iter().any(|s| s.repo == grammar.repo);
                if !already_has {
                    def.grammar.sources.push(GrammarSourceEntry {
                        repo: grammar.repo.clone(),
                        origin: Some(provider.name().to_string()),
                        quality: grammar.quality,
                    });
                }

                // Set grammar build metadata from highest-priority provider
                def.grammar.has_cpp_scanner = grammar.has_cpp_scanner;
                if grammar.src_subdir.is_some() {
                    def.grammar.src_subdir = grammar.src_subdir.clone();
                }
                if grammar.symbol_name.is_some() {
                    def.grammar.symbol_name = grammar.symbol_name.clone();
                }
                if grammar.archive_branch.is_some() {
                    def.grammar.archive_branch = grammar.archive_branch.clone();
                }

                def.sources.grammar = Some(provider.name().to_string());
            }

            // Merge LSP entries
            for lsp in &data.lsp_servers {
                let id = lsp.language.to_lowercase();
                let def = merged
                    .entry(id.clone())
                    .or_insert_with(|| LanguageDefinition {
                        language: lsp.language.clone(),
                        aliases: Vec::new(),
                        extensions: Vec::new(),
                        language_type: super::types::LanguageType::Programming,
                        grammar: Default::default(),
                        semantic_patterns: None,
                        lsp_servers: Vec::new(),
                        sources: SourceMetadata::default(),
                    });

                // Dedup LSP servers by binary name
                let already_has = def
                    .lsp_servers
                    .iter()
                    .any(|s| s.binary == lsp.server.binary);
                if !already_has {
                    def.lsp_servers.push(lsp.server.clone());
                }

                def.sources.lsp = Some(provider.name().to_string());
            }
        }

        // Sort grammar sources by quality (curated first)
        for def in merged.values_mut() {
            def.grammar.sources.sort_by_key(|s| s.quality);
            def.lsp_servers.sort_by_key(|s| s.priority);
        }

        // Load semantic patterns from providers with full definitions
        self.load_full_definition_extras(&mut merged);

        // Load user overrides last (highest precedence)
        if let Some(ref user_dir) = self.user_dir {
            self.load_user_overrides(user_dir, &mut merged).await;
        }

        *self.languages.write().await = merged;
        Ok(())
    }

    /// Load semantic patterns from providers that have full definitions.
    ///
    /// Standard provider methods return flat entries (LanguageEntry, GrammarEntry,
    /// LspEntry). Providers with `full_definitions()` (like RegistryProvider) can
    /// also contribute semantic_patterns, which the merge loop doesn't cover.
    fn load_full_definition_extras(&self, merged: &mut LanguageMap) {
        // Apply in reverse priority order so higher-priority providers win
        let mut providers_with_defs: Vec<_> = self
            .providers
            .iter()
            .filter(|p| p.full_definitions().is_some())
            .collect();
        providers_with_defs.sort_by_key(|p| std::cmp::Reverse(p.priority()));

        for provider in providers_with_defs {
            if let Some(defs) = provider.full_definitions() {
                for def in defs {
                    let id = def.id();
                    if let Some(merged_def) = merged.get_mut(&id) {
                        // Only set semantic_patterns if not already set by a
                        // higher-priority provider
                        if merged_def.semantic_patterns.is_none() {
                            merged_def.semantic_patterns = def.semantic_patterns.clone();
                        }
                    }
                }
            }
        }
    }

    /// Load user-local YAML overrides from the given directory.
    async fn load_user_overrides(&self, dir: &Path, merged: &mut LanguageMap) {
        if !dir.exists() {
            return;
        }

        let entries = match std::fs::read_dir(dir) {
            Ok(e) => e,
            Err(e) => {
                tracing::warn!(dir = %dir.display(), error = %e, "Failed to read user language dir");
                return;
            }
        };

        for entry in entries.flatten() {
            let path = entry.path();
            if path.extension().is_some_and(|e| e == "yaml" || e == "yml") {
                match std::fs::read_to_string(&path) {
                    Ok(content) => match serde_yaml_ng::from_str::<LanguageDefinition>(&content) {
                        Ok(def) => {
                            let id = def.id();
                            tracing::info!(
                                language = %id,
                                path = %path.display(),
                                "Loaded user language override"
                            );
                            merged.insert(id, def);
                        }
                        Err(e) => {
                            tracing::warn!(
                                path = %path.display(),
                                error = %e,
                                "Failed to parse user language YAML"
                            );
                        }
                    },
                    Err(e) => {
                        tracing::warn!(
                            path = %path.display(),
                            error = %e,
                            "Failed to read user language file"
                        );
                    }
                }
            }
        }
    }

    /// Refresh all enabled providers and re-merge.
    pub async fn refresh(&self) -> Result<(), DaemonError> {
        self.load().await
    }

    /// Get a language definition by ID or alias.
    pub async fn get(&self, language: &str) -> Option<LanguageDefinition> {
        let languages = self.languages.read().await;
        let normalized = language.to_lowercase();

        // Direct ID match
        if let Some(def) = languages.get(&normalized) {
            return Some(def.clone());
        }

        // Search by alias
        for def in languages.values() {
            if def.aliases.iter().any(|a| a.to_lowercase() == normalized) {
                return Some(def.clone());
            }
        }

        None
    }

    /// Get all language definitions.
    pub async fn all(&self) -> LanguageMap {
        self.languages.read().await.clone()
    }

    /// Get a list of all language IDs.
    pub async fn language_ids(&self) -> Vec<String> {
        self.languages.read().await.keys().cloned().collect()
    }

    /// Get languages that have grammar sources available.
    pub async fn languages_with_grammars(&self) -> Vec<LanguageDefinition> {
        self.languages
            .read()
            .await
            .values()
            .filter(|d| d.has_grammar())
            .cloned()
            .collect()
    }

    /// Get languages that have semantic patterns defined.
    pub async fn languages_with_patterns(&self) -> Vec<LanguageDefinition> {
        self.languages
            .read()
            .await
            .values()
            .filter(|d| d.has_semantic_patterns())
            .cloned()
            .collect()
    }

    /// Get the number of registered languages.
    pub async fn count(&self) -> usize {
        self.languages.read().await.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::language_registry::providers::registry::RegistryProvider;
    use crate::language_registry::types::GrammarQuality;

    #[tokio::test]
    async fn test_registry_loads_bundled() {
        let mut registry = LanguageRegistry::new();
        let bundled = RegistryProvider::new().unwrap();
        registry.register_provider(Box::new(bundled));

        registry.load().await.unwrap();
        let count = registry.count().await;
        assert!(count >= 40, "Expected 40+ languages, got {count}");
    }

    #[tokio::test]
    async fn test_registry_get_by_id() {
        let mut registry = LanguageRegistry::new();
        registry.register_provider(Box::new(RegistryProvider::new().unwrap()));
        registry.load().await.unwrap();

        let rust = registry.get("rust").await;
        assert!(rust.is_some());
        let rust = rust.unwrap();
        assert!(rust.has_grammar());
    }

    #[tokio::test]
    async fn test_registry_get_by_alias() {
        let mut registry = LanguageRegistry::new();
        registry.register_provider(Box::new(RegistryProvider::new().unwrap()));
        registry.load().await.unwrap();

        // "py" is an alias for python
        let python = registry.get("py").await;
        assert!(python.is_some());
    }

    #[tokio::test]
    async fn test_registry_unknown_language() {
        let mut registry = LanguageRegistry::new();
        registry.register_provider(Box::new(RegistryProvider::new().unwrap()));
        registry.load().await.unwrap();

        let unknown = registry.get("brainfuck").await;
        assert!(unknown.is_none());
    }

    #[tokio::test]
    async fn test_grammar_sources_sorted_by_quality() {
        let mut registry = LanguageRegistry::new();
        registry.register_provider(Box::new(RegistryProvider::new().unwrap()));
        registry.load().await.unwrap();

        let languages = registry.all().await;
        for def in languages.values() {
            // Verify sources are sorted: Curated < Official < Community
            let qualities: Vec<GrammarQuality> =
                def.grammar.sources.iter().map(|s| s.quality).collect();
            let mut sorted = qualities.clone();
            sorted.sort();
            assert_eq!(
                qualities, sorted,
                "Grammar sources for {} not sorted by quality",
                def.language
            );
        }
    }

    #[tokio::test]
    async fn test_user_override_takes_precedence() {
        let tmp_dir = tempfile::tempdir().unwrap();
        let user_yaml = tmp_dir.path().join("test-lang.yaml");
        std::fs::write(
            &user_yaml,
            r#"
language: test-lang
aliases: [tl]
extensions: [".tl"]
type: programming
grammar:
  sources: []
"#,
        )
        .unwrap();

        let mut registry = LanguageRegistry::new().with_user_dir(tmp_dir.path().to_path_buf());
        registry.register_provider(Box::new(RegistryProvider::new().unwrap()));
        registry.load().await.unwrap();

        let test_lang = registry.get("test-lang").await;
        assert!(test_lang.is_some());
        assert_eq!(test_lang.unwrap().extensions, vec![".tl"]);
    }

    #[tokio::test]
    async fn test_languages_with_grammars() {
        let mut registry = LanguageRegistry::new();
        registry.register_provider(Box::new(RegistryProvider::new().unwrap()));
        registry.load().await.unwrap();

        let with_grammars = registry.languages_with_grammars().await;
        assert!(with_grammars.len() >= 40);
    }
}
