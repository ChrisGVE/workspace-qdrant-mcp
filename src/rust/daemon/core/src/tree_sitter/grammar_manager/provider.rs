//! Synchronous `LanguageProvider` backed by pre-loaded grammars.

use super::GrammarManager;
use std::collections::HashMap;
use tree_sitter::Language;

/// A synchronous LanguageProvider backed by pre-loaded grammars.
///
/// This provider is created from a GrammarManager's loaded grammars and can be
/// used with SemanticChunker. Since loading grammars is an async operation,
/// this provider only returns grammars that were already loaded.
///
/// # Example
///
/// ```ignore
/// // Preload grammars (async)
/// let manager = GrammarManager::new(config);
/// manager.preload_required().await;
///
/// // Create sync provider from loaded grammars
/// let provider = manager.create_language_provider();
///
/// // Use with chunker
/// let chunker = SemanticChunker::with_provider(8000, Arc::new(provider));
/// ```
#[derive(Debug, Clone)]
pub struct LoadedGrammarsProvider {
    grammars: HashMap<String, Language>,
}

impl LoadedGrammarsProvider {
    /// Create a new provider with no loaded grammars.
    pub fn new() -> Self {
        Self {
            grammars: HashMap::new(),
        }
    }

    /// Create a provider from a map of loaded grammars.
    pub fn from_loaded(grammars: HashMap<String, Language>) -> Self {
        Self { grammars }
    }

    /// Add a grammar to the provider.
    pub fn add_grammar(&mut self, language: &str, grammar: Language) {
        self.grammars.insert(language.to_string(), grammar);
    }

    /// Get the number of loaded grammars.
    pub fn len(&self) -> usize {
        self.grammars.len()
    }

    /// Check if the provider has no loaded grammars.
    pub fn is_empty(&self) -> bool {
        self.grammars.is_empty()
    }
}

impl Default for LoadedGrammarsProvider {
    fn default() -> Self {
        Self::new()
    }
}

impl crate::tree_sitter::parser::LanguageProvider for LoadedGrammarsProvider {
    fn get_language(&self, name: &str) -> Option<Language> {
        self.grammars.get(name).cloned()
    }

    fn supports_language(&self, name: &str) -> bool {
        self.grammars.contains_key(name)
    }

    fn available_languages(&self) -> Vec<&str> {
        self.grammars.keys().map(|s| s.as_str()).collect()
    }
}

impl GrammarManager {
    /// Create a synchronous LanguageProvider from the currently loaded grammars.
    ///
    /// This takes a snapshot of the loaded grammars. Any grammars loaded after
    /// calling this method won't be available in the provider.
    pub fn create_language_provider(&self) -> LoadedGrammarsProvider {
        LoadedGrammarsProvider::from_loaded(self.loaded_grammars.clone())
    }
}
