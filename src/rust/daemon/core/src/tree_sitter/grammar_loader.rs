//! Dynamic grammar loading using libloading.
//!
//! This module provides functionality to load tree-sitter grammars from
//! shared library files (.so/.dylib/.dll) at runtime, enabling language
//! support to be added without recompiling the daemon.
//!
//! # Symbol Convention
//!
//! Grammar libraries must export a C function with the name:
//! `tree_sitter_{language}` (e.g., `tree_sitter_rust`, `tree_sitter_python`)
//!
//! This function returns a pointer to the language structure that tree-sitter uses.

use libloading::{Library, Symbol};
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use thiserror::Error;
use tree_sitter::Language;

use super::grammar_cache::{grammar_filename, GrammarCachePaths};

/// Errors that can occur during grammar loading.
#[derive(Debug, Error)]
pub enum GrammarLoadError {
    #[error("Grammar not found for language: {0}")]
    NotFound(String),

    #[error("Failed to load library: {0}")]
    LibraryLoadFailed(String),

    #[error("Symbol not found in library: {0}")]
    SymbolNotFound(String),

    #[error("IO error: {0}")]
    IoError(#[from] std::io::Error),

    #[error("Library error: {0}")]
    LibLoadingError(String),
}

/// Result type for grammar loading operations.
pub type GrammarResult<T> = Result<T, GrammarLoadError>;

/// A loaded grammar library that provides a tree-sitter Language.
///
/// This struct keeps the loaded library alive as long as the grammar is in use.
/// Dropping this struct will unload the library.
pub struct LoadedGrammar {
    /// The tree-sitter Language instance.
    pub language: Language,
    /// The loaded library (kept alive to prevent unloading).
    _library: Arc<Library>,
    /// Language name (e.g., "rust", "python").
    pub name: String,
    /// Path to the loaded library file.
    pub path: PathBuf,
}

impl std::fmt::Debug for LoadedGrammar {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("LoadedGrammar")
            .field("name", &self.name)
            .field("path", &self.path)
            .finish()
    }
}

/// Dynamic grammar loader using libloading.
///
/// This struct manages loading tree-sitter grammars from shared library files
/// and keeps track of loaded libraries to ensure they remain valid while in use.
pub struct GrammarLoader {
    /// Grammar cache paths for locating grammar files.
    cache_paths: GrammarCachePaths,
    /// Loaded libraries indexed by language name.
    /// Using Arc to allow sharing the library across multiple grammars.
    loaded_libraries: HashMap<String, Arc<Library>>,
}

impl GrammarLoader {
    /// Create a new grammar loader with the specified cache paths.
    pub fn new(cache_paths: GrammarCachePaths) -> Self {
        Self {
            cache_paths,
            loaded_libraries: HashMap::new(),
        }
    }

    /// Create a grammar loader with default cache paths.
    pub fn with_default_paths(tree_sitter_version: impl Into<String>) -> Self {
        Self::new(GrammarCachePaths::new(tree_sitter_version))
    }

    /// Load a grammar for the specified language.
    ///
    /// If the grammar is already loaded, returns a clone of the existing Language.
    /// Otherwise, loads the grammar from the cache directory.
    ///
    /// # Arguments
    ///
    /// * `language` - The language name (e.g., "rust", "python")
    ///
    /// # Returns
    ///
    /// A `LoadedGrammar` containing the tree-sitter Language and metadata.
    pub fn load_grammar(&mut self, language: &str) -> GrammarResult<LoadedGrammar> {
        // Check if already loaded
        if let Some(library) = self.loaded_libraries.get(language) {
            // Re-extract the language from the already-loaded library
            return self.extract_language(language, library.clone(), self.cache_paths.grammar_path(language));
        }

        // Find the grammar file
        let grammar_path = self.cache_paths.grammar_path(language);
        if !grammar_path.exists() {
            return Err(GrammarLoadError::NotFound(language.to_string()));
        }

        // Load the library
        let library = self.load_library(&grammar_path)?;
        let library = Arc::new(library);

        // Store the library to keep it alive
        self.loaded_libraries.insert(language.to_string(), library.clone());

        // Extract the language
        self.extract_language(language, library, grammar_path)
    }

    /// Load a grammar from a specific path (for testing or custom locations).
    pub fn load_grammar_from_path(&mut self, language: &str, path: &Path) -> GrammarResult<LoadedGrammar> {
        if !path.exists() {
            return Err(GrammarLoadError::NotFound(path.display().to_string()));
        }

        let library = self.load_library(path)?;
        let library = Arc::new(library);

        // Store with a path-based key to avoid conflicts
        let key = format!("{}:{}", language, path.display());
        self.loaded_libraries.insert(key, library.clone());

        self.extract_language(language, library, path.to_path_buf())
    }

    /// Check if a grammar is already loaded.
    pub fn is_loaded(&self, language: &str) -> bool {
        self.loaded_libraries.contains_key(language)
    }

    /// Get the list of currently loaded languages.
    pub fn loaded_languages(&self) -> Vec<&str> {
        self.loaded_libraries.keys().map(|s| s.as_str()).collect()
    }

    /// Unload a grammar (the library will be unloaded when the last reference is dropped).
    pub fn unload_grammar(&mut self, language: &str) -> bool {
        self.loaded_libraries.remove(language).is_some()
    }

    /// Unload all grammars.
    pub fn unload_all(&mut self) {
        self.loaded_libraries.clear();
    }

    /// Get the cache paths.
    pub fn cache_paths(&self) -> &GrammarCachePaths {
        &self.cache_paths
    }

    /// Load a shared library from the given path.
    fn load_library(&self, path: &Path) -> GrammarResult<Library> {
        // SAFETY: Loading a shared library is inherently unsafe.
        // We trust that grammar libraries from our cache are well-formed.
        unsafe {
            Library::new(path).map_err(|e| {
                GrammarLoadError::LibLoadingError(format!(
                    "Failed to load {}: {}",
                    path.display(),
                    e
                ))
            })
        }
    }

    /// Extract the Language from a loaded library.
    fn extract_language(
        &self,
        language: &str,
        library: Arc<Library>,
        path: PathBuf,
    ) -> GrammarResult<LoadedGrammar> {
        // Build the symbol name: tree_sitter_rust, tree_sitter_python, etc.
        let symbol_name = format!("tree_sitter_{}", language);
        let symbol_bytes = symbol_name.as_bytes();

        // Get the symbol from the library
        // The symbol is a function that returns a *const Language
        type LanguageFn = unsafe extern "C" fn() -> Language;

        let language_fn: Symbol<LanguageFn> = unsafe {
            library.get(symbol_bytes).map_err(|e| {
                GrammarLoadError::SymbolNotFound(format!(
                    "Symbol '{}' not found: {}",
                    symbol_name, e
                ))
            })?
        };

        // Call the function to get the Language
        let ts_language = unsafe { language_fn() };

        Ok(LoadedGrammar {
            language: ts_language,
            _library: library,
            name: language.to_string(),
            path,
        })
    }
}

impl std::fmt::Debug for GrammarLoader {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("GrammarLoader")
            .field("cache_paths", &self.cache_paths)
            .field("loaded_languages", &self.loaded_languages())
            .finish()
    }
}

/// Get the expected symbol name for a language.
pub fn grammar_symbol_name(language: &str) -> String {
    format!("tree_sitter_{}", language)
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    #[test]
    fn test_grammar_symbol_name() {
        assert_eq!(grammar_symbol_name("rust"), "tree_sitter_rust");
        assert_eq!(grammar_symbol_name("python"), "tree_sitter_python");
        assert_eq!(grammar_symbol_name("javascript"), "tree_sitter_javascript");
    }

    #[test]
    fn test_grammar_loader_creation() {
        let temp_dir = TempDir::new().unwrap();
        let cache_paths = GrammarCachePaths::with_root(temp_dir.path(), "0.24");
        let loader = GrammarLoader::new(cache_paths);

        assert!(loader.loaded_languages().is_empty());
    }

    #[test]
    fn test_grammar_loader_with_default_paths() {
        let loader = GrammarLoader::with_default_paths("0.24");
        assert!(loader.loaded_languages().is_empty());
    }

    #[test]
    fn test_load_grammar_not_found() {
        let temp_dir = TempDir::new().unwrap();
        let cache_paths = GrammarCachePaths::with_root(temp_dir.path(), "0.24");
        let mut loader = GrammarLoader::new(cache_paths);

        let result = loader.load_grammar("nonexistent");
        assert!(result.is_err());

        if let Err(GrammarLoadError::NotFound(lang)) = result {
            assert_eq!(lang, "nonexistent");
        } else {
            panic!("Expected NotFound error");
        }
    }

    #[test]
    fn test_is_loaded() {
        let temp_dir = TempDir::new().unwrap();
        let cache_paths = GrammarCachePaths::with_root(temp_dir.path(), "0.24");
        let loader = GrammarLoader::new(cache_paths);

        assert!(!loader.is_loaded("rust"));
    }

    #[test]
    fn test_unload_grammar() {
        let temp_dir = TempDir::new().unwrap();
        let cache_paths = GrammarCachePaths::with_root(temp_dir.path(), "0.24");
        let mut loader = GrammarLoader::new(cache_paths);

        // Unloading a non-existent grammar should return false
        assert!(!loader.unload_grammar("rust"));
    }

    #[test]
    fn test_unload_all() {
        let temp_dir = TempDir::new().unwrap();
        let cache_paths = GrammarCachePaths::with_root(temp_dir.path(), "0.24");
        let mut loader = GrammarLoader::new(cache_paths);

        loader.unload_all();
        assert!(loader.loaded_languages().is_empty());
    }

    // Note: Testing actual grammar loading requires real grammar .so/.dylib files.
    // These tests verify the API and error handling without requiring real grammar files.
    // Integration tests with real grammars should be in a separate test suite.
}
