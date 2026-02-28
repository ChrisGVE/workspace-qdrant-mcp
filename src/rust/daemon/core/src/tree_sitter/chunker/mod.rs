//! Semantic chunker that extracts meaningful code units.
//!
//! This module provides the primary interface for extracting semantic code chunks
//! from source files. It supports both static (compiled-in) and dynamic (runtime-loaded)
//! tree-sitter grammars through the `LanguageProvider` trait.

pub mod helpers;
mod splitting;
mod strategy;

#[cfg(test)]
mod tests;

use std::path::Path;
use std::sync::Arc;

use crate::error::DaemonError;
use crate::tree_sitter::parser::LanguageProvider;
use crate::tree_sitter::types::SemanticChunk;

// Re-export public items for consumers
pub use helpers::{extract_function_calls, find_child_by_kind, find_children_by_kind, node_text};
pub use splitting::text_chunk_fallback;

/// Default maximum chunk size in estimated tokens.
const DEFAULT_MAX_CHUNK_SIZE: usize = 8000;

/// Semantic chunker that extracts code units from source files.
///
/// The chunker can optionally use a `LanguageProvider` for dynamic grammar loading,
/// which allows supporting languages beyond those compiled into the binary.
pub struct SemanticChunker {
    pub(crate) max_chunk_size: usize,
    /// Optional language provider for dynamic grammar loading.
    language_provider: Option<Arc<dyn LanguageProvider>>,
}

impl SemanticChunker {
    /// Create a new semantic chunker with the specified max chunk size.
    ///
    /// Uses only statically compiled grammars.
    pub fn new(max_chunk_size: usize) -> Self {
        Self {
            max_chunk_size,
            language_provider: None,
        }
    }

    /// Create a chunker with default settings.
    ///
    /// Uses only statically compiled grammars.
    pub fn default() -> Self {
        Self::new(DEFAULT_MAX_CHUNK_SIZE)
    }

    /// Create a chunker with a language provider for dynamic grammar support.
    ///
    /// The provider is used as a fallback when a language is not available
    /// statically. This enables support for additional languages without
    /// recompilation.
    pub fn with_provider(max_chunk_size: usize, provider: Arc<dyn LanguageProvider>) -> Self {
        Self {
            max_chunk_size,
            language_provider: Some(provider),
        }
    }

    /// Set the language provider for dynamic grammar loading.
    ///
    /// Returns self for method chaining.
    pub fn set_provider(mut self, provider: Arc<dyn LanguageProvider>) -> Self {
        self.language_provider = Some(provider);
        self
    }

    /// Get the language provider, if one is configured.
    pub fn language_provider(&self) -> Option<&Arc<dyn LanguageProvider>> {
        self.language_provider.as_ref()
    }

    /// Chunk source code using the appropriate language extractor.
    ///
    /// Uses statically compiled grammars by default. If a language provider
    /// is configured, it will be used to provide grammars for languages
    /// beyond the built-in set.
    pub fn chunk_source(
        &self,
        source: &str,
        file_path: &Path,
        language: &str,
    ) -> Result<Vec<SemanticChunk>, DaemonError> {
        // Try to get dynamic grammar first if provider is available
        let dynamic_lang = strategy::get_language_from_provider(
            self.language_provider.as_deref(),
            language,
        );

        // Get the appropriate extractor
        let extractor = match strategy::create_extractor(language, dynamic_lang) {
            Some(ext) => ext,
            None => {
                // Fall back to text chunking
                return Ok(text_chunk_fallback(source, file_path, self.max_chunk_size));
            }
        };

        // Extract chunks using the language-specific extractor
        let mut chunks = extractor.extract_chunks(source, file_path)?;

        // Process chunks that exceed the size limit
        chunks = splitting::handle_oversized_chunks(chunks, source, self.max_chunk_size);

        Ok(chunks)
    }

    /// Split a single oversized chunk into fragments.
    ///
    /// Exposed for testing; production code uses `chunk_source` which calls
    /// `handle_oversized_chunks` internally.
    #[cfg(test)]
    pub(crate) fn split_oversized_chunk(&self, chunk: &SemanticChunk) -> Vec<SemanticChunk> {
        splitting::handle_oversized_chunks(vec![chunk.clone()], "", self.max_chunk_size)
    }
}
