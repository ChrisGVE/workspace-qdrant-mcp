//! Abstraction layer decoupling the daemon from tree-sitter details.
//!
//! The `SyntaxProvider` trait allows future enrichment or replacement of
//! tree-sitter with alternative parsing backends, while the current
//! `TreeSitterSyntaxProvider` wraps the existing grammar pipeline.

use std::path::Path;

use async_trait::async_trait;

use crate::error::DaemonError;
use crate::tree_sitter::types::SemanticChunk;

use super::types::SemanticPatterns;

/// Abstraction over a syntax parsing backend (currently tree-sitter).
///
/// This trait decouples the daemon from tree-sitter implementation details,
/// enabling future enrichment or replacement without changing the extraction
/// pipeline.
#[async_trait]
pub trait SyntaxProvider: Send + Sync {
    /// Provider name (e.g., "tree-sitter").
    fn name(&self) -> &str;

    /// List languages that are currently available (grammar installed).
    fn available_languages(&self) -> Vec<String>;

    /// Check if a language grammar is installed and ready.
    fn is_available(&self, language: &str) -> bool;

    /// Install a grammar for a language (download + compile).
    async fn install_language(&self, language: &str) -> Result<(), DaemonError>;

    /// Get the ABI version of an installed grammar.
    fn language_version(&self, language: &str) -> Option<String>;

    /// Parse source code and extract semantic chunks using patterns.
    ///
    /// This is the main entry point for the generic extraction pipeline.
    /// If `patterns` is provided, uses the generic AST walker; otherwise
    /// falls back to language-specific extractors or text chunking.
    fn extract_chunks(
        &self,
        source: &str,
        language: &str,
        file_path: &Path,
        patterns: Option<&SemanticPatterns>,
    ) -> Result<Vec<SemanticChunk>, DaemonError>;
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Verify the trait is object-safe.
    fn _assert_object_safe(_: &dyn SyntaxProvider) {}
}
