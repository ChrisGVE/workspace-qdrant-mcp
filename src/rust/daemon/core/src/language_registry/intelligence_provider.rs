//! Abstraction layer for LSP and future code intelligence sources.
//!
//! The `CodeIntelligenceProvider` trait allows multiple intelligence backends
//! (LSP, tree-sitter queries, static analysis) to enrich code metadata.

use std::path::Path;

use async_trait::async_trait;

use crate::error::DaemonError;

/// Capabilities a code intelligence provider can offer.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Capability {
    /// Find references to a symbol.
    References,
    /// Get type information for a symbol.
    TypeInfo,
    /// Get hover documentation.
    Hover,
    /// Get diagnostics (errors, warnings).
    Diagnostics,
    /// Semantic token highlighting.
    SemanticTokens,
    /// Go to definition.
    Definition,
    /// Find implementations of a trait/interface.
    Implementation,
    /// Code completion.
    Completion,
    /// Rename symbol.
    Rename,
}

/// Enriched symbol information from a code intelligence provider.
#[derive(Debug, Clone, Default)]
pub struct EnrichedSymbols {
    /// Resolved type annotations for symbols.
    pub type_annotations: Vec<SymbolTypeInfo>,
    /// Cross-references (definition ↔ usage links).
    pub references: Vec<SymbolReference>,
}

/// Type information for a symbol.
#[derive(Debug, Clone)]
pub struct SymbolTypeInfo {
    /// Symbol name.
    pub name: String,
    /// Resolved type string (e.g., "fn(i32) -> String").
    pub type_str: String,
    /// Line number in the source file.
    pub line: usize,
}

/// A cross-reference from one symbol to another.
#[derive(Debug, Clone)]
pub struct SymbolReference {
    /// Source symbol name.
    pub from_symbol: String,
    /// Target symbol name.
    pub to_symbol: String,
    /// File path of the reference target.
    pub target_file: String,
    /// Line number of the reference target.
    pub target_line: usize,
}

/// Abstraction over code intelligence backends (LSP, static analysis, etc.).
///
/// This trait enables pluggable intelligence sources that can enrich
/// semantic chunks with type information, cross-references, and more.
#[async_trait]
pub trait CodeIntelligenceProvider: Send + Sync {
    /// Provider name (e.g., "lsp", "tree-sitter-queries").
    fn name(&self) -> &str;

    /// Capabilities this provider offers.
    fn capabilities(&self) -> Vec<Capability>;

    /// Check if this provider is available for a project.
    async fn is_available(&self, project_root: &Path) -> bool;

    /// Start the provider for a project.
    async fn start(&self, project_root: &Path) -> Result<(), DaemonError>;

    /// Stop the provider.
    async fn stop(&self) -> Result<(), DaemonError>;

    /// Enrich symbols in a file with additional intelligence data.
    async fn enrich(
        &self,
        file_path: &Path,
        symbols: &[String],
    ) -> Result<EnrichedSymbols, DaemonError>;
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Verify the trait is object-safe.
    fn _assert_object_safe(_: &dyn CodeIntelligenceProvider) {}

    #[test]
    fn test_enriched_symbols_default() {
        let enriched = EnrichedSymbols::default();
        assert!(enriched.type_annotations.is_empty());
        assert!(enriched.references.is_empty());
    }
}
