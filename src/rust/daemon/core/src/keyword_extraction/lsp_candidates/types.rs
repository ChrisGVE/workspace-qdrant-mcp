//! Types for LSP candidate extraction.

/// A candidate extracted from LSP or code structure analysis.
#[derive(Debug, Clone)]
pub struct LspCandidate {
    /// Normalized phrase (for semantic search)
    pub phrase: String,
    /// Original identifier (for exact search)
    pub identifier: String,
    /// Source of the candidate
    pub source: CandidateSource,
    /// Priority boost (LSP candidates rank higher)
    pub priority_boost: f64,
}

/// Source of an LSP candidate.
#[derive(Debug, Clone, PartialEq)]
pub enum CandidateSource {
    /// Extracted from public symbol definition
    PublicSymbol,
    /// Extracted from import/use statement
    Import,
    /// Extracted from frequently referenced name
    Reference,
}

/// Configuration for LSP candidate extraction.
#[derive(Debug, Clone)]
pub struct LspCandidateConfig {
    /// Boost factor for LSP candidates in combined scoring
    pub priority_boost: f64,
    /// Minimum identifier length to consider
    pub min_identifier_len: usize,
    /// Suffixes to strip from identifiers
    pub strip_suffixes: Vec<String>,
}

impl Default for LspCandidateConfig {
    fn default() -> Self {
        Self {
            priority_boost: 1.5,
            min_identifier_len: 3,
            strip_suffixes: vec![
                "Impl".to_string(),
                "Manager".to_string(),
                "Handler".to_string(),
                "Helper".to_string(),
                "Util".to_string(),
                "Utils".to_string(),
                "Factory".to_string(),
                "Builder".to_string(),
            ],
        }
    }
}
