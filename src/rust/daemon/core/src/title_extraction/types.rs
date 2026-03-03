/// Result of title extraction, including source attribution.
#[derive(Debug, Clone)]
pub struct TitleResult {
    /// The extracted title
    pub title: String,
    /// How the title was obtained
    pub source: TitleSource,
    /// Extracted authors, if available
    pub authors: Vec<String>,
}

/// How the title was obtained
#[derive(Debug, Clone, PartialEq)]
pub enum TitleSource {
    /// Extracted from embedded document metadata
    Metadata,
    /// Extracted from content heuristics (first heading, etc.)
    ContentHeuristic,
    /// Derived from filename
    FilenameFallback,
}
