//! Detection types: confidence levels, results, methods, and statistics.

/// Language detection confidence levels
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub enum DetectionConfidence {
    /// Very high confidence (exact extension match, known shebang)
    VeryHigh = 100,
    /// High confidence (common extension, typical shebang pattern)
    High = 80,
    /// Medium confidence (keyword patterns, contextual clues)
    Medium = 60,
    /// Low confidence (fallback patterns, weak indicators)
    Low = 40,
    /// Unknown (no indicators found)
    Unknown = 0,
}

/// Language detection result
#[derive(Debug, Clone)]
pub struct DetectionResult {
    pub language: Option<String>,
    pub confidence: DetectionConfidence,
    pub detection_method: DetectionMethod,
    pub details: String,
}

/// Detection method used to identify the language
#[derive(Debug, Clone)]
pub enum DetectionMethod {
    /// Detected via file extension
    Extension(String),
    /// Detected via shebang line
    Shebang(String),
    /// Detected via content keywords
    Keywords(Vec<String>),
    /// Detected via filename pattern
    Filename(String),
    /// Multiple methods agreed
    Consensus(Vec<DetectionMethod>),
    /// No detection possible
    None,
}

/// Detector statistics
#[derive(Debug, Clone)]
pub struct DetectorStats {
    pub total_extensions: usize,
    pub case_insensitive_extensions: usize,
    pub shebang_patterns: usize,
    pub keyword_patterns: usize,
    pub unique_languages: usize,
}
