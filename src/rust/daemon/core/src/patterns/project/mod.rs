//! Project detection system for identifying build systems and project ecosystems
//!
//! This module provides comprehensive project detection capabilities using the
//! embedded configuration data. Detects languages, build systems, frameworks,
//! and project characteristics from file patterns and directory structures.

mod detector;
mod helpers;
#[cfg(test)]
mod tests;

pub use detector::ProjectDetector;

/// Project detection confidence levels
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum ProjectConfidence {
    /// Very high confidence (multiple strong indicators)
    VeryHigh = 100,
    /// High confidence (strong primary indicator)
    High = 80,
    /// Medium confidence (some indicators present)
    Medium = 60,
    /// Low confidence (weak or ambiguous indicators)
    Low = 40,
    /// Unknown (no clear indicators)
    Unknown = 0,
}

/// Detected project information
#[derive(Debug, Clone)]
pub struct ProjectInfo {
    /// Primary language detected
    pub primary_language: Option<String>,
    /// All languages found in the project
    pub languages: Vec<String>,
    /// Build system(s) detected
    pub build_systems: Vec<BuildSystemInfo>,
    /// Frameworks detected
    pub frameworks: Vec<String>,
    /// Project type (e.g., "library", "application", "monorepo")
    pub project_type: ProjectType,
    /// Overall detection confidence
    pub confidence: ProjectConfidence,
    /// Detailed detection reasoning
    pub detection_details: DetectionDetails,
}

/// Build system information
#[derive(Debug, Clone)]
pub struct BuildSystemInfo {
    pub name: String,
    pub language: String,
    pub config_files: Vec<String>,
    pub commands: Vec<String>,
    pub confidence: ProjectConfidence,
}

/// Project type classification
#[derive(Debug, Clone, PartialEq)]
pub enum ProjectType {
    /// Single language application
    Application,
    /// Reusable library or package
    Library,
    /// Multiple projects in one repository
    Monorepo,
    /// Documentation project
    Documentation,
    /// Configuration or infrastructure
    Configuration,
    /// Unknown or mixed type
    Unknown,
}

/// Detailed detection information
#[derive(Debug, Clone)]
pub struct DetectionDetails {
    /// Files analyzed
    pub files_analyzed: usize,
    /// Pattern matches found
    pub pattern_matches: Vec<PatternMatch>,
    /// Detection methods used
    pub methods_used: Vec<String>,
    /// Reasoning for final decision
    pub reasoning: String,
}

/// Pattern match result
#[derive(Debug, Clone)]
pub struct PatternMatch {
    pub pattern: String,
    pub matched_files: Vec<String>,
    pub confidence: ProjectConfidence,
    pub category: String,
}

/// Convenient function for quick project analysis
pub fn analyze_project_from_files(files: &[String]) -> ProjectInfo {
    match ProjectDetector::global() {
        Ok(detector) => detector.analyze_project(files),
        Err(e) => ProjectInfo {
            primary_language: None,
            languages: Vec::new(),
            build_systems: Vec::new(),
            frameworks: Vec::new(),
            project_type: ProjectType::Unknown,
            confidence: ProjectConfidence::Unknown,
            detection_details: DetectionDetails {
                files_analyzed: files.len(),
                pattern_matches: Vec::new(),
                methods_used: Vec::new(),
                reasoning: format!("Detector initialization failed: {}", e),
            },
        },
    }
}
