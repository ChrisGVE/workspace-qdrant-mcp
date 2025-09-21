//! Pattern Management System for workspace-qdrant-mcp
//!
//! This module provides compile-time embedded YAML pattern loading for project detection,
//! file filtering, and language recognition. Patterns are embedded using include_str! for
//! performance and distribution simplicity.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use thiserror::Error;

pub mod manager;
pub mod comprehensive;
pub mod detection;
pub mod project;

pub use manager::PatternManager;
pub use comprehensive::{
    ComprehensivePatternManager, ComprehensiveStats, ComprehensiveResult,
    ComprehensivePatternError, InternalConfiguration, LspServerConfig,
    TreeSitterConfig, BuildSystemConfig, ContentSignaturesConfig
};
pub use detection::{
    LanguageDetector, DetectionResult, DetectionConfidence, DetectionMethod,
    DetectorStats, detect_language_from_path, detect_language_from_content
};
pub use project::{
    ProjectDetector, ProjectInfo, BuildSystemInfo, ProjectType, ProjectConfidence,
    DetectionDetails, PatternMatch, analyze_project_from_files
};

/// Errors that can occur during pattern operations
#[derive(Error, Debug)]
pub enum PatternError {
    #[error("YAML parsing error: {0}")]
    YamlParse(#[from] serde_yaml::Error),

    #[error("Pattern validation error: {0}")]
    Validation(String),

    #[error("Pattern not found: {0}")]
    NotFound(String),

    #[error("Invalid pattern format: {0}")]
    InvalidFormat(String),
}

pub type PatternResult<T> = Result<T, PatternError>;

/// Project ecosystem indicator with confidence scoring
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProjectIndicator {
    pub pattern: String,
    pub confidence: ConfidenceLevel,
    pub rationale: String,
    pub weight: u8,
}

/// Confidence levels for pattern matching
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum ConfidenceLevel {
    High,
    Medium,
    Low,
}

impl ConfidenceLevel {
    /// Get numeric weight for confidence level
    pub fn weight(&self) -> f32 {
        match self {
            ConfidenceLevel::High => 1.0,
            ConfidenceLevel::Medium => 0.8,
            ConfidenceLevel::Low => 0.6,
        }
    }
}

/// Ecosystem definition with indicators
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Ecosystem {
    pub name: String,
    pub description: String,
    pub confidence_levels: HashMap<String, String>,
    pub indicators: Vec<ProjectIndicator>,
}

/// Project indicators configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProjectIndicators {
    pub version: String,
    pub last_updated: String,
    pub research_coverage: String,
    pub ecosystems: HashMap<String, Ecosystem>,
}

/// Pattern with metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PatternWithMetadata {
    pub pattern: String,
    pub description: String,
    pub ecosystems: Vec<String>,
}

/// Exclude patterns configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExcludePatterns {
    pub build_artifacts: Vec<PatternWithMetadata>,
    pub compiled_code: Vec<PatternWithMetadata>,
    pub environments: Vec<PatternWithMetadata>,
    pub caches: Vec<PatternWithMetadata>,
    pub version_control: Vec<PatternWithMetadata>,
    pub editor_files: Vec<PatternWithMetadata>,
    pub system_files: Vec<PatternWithMetadata>,
    pub logs_and_temp: Vec<PatternWithMetadata>,
    pub media_files: Vec<PatternWithMetadata>,
    pub large_binaries: Vec<PatternWithMetadata>,
    pub security: Vec<PatternWithMetadata>,
    pub test_artifacts: Vec<PatternWithMetadata>,
}

/// Include patterns configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IncludePatterns {
    pub version: String,
    pub last_updated: String,
    pub research_coverage: String,
    pub source_code: Vec<PatternWithMetadata>,
    pub documentation: Vec<PatternWithMetadata>,
    pub configuration: Vec<PatternWithMetadata>,
    pub schema_and_data: Vec<PatternWithMetadata>,
    pub templates_and_resources: Vec<PatternWithMetadata>,
    pub project_management: Vec<PatternWithMetadata>,
    pub special_patterns: Vec<PatternWithMetadata>,
}

/// Language extension information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LanguageInfo {
    pub name: String,
    pub category: String,
    pub mime_types: Vec<String>,
    pub aliases: Vec<String>,
    pub tree_sitter_name: Option<String>,
    pub comment_syntax: Option<CommentSyntax>,
}

/// Comment syntax for a language
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CommentSyntax {
    pub line: Option<String>,
    pub block_start: Option<String>,
    pub block_end: Option<String>,
}

/// Language group for categorizing languages
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LanguageGroup {
    #[serde(default)]
    pub extensions: Vec<String>,
    #[serde(default)]
    pub filenames: Vec<String>,
    pub lsp_id: String,
    pub category: String,
}

/// Language extensions configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LanguageExtensions {
    pub programming_languages: HashMap<String, LanguageGroup>,
    pub web_technologies: HashMap<String, LanguageGroup>,
    pub markup_languages: HashMap<String, LanguageGroup>,
    pub configuration_files: HashMap<String, LanguageGroup>,
    pub shell_scripting: HashMap<String, LanguageGroup>,
    pub data_formats: HashMap<String, LanguageGroup>,
    pub specialized_formats: HashMap<String, LanguageGroup>,
    pub extensions_to_languages: HashMap<String, String>,
    pub filenames_to_languages: HashMap<String, String>,
    pub metadata: HashMap<String, serde_yaml::Value>,
}

/// All pattern types combined
#[derive(Debug, Clone)]
pub struct AllPatterns {
    pub project_indicators: ProjectIndicators,
    pub exclude_patterns: ExcludePatterns,
    pub include_patterns: IncludePatterns,
    pub language_extensions: LanguageExtensions,
}

impl AllPatterns {
    /// Get all exclude patterns as a flat vector
    pub fn all_exclude_patterns(&self) -> Vec<&PatternWithMetadata> {
        let mut patterns = Vec::new();
        patterns.extend(&self.exclude_patterns.build_artifacts);
        patterns.extend(&self.exclude_patterns.compiled_code);
        patterns.extend(&self.exclude_patterns.environments);
        patterns.extend(&self.exclude_patterns.caches);
        patterns.extend(&self.exclude_patterns.version_control);
        patterns.extend(&self.exclude_patterns.editor_files);
        patterns.extend(&self.exclude_patterns.system_files);
        patterns.extend(&self.exclude_patterns.logs_and_temp);
        patterns.extend(&self.exclude_patterns.media_files);
        patterns.extend(&self.exclude_patterns.large_binaries);
        patterns.extend(&self.exclude_patterns.security);
        patterns.extend(&self.exclude_patterns.test_artifacts);
        patterns
    }

    /// Get all include patterns as a flat vector
    pub fn all_include_patterns(&self) -> Vec<&PatternWithMetadata> {
        let mut patterns = Vec::new();
        patterns.extend(&self.include_patterns.source_code);
        patterns.extend(&self.include_patterns.documentation);
        patterns.extend(&self.include_patterns.configuration);
        patterns.extend(&self.include_patterns.schema_and_data);
        patterns.extend(&self.include_patterns.templates_and_resources);
        patterns.extend(&self.include_patterns.project_management);
        patterns.extend(&self.include_patterns.special_patterns);
        patterns
    }

    /// Get language info by file extension
    pub fn language_by_extension(&self, extension: &str) -> Option<&LanguageGroup> {
        // Search across all language categories
        for category in [
            &self.language_extensions.programming_languages,
            &self.language_extensions.web_technologies,
            &self.language_extensions.markup_languages,
            &self.language_extensions.configuration_files,
            &self.language_extensions.shell_scripting,
            &self.language_extensions.data_formats,
            &self.language_extensions.specialized_formats,
        ] {
            for (_, lang_group) in category {
                // Check extensions
                if lang_group.extensions.iter().any(|ext| ext.trim_start_matches('.') == extension) {
                    return Some(lang_group);
                }
                // Check filenames (for exact matches like "Dockerfile")
                if lang_group.filenames.iter().any(|filename| filename == extension) {
                    return Some(lang_group);
                }
            }
        }
        None
    }

    /// Get ecosystem by name
    pub fn ecosystem_by_name(&self, name: &str) -> Option<&Ecosystem> {
        self.project_indicators.ecosystems.get(name)
    }

    /// Calculate ecosystem confidence score for given indicators
    pub fn calculate_ecosystem_confidence(&self, ecosystem_name: &str, found_patterns: &[String]) -> f32 {
        if let Some(ecosystem) = self.ecosystem_by_name(ecosystem_name) {
            let mut total_score = 0.0;
            let mut max_possible_score = 0.0;

            for indicator in &ecosystem.indicators {
                let weight = indicator.weight as f32;
                let confidence_weight = indicator.confidence.weight();
                let indicator_score = weight * confidence_weight;

                max_possible_score += indicator_score;

                // Check if this indicator pattern was found
                if found_patterns.iter().any(|p| p == &indicator.pattern) {
                    total_score += indicator_score;
                }
            }

            if max_possible_score > 0.0 {
                total_score / max_possible_score
            } else {
                0.0
            }
        } else {
            0.0
        }
    }

    /// Get all ecosystem names
    pub fn ecosystem_names(&self) -> Vec<&String> {
        self.project_indicators.ecosystems.keys().collect()
    }

    /// Get all supported file extensions
    pub fn supported_extensions(&self) -> Vec<String> {
        let mut extensions = Vec::new();

        for category in [
            &self.language_extensions.programming_languages,
            &self.language_extensions.web_technologies,
            &self.language_extensions.markup_languages,
            &self.language_extensions.configuration_files,
            &self.language_extensions.shell_scripting,
            &self.language_extensions.data_formats,
            &self.language_extensions.specialized_formats,
        ] {
            for (_, lang_group) in category {
                // Add extensions
                for ext in &lang_group.extensions {
                    extensions.push(ext.trim_start_matches('.').to_string());
                }
                // Add filenames as extensions too
                for filename in &lang_group.filenames {
                    extensions.push(filename.clone());
                }
            }
        }

        extensions.sort();
        extensions.dedup();
        extensions
    }
}