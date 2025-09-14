//! PatternManager - Compile-time YAML pattern embedding and runtime access
//!
//! This module provides the PatternManager struct that embeds YAML pattern files
//! at compile time using include_str! macro and provides efficient runtime access
//! to parsed pattern data.

use super::{
    AllPatterns, ExcludePatterns, IncludePatterns, LanguageExtensions, PatternError,
    PatternResult, ProjectIndicators,
};
use once_cell::sync::Lazy;
use std::sync::Arc;

/// Embedded YAML content - loaded at compile time
struct EmbeddedPatterns {
    project_indicators: &'static str,
    exclude_patterns: &'static str,
    include_patterns: &'static str,
    language_extensions: &'static str,
}

/// Compile-time embedded pattern files using include_str! macro
static EMBEDDED_PATTERNS: EmbeddedPatterns = EmbeddedPatterns {
    project_indicators: include_str!("../../../../../../patterns/project_indicators.yaml"),
    exclude_patterns: include_str!("../../../../../../patterns/exclude_patterns.yaml"),
    include_patterns: include_str!("../../../../../../patterns/include_patterns.yaml"),
    language_extensions: include_str!("../../../../../../patterns/language_extensions.yaml"),
};

/// Global parsed patterns - lazily initialized on first access
static PARSED_PATTERNS: Lazy<Result<Arc<AllPatterns>, PatternError>> = Lazy::new(|| {
    let project_indicators: ProjectIndicators = serde_yaml::from_str(EMBEDDED_PATTERNS.project_indicators)
        .map_err(|e| PatternError::YamlParse(e))?;

    let exclude_patterns: ExcludePatterns = serde_yaml::from_str(EMBEDDED_PATTERNS.exclude_patterns)
        .map_err(|e| PatternError::YamlParse(e))?;

    let include_patterns: IncludePatterns = serde_yaml::from_str(EMBEDDED_PATTERNS.include_patterns)
        .map_err(|e| PatternError::YamlParse(e))?;

    let language_extensions: LanguageExtensions = serde_yaml::from_str(EMBEDDED_PATTERNS.language_extensions)
        .map_err(|e| PatternError::YamlParse(e))?;

    let all_patterns = AllPatterns {
        project_indicators,
        exclude_patterns,
        include_patterns,
        language_extensions,
    };

    // Validate patterns after loading
    validate_patterns(&all_patterns)?;

    Ok(Arc::new(all_patterns))
});

/// PatternManager provides efficient access to embedded pattern data
#[derive(Debug, Clone)]
pub struct PatternManager {
    patterns: Arc<AllPatterns>,
}

impl PatternManager {
    /// Create a new PatternManager with embedded patterns
    ///
    /// # Errors
    /// Returns an error if the embedded YAML patterns cannot be parsed or are invalid
    pub fn new() -> PatternResult<Self> {
        match PARSED_PATTERNS.as_ref() {
            Ok(patterns) => Ok(Self {
                patterns: Arc::clone(patterns),
            }),
            Err(e) => Err(PatternError::Validation(format!(
                "Failed to load embedded patterns: {}",
                e
            ))),
        }
    }

    /// Get all patterns
    pub fn patterns(&self) -> &AllPatterns {
        &self.patterns
    }

    /// Get project indicators for ecosystem detection
    pub fn project_indicators(&self) -> &ProjectIndicators {
        &self.patterns.project_indicators
    }

    /// Get exclude patterns for file filtering
    pub fn exclude_patterns(&self) -> &ExcludePatterns {
        &self.patterns.exclude_patterns
    }

    /// Get include patterns for file selection
    pub fn include_patterns(&self) -> &IncludePatterns {
        &self.patterns.include_patterns
    }

    /// Get language extensions for file type detection
    pub fn language_extensions(&self) -> &LanguageExtensions {
        &self.patterns.language_extensions
    }

    /// Check if a file path should be excluded based on patterns
    pub fn should_exclude(&self, file_path: &str) -> bool {
        let exclude_patterns = self.patterns.all_exclude_patterns();

        for pattern_meta in exclude_patterns {
            if glob_match(&pattern_meta.pattern, file_path) {
                tracing::debug!(
                    "File '{}' excluded by pattern '{}': {}",
                    file_path,
                    pattern_meta.pattern,
                    pattern_meta.description
                );
                return true;
            }
        }

        false
    }

    /// Check if a file path should be included based on patterns
    pub fn should_include(&self, file_path: &str) -> bool {
        let include_patterns = self.patterns.all_include_patterns();

        for pattern_meta in include_patterns {
            if glob_match(&pattern_meta.pattern, file_path) {
                tracing::debug!(
                    "File '{}' included by pattern '{}': {}",
                    file_path,
                    pattern_meta.pattern,
                    pattern_meta.description
                );
                return true;
            }
        }

        false
    }

    /// Detect ecosystem for a given list of file paths
    pub fn detect_ecosystem(&self, file_paths: &[String]) -> Option<(String, f32)> {
        let mut best_ecosystem: Option<(String, f32)> = None;

        for ecosystem_name in self.patterns.ecosystem_names() {
            let confidence = self.patterns.calculate_ecosystem_confidence(ecosystem_name, file_paths);

            if confidence > 0.0 {
                match &best_ecosystem {
                    Some((_, best_confidence)) => {
                        if confidence > *best_confidence {
                            best_ecosystem = Some((ecosystem_name.clone(), confidence));
                        }
                    }
                    None => {
                        best_ecosystem = Some((ecosystem_name.clone(), confidence));
                    }
                }
            }
        }

        best_ecosystem
    }

    /// Get language information by file extension
    pub fn language_info(&self, extension: &str) -> Option<&super::LanguageGroup> {
        self.patterns.language_by_extension(extension)
    }

    /// Get supported file extensions
    pub fn supported_extensions(&self) -> Vec<String> {
        self.patterns.supported_extensions()
    }

    /// Get ecosystem information by name
    pub fn ecosystem_info(&self, name: &str) -> Option<&super::Ecosystem> {
        self.patterns.ecosystem_by_name(name)
    }

    /// Get pattern statistics for debugging
    pub fn pattern_stats(&self) -> PatternStats {
        PatternStats {
            ecosystems_count: self.patterns.project_indicators.ecosystems.len(),
            exclude_patterns_count: self.patterns.all_exclude_patterns().len(),
            include_patterns_count: self.patterns.all_include_patterns().len(),
            languages_count: self.patterns.supported_extensions().len(),
        }
    }
}

impl Default for PatternManager {
    fn default() -> Self {
        Self::new().expect("Failed to initialize PatternManager with embedded patterns")
    }
}

/// Pattern loading statistics
#[derive(Debug, Clone)]
pub struct PatternStats {
    pub ecosystems_count: usize,
    pub exclude_patterns_count: usize,
    pub include_patterns_count: usize,
    pub languages_count: usize,
}

/// Validate that all patterns are well-formed and consistent
fn validate_patterns(patterns: &AllPatterns) -> PatternResult<()> {
    // Validate project indicators
    for (ecosystem_name, ecosystem) in &patterns.project_indicators.ecosystems {
        if ecosystem.name.is_empty() {
            return Err(PatternError::Validation(format!(
                "Ecosystem '{}' has empty name",
                ecosystem_name
            )));
        }

        for indicator in &ecosystem.indicators {
            if indicator.pattern.is_empty() {
                return Err(PatternError::Validation(format!(
                    "Ecosystem '{}' has empty pattern",
                    ecosystem_name
                )));
            }

            if indicator.weight == 0 || indicator.weight > 100 {
                return Err(PatternError::Validation(format!(
                    "Ecosystem '{}' indicator '{}' has invalid weight: {}",
                    ecosystem_name, indicator.pattern, indicator.weight
                )));
            }
        }
    }

    // Validate exclude patterns
    let exclude_patterns = patterns.all_exclude_patterns();
    for pattern_meta in &exclude_patterns {
        if pattern_meta.pattern.is_empty() {
            return Err(PatternError::Validation(
                "Found empty exclude pattern".to_string(),
            ));
        }
    }

    // Validate include patterns
    let include_patterns = patterns.all_include_patterns();
    for pattern_meta in &include_patterns {
        if pattern_meta.pattern.is_empty() {
            return Err(PatternError::Validation(
                "Found empty include pattern".to_string(),
            ));
        }
    }

    // Validate language extensions - check all categories
    let all_lang_groups = [
        &patterns.language_extensions.programming_languages,
        &patterns.language_extensions.web_technologies,
        &patterns.language_extensions.markup_languages,
        &patterns.language_extensions.configuration_files,
        &patterns.language_extensions.shell_scripting,
        &patterns.language_extensions.data_formats,
        &patterns.language_extensions.specialized_formats,
    ];

    for category in all_lang_groups {
        for (lang_name, lang_group) in category {
            if lang_group.lsp_id.is_empty() {
                return Err(PatternError::Validation(format!(
                    "Language '{}' has empty lsp_id",
                    lang_name
                )));
            }
            if lang_group.extensions.is_empty() && lang_group.filenames.is_empty() {
                return Err(PatternError::Validation(format!(
                    "Language '{}' has no extensions or filenames",
                    lang_name
                )));
            }
        }
    }

    tracing::debug!(
        "Pattern validation complete: {} ecosystems, {} exclude patterns, {} include patterns, {} languages",
        patterns.project_indicators.ecosystems.len(),
        exclude_patterns.len(),
        include_patterns.len(),
        patterns.supported_extensions().len()
    );

    Ok(())
}

/// Simple glob pattern matching
///
/// This is a basic implementation for the most common patterns.
/// For production use, consider using the `glob` crate for full glob support.
fn glob_match(pattern: &str, text: &str) -> bool {
    // Handle the most common patterns used in our YAML files

    // Exact match
    if pattern == text {
        return true;
    }

    // Directory patterns ending with /
    if pattern.ends_with('/') && text.starts_with(&pattern[..pattern.len() - 1]) {
        return true;
    }

    // Double star patterns (**/*.ext) - match any depth
    if pattern.starts_with("**/") {
        let suffix_pattern = &pattern[3..]; // Remove "**/"
        // Check if the text ends with the suffix pattern
        if suffix_pattern.starts_with("*.") {
            let extension = &suffix_pattern[2..];
            if let Some(file_extension) = text.rsplit('.').next() {
                return file_extension == extension;
            }
        }
        return text.ends_with(suffix_pattern);
    }

    // File extension patterns (*.ext)
    if pattern.starts_with("*.") {
        let extension = &pattern[2..];
        if let Some(file_extension) = text.rsplit('.').next() {
            return file_extension == extension;
        }
    }

    // Simple wildcard patterns (*name*)
    if pattern.starts_with('*') && pattern.ends_with('*') && pattern.len() > 2 {
        let inner = &pattern[1..pattern.len() - 1];
        return text.contains(inner);
    }

    // Prefix patterns (*name)
    if pattern.starts_with('*') && pattern.len() > 1 {
        let suffix = &pattern[1..];
        return text.ends_with(suffix);
    }

    // Suffix patterns (name*)
    if pattern.ends_with('*') && pattern.len() > 1 {
        let prefix = &pattern[..pattern.len() - 1];
        return text.starts_with(prefix);
    }

    // For complex patterns, fall back to basic contains check
    // In production, this should use a proper glob library
    false
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pattern_manager_creation() {
        let manager = PatternManager::new();
        assert!(manager.is_ok(), "PatternManager should initialize successfully");
    }

    #[test]
    fn test_pattern_stats() {
        let manager = PatternManager::new().unwrap();
        let stats = manager.pattern_stats();

        // These should be non-zero if patterns are loaded correctly
        assert!(stats.ecosystems_count > 0, "Should have ecosystems");
        assert!(stats.exclude_patterns_count > 0, "Should have exclude patterns");
        assert!(stats.include_patterns_count > 0, "Should have include patterns");
        assert!(stats.languages_count > 0, "Should have language definitions");
    }

    #[test]
    fn test_glob_matching() {
        // Test exact match
        assert!(glob_match("test.txt", "test.txt"));

        // Test directory patterns
        assert!(glob_match("node_modules/", "node_modules"));
        assert!(glob_match("node_modules/", "node_modules/package"));

        // Test extension patterns
        assert!(glob_match("*.txt", "file.txt"));
        assert!(glob_match("*.rs", "main.rs"));
        assert!(!glob_match("*.txt", "file.rs"));

        // Test wildcard patterns
        assert!(glob_match("*test*", "mytest.txt"));
        assert!(glob_match("test*", "test.txt"));
        assert!(glob_match("*test", "mytest"));

        // Test double star patterns
        assert!(glob_match("**/*.rs", "src/main.rs"));
        assert!(glob_match("**/*.js", "src/components/app.js"));
        assert!(glob_match("**/*.md", "docs/README.md"));
        assert!(!glob_match("**/*.rs", "src/main.py"));
    }

    #[test]
    fn test_should_exclude() {
        let manager = PatternManager::new().unwrap();

        // Test some common exclude patterns
        // Note: These tests depend on the actual patterns in the YAML files
        assert!(manager.should_exclude("node_modules/package.json"));
        assert!(manager.should_exclude("target/debug/main"));
        assert!(!manager.should_exclude("src/main.rs"));
    }

    #[test]
    fn test_should_include() {
        let manager = PatternManager::new().unwrap();

        // Test some common include patterns
        // Note: These tests depend on the actual patterns in the YAML files

        // Test with patterns that should work with our simple glob implementation
        assert!(manager.should_include("main.rs"));  // *.rs pattern
        assert!(manager.should_include("README.md")); // *.md pattern
        assert!(manager.should_include("package.json")); // *.json pattern

        // Print some debug info to understand what patterns are loaded
        let include_patterns = manager.patterns.all_include_patterns();
        println!("Total include patterns: {}", include_patterns.len());
        for (i, pattern) in include_patterns.iter().take(5).enumerate() {
            println!("Pattern {}: {} - {}", i, pattern.pattern, pattern.description);
        }
    }

    #[test]
    fn test_language_detection() {
        let manager = PatternManager::new().unwrap();

        let rust_info = manager.language_info("rs");
        assert!(rust_info.is_some(), "Should recognize Rust files");

        let python_info = manager.language_info("py");
        assert!(python_info.is_some(), "Should recognize Python files");
    }

    #[test]
    fn test_ecosystem_detection() {
        let manager = PatternManager::new().unwrap();

        // Test with typical Rust project files
        let rust_files = vec![
            "Cargo.toml".to_string(),
            "src/main.rs".to_string(),
            "Cargo.lock".to_string(),
        ];

        let (ecosystem, confidence) = manager.detect_ecosystem(&rust_files).unwrap();
        assert!(confidence > 0.0, "Should detect some ecosystem with confidence");
        println!("Detected ecosystem: {} with confidence: {}", ecosystem, confidence);
    }
}