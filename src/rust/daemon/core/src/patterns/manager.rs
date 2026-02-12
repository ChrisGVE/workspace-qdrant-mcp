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
use std::collections::HashMap;

/// Embedded YAML content - loaded at compile time
struct _EmbeddedPatterns {
    project_indicators: &'static str,
    exclude_patterns: &'static str,
    include_patterns: &'static str,
    language_extensions: &'static str,
}

/// Compile-time embedded pattern files using include_str! macro
static _EMBEDDED_PATTERNS: _EmbeddedPatterns = _EmbeddedPatterns {
    project_indicators: include_str!("../../../../../../patterns/project_indicators.yaml"),
    exclude_patterns: include_str!("../../../../../../patterns/exclude_patterns.yaml"),
    include_patterns: include_str!("../../../../../../patterns/include_patterns.yaml"),
    language_extensions: include_str!("../../../../../../patterns/language_extensions.yaml"),
};

/// Global comprehensive pattern manager - lazily initialized on first access
static COMPREHENSIVE_MANAGER: Lazy<Result<Arc<super::comprehensive::ComprehensivePatternManager>, PatternError>> = Lazy::new(|| {
    super::comprehensive::ComprehensivePatternManager::new()
        .map(Arc::new)
        .map_err(|e| PatternError::Validation(format!("Comprehensive pattern manager failed: {}", e)))
});

/// Global patterns derived from comprehensive configuration - lazily initialized
static DERIVED_PATTERNS: Lazy<Result<Arc<AllPatterns>, PatternError>> = Lazy::new(|| {
    let comprehensive = COMPREHENSIVE_MANAGER.as_ref()
        .map_err(|e| PatternError::Validation(format!("Failed to load comprehensive manager: {}", e)))?;

    let all_patterns = convert_comprehensive_to_patterns(comprehensive)?;
    validate_patterns(&all_patterns)?;

    Ok(Arc::new(all_patterns))
});

/// PatternManager provides efficient access to embedded pattern data
#[derive(Debug, Clone)]
pub struct PatternManager {
    patterns: Arc<AllPatterns>,
}

impl PatternManager {
    /// Create a new PatternManager with comprehensive embedded configuration
    ///
    /// # Errors
    /// Returns an error if the comprehensive configuration cannot be parsed or is invalid
    pub fn new() -> PatternResult<Self> {
        match DERIVED_PATTERNS.as_ref() {
            Ok(patterns) => Ok(Self {
                patterns: Arc::clone(patterns),
            }),
            Err(e) => Err(PatternError::Validation(format!(
                "Failed to load comprehensive configuration: {}",
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
        Self::new().expect("Failed to initialize PatternManager with comprehensive configuration")
    }
}

/// Get access to the comprehensive pattern manager
pub fn comprehensive_manager() -> PatternResult<Arc<super::comprehensive::ComprehensivePatternManager>> {
    COMPREHENSIVE_MANAGER.as_ref()
        .map(Arc::clone)
        .map_err(|e| PatternError::Validation(format!("Comprehensive manager unavailable: {}", e)))
}

/// Convert comprehensive configuration to legacy AllPatterns structure
fn convert_comprehensive_to_patterns(
    comprehensive: &super::comprehensive::ComprehensivePatternManager
) -> PatternResult<AllPatterns> {
    use super::{
        ProjectIndicators, ExcludePatterns, IncludePatterns, LanguageExtensions,
        Ecosystem, ProjectIndicator, ConfidenceLevel, LanguageGroup,
        PatternWithMetadata
    };

    let config = comprehensive.config();

    fn normalize_pattern(pattern: &str) -> String {
        if pattern.contains('*') || pattern.contains('?') || pattern.contains('[') {
            pattern.to_string()
        } else {
            format!("*{}*", pattern)
        }
    }

    // Create project indicators from comprehensive config
    let mut ecosystems = HashMap::new();

    // Create a rust ecosystem from build system detection
    if let Some(rust_build) = config.build_systems.get("cargo") {
        let indicators = rust_build.files.iter().map(|file| ProjectIndicator {
            pattern: file.clone(),
            confidence: ConfidenceLevel::High,
            rationale: "Cargo build system indicator".to_string(),
            weight: 90,
        }).collect();

        ecosystems.insert("rust".to_string(), Ecosystem {
            name: "Rust".to_string(),
            description: "Rust programming ecosystem".to_string(),
            confidence_levels: HashMap::new(),
            indicators,
        });
    }

    // Create similar ecosystems for other major languages
    for (name, build_config) in &config.build_systems {
        if !ecosystems.contains_key(&build_config.language) {
            let indicators = build_config.files.iter().map(|file| ProjectIndicator {
                pattern: file.clone(),
                confidence: ConfidenceLevel::High,
                rationale: format!("{} build system indicator", name),
                weight: 80,
            }).collect();

            ecosystems.insert(build_config.language.clone(), Ecosystem {
                name: build_config.language.clone(),
                description: format!("{} programming ecosystem", build_config.language),
                confidence_levels: HashMap::new(),
                indicators,
            });
        }
    }

    let project_indicators = ProjectIndicators {
        version: "1.0.0".to_string(),
        last_updated: wqm_common::timestamps::now_utc(),
        research_coverage: "500+ languages from comprehensive A-Z research".to_string(),
        ecosystems,
    };

    // Create exclude patterns from comprehensive config
    let mut all_exclude_patterns = Vec::new();
    let exclusions = &config.exclusion_patterns;

    for patterns in [
        &exclusions.version_control,
        &exclusions.build_outputs,
        &exclusions.cache_directories,
        &exclusions.virtual_environments,
        &exclusions.ide_files,
        &exclusions.temporary_files,
        &exclusions.binary_files,
        &exclusions.media_files,
        &exclusions.archive_files,
        &exclusions.package_files,
    ] {
        for pattern in patterns {
            all_exclude_patterns.push(PatternWithMetadata {
                pattern: normalize_pattern(pattern),
                description: format!("Auto-generated from comprehensive config: {}", pattern),
                ecosystems: vec!["all".to_string()],
            });
        }
    }

    let exclude_patterns = ExcludePatterns {
        build_artifacts: all_exclude_patterns.clone(),
        compiled_code: Vec::new(),
        environments: Vec::new(),
        caches: Vec::new(),
        version_control: Vec::new(),
        editor_files: Vec::new(),
        system_files: Vec::new(),
        logs_and_temp: Vec::new(),
        media_files: Vec::new(),
        large_binaries: Vec::new(),
        security: Vec::new(),
        test_artifacts: Vec::new(),
    };

    // Create include patterns for source code
    let mut source_code_patterns = Vec::new();
    for ext in config.file_extensions.keys() {
        source_code_patterns.push(PatternWithMetadata {
            pattern: format!("*.{}", ext.trim_start_matches('.')),
            description: format!("Source code pattern for {}", ext),
            ecosystems: vec!["all".to_string()],
        });
    }

    let include_patterns = IncludePatterns {
        version: "1.0.0".to_string(),
        last_updated: wqm_common::timestamps::now_utc(),
        research_coverage: "500+ languages from comprehensive A-Z research".to_string(),
        source_code: source_code_patterns,
        documentation: Vec::new(),
        configuration: {
            let mut config_patterns = Vec::new();
            for patterns in [
                &config.project_indicators.version_control,
                &config.project_indicators.language_ecosystems,
                &config.project_indicators.build_systems,
                &config.project_indicators.ci_cd,
                &config.project_indicators.containerization,
                &config.project_indicators.config_management,
            ] {
                for pattern in patterns {
                    config_patterns.push(PatternWithMetadata {
                        pattern: normalize_pattern(pattern),
                        description: format!("Config indicator: {}", pattern),
                        ecosystems: vec!["all".to_string()],
                    });
                }
            }
            for build in config.build_systems.values() {
                for pattern in &build.files {
                    config_patterns.push(PatternWithMetadata {
                        pattern: normalize_pattern(pattern),
                        description: format!("Build system file: {}", pattern),
                        ecosystems: vec!["all".to_string()],
                    });
                }
            }
            config_patterns
        },
        schema_and_data: Vec::new(),
        templates_and_resources: Vec::new(),
        project_management: Vec::new(),
        special_patterns: Vec::new(),
    };

    // Create language extensions from comprehensive config
    let mut programming_languages = HashMap::new();
    let mut web_technologies = HashMap::new();
    let mut markup_languages = HashMap::new();
    let mut configuration_files = HashMap::new();
    let mut shell_scripting = HashMap::new();
    let mut data_formats = HashMap::new();
    let mut specialized_formats = HashMap::new();
    let mut extensions_to_languages = HashMap::new();
    let filenames_to_languages = HashMap::new();

    // Group languages by category and create language groups
    for (ext, language) in &config.file_extensions {
        extensions_to_languages.insert(ext.clone(), language.clone());

        let lsp_id = config.lsp_servers.get(language)
            .map(|lsp| lsp.primary.clone())
            .unwrap_or_else(|| format!("{}-lsp", language));

        let lang_group = LanguageGroup {
            extensions: vec![ext.clone()],
            filenames: Vec::new(),
            lsp_id,
            category: categorize_language(language),
        };

        match categorize_language(language).as_str() {
            "programming" => {
                programming_languages.insert(language.clone(), lang_group);
            },
            "web" => {
                web_technologies.insert(language.clone(), lang_group);
            },
            "markup" => {
                markup_languages.insert(language.clone(), lang_group);
            },
            "config" => {
                configuration_files.insert(language.clone(), lang_group);
            },
            "shell" => {
                shell_scripting.insert(language.clone(), lang_group);
            },
            "data" => {
                data_formats.insert(language.clone(), lang_group);
            },
            _ => {
                specialized_formats.insert(language.clone(), lang_group);
            },
        }
    }

    let language_extensions = LanguageExtensions {
        programming_languages,
        web_technologies,
        markup_languages,
        configuration_files,
        shell_scripting,
        data_formats,
        specialized_formats,
        extensions_to_languages,
        filenames_to_languages,
        metadata: HashMap::new(),
    };

    Ok(AllPatterns {
        project_indicators,
        exclude_patterns,
        include_patterns,
        language_extensions,
    })
}

/// Categorize a language into a general category
fn categorize_language(language: &str) -> String {
    match language {
        "rust" | "go" | "python" | "java" | "cpp" | "c" | "csharp" | "swift" | "kotlin" | "scala" | "clojure" | "haskell" | "ocaml" | "julia" | "dart" | "crystal" | "nim" | "zig" | "fortran" | "cobol" | "ada" | "d" | "elixir" | "erlang" | "fsharp" | "groovy" | "lua" | "perl" | "php" | "r" | "ruby" => "programming".to_string(),
        "javascript" | "typescript" | "css" | "scss" | "sass" | "less" | "html" | "jsx" | "tsx" => "web".to_string(),
        "markdown" | "asciidoc" | "restructuredtext" | "latex" | "xml" => "markup".to_string(),
        "json" | "yaml" | "toml" | "ini" | "cfg" | "conf" => "config".to_string(),
        "bash" | "zsh" | "fish" | "powershell" | "bat" | "cmd" => "shell".to_string(),
        "csv" | "tsv" | "sql" | "graphql" => "data".to_string(),
        _ => "specialized".to_string(),
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
    if let Some(suffix_pattern) = pattern.strip_prefix("**/") {
        // Remove "**/"
        // Check if the text ends with the suffix pattern
        if let Some(extension) = suffix_pattern.strip_prefix("*.") {
            if let Some(file_extension) = text.rsplit('.').next() {
                return file_extension == extension;
            }
        }
        return text.ends_with(suffix_pattern);
    }

    // File extension patterns (*.ext)
    if let Some(extension) = pattern.strip_prefix("*.") {
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
