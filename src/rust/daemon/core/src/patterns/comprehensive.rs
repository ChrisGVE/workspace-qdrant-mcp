//! Comprehensive embedded pattern system with 500+ language support
//!
//! This module provides compile-time embedding of the comprehensive internal_configuration.yaml
//! containing research from 500+ languages, 80+ LSP servers, and 180+ Tree-sitter grammars.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use once_cell::sync::Lazy;
use std::sync::Arc;
use thiserror::Error;

/// Errors specific to comprehensive pattern operations
#[derive(Error, Debug)]
pub enum ComprehensivePatternError {
    #[error("YAML parsing error: {0}")]
    YamlParse(#[from] serde_yml::Error),

    #[error("Configuration validation error: {0}")]
    Validation(String),

    #[error("Pattern not found: {0}")]
    NotFound(String),

    #[error("Language not supported: {0}")]
    UnsupportedLanguage(String),
}

pub type ComprehensiveResult<T> = Result<T, ComprehensivePatternError>;

/// Main comprehensive internal configuration structure
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InternalConfiguration {
    pub project_indicators: ProjectIndicatorsConfig,
    pub file_extensions: HashMap<String, String>,
    pub lsp_servers: HashMap<String, LspServerConfig>,
    pub exclusion_patterns: ExclusionPatternsConfig,
    pub tree_sitter_grammars: TreeSitterConfig,
    pub content_signatures: ContentSignaturesConfig,
    pub build_systems: HashMap<String, BuildSystemConfig>,
    pub metadata_schemas: HashMap<String, MetadataSchema>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProjectIndicatorsConfig {
    pub version_control: Vec<String>,
    pub language_ecosystems: Vec<String>,
    pub build_systems: Vec<String>,
    pub ci_cd: Vec<String>,
    pub containerization: Vec<String>,
    pub config_management: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LspServerConfig {
    pub primary: String,
    pub features: Vec<String>,
    pub rationale: String,
    pub install_notes: String,
    #[serde(default)]
    pub alternatives: Vec<LspAlternative>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LspAlternative {
    pub name: String,
    pub features: Vec<String>,
    pub notes: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExclusionPatternsConfig {
    pub version_control: Vec<String>,
    pub build_outputs: Vec<String>,
    pub cache_directories: Vec<String>,
    pub virtual_environments: Vec<String>,
    pub ide_files: Vec<String>,
    pub temporary_files: Vec<String>,
    pub binary_files: Vec<String>,
    pub media_files: Vec<String>,
    pub archive_files: Vec<String>,
    pub package_files: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TreeSitterConfig {
    pub available: Vec<String>,
    pub quality_tiers: QualityTiers,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QualityTiers {
    pub excellent: Vec<String>,
    pub good: Vec<String>,
    pub basic: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ContentSignaturesConfig {
    pub shebangs: HashMap<String, String>,
    pub keyword_patterns: HashMap<String, Vec<String>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BuildSystemConfig {
    pub files: Vec<String>,
    pub language: String,
    pub commands: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MetadataSchema {
    pub required: Vec<String>,
    pub optional: Vec<String>,
}

/// Compile-time embedded comprehensive configuration
static EMBEDDED_CONFIG: &str = include_str!("../../../../../../assets/internal_configuration.yaml");

/// Global parsed configuration - lazily initialized on first access
static PARSED_CONFIG: Lazy<Result<Arc<InternalConfiguration>, ComprehensivePatternError>> = Lazy::new(|| {
    let config: InternalConfiguration = serde_yml::from_str(EMBEDDED_CONFIG)
        .map_err(ComprehensivePatternError::YamlParse)?;

    // Validate configuration after loading
    validate_configuration(&config)?;

    tracing::info!(
        "Loaded comprehensive configuration: {} languages, {} LSP servers, {} Tree-sitter grammars",
        config.file_extensions.len(),
        config.lsp_servers.len(),
        config.tree_sitter_grammars.available.len()
    );

    Ok(Arc::new(config))
});

/// Comprehensive pattern manager with 500+ language support
#[derive(Debug, Clone)]
pub struct ComprehensivePatternManager {
    config: Arc<InternalConfiguration>,
}

impl ComprehensivePatternManager {
    /// Create a new comprehensive pattern manager
    pub fn new() -> ComprehensiveResult<Self> {
        match PARSED_CONFIG.as_ref() {
            Ok(config) => Ok(Self {
                config: Arc::clone(config),
            }),
            Err(e) => Err(ComprehensivePatternError::Validation(format!(
                "Failed to load comprehensive configuration: {}", e
            ))),
        }
    }

    /// Get the full internal configuration
    pub fn config(&self) -> &InternalConfiguration {
        &self.config
    }

    /// Get language from file extension (500+ languages supported)
    pub fn language_from_extension(&self, extension: &str) -> Option<&String> {
        // Try with and without leading dot
        let clean_ext = extension.trim_start_matches('.');
        self.config.file_extensions.get(clean_ext)
            .or_else(|| self.config.file_extensions.get(&format!(".{}", clean_ext)))
    }

    /// Get LSP server configuration for a language
    pub fn lsp_config(&self, language: &str) -> Option<&LspServerConfig> {
        self.config.lsp_servers.get(language)
    }

    /// Check if Tree-sitter grammar is available
    pub fn tree_sitter_available(&self, language: &str) -> bool {
        self.config.tree_sitter_grammars.available.contains(&language.to_string())
    }

    /// Get Tree-sitter grammar quality tier
    pub fn tree_sitter_quality(&self, language: &str) -> Option<&str> {
        let grammars = &self.config.tree_sitter_grammars;
        if grammars.quality_tiers.excellent.contains(&language.to_string()) {
            Some("excellent")
        } else if grammars.quality_tiers.good.contains(&language.to_string()) {
            Some("good")
        } else if grammars.quality_tiers.basic.contains(&language.to_string()) {
            Some("basic")
        } else {
            None
        }
    }

    /// Check if file should be excluded
    pub fn should_exclude(&self, file_path: &str) -> bool {
        let exclusions = &self.config.exclusion_patterns;

        // Check all exclusion pattern categories
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
                if glob_match(pattern, file_path) {
                    return true;
                }
            }
        }
        false
    }

    /// Detect language from shebang
    pub fn detect_language_from_shebang(&self, content: &str) -> Option<&String> {
        for line in content.lines().take(3) {
            if line.starts_with("#!") {
                for (shebang, language) in &self.config.content_signatures.shebangs {
                    if line.contains(shebang) {
                        return Some(language);
                    }
                }
            }
        }
        None
    }

    /// Detect language from content keywords
    pub fn detect_language_from_keywords(&self, content: &str) -> Option<String> {
        let mut language_scores: HashMap<String, usize> = HashMap::new();

        for (language, keywords) in &self.config.content_signatures.keyword_patterns {
            let mut score = 0;
            for keyword in keywords {
                if content.contains(keyword) {
                    score += 1;
                }
            }
            if score > 0 {
                language_scores.insert(language.clone(), score);
            }
        }

        // Return language with highest keyword score
        language_scores.into_iter()
            .max_by_key(|(_, score)| *score)
            .map(|(language, _)| language)
    }

    /// Detect build system for project
    pub fn detect_build_system(&self, file_paths: &[String]) -> Option<(&String, &BuildSystemConfig)> {
        for (name, config) in &self.config.build_systems {
            for file in &config.files {
                if file_paths.iter().any(|path| glob_match(file, path)) {
                    return Some((name, config));
                }
            }
        }
        None
    }

    /// Get all supported languages
    pub fn all_languages(&self) -> Vec<&String> {
        self.config.file_extensions.values().collect()
    }

    /// Get all supported file extensions
    pub fn all_extensions(&self) -> Vec<&String> {
        self.config.file_extensions.keys().collect()
    }

    /// Get comprehensive statistics
    pub fn stats(&self) -> ComprehensiveStats {
        ComprehensiveStats {
            total_languages: self.config.file_extensions.values().collect::<std::collections::HashSet<_>>().len(),
            total_extensions: self.config.file_extensions.len(),
            lsp_servers: self.config.lsp_servers.len(),
            tree_sitter_grammars: self.config.tree_sitter_grammars.available.len(),
            excellent_grammars: self.config.tree_sitter_grammars.quality_tiers.excellent.len(),
            good_grammars: self.config.tree_sitter_grammars.quality_tiers.good.len(),
            basic_grammars: self.config.tree_sitter_grammars.quality_tiers.basic.len(),
            build_systems: self.config.build_systems.len(),
            exclusion_patterns: self.total_exclusion_patterns(),
        }
    }

    fn total_exclusion_patterns(&self) -> usize {
        let exclusions = &self.config.exclusion_patterns;
        exclusions.version_control.len() +
        exclusions.build_outputs.len() +
        exclusions.cache_directories.len() +
        exclusions.virtual_environments.len() +
        exclusions.ide_files.len() +
        exclusions.temporary_files.len() +
        exclusions.binary_files.len() +
        exclusions.media_files.len() +
        exclusions.archive_files.len() +
        exclusions.package_files.len()
    }
}

impl Default for ComprehensivePatternManager {
    fn default() -> Self {
        Self::new().expect("Failed to initialize ComprehensivePatternManager")
    }
}

/// Comprehensive statistics about the embedded configuration
#[derive(Debug, Clone)]
pub struct ComprehensiveStats {
    pub total_languages: usize,
    pub total_extensions: usize,
    pub lsp_servers: usize,
    pub tree_sitter_grammars: usize,
    pub excellent_grammars: usize,
    pub good_grammars: usize,
    pub basic_grammars: usize,
    pub build_systems: usize,
    pub exclusion_patterns: usize,
}

/// Validate the comprehensive configuration
fn validate_configuration(config: &InternalConfiguration) -> ComprehensiveResult<()> {
    // Validate file extensions are not empty
    if config.file_extensions.is_empty() {
        return Err(ComprehensivePatternError::Validation(
            "No file extensions defined".to_string()
        ));
    }

    // Validate LSP servers have required fields
    for (language, lsp_config) in &config.lsp_servers {
        if lsp_config.primary.is_empty() {
            return Err(ComprehensivePatternError::Validation(
                format!("LSP server for '{}' has empty primary field", language)
            ));
        }
        if lsp_config.features.is_empty() {
            return Err(ComprehensivePatternError::Validation(
                format!("LSP server for '{}' has no features defined", language)
            ));
        }
    }

    // Validate Tree-sitter grammars
    if config.tree_sitter_grammars.available.is_empty() {
        return Err(ComprehensivePatternError::Validation(
            "No Tree-sitter grammars defined".to_string()
        ));
    }

    // Validate build systems
    for (name, build_config) in &config.build_systems {
        if build_config.files.is_empty() {
            return Err(ComprehensivePatternError::Validation(
                format!("Build system '{}' has no files defined", name)
            ));
        }
        if build_config.language.is_empty() {
            return Err(ComprehensivePatternError::Validation(
                format!("Build system '{}' has no language defined", name)
            ));
        }
    }

    tracing::debug!(
        "Configuration validation complete: {} languages, {} LSP servers, {} Tree-sitter grammars",
        config.file_extensions.len(),
        config.lsp_servers.len(),
        config.tree_sitter_grammars.available.len()
    );

    Ok(())
}

/// Simple glob pattern matching
fn glob_match(pattern: &str, text: &str) -> bool {
    // Handle exact match
    if pattern == text {
        return true;
    }

    // Handle directory patterns ending with /
    if pattern.ends_with('/') && text.starts_with(&pattern[..pattern.len() - 1]) {
        return true;
    }

    // Handle wildcard patterns (*.ext)
    if let Some(extension) = pattern.strip_prefix("*.") {
        if let Some(file_extension) = text.rsplit('.').next() {
            return file_extension == extension;
        }
    }

    // Handle prefix patterns (*name)
    if pattern.starts_with('*') && pattern.len() > 1 {
        let suffix = &pattern[1..];
        return text.ends_with(suffix);
    }

    // Handle suffix patterns (name*)
    if pattern.ends_with('*') && pattern.len() > 1 {
        let prefix = &pattern[..pattern.len() - 1];
        return text.starts_with(prefix);
    }

    // Fallback: treat plain patterns as substrings for path matching
    if !pattern.contains('*') && !pattern.contains('?') && !pattern.contains('[') {
        return text.contains(pattern);
    }

    false
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_comprehensive_pattern_manager_creation() {
        let manager = ComprehensivePatternManager::new();
        assert!(manager.is_ok(), "Should initialize comprehensive pattern manager");
    }

    #[test]
    fn test_language_detection() {
        let manager = ComprehensivePatternManager::new().unwrap();

        // Test common extensions
        assert!(manager.language_from_extension("rs").is_some());
        assert!(manager.language_from_extension("py").is_some());
        assert!(manager.language_from_extension("js").is_some());
        assert!(manager.language_from_extension("ts").is_some());
    }

    #[test]
    fn test_lsp_server_configs() {
        let manager = ComprehensivePatternManager::new().unwrap();

        // Test LSP configs for major languages
        assert!(manager.lsp_config("rust").is_some());
        assert!(manager.lsp_config("python").is_some());
    }

    #[test]
    fn test_tree_sitter_availability() {
        let manager = ComprehensivePatternManager::new().unwrap();

        // Test Tree-sitter grammar availability
        assert!(manager.tree_sitter_available("rust"));
        assert!(manager.tree_sitter_available("python"));
        assert!(manager.tree_sitter_available("javascript"));
    }

    #[test]
    fn test_exclusion_patterns() {
        let manager = ComprehensivePatternManager::new().unwrap();

        // Test common exclusion patterns
        assert!(manager.should_exclude("node_modules/package.json"));
        assert!(manager.should_exclude("target/debug/main"));
        assert!(manager.should_exclude(".git/config"));
        assert!(!manager.should_exclude("src/main.rs"));
    }

    #[test]
    fn test_shebang_detection() {
        let manager = ComprehensivePatternManager::new().unwrap();

        let python_script = "#!/usr/bin/env python3\nprint('hello')";
        assert_eq!(manager.detect_language_from_shebang(python_script), Some(&"python".to_string()));

        let bash_script = "#!/bin/bash\necho hello";
        assert!(manager.detect_language_from_shebang(bash_script).is_some());
    }

    #[test]
    fn test_comprehensive_stats() {
        let manager = ComprehensivePatternManager::new().unwrap();
        let stats = manager.stats();

        // Verify we have non-empty coverage
        assert!(stats.total_languages > 0, "Should support at least one language");
        assert!(stats.lsp_servers > 0, "Should have at least one LSP server");
        assert!(stats.tree_sitter_grammars > 0, "Should have at least one Tree-sitter grammar");
        assert!(stats.build_systems > 0, "Should have at least one build system");
        assert!(stats.exclusion_patterns > 0, "Should have at least one exclusion pattern");
    }
}
