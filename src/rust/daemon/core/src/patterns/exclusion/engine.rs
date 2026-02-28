//! ExclusionEngine implementation for pattern-based file exclusion.

use std::collections::HashSet;
use std::path::Path;

use once_cell::sync::Lazy;

use crate::patterns::comprehensive::{ComprehensivePatternManager, ComprehensiveResult};
use super::{
    ExclusionCategory, ExclusionResult, ExclusionRule, ExclusionStats,
};
use super::helpers::{classify_and_store_pattern, get_critical_exclusion_patterns};

/// Global exclusion engine instance
static EXCLUSION_ENGINE: Lazy<Result<ExclusionEngine, String>> = Lazy::new(|| {
    ExclusionEngine::new().map_err(|e| format!("Failed to initialize exclusion engine: {}", e))
});

/// High-performance exclusion engine
#[derive(Debug)]
pub struct ExclusionEngine {
    /// Fast lookup sets for common patterns
    exact_matches: HashSet<String>,
    prefix_patterns: Vec<String>,
    suffix_patterns: Vec<String>,
    contains_patterns: Vec<String>,
    /// All rules for detailed reporting
    all_rules: Vec<ExclusionRule>,
}

impl ExclusionEngine {
    /// Create a new exclusion engine
    pub fn new() -> ComprehensiveResult<Self> {
        let comprehensive = ComprehensivePatternManager::new()?;
        let config = comprehensive.config();

        let mut exact_matches = HashSet::new();
        let mut prefix_patterns = Vec::new();
        let mut suffix_patterns = Vec::new();
        let mut contains_patterns = Vec::new();
        let mut all_rules = Vec::new();

        // Process all exclusion patterns from the configuration
        let exclusions = &config.exclusion_patterns;

        // Version control patterns
        for pattern in &exclusions.version_control {
            let rule = ExclusionRule {
                pattern: pattern.clone(),
                category: ExclusionCategory::VersionControl,
                reason: "Version control metadata".to_string(),
                is_regex: false,
                case_sensitive: true,
            };
            classify_and_store_pattern(pattern, &rule, &mut exact_matches, &mut prefix_patterns, &mut suffix_patterns, &mut contains_patterns);
            all_rules.push(rule);
        }

        // Build output patterns
        for pattern in &exclusions.build_outputs {
            let rule = ExclusionRule {
                pattern: pattern.clone(),
                category: ExclusionCategory::BuildArtifacts,
                reason: "Build artifacts and generated files".to_string(),
                is_regex: false,
                case_sensitive: false,
            };
            classify_and_store_pattern(pattern, &rule, &mut exact_matches, &mut prefix_patterns, &mut suffix_patterns, &mut contains_patterns);
            all_rules.push(rule);
        }

        // Cache patterns
        for pattern in &exclusions.cache_directories {
            let rule = ExclusionRule {
                pattern: pattern.clone(),
                category: ExclusionCategory::Cache,
                reason: "Cache and temporary files".to_string(),
                is_regex: false,
                case_sensitive: false,
            };
            classify_and_store_pattern(pattern, &rule, &mut exact_matches, &mut prefix_patterns, &mut suffix_patterns, &mut contains_patterns);
            all_rules.push(rule);
        }

        // IDE files
        for pattern in &exclusions.ide_files {
            let rule = ExclusionRule {
                pattern: pattern.clone(),
                category: ExclusionCategory::IdeFiles,
                reason: "IDE and editor configuration".to_string(),
                is_regex: false,
                case_sensitive: false,
            };
            classify_and_store_pattern(pattern, &rule, &mut exact_matches, &mut prefix_patterns, &mut suffix_patterns, &mut contains_patterns);
            all_rules.push(rule);
        }

        // Add additional critical patterns
        let critical_patterns = get_critical_exclusion_patterns();
        for (pattern, reason) in critical_patterns {
            let rule = ExclusionRule {
                pattern: pattern.clone(),
                category: ExclusionCategory::Critical,
                reason,
                is_regex: false,
                case_sensitive: true,
            };
            classify_and_store_pattern(&pattern, &rule, &mut exact_matches, &mut prefix_patterns, &mut suffix_patterns, &mut contains_patterns);
            all_rules.push(rule);
        }

        tracing::debug!(
            "Exclusion engine initialized: {} exact, {} prefix, {} suffix, {} contains patterns",
            exact_matches.len(),
            prefix_patterns.len(),
            suffix_patterns.len(),
            contains_patterns.len()
        );

        Ok(Self {
            exact_matches,
            prefix_patterns,
            suffix_patterns,
            contains_patterns,
            all_rules,
        })
    }

    /// Get the global exclusion engine instance
    pub fn global() -> Result<&'static ExclusionEngine, &'static str> {
        EXCLUSION_ENGINE.as_ref().map_err(|e| e.as_str())
    }

    /// Check if a file should be excluded
    pub fn should_exclude(&self, file_path: &str) -> ExclusionResult {
        // Whitelist check: .github/ is explicitly allowed
        if self.is_github_path(file_path) {
            return ExclusionResult {
                excluded: false,
                rule: None,
                reason: "Whitelisted: .github/ directory (CI/CD workflows)".to_string(),
            };
        }

        // Second check: hidden files/directories at ANY depth
        if let Some(result) = self.check_hidden_components(file_path) {
            return result;
        }

        // Fast path: exact match check
        if self.exact_matches.contains(file_path) {
            return ExclusionResult {
                excluded: true,
                rule: self.find_rule_for_pattern(file_path),
                reason: "Exact pattern match".to_string(),
            };
        }

        // Extract filename for filename-based checks
        let filename = Path::new(file_path)
            .file_name()
            .and_then(|f| f.to_str())
            .unwrap_or(file_path);

        // Check exact filename match
        if self.exact_matches.contains(filename) {
            return ExclusionResult {
                excluded: true,
                rule: self.find_rule_for_pattern(filename),
                reason: "Filename exact match".to_string(),
            };
        }

        // Prefix patterns (e.g., "tmp", "temp")
        for pattern in &self.prefix_patterns {
            if file_path.starts_with(pattern) || filename.starts_with(pattern) {
                return ExclusionResult {
                    excluded: true,
                    rule: self.find_rule_for_pattern(pattern),
                    reason: format!("Prefix pattern match: {}", pattern),
                };
            }
        }

        // Suffix patterns (e.g., ".tmp", ".bak")
        for pattern in &self.suffix_patterns {
            if file_path.ends_with(pattern) || filename.ends_with(pattern) {
                return ExclusionResult {
                    excluded: true,
                    rule: self.find_rule_for_pattern(pattern),
                    reason: format!("Suffix pattern match: {}", pattern),
                };
            }
        }

        // Contains patterns (e.g., "node_modules")
        for pattern in &self.contains_patterns {
            if file_path.contains(pattern) {
                return ExclusionResult {
                    excluded: true,
                    rule: self.find_rule_for_pattern(pattern),
                    reason: format!("Contains pattern match: {}", pattern),
                };
            }
        }

        ExclusionResult {
            excluded: false,
            rule: None,
            reason: "No exclusion rules matched".to_string(),
        }
    }

    /// Check if a file should be excluded with detailed context
    pub fn check_with_context(&self, file_path: &str, project_type: Option<&str>) -> ExclusionResult {
        let base_result = self.should_exclude(file_path);

        if base_result.excluded {
            return base_result;
        }

        if let Some(project_type) = project_type {
            if let Some(context_rule) = self.check_contextual_exclusion(file_path, project_type) {
                return ExclusionResult {
                    excluded: true,
                    rule: Some(context_rule),
                    reason: "Contextual exclusion based on project type".to_string(),
                };
            }
        }

        base_result
    }

    /// Get all exclusion rules for inspection
    pub fn get_all_rules(&self) -> &[ExclusionRule] {
        &self.all_rules
    }

    /// Get exclusion statistics
    pub fn stats(&self) -> ExclusionStats {
        let mut category_counts = std::collections::HashMap::new();
        for rule in &self.all_rules {
            *category_counts.entry(format!("{:?}", rule.category)).or_insert(0) += 1;
        }

        ExclusionStats {
            total_rules: self.all_rules.len(),
            exact_matches: self.exact_matches.len(),
            prefix_patterns: self.prefix_patterns.len(),
            suffix_patterns: self.suffix_patterns.len(),
            contains_patterns: self.contains_patterns.len(),
            category_counts,
        }
    }

    /// Find the rule that matches a specific pattern
    fn find_rule_for_pattern(&self, pattern: &str) -> Option<ExclusionRule> {
        self.all_rules.iter()
            .find(|rule| rule.pattern == pattern)
            .cloned()
    }

    /// Check if path is inside .github/ directory (whitelisted)
    fn is_github_path(&self, file_path: &str) -> bool {
        file_path.starts_with(".github/")
            || file_path.starts_with(".github\\")
            || file_path.contains("/.github/")
            || file_path.contains("\\.github\\")
            || file_path == ".github"
    }

    /// Check for hidden files/directories at any depth in the path
    fn check_hidden_components(&self, file_path: &str) -> Option<ExclusionResult> {
        for component in file_path.split('/') {
            if component.is_empty() {
                continue;
            }

            if component.starts_with('.') {
                if component == ".github" {
                    continue;
                }

                return Some(ExclusionResult {
                    excluded: true,
                    rule: Some(ExclusionRule {
                        pattern: format!(".* (hidden: {})", component),
                        category: ExclusionCategory::IdeFiles,
                        reason: format!("Hidden file/directory excluded: {}", component),
                        is_regex: false,
                        case_sensitive: true,
                    }),
                    reason: format!("Hidden path component: {}", component),
                });
            }
        }

        None
    }

    /// Check for contextual exclusions based on project type
    fn check_contextual_exclusion(&self, file_path: &str, project_type: &str) -> Option<ExclusionRule> {
        match project_type {
            "rust" => {
                if file_path.starts_with("target/") || file_path.contains("/target/") {
                    return Some(ExclusionRule {
                        pattern: "target/".to_string(),
                        category: ExclusionCategory::BuildArtifacts,
                        reason: "Rust build directory".to_string(),
                        is_regex: false,
                        case_sensitive: true,
                    });
                }
            },
            "javascript" | "typescript" => {
                if file_path.starts_with("node_modules/") || file_path.contains("/node_modules/") {
                    return Some(ExclusionRule {
                        pattern: "node_modules/".to_string(),
                        category: ExclusionCategory::BuildArtifacts,
                        reason: "Node.js dependencies".to_string(),
                        is_regex: false,
                        case_sensitive: true,
                    });
                }
            },
            "python" => {
                if file_path.contains("__pycache__") || file_path.ends_with(".pyc") {
                    return Some(ExclusionRule {
                        pattern: "__pycache__".to_string(),
                        category: ExclusionCategory::BuildArtifacts,
                        reason: "Python bytecode cache".to_string(),
                        is_regex: false,
                        case_sensitive: true,
                    });
                }
            },
            _ => {},
        }
        None
    }
}
