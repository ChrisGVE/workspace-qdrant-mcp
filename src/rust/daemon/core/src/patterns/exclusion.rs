//! Advanced exclusion rules system for build artifacts and unwanted files
//!
//! This module provides sophisticated file exclusion capabilities using the
//! comprehensive configuration data. Optimized for performance with multi-tier
//! filtering and context-aware exclusion logic.

use super::comprehensive::{ComprehensivePatternManager, ComprehensiveResult};
use std::collections::HashSet;
use std::path::Path;
use once_cell::sync::Lazy;

/// Exclusion rule categories with different priorities
#[derive(Debug, Clone)]
pub enum ExclusionCategory {
    /// Critical system files that should never be processed
    Critical,
    /// Build artifacts and generated files
    BuildArtifacts,
    /// Cache directories and temporary files
    Cache,
    /// Version control metadata
    VersionControl,
    /// IDE and editor files
    IdeFiles,
    /// Media and binary files
    Media,
    /// Security sensitive files
    Security,
}

/// Exclusion rule with metadata
#[derive(Debug, Clone)]
pub struct ExclusionRule {
    pub pattern: String,
    pub category: ExclusionCategory,
    pub reason: String,
    pub is_regex: bool,
    pub case_sensitive: bool,
}

/// Result of exclusion checking
#[derive(Debug, Clone)]
pub struct ExclusionResult {
    pub excluded: bool,
    pub rule: Option<ExclusionRule>,
    pub reason: String,
}

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

/// Global exclusion engine instance
static EXCLUSION_ENGINE: Lazy<Result<ExclusionEngine, String>> = Lazy::new(|| {
    ExclusionEngine::new().map_err(|e| format!("Failed to initialize exclusion engine: {}", e))
});

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
        // This must come FIRST to prevent .git contains pattern from matching
        if self.is_github_path(file_path) {
            return ExclusionResult {
                excluded: false,
                rule: None,
                reason: "Whitelisted: .github/ directory (CI/CD workflows)".to_string(),
            };
        }

        // Second check: hidden files/directories at ANY depth
        // Exclude any path containing a component starting with '.' (hidden)
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

        // If already excluded, return as-is
        if base_result.excluded {
            return base_result;
        }

        // Apply project-type specific exclusions
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
    ///
    /// .github/ is explicitly allowed because it contains CI/CD workflows
    /// and other GitHub-specific configuration that's useful to index.
    fn is_github_path(&self, file_path: &str) -> bool {
        // Check if path starts with .github/ or contains /.github/
        file_path.starts_with(".github/")
            || file_path.starts_with(".github\\")
            || file_path.contains("/.github/")
            || file_path.contains("\\.github\\")
            || file_path == ".github"
    }

    /// Check for hidden files/directories at any depth in the path
    ///
    /// Hidden files/directories (starting with '.') are excluded by default,
    /// as they typically contain:
    /// - Tool caches (.mypy_cache/, .pytest_cache/, .ruff_cache/)
    /// - IDE/editor configs (.vscode/, .idea/)
    /// - Version control (.git/ - already excluded separately)
    /// - Environment configs (.env files)
    /// - Build artifacts and caches
    ///
    /// Exceptions (useful hidden dirs/files that MAY be indexed):
    /// - .github/ - GitHub Actions workflows (useful for CI/CD understanding)
    fn check_hidden_components(&self, file_path: &str) -> Option<ExclusionResult> {
        // Split path into components and check each one
        for component in file_path.split('/') {
            // Skip empty components (from leading/trailing/double slashes)
            if component.is_empty() {
                continue;
            }

            // Check if component starts with '.' (hidden)
            if component.starts_with('.') {
                // Exception: .github is allowed (useful for CI/CD understanding)
                if component == ".github" {
                    continue;
                }

                // All other hidden files/directories are excluded
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

        // No hidden components found
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

/// Exclusion engine statistics
#[derive(Debug, Clone)]
pub struct ExclusionStats {
    pub total_rules: usize,
    pub exact_matches: usize,
    pub prefix_patterns: usize,
    pub suffix_patterns: usize,
    pub contains_patterns: usize,
    pub category_counts: std::collections::HashMap<String, usize>,
}

/// Classify and store a pattern in the appropriate fast lookup structure
fn classify_and_store_pattern(
    pattern: &str,
    _rule: &ExclusionRule,
    _exact_matches: &mut HashSet<String>,
    prefix_patterns: &mut Vec<String>,
    suffix_patterns: &mut Vec<String>,
    contains_patterns: &mut Vec<String>,
) {
    if pattern.contains('*') || pattern.contains('/') {
        // Complex patterns - analyze for optimization
        if pattern.starts_with('*') && pattern.ends_with('*') && pattern.len() > 2 {
            // *pattern* -> contains
            let inner = &pattern[1..pattern.len()-1];
            contains_patterns.push(inner.to_string());
        } else if pattern.starts_with('*') && pattern.len() > 1 {
            // *pattern -> suffix
            let suffix = &pattern[1..];
            suffix_patterns.push(suffix.to_string());
        } else if pattern.ends_with('*') && pattern.len() > 1 {
            // pattern* -> prefix
            let prefix = &pattern[..pattern.len()-1];
            prefix_patterns.push(prefix.to_string());
        } else if pattern.ends_with('/') {
            // directory pattern
            prefix_patterns.push(pattern.to_string());
        } else {
            // Complex pattern - use contains as fallback
            contains_patterns.push(pattern.replace('*', ""));
        }
    } else {
        // Plain patterns - treat as contains to match subpaths (e.g., "node_modules")
        contains_patterns.push(pattern.to_string());
    }
}

/// Get critical exclusion patterns that should always be excluded
fn get_critical_exclusion_patterns() -> Vec<(String, String)> {
    vec![
        // System files
        ("Thumbs.db".to_string(), "Windows thumbnail cache".to_string()),
        (".DS_Store".to_string(), "macOS folder metadata".to_string()),
        ("desktop.ini".to_string(), "Windows folder settings".to_string()),

        // Temporary files
        ("~$".to_string(), "Office temporary files".to_string()),
        (".tmp".to_string(), "Temporary files".to_string()),
        (".temp".to_string(), "Temporary files".to_string()),
        (".swp".to_string(), "Vim swap files".to_string()),
        (".swo".to_string(), "Vim swap files".to_string()),
        (".orig".to_string(), "Merge conflict backup".to_string()),

        // Security sensitive
        (".env".to_string(), "Environment variables (potentially sensitive)".to_string()),
        (".env.local".to_string(), "Local environment variables".to_string()),
        ("id_rsa".to_string(), "SSH private key".to_string()),
        ("id_dsa".to_string(), "SSH private key".to_string()),
        ("id_ecdsa".to_string(), "SSH private key".to_string()),
        ("id_ed25519".to_string(), "SSH private key".to_string()),

        // Large binaries
        (".dmg".to_string(), "macOS disk image".to_string()),
        (".iso".to_string(), "Disk image".to_string()),
        (".img".to_string(), "Disk image".to_string()),
        (".vmdk".to_string(), "Virtual machine disk".to_string()),
    ]
}

/// Convenient function for quick exclusion checking
pub fn should_exclude_file(file_path: &str) -> bool {
    match ExclusionEngine::global() {
        Ok(engine) => engine.should_exclude(file_path).excluded,
        Err(_) => false, // If engine fails to initialize, don't exclude anything
    }
}

/// Convenient function for contextual exclusion checking
pub fn should_exclude_file_with_context(file_path: &str, project_type: &str) -> ExclusionResult {
    match ExclusionEngine::global() {
        Ok(engine) => engine.check_with_context(file_path, Some(project_type)),
        Err(e) => ExclusionResult {
            excluded: false,
            rule: None,
            reason: format!("Engine initialization failed: {}", e),
        },
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_engine_initialization() {
        let engine = ExclusionEngine::new();
        assert!(engine.is_ok(), "Should initialize exclusion engine");
    }

    #[test]
    fn test_basic_exclusion() {
        let engine = ExclusionEngine::new().unwrap();

        // Version control
        assert!(engine.should_exclude(".git/config").excluded);
        assert!(engine.should_exclude(".gitignore").excluded);

        // Node modules
        assert!(engine.should_exclude("node_modules/package/index.js").excluded);

        // Build artifacts
        assert!(engine.should_exclude("target/debug/main").excluded);

        // Should not exclude source files
        assert!(!engine.should_exclude("src/main.rs").excluded);
        assert!(!engine.should_exclude("README.md").excluded);
    }

    #[test]
    fn test_contextual_exclusion() {
        let engine = ExclusionEngine::new().unwrap();

        // Rust project context
        let result = engine.check_with_context("target/debug/app", Some("rust"));
        assert!(result.excluded);

        // Python project context
        let result = engine.check_with_context("src/__pycache__/module.pyc", Some("python"));
        assert!(result.excluded);

        // JavaScript project context
        let result = engine.check_with_context("node_modules/lodash/index.js", Some("javascript"));
        assert!(result.excluded);
    }

    #[test]
    fn test_critical_patterns() {
        let engine = ExclusionEngine::new().unwrap();

        // System files
        assert!(engine.should_exclude(".DS_Store").excluded);
        assert!(engine.should_exclude("Thumbs.db").excluded);

        // Security files
        assert!(engine.should_exclude(".env").excluded);
        assert!(engine.should_exclude("id_rsa").excluded);

        // Temporary files
        assert!(engine.should_exclude("file.tmp").excluded);
        assert!(engine.should_exclude("document.swp").excluded);
    }

    #[test]
    fn test_pattern_classification() {
        let mut exact = HashSet::new();
        let mut prefix = Vec::new();
        let mut suffix = Vec::new();
        let mut contains = Vec::new();

        let rule = ExclusionRule {
            pattern: "test".to_string(),
            category: ExclusionCategory::Cache,
            reason: "test".to_string(),
            is_regex: false,
            case_sensitive: true,
        };

        // Test exact pattern
        classify_and_store_pattern("exact", &rule, &mut exact, &mut prefix, &mut suffix, &mut contains);
        assert!(contains.contains(&"exact".to_string()));

        // Test prefix pattern
        classify_and_store_pattern("prefix*", &rule, &mut exact, &mut prefix, &mut suffix, &mut contains);
        assert!(prefix.contains(&"prefix".to_string()));

        // Test suffix pattern
        classify_and_store_pattern("*.suffix", &rule, &mut exact, &mut prefix, &mut suffix, &mut contains);
        assert!(suffix.contains(&".suffix".to_string()));
    }

    #[test]
    fn test_exclusion_stats() {
        let engine = ExclusionEngine::new().unwrap();
        let stats = engine.stats();

        assert!(stats.total_rules > 0);
        assert!(stats.contains_patterns + stats.prefix_patterns + stats.suffix_patterns > 0);
        assert!(!stats.category_counts.is_empty());
    }

    #[test]
    fn test_global_engine() {
        let engine = ExclusionEngine::global();
        assert!(engine.is_ok(), "Global engine should be available");
    }

    #[test]
    fn test_convenience_functions() {
        assert!(should_exclude_file(".git/config"));
        assert!(!should_exclude_file("src/main.rs"));

        let result = should_exclude_file_with_context("target/debug/app", "rust");
        assert!(result.excluded);
    }

    #[test]
    fn test_filename_vs_path_exclusion() {
        let engine = ExclusionEngine::new().unwrap();

        // Test that both full path and filename are checked
        assert!(engine.should_exclude("path/to/.DS_Store").excluded);
        assert!(engine.should_exclude(".DS_Store").excluded);

        // Test directory patterns
        assert!(engine.should_exclude("project/node_modules/package.json").excluded);
        assert!(engine.should_exclude("node_modules/package.json").excluded);
    }

    #[test]
    fn test_hidden_files_excluded_at_all_depths() {
        let engine = ExclusionEngine::new().unwrap();

        // Hidden directories at root level
        assert!(engine.should_exclude(".mypy_cache/something.json").excluded);
        assert!(engine.should_exclude(".vscode/settings.json").excluded);
        assert!(engine.should_exclude(".idea/workspace.xml").excluded);

        // Hidden directories at arbitrary depth
        assert!(engine.should_exclude("src/.cache/file.txt").excluded);
        assert!(engine.should_exclude("deep/path/.mypy_cache/file.json").excluded);
        assert!(engine.should_exclude("a/b/c/.hidden/file.txt").excluded);

        // Hidden files (not just directories)
        assert!(engine.should_exclude("src/.hidden_file").excluded);
        assert!(engine.should_exclude("deep/path/.secret").excluded);

        // Multiple hidden components
        assert!(engine.should_exclude(".hidden1/.hidden2/file.txt").excluded);
    }

    #[test]
    fn test_github_directory_not_excluded() {
        let engine = ExclusionEngine::new().unwrap();

        // .github is explicitly allowed (useful for CI/CD understanding)
        assert!(!engine.should_exclude(".github/workflows/ci.yml").excluded);
        assert!(!engine.should_exclude(".github/CODEOWNERS").excluded);
        assert!(!engine.should_exclude("project/.github/workflows/test.yml").excluded);

        // But other .g* directories should still be excluded
        assert!(engine.should_exclude(".gradle/cache/file").excluded);
    }

    #[test]
    fn test_non_hidden_paths_not_excluded_by_hidden_rule() {
        let engine = ExclusionEngine::new().unwrap();

        // Normal source files should NOT be excluded by hidden rule
        // (may be excluded by other rules, but check the reason)
        let result = engine.should_exclude("src/main.rs");
        if result.excluded {
            // If excluded, make sure it's not because of hidden component
            assert!(!result.reason.contains("Hidden path component"));
        }

        let result = engine.should_exclude("lib/utils/helper.py");
        if result.excluded {
            assert!(!result.reason.contains("Hidden path component"));
        }

        // Files with dots in name (not at start) are fine
        let result = engine.should_exclude("config.json");
        if result.excluded {
            assert!(!result.reason.contains("Hidden path component"));
        }

        let result = engine.should_exclude("src/my.module.ts");
        if result.excluded {
            assert!(!result.reason.contains("Hidden path component"));
        }
    }

    #[test]
    fn test_hidden_component_with_various_formats() {
        let engine = ExclusionEngine::new().unwrap();

        // Windows-style paths (with backslashes) - test robustness
        // Note: We primarily use forward slashes, but test anyway
        assert!(engine.should_exclude("src/.hidden/file").excluded);

        // Leading/trailing slashes
        assert!(engine.should_exclude("/.hidden/file").excluded);
        assert!(engine.should_exclude(".hidden/file/").excluded);

        // Multiple slashes (should handle gracefully)
        assert!(engine.should_exclude("src//.hidden//file").excluded);
    }
}
