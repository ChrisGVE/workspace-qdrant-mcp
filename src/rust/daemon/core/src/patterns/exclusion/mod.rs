//! Advanced exclusion rules system for build artifacts and unwanted files
//!
//! This module provides sophisticated file exclusion capabilities using the
//! comprehensive configuration data. Optimized for performance with multi-tier
//! filtering and context-aware exclusion logic.

mod engine;
pub(crate) mod helpers;
#[cfg(test)]
mod tests;

pub use engine::ExclusionEngine;

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

/// Convenient function for quick exclusion checking
pub fn should_exclude_file(file_path: &str) -> bool {
    match ExclusionEngine::global() {
        Ok(engine) => engine.should_exclude(file_path).excluded,
        Err(_) => false, // If engine fails to initialize, don't exclude anything
    }
}

/// Check if a directory should be skipped entirely during filesystem walks.
///
/// Uses the existing exclusion engine by testing if a synthetic file path
/// under this directory would be excluded. This allows WalkDir's `filter_entry`
/// to skip entire subtrees (e.g., target/, node_modules/, .git/) without
/// enumerating their contents.
pub fn should_exclude_directory(dir_name: &str) -> bool {
    // .github is explicitly whitelisted — never skip it
    if dir_name == ".github" {
        return false;
    }
    // Hidden directories (start with '.') are always excluded
    if dir_name.starts_with('.') {
        return true;
    }
    // Check if the exclusion engine would exclude files under this directory
    match ExclusionEngine::global() {
        Ok(engine) => {
            let synthetic_path = format!("{}/placeholder.txt", dir_name);
            engine.should_exclude(&synthetic_path).excluded
        }
        Err(_) => false,
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
