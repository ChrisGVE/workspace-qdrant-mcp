//! Helper functions for the exclusion engine.

use std::collections::HashSet;

use super::ExclusionRule;

/// Classify and store a pattern in the appropriate fast lookup structure
pub(crate) fn classify_and_store_pattern(
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
            let inner = &pattern[1..pattern.len() - 1];
            contains_patterns.push(inner.to_string());
        } else if pattern.starts_with('*') && pattern.len() > 1 {
            // *pattern -> suffix
            let suffix = &pattern[1..];
            suffix_patterns.push(suffix.to_string());
        } else if pattern.ends_with('*') && pattern.len() > 1 {
            // pattern* -> prefix
            let prefix = &pattern[..pattern.len() - 1];
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
pub(crate) fn get_critical_exclusion_patterns() -> Vec<(String, String)> {
    vec![
        // System files
        (
            "Thumbs.db".to_string(),
            "Windows thumbnail cache".to_string(),
        ),
        (".DS_Store".to_string(), "macOS folder metadata".to_string()),
        (
            "desktop.ini".to_string(),
            "Windows folder settings".to_string(),
        ),
        // Temporary files
        ("~$".to_string(), "Office temporary files".to_string()),
        (".tmp".to_string(), "Temporary files".to_string()),
        (".temp".to_string(), "Temporary files".to_string()),
        (".swp".to_string(), "Vim swap files".to_string()),
        (".swo".to_string(), "Vim swap files".to_string()),
        (".orig".to_string(), "Merge conflict backup".to_string()),
        // Security sensitive
        (
            ".env".to_string(),
            "Environment variables (potentially sensitive)".to_string(),
        ),
        (
            ".env.local".to_string(),
            "Local environment variables".to_string(),
        ),
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
