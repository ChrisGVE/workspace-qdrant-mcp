//! Shared types and generators for property-based file monitoring tests
//!
//! Provides `FileOperation` enum with `Arbitrary` implementation, filename
//! sanitisation, and `prop_compose!` generators used across the split
//! property-file-monitoring test crates.

use proptest::prelude::*;

use workspace_qdrant_core::{config::Config, ChunkingConfig};

// ============================================================================
// FILE MONITORING PROPERTY GENERATORS
// ============================================================================

/// Random file operation for monitoring property tests.
#[derive(Debug, Clone)]
pub enum FileOperation {
    Create(String, String),  // filename, content
    Modify(String, String),  // filename, new_content
    Delete(String),          // filename
    Move(String, String),    // old_name, new_name
    CreateDirectory(String), // dirname
    DeleteDirectory(String), // dirname
}

impl Arbitrary for FileOperation {
    type Parameters = ();
    type Strategy = BoxedStrategy<Self>;

    fn arbitrary_with(_args: ()) -> Self::Strategy {
        prop_oneof![
            (any::<String>(), any::<String>()).prop_map(|(name, content)| {
                FileOperation::Create(sanitize_filename(name), content)
            }),
            (any::<String>(), any::<String>()).prop_map(|(name, content)| {
                FileOperation::Modify(sanitize_filename(name), content)
            }),
            any::<String>().prop_map(|name| FileOperation::Delete(sanitize_filename(name))),
            (any::<String>(), any::<String>()).prop_map(|(old, new)| {
                FileOperation::Move(sanitize_filename(old), sanitize_filename(new))
            }),
            any::<String>()
                .prop_map(|name| FileOperation::CreateDirectory(sanitize_filename(name))),
            any::<String>()
                .prop_map(|name| FileOperation::DeleteDirectory(sanitize_filename(name))),
        ]
        .boxed()
    }
}

/// Sanitize filename for cross-platform compatibility.
fn sanitize_filename(name: String) -> String {
    name.chars()
        .filter(|c| c.is_alphanumeric() || *c == '_' || *c == '-' || *c == '.')
        .take(50) // Limit length
        .collect::<String>()
        .trim_start_matches('.')
        .to_string()
        .trim()
        .to_string()
        .replace("", "file") // Handle empty string
}

// Generate random file patterns for include/exclude testing.
prop_compose! {
    pub fn arb_file_pattern()(
        pattern_type in prop_oneof!["glob", "extension", "directory", "exact"],
        pattern in "[a-zA-Z0-9*?._/-]{1,20}",
    ) -> String {
        match pattern_type.as_str() {
            "glob" => format!("*.{}", pattern.replace("*", "").replace("?", "")),
            "extension" => format!(".{}", pattern.replace(".", "")),
            "directory" => format!("{}/", pattern.replace("/", "")),
            "exact" => pattern,
            _ => "*.txt".to_string(),
        }
    }
}

// Generate random concurrent file operations.
prop_compose! {
    pub fn arb_concurrent_operations()(
        operations in prop::collection::vec(any::<FileOperation>(), 1..20),
        concurrent_count in 1..10usize,
    ) -> (Vec<FileOperation>, usize) {
        (operations, concurrent_count)
    }
}

// Generate random processing configurations.
prop_compose! {
    pub fn arb_processing_config()(
        max_concurrent in 1..20usize,
        timeout_ms in 100..30000u64,
        chunk_size in 50..5000usize,
        overlap_size in 0..1000usize,
    ) -> (Config, ChunkingConfig) {
        let config = Config {
            max_concurrent_tasks: Some(max_concurrent),
            default_timeout_ms: Some(timeout_ms),
            ..Default::default()
        };
        let chunking_config = ChunkingConfig {
            chunk_size,
            overlap_size: std::cmp::min(overlap_size, chunk_size / 4),
            preserve_paragraphs: true,
            ..ChunkingConfig::default()
        };
        (config, chunking_config)
    }
}

/// Simple pattern matching helper (basic implementation for testing).
pub fn matches_pattern(filename: &str, pattern: &str) -> bool {
    if pattern.contains('*') {
        // Basic glob matching
        let prefix = pattern.split('*').next().unwrap_or("");
        let suffix = pattern.split('*').next_back().unwrap_or("");
        filename.starts_with(prefix) && filename.ends_with(suffix)
    } else if pattern.starts_with('.') {
        // Extension matching
        filename.ends_with(pattern)
    } else if pattern.ends_with('/') {
        // Directory matching (not applicable to filenames)
        false
    } else {
        // Exact matching
        filename == pattern
    }
}
