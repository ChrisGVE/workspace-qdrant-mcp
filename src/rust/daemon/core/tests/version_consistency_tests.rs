//! Version consistency tests
//!
//! Validates that hardcoded version strings in configuration defaults
//! match the actual crate versions from Cargo.lock. These tests catch
//! drift that could occur if someone updates a dependency without
//! updating the corresponding configuration defaults.

use workspace_qdrant_core::config::GrammarConfig;
use workspace_qdrant_core::tree_sitter::grammar_cache::tree_sitter_runtime_version;
use workspace_qdrant_core::tree_sitter::version_checker::tree_sitter_version_string;

/// Verify that GrammarConfig's default tree_sitter_version matches the
/// actual tree-sitter crate version derived from Cargo.lock via build.rs.
///
/// If this test fails, it means the build.rs env var derivation is broken.
#[test]
fn tree_sitter_config_default_matches_crate_version() {
    let config = GrammarConfig::default();
    let build_rs_version = env!("TREE_SITTER_VERSION_MAJOR_MINOR");

    assert_eq!(
        config.tree_sitter_version, build_rs_version,
        "GrammarConfig default tree_sitter_version ('{}') does not match \
         build.rs-derived TREE_SITTER_VERSION_MAJOR_MINOR ('{}')",
        config.tree_sitter_version, build_rs_version,
    );
}

/// Verify that tree_sitter_runtime_version() (used for cache paths)
/// matches the config default version.
#[test]
fn tree_sitter_runtime_version_matches_config_default() {
    let config = GrammarConfig::default();
    let runtime_version = tree_sitter_runtime_version();

    assert_eq!(
        runtime_version, config.tree_sitter_version,
        "tree_sitter_runtime_version() ('{}') does not match \
         GrammarConfig default tree_sitter_version ('{}')",
        runtime_version, config.tree_sitter_version,
    );
}

/// Verify that the full version string from version_checker starts with
/// the major.minor version used in config.
#[test]
fn tree_sitter_full_version_starts_with_config_major_minor() {
    let config = GrammarConfig::default();
    let full_version = tree_sitter_version_string();

    assert!(
        full_version.starts_with(&config.tree_sitter_version),
        "Full tree-sitter version '{}' does not start with config \
         major.minor version '{}'",
        full_version,
        config.tree_sitter_version,
    );
}
